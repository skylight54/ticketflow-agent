"""评测向量检索的文档命中率；无答案用例仅观察，不计算拒答准确率。"""

import argparse
import json
from pathlib import Path

from knowledge_chunks import load_chunks


BASE_DIR = Path(__file__).resolve().parent
CASES_DIR = BASE_DIR / "cases"
RESULTS_DIR = BASE_DIR / "results"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def check_length(model, text: str, label: str) -> None:
    count = len(model.tokenizer(
        text, add_special_tokens=True, truncation=False,
    )["input_ids"])
    if count > model.max_seq_length:
        raise ValueError(
            f"{label} 有 {count} tokens，超过上限 {model.max_seq_length}，请先切分。"
        )


def evaluate_cases(
    cases: list[dict], model, chunks: list[dict], *, reranker=None,
) -> list[dict]:
    if not chunks:
        raise ValueError("知识库没有可检索的块。")

    for chunk in chunks:
        check_length(model, chunk["text"], chunk["chunk_id"])
    queries = [QUERY_PREFIX + case["question"].strip() for case in cases]
    for index, query in enumerate(queries, start=1):
        check_length(model, query, f"第 {index} 个问题")

    # 同一轮评测只编码一次知识库，所有问题共用这份矩阵。
    document_vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_vectors = model.encode(
        queries, normalize_embeddings=True, convert_to_numpy=True,
    )
    results = []
    for case, query_vector in zip(cases, query_vectors):
        scores = query_vector @ document_vectors.T
        rerank_by_index = {}
        candidates = []
        candidate_chunk_hit = None
        if reranker is None:
            indices = scores.argsort()[::-1][:3]
        else:
            candidate_indices = scores.argsort()[::-1][:5]
            expected_chunks = set(case["expected_chunk_ids"])

            candidate_chunk_hit = (
                any(
                    chunks[index]["chunk_id"] in expected_chunks
                    for index in candidate_indices
                )
                if case["expected_doc_id"] is not None
                else None
            )
            pairs = [
                (case["question"], chunks[index]["text"])
                for index in candidate_indices
            ]
            for pair, index in zip(pairs, candidate_indices):
                encoded = reranker.tokenizer(
                    pair[0], pair[1],
                    add_special_tokens=True,
                    truncation=False,
                )
                if len(encoded["input_ids"]) > reranker.max_length:
                    raise ValueError(
                        f"{chunks[index]['chunk_id']} 的问题与正文组合过长"
                    )

            rerank_scores = reranker.predict(pairs)
            rerank_by_index = {
                int(index): float(score)
                for index, score in zip(candidate_indices, rerank_scores)
            }
            ranked = sorted(
                zip(candidate_indices, rerank_scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )[:3]
            indices = [index for index, _ in ranked]
            candidates = [
                {
                    "chunk_id": chunks[index]["chunk_id"],
                    "doc_id": chunks[index]["doc_id"],
                    "embedding_score": float(scores[index]),
                    "rerank_score": rerank_by_index[int(index)],
                }
                for index in candidate_indices
            ]
        hits = [
            {
                "chunk_id": chunks[index]["chunk_id"],
                "doc_id": chunks[index]["doc_id"],
                "section": chunks[index]["section"],
                "embedding_score": float(scores[index]),
                "rerank_score": rerank_by_index.get(int(index)),
            }
            for index in indices
        ]
        expected = case["expected_doc_id"]
        expected_chunks = set(case["expected_chunk_ids"])
        chunk_hit_at_1 = hits[0]["chunk_id"] in expected_chunks
        chunk_hit_at_3 = any(
            hit["chunk_id"] in expected_chunks
            for hit in hits
        )
        results.append({
            "question": case["question"],
            "expected_doc_id": expected,
            "status": "evaluated" if expected is not None else "pending_refusal",
            "hit_at_1": hits[0]["doc_id"] == expected if expected is not None else None,
            "hit_at_3": any(hit["doc_id"] == expected for hit in hits)
            if expected is not None else None,
            "top_score": (
                hits[0]["rerank_score"]
                if reranker is not None
                else hits[0]["embedding_score"]
            ),
            "hits": hits,
            "expected_chunk_ids": case["expected_chunk_ids"],
            "chunk_hit_at_1": chunk_hit_at_1 if expected is not None else None,
            "chunk_hit_at_3": chunk_hit_at_3 if expected is not None else None,
            "candidates": candidates,
            "candidate_chunk_hit_at_5": candidate_chunk_hit,
        })
    return results


def summarize_results(results: list[dict]) -> dict:
    evaluated = [item for item in results if item["expected_doc_id"] is not None]
    total = len(evaluated)
    hit_1 = sum(item["hit_at_1"] for item in evaluated)
    hit_3 = sum(item["hit_at_3"] for item in evaluated)
    chunk_hit_1 = sum(item["chunk_hit_at_1"] for item in evaluated)
    chunk_hit_3 = sum(item["chunk_hit_at_3"] for item in evaluated)
    candidate_evaluated = [
        item for item in evaluated
        if item["candidate_chunk_hit_at_5"] is not None
    ]
    candidate_total = len(candidate_evaluated)
    candidate_hits = sum(
        item["candidate_chunk_hit_at_5"]
        for item in candidate_evaluated
    )
    return {
        "total": len(results),
        "answerable_total": total,
        "pending_refusal_total": len(results) - total,
        "hit_at_1_count": hit_1,
        "hit_at_3_count": hit_3,
        "doc_hit_at_1": hit_1 / total if total else None,
        "doc_hit_at_3": hit_3 / total if total else None,
        "chunk_hit_at_1_count": chunk_hit_1,
        "chunk_hit_at_3_count": chunk_hit_3,
        "chunk_hit_at_1": chunk_hit_1 / total if total else None,
        "chunk_hit_at_3": chunk_hit_3 / total if total else None,
        "candidate_chunk_total": candidate_total,
        "candidate_chunk_hit_at_5_count": candidate_hits,
        "candidate_chunk_hit_at_5": (
            candidate_hits / candidate_total
            if candidate_total else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path,
                        default=CASES_DIR / "retrieval_holdout_cases.json")
    parser.add_argument("--output", type=Path,
                        default=RESULTS_DIR / "embedding_results.json")
    parser.add_argument("--offline", action="store_true", help="仅使用本地缓存模型")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("用例文件必须是非空 JSON 数组。")
    for index, case in enumerate(cases, start=1):
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("question"), str)
            or not case["question"].strip()
            or "expected_doc_id" not in case
            or (case["expected_doc_id"] is not None and (
                not isinstance(case["expected_doc_id"], str)
                or not case["expected_doc_id"].strip()
            ))
        ):
            raise ValueError(f"第 {index} 个用例需要非空 question 和 expected_doc_id（字符串或 null）。")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        local_files_only=args.offline,
    )
    reranker = None
    if args.rerank:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(
            "BAAI/bge-reranker-base",
            device="cpu",
            max_length=512,
            local_files_only=args.offline,
        )
    results = evaluate_cases(
        cases, model, load_chunks(), reranker=reranker,
    )
    summary = summarize_results(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": MODEL_NAME,
        "cases_path": str(args.cases.resolve()),
        "summary": summary,
        "results": results,
        "rerank_enabled": args.rerank,
        "reranker_model": "BAAI/bge-reranker-base" if args.rerank else None,
        "candidate_k": 5 if args.rerank else 3,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for item in results:
        print(f"\n问题：{item['question']}")
        if item["expected_doc_id"] is None:
            print(f"  待验证拒答；最高分={item['top_score']:.4f}（不计入命中率）")
        else:
            print(f"  预期：{item['expected_doc_id']} | Hit@1={item['hit_at_1']} | Hit@3={item['hit_at_3']}")
        for rank, hit in enumerate(item["hits"], start=1):
            rerank_text = (
                f"{hit['rerank_score']:.4f}"
                if hit["rerank_score"] is not None
                else "未启用"
            )
            print(
                f"  {rank}. {hit['chunk_id']} | {hit['section']} | "
                f"embedding={hit['embedding_score']:.4f} | "
                f"rerank={rerank_text}"
            )

    total = summary["answerable_total"]
    for k in (1, 3):
        rate = summary[f"doc_hit_at_{k}"]
        display = f"{rate:.1%}" if rate is not None else "N/A"
        print(f"DocHit@{k}：{display}（{summary[f'hit_at_{k}_count']}/{total}）")
    for k in (1, 3):
        rate = summary[f"chunk_hit_at_{k}"]
        display = f"{rate:.1%}" if rate is not None else "N/A"
        print(f"ChunkHit@{k}：{display}（{summary[f'chunk_hit_at_{k}_count']}/{total}）")
        print(f"结果文件：{args.output.resolve()}")


if __name__ == "__main__":
    main()
