import json
from pathlib import Path
import argparse

from knowledge_search import search_knowledge

BASE_DIR = Path(__file__).resolve().parent
CASES_DIR = BASE_DIR / "cases"
RESULTS_DIR = BASE_DIR / "results"
CASES_PATH = CASES_DIR / "retrieval_cases.json"
RESULTS_PATH = RESULTS_DIR / "eval_retrieval_results.json"


def score_retrieval_case(case: dict, hits: list[dict]) -> list[str]:
    """只看第一条结果；返回失败原因，空列表表示通过。"""
    expected = case.get("expected_doc_id")
    actual = hits[0]["doc_id"] if hits else None

    if expected is None:
        if actual is not None:
            return [f"预期无匹配文档，实际返回 {actual}"]
        return []

    if actual is None:
        return [f"未检索到任何文档，预期 {expected}"]

    if actual != expected:
        return [f"检索文档不匹配，预期 {expected}，实际 {actual}"]

    return []


def evaluate_cases(
    cases: list[dict],
    *,
    expand: bool = False,
) -> list[dict]:
    results = []

    for case in cases:
        question = case["question"]
        hits = search_knowledge(question, top_k=1, expand=expand)
        reasons = score_retrieval_case(case, hits)
        top = hits[0] if hits else None

        results.append({
            "question": question,
            "expected_doc_id": case.get("expected_doc_id"),
            "actual_doc_id": top["doc_id"] if top else None,
            "score": top["score"] if top else 0,
            "matched_keywords": top["matched_keywords"] if top else [],
            "passed": not reasons,
            "reasons": reasons,
        })

    return results


def summarize_results(results: list[dict]) -> dict:
    """汇总评测结果；空列表返回全 0，避免除零。"""
    total = len(results)
    passed = sum(1 for item in results if item["passed"])

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="知识库检索评测")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--expand", action="store_true")
    args = parser.parse_args()

    print(f"用例文件：{args.cases.resolve()}")
    print(f"结果文件：{args.output.resolve()}")

    with args.cases.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"查询扩展：{args.expand}")
    results = evaluate_cases(cases, expand=args.expand)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 后面的汇总和打印代码保持原样

    summary = summarize_results(results)
    print(f"通过 {summary['passed']}/{summary['total']}")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['question']}")
        print(
            f"  预期 {item['expected_doc_id']}，"
            f"实际 {item['actual_doc_id']}（score={item['score']}）"
        )
        for reason in item["reasons"]:
            print(f"  - {reason}")
    print(
        f"用例通过率：{summary['pass_rate']:.1%}"
        f"（{summary['passed']}/{summary['total']}）"
    )


if __name__ == "__main__":
    main()
