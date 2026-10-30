from sentence_transformers import SentenceTransformer

from knowledge_chunks import load_chunks


QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def main():
    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu",
    )
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("知识库没有可检索的块。")

    # 知识库向量只生成一次，后续问题重复使用。
    document_vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    while True:
        question = input("\n请输入问题（输入 exit 退出）：").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        query_text = QUERY_PREFIX + question

        # 查询也要检查长度，避免模型静默截断。
        token_count = len(model.tokenizer(
            query_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"])

        if token_count > model.max_seq_length:
            print("问题过长，请缩短后重试。")
            continue

        query_vector = model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = (query_vector @ document_vectors.T)[0]
        indices = scores.argsort()[::-1][:3]

        for rank, index in enumerate(indices, start=1):
            chunk = chunks[index]
            print(
                f"{rank}. score={float(scores[index]):.4f} | "
                f"{chunk['chunk_id']} | {chunk['section']}"
            )
            print(chunk["text"][:160])


if __name__ == "__main__":
    main()