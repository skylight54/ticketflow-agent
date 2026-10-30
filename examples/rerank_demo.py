from sentence_transformers import CrossEncoder

from knowledge_chunks import load_chunks


def main():
    model = CrossEncoder(
        "BAAI/bge-reranker-base",
        device="cpu",
        max_length=512,
    )
    chunks = load_chunks()
    question = "输完口令后始终进不了系统，应该先检查什么？"

    # 重排序使用原始问题，不加 embedding 的查询前缀。
    pairs = [(question, chunk["text"]) for chunk in chunks]

    # 新模型的 tokenizer 不同，需要重新检查输入长度。
    for pair, chunk in zip(pairs, chunks):
        encoded = model.tokenizer(
            pair[0], pair[1],
            add_special_tokens=True,
            truncation=False,
        )
        if len(encoded["input_ids"]) > 512:
            raise ValueError(f"{chunk['chunk_id']} 的问题与正文组合过长")

    scores = model.predict(pairs, show_progress_bar=True)
    indices = scores.argsort()[::-1][:5]

    for rank, index in enumerate(indices, start=1):
        chunk = chunks[index]
        print(
            f"{rank}. score={float(scores[index]):.4f} | "
            f"{chunk['chunk_id']} | {chunk['section']}"
        )


if __name__ == "__main__":
    main()