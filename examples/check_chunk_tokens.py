from sentence_transformers import SentenceTransformer

from knowledge_chunks import load_chunks


def main():
    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu",
    )
    limit = model.max_seq_length
    print(f"模型编码长度上限：{limit} tokens")

    oversized = []

    for chunk in load_chunks():
        encoded = model.tokenizer(
            chunk["text"],
            add_special_tokens=True,
            truncation=False,
        )
        count = len(encoded["input_ids"])
        status = "OVER" if count > limit else "OK"

        print(
            f"[{status}] {chunk['chunk_id']} "
            f"{chunk['section']}：{count} tokens"
        )

        if count > limit:
            oversized.append(chunk["chunk_id"])

    if oversized:
        raise SystemExit(
            f"存在超长块，需要继续切分：{', '.join(oversized)}"
        )

    print("所有块均未超过编码长度上限。")


if __name__ == "__main__":
    main()