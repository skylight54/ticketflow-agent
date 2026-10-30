from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu",
)

question = "输完口令后始终进不了系统，应该先检查什么？"

passages = [
    "登录失败时，检查账号、密码、验证码和账号锁定状态。",
    "支付成功但订单未更新时，核对扣款记录和支付回调。",
    "待处理工单只能流转到处理中，不能直接完成。",
]

query_vector = model.encode(
    ["为这个句子生成表示以用于检索相关文章：" + question],
    normalize_embeddings=True,
)
document_vectors = model.encode(
    passages,
    normalize_embeddings=True,
)

scores = (query_vector @ document_vectors.T)[0]

for index in scores.argsort()[::-1]:
    print(f"{float(scores[index]):.4f}  {passages[index]}")