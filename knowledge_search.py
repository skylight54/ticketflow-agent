from pathlib import Path


KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

# 关键词命中比标题命中更可靠，权重高一些
KEYWORD_WEIGHT = 3
TITLE_WEIGHT = 1


def _read_field(path: Path, lines: list[str], name: str) -> str:
    prefix = f"{name}："
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise ValueError(f"{path.name} 缺少字段：{name}")


def _read_title(path: Path, lines: list[str]) -> str:
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{path.name} 第一行必须是 # 标题")
    return lines[0][2:].strip()


def load_documents(directory: Path = KNOWLEDGE_DIR) -> list[dict]:
    """读取目录下全部 Markdown，按文件名排序，保证检索结果稳定。"""
    documents = []

    for path in sorted(Path(directory).glob("*.md")):
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()

        documents.append({
            "doc_id": _read_field(path, lines, "文档编号"),
            "title": _read_title(path, lines),
            "keywords": [
                word.strip()
                for word in _read_field(path, lines, "关键词").split("、")
                if word.strip()
            ],
            "content": content,
            "source": path.name,
        })

    return documents


def match_document(question: str, document: dict) -> tuple[int, list[str]]:
    """返回（得分，命中的关键词）；分数只用于排序，不代表相关度绝对值。"""
    normalized_question = question.casefold()

    matched = [
        keyword
        for keyword in document["keywords"]
        if keyword.casefold() in normalized_question
    ]

    score = len(matched) * KEYWORD_WEIGHT

    if (
        document["title"]
        and document["title"].casefold() in normalized_question
    ):
        score += TITLE_WEIGHT

    return score, matched

QUERY_EXPANSIONS = {
    "进不去": "登录",
    "登不上": "登录",
    "钱扣走": "扣款",
    "付款成功": "支付",
    "结单": "已完成",
}

def expand_query(question: str) -> str:
    normalized = question.strip().casefold()
    additions = {
        keyword
        for phrase, keyword in QUERY_EXPANSIONS.items()
        if phrase.casefold() in normalized
    }

    return " ".join([normalized, *sorted(additions)]).strip()

def search_knowledge(
    question: str,
    top_k: int = 3,
    *,
    expand: bool = False,
) -> list[dict]:
    """按关键词检索知识库，返回得分最高的前 top_k 篇；没有命中返回空列表。"""
    question = question.strip()
    if question == "":
        return []
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if expand:
        question = expand_query(question)
    result = []

    for document in load_documents():
        score, matched = match_document(question, document)
        if score == 0:
            continue
        result.append({
            "doc_id": document["doc_id"],
            "title": document["title"],
            "score": score,
            "matched_keywords": matched,
            "content": document["content"],
            "source": document["source"],
        })

    # 同分时按文档编号排序，避免依赖文件系统顺序
    result.sort(key=lambda item: (-item["score"], item["doc_id"]))
    return result[:top_k]
