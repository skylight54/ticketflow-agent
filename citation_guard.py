import re

from deepseek_demo import AgentResponseError


# 大小写不敏感：模型可能写成 [kb-login-001]，编号集合本身仍然严格比对
CITATION_PATTERN = re.compile(
    r"\[(KB-[A-Z0-9]+(?:-[A-Z0-9]+)*)\]",
    re.IGNORECASE,
)


def _retrieved_documents(tool_results: list[dict]) -> dict[str, dict]:
    """本轮检索成功返回的文档，按大写编号索引；失败的结果不算依据。"""
    index: dict[str, dict] = {}

    for trace in tool_results:
        if not isinstance(trace, dict):
            continue
        if trace.get("name") != "search_knowledge":
            continue

        result = trace.get("result")
        if not isinstance(result, dict) or result.get("ok") is not True:
            continue

        data = result.get("data")
        items = data.get("documents") if isinstance(data, dict) else []
        for item in items or []:
            doc_id = item.get("doc_id") if isinstance(item, dict) else None
            if isinstance(doc_id, str) and doc_id:
                index.setdefault(doc_id.upper(), item)

    return index


def _allowed_doc_ids(tool_results: list[dict]) -> set[str]:
    return set(_retrieved_documents(tool_results))


def validate_citations(answer: str, tool_results: list[dict]) -> None:
    """回答只能引用本轮检索到的文档；检索到了文档就必须标注来源。"""
    allowed_ids = _allowed_doc_ids(tool_results)
    cited_ids = {match.upper() for match in CITATION_PATTERN.findall(answer)}

    unauthorized = cited_ids - allowed_ids
    if unauthorized:
        raise AgentResponseError(
            "INVALID_CITATION",
            f"回答引用了本轮未检索到的文档：{', '.join(sorted(unauthorized))}",
        )

    if allowed_ids and not cited_ids:
        raise AgentResponseError(
            "MISSING_CITATION",
            "检索到了文档却没有标注来源，本轮未提交",
        )
    
def build_sources(answer: str, tool_results: list[dict]) -> list[dict]:
    """按回答中的引用顺序返回来源；内容全部取自本轮工具结果，模型无法填写。"""
    # 校验不通过时直接抛出，来源构建不会掩盖引用问题
    validate_citations(answer, tool_results)
    index = _retrieved_documents(tool_results)

    sources = []
    seen = set()

    for match in CITATION_PATTERN.finditer(answer):
        doc_id = match.group(1).upper()
        if doc_id in seen:
            continue

        document = index.get(doc_id)
        if document is None:
            continue

        seen.add(doc_id)
        sources.append({
            "doc_id": document.get("doc_id"),
            "title": document.get("title") or "",
            "source": document.get("source") or "",
        })

    return sources
