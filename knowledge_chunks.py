"""按 Markdown 二级标题分块；暂不负责 token 限制或向量编码。"""

import re
from pathlib import Path

from knowledge_search import KNOWLEDGE_DIR, load_documents


METADATA_FIELDS = ("文档编号", "适用场景", "关键词", "相关文档")
SECTION_HEADING = re.compile(r"^##[ \t]+(.+?)\s*$")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def _split_sections(content: str) -> list[tuple[str, str]]:
    """保留正文中的列表、表格和代码块，不把代码中的 ## 当成标题。"""
    sections = []
    section = "文档正文"
    body: list[str] = []
    in_preamble = True
    fence_char = ""
    fence_length = 0

    for index, line in enumerate(content.splitlines()):
        # load_documents 已验证第一行是文档标题，后面统一给每块补上。
        if index == 0 and line.startswith("# "):
            continue

        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_length = marker[0], len(marker)
            elif (
                marker[0] == fence_char
                and len(marker) >= fence_length
                and not line[fence.end():].strip()
            ):
                fence_char, fence_length = "", 0
            body.append(line)
            continue

        heading = SECTION_HEADING.match(line) if not fence_char else None
        if heading:
            text = "\n".join(body).strip()
            if text:
                sections.append((section, text))
            section = heading.group(1).strip()
            body = []
            in_preamble = False
            continue

        if in_preamble and not fence_char:
            stripped = line.strip()
            if any(
                stripped.startswith((field + "：", field + ":"))
                for field in METADATA_FIELDS
            ):
                continue
        body.append(line)

    text = "\n".join(body).strip()
    if text:
        sections.append((section, text))
    return sections


def load_chunks(directory: Path = KNOWLEDGE_DIR) -> list[dict]:
    """加载知识文档并按章节生成带来源的块，空章节不生成块。"""
    chunks = []
    for document in load_documents(directory):
        for index, (section, body) in enumerate(
            _split_sections(document["content"]), start=1
        ):
            chunks.append({
                "chunk_id": f"{document['doc_id']}:{index}",
                "doc_id": document["doc_id"],
                "title": document["title"],
                "section": section,
                "source": document["source"],
                "text": f"{document['title']}\n{section}\n{body}",
            })
    return chunks


if __name__ == "__main__":
    chunks = load_chunks()
    for chunk in chunks:
        print(
            f"{chunk['chunk_id']} | {chunk['section']} | "
            f"{len(chunk['text'])} 字符 | {chunk['source']}"
        )
    print(f"共 {len(chunks)} 个块；字符数不等于 token 数。")
