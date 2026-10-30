import unittest

from citation_guard import build_sources, validate_citations
from deepseek_demo import AgentResponseError


def search_trace(*documents: str | dict, ok: bool = True) -> dict:
    """构造一条 search_knowledge 的工具结果，固定数据，不调用模型。"""
    items = [
        {"doc_id": document} if isinstance(document, str) else dict(document)
        for document in documents
    ]
    return {
        "name": "search_knowledge",
        "result": {
            "ok": ok,
            "data": {"documents": items} if ok else None,
            "error": (
                None
                if ok
                else {"code": "MODEL_TIMEOUT", "message": "检索失败"}
            ),
        },
    }


class CitationGuardTests(unittest.TestCase):
    """只认可本轮检索到的文档；固定数据，不调用模型。"""

    def test_cited_document_retrieved_this_turn_passes(self):
        # 检索到 KB-LOGIN-001，回答引用它 → 通过
        traces = [search_trace("KB-LOGIN-001")]
        answer = "请按登录规范排查 [KB-LOGIN-001]。"

        self.assertIsNone(validate_citations(answer, traces))

    def test_citing_document_not_retrieved_is_rejected(self):
        # 本轮只检索到 KB-LOGIN-001，却引用了支付规范 → INVALID_CITATION
        traces = [search_trace("KB-LOGIN-001")]
        answer = "请按退款规范处理 [KB-PAYMENT-001]。"

        with self.assertRaises(AgentResponseError) as context:
            validate_citations(answer, traces)

        self.assertEqual(context.exception.code, "INVALID_CITATION")
        self.assertIn("KB-PAYMENT-001", str(context.exception))

    def test_retrieved_but_not_cited_is_rejected(self):
        # 检索到了文档却没有任何引用 → MISSING_CITATION
        traces = [search_trace("KB-LOGIN-001")]
        answer = "请先确认账号是否存在，再检查验证码。"

        with self.assertRaises(AgentResponseError) as context:
            validate_citations(answer, traces)

        self.assertEqual(context.exception.code, "MISSING_CITATION")

    def test_no_retrieval_and_no_citation_passes(self):
        # 只查工单、没有检索，也不需要引用
        traces = [{
            "name": "get_ticket",
            "result": {"ok": True, "data": {"ticket_id": "T103"}, "error": None},
        }]

        self.assertIsNone(validate_citations("工单 T103 处于待处理状态。", traces))

    def test_lowercase_citation_is_accepted(self):
        # 大小写不敏感，编号本身仍在白名单内
        traces = [search_trace("KB-LOGIN-001")]

        self.assertIsNone(validate_citations("见 [kb-login-001]。", traces))

    def test_failed_retrieval_cannot_be_cited(self):
        # 检索失败的结果不能作为引用依据
        traces = [search_trace("KB-LOGIN-001", ok=False)]
        answer = "见 [KB-LOGIN-001]。"

        with self.assertRaises(AgentResponseError) as context:
            validate_citations(answer, traces)

        self.assertEqual(context.exception.code, "INVALID_CITATION")

    def test_one_unauthorized_citation_rejects_the_answer(self):
        # 合法引用和非法引用混在一起时，整条回答不通过
        traces = [search_trace("KB-LOGIN-001")]
        answer = "见 [KB-LOGIN-001] 和 [KB-PAYMENT-001]。"

        with self.assertRaises(AgentResponseError) as context:
            validate_citations(answer, traces)

        self.assertEqual(context.exception.code, "INVALID_CITATION")

    def test_citation_from_another_tool_is_not_allowed(self):
        # 只有 search_knowledge 的返回算依据，其它工具结果里的编号不算
        traces = [{
            "name": "get_ticket",
            "result": {
                "ok": True,
                "data": {"ticket_id": "T103", "status": "待处理"},
                "error": None,
            },
        }]

        with self.assertRaises(AgentResponseError) as context:
            validate_citations("见 [KB-LOGIN-001]。", traces)

        self.assertEqual(context.exception.code, "INVALID_CITATION")


class BuildSourcesTests(unittest.TestCase):
    """来源由程序从工具结果生成，按回答中的引用顺序去重。"""

    documents = [
        {
            "doc_id": "KB-LOGIN-001",
            "title": "无法登录处理规范",
            "source": "login.md",
        },
        {
            "doc_id": "KB-STATUS-001",
            "title": "工单状态流转规则",
            "source": "ticket_status.md",
        },
    ]

    def test_only_cited_documents_are_returned(self):
        # 检索回两篇，回答只引用一篇 → sources 只含被引用的那篇
        traces = [search_trace(*self.documents)]
        answer = "按登录规范排查 [KB-LOGIN-001]。"

        self.assertEqual(build_sources(answer, traces), [self.documents[0]])

    def test_order_follows_citation_in_answer(self):
        traces = [search_trace(*self.documents)]
        answer = "先看 [KB-STATUS-001]，再看 [KB-LOGIN-001]。"

        sources = build_sources(answer, traces)

        self.assertEqual(
            [item["doc_id"] for item in sources],
            ["KB-STATUS-001", "KB-LOGIN-001"],
        )

    def test_duplicate_citation_appears_once(self):
        traces = [search_trace(*self.documents)]
        answer = "见 [KB-LOGIN-001]，补充仍见 [KB-LOGIN-001]。"

        sources = build_sources(answer, traces)

        self.assertEqual(sources, [self.documents[0]])

    def test_no_citation_returns_empty_list(self):
        # 没有检索也没有引用 → 空列表
        self.assertEqual(build_sources("工单 T103 处于待处理状态。", []), [])

    def test_lowercase_citation_uses_document_data(self):
        traces = [search_trace(*self.documents)]
        answer = "见 [kb-login-001]。"

        self.assertEqual(build_sources(answer, traces), [self.documents[0]])

    def test_invalid_citation_raises_before_building(self):
        traces = [search_trace("KB-LOGIN-001")]

        with self.assertRaises(AgentResponseError) as context:
            build_sources("见 [KB-PAYMENT-001]。", traces)

        self.assertEqual(context.exception.code, "INVALID_CITATION")


if __name__ == "__main__":
    unittest.main()
