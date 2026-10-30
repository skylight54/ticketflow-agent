import tempfile
import unittest
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import db
from action_service import confirm_proposed_action
from approval_graph import ApprovalState, build_approval_graph


class ApprovalGraphTests(unittest.TestCase):
    def setUp(self):
        self.original_path = db.DB_PATH
        self.directory = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        )
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self._restore_db_path)

        # 用和业务代码完全相同的建表/插入函数，避免手写 schema 漂移
        db.DB_PATH = Path(self.directory.name) / "test.db"
        db.init_db()
        self.sid = db.create_session()
        db.InsertTicket("T109", "待处理", "技术支持", "无法登录")
        self.action_id = db.insert_proposed_action(
            self.sid, "T109", "待处理", "处理中"
        )
        self.config: RunnableConfig = {
            "configurable": {"thread_id": "approval-001"},
        }

    def _restore_db_path(self):
        db.DB_PATH = self.original_path

    def start(self):
        """启动图，执行到 interrupt() 后暂停。"""
        graph = build_approval_graph(InMemorySaver())
        state: ApprovalState = {
            "session_id": self.sid,
            "action_id": self.action_id,
        }
        result = graph.invoke(state, self.config)
        return graph, result

    def load_ticket(self):
        ticket = db.SelectTicket("T109")
        self.assertIsNotNone(ticket)
        assert ticket is not None
        return ticket

    def load_action(self):
        action = db.load_proposed_action(self.action_id)
        self.assertIsNotNone(action)
        assert action is not None
        return action

    def test_start_pauses_before_approval(self):
        graph, result = self.start()

        interrupts = result["__interrupt__"]
        self.assertEqual(len(interrupts), 1)
        self.assertEqual(
            interrupts[0].value["question"],
            "是否同意将工单T109由待处理改为处理中？",
        )
        self.assertEqual(graph.get_state(self.config).next, ("wait",))

        # load 节点已经读到建议，但还没有执行任何修改
        self.assertEqual(result["action"]["action_id"], self.action_id)
        self.assertEqual(self.load_ticket()["status"], "待处理")
        self.assertEqual(self.load_action()["status"], "pending")

    def test_approve_executes_action(self):
        graph, _ = self.start()

        result = graph.invoke(Command(resume=True), self.config)

        self.assertTrue(result["result"]["ok"])
        self.assertEqual(result["result"]["data"]["status"], "executed")
        self.assertEqual(self.load_ticket()["status"], "处理中")
        self.assertEqual(self.load_action()["status"], "executed")
        self.assertEqual(graph.get_state(self.config).next, ())

    def test_reject_keeps_ticket_unchanged(self):
        graph, _ = self.start()

        result = graph.invoke(Command(resume=False), self.config)

        self.assertTrue(result["result"]["ok"])
        self.assertEqual(result["result"]["data"]["status"], "rejected")
        self.assertEqual(self.load_ticket()["status"], "待处理")
        self.assertEqual(self.load_action()["status"], "rejected")
        self.assertEqual(graph.get_state(self.config).next, ())

    def test_unknown_action_fails_before_interrupt(self):
        graph = build_approval_graph(InMemorySaver())
        with self.assertRaises(ValueError):
            graph.invoke(
                {"session_id": self.sid, "action_id": "missing"},
                self.config,
            )

    def test_action_from_another_session_is_rejected(self):
        other_sid = db.create_session()
        graph = build_approval_graph(InMemorySaver())
        with self.assertRaises(ValueError):
            graph.invoke(
                {"session_id": other_sid, "action_id": self.action_id},
                self.config,
            )

    def test_ticket_changed_while_waiting_marks_action_stale(self):
        graph, _ = self.start()

        # 等待审批期间，另一个操作抢先完成了同一个状态流转
        changed = db.UpdateTicketStatus("T109", "待处理", "处理中")
        self.assertTrue(changed)

        result = graph.invoke(Command(resume=True), self.config)

        # 审批没有执行：旧状态已经不存在，建议被标记为 stale
        self.assertFalse(result["result"]["ok"])
        self.assertEqual(result["result"]["error"]["code"], "ACTION_STALE")
        self.assertIsNone(result["result"]["data"])
        self.assertEqual(self.load_action()["status"], "stale")
        self.assertEqual(self.load_ticket()["status"], "处理中")
        self.assertEqual(graph.get_state(self.config).next, ())

    def test_resume_after_direct_confirm_does_not_revert_ticket(self):
        graph, _ = self.start()

        # 模拟“业务库已提交，但图检查点仍停在 wait”：
        # 不恢复图，直接走确认事务
        confirmed = confirm_proposed_action(self.sid, self.action_id)
        self.assertTrue(confirmed["ok"])
        self.assertEqual(confirmed["data"]["status"], "executed")
        self.assertEqual(self.load_action()["status"], "executed")
        self.assertEqual(self.load_ticket()["status"], "处理中")
        # 图确实还停在旧检查点
        self.assertEqual(graph.get_state(self.config).next, ("wait",))

        # 工单继续正常流转，离开建议里的目标状态
        self.assertTrue(db.UpdateTicketStatus("T109", "处理中", "已完成"))
        self.assertEqual(self.load_ticket()["status"], "已完成")

        # 从旧检查点恢复：重复确认必须幂等，不能把工单改回处理中
        result = graph.invoke(Command(resume=True), self.config)

        self.assertTrue(result["result"]["ok"])
        self.assertEqual(result["result"]["data"]["status"], "executed")
        self.assertEqual(self.load_action()["status"], "executed")
        self.assertEqual(self.load_ticket()["status"], "已完成")
        self.assertEqual(graph.get_state(self.config).next, ())


if __name__ == "__main__":
    unittest.main()
