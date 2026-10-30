import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

import db
from action_service import reject_proposed_action
from approval_graph import build_approval_graph
from approval_service import resume_approval, start_approval


class ApprovalServiceTests(unittest.TestCase):
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

        self.graph = build_approval_graph(InMemorySaver())
        self.config: RunnableConfig = {
            "configurable": {"thread_id": f"approval:{self.action_id}"}
        }

    def _restore_db_path(self):
        db.DB_PATH = self.original_path

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

    def start(self):
        result = start_approval(self.graph, self.sid, self.action_id)
        self.assertTrue(result["ok"], result)
        return result

    def test_start_pauses_and_keeps_ticket_unchanged(self):
        result = self.start()

        self.assertEqual(result["data"]["status"], "waiting")
        self.assertEqual(len(result["data"]["interrupts"]), 1)
        self.assertEqual(
            result["data"]["interrupts"][0]["question"],
            "是否同意将工单T109由待处理改为处理中？",
        )
        self.assertEqual(self.graph.get_state(self.config).next, ("wait",))

        # 只是停在确认节点，工单和建议都还没有被修改
        self.assertEqual(self.load_ticket()["status"], "待处理")
        self.assertEqual(self.load_action()["status"], "pending")

    def test_approve_executes_action(self):
        self.start()

        result = resume_approval(self.graph, self.sid, self.action_id, True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["status"], "executed")
        self.assertEqual(self.load_ticket()["status"], "处理中")
        self.assertEqual(self.load_action()["status"], "executed")
        self.assertEqual(self.graph.get_state(self.config).next, ())

    def test_reject_keeps_ticket_unchanged(self):
        self.start()

        result = resume_approval(self.graph, self.sid, self.action_id, False)

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["status"], "rejected")
        self.assertEqual(self.load_ticket()["status"], "待处理")
        self.assertEqual(self.load_action()["status"], "rejected")
        self.assertEqual(self.graph.get_state(self.config).next, ())

    def test_start_twice_returns_already_started(self):
        self.start()

        again = start_approval(self.graph, self.sid, self.action_id)

        self.assertFalse(again["ok"])
        self.assertEqual(again["error"]["code"], "APPROVAL_ALREADY_STARTED")
        # 没有产生第二个中断
        self.assertEqual(len(self.graph.get_state(self.config).interrupts), 1)
        self.assertEqual(self.graph.get_state(self.config).next, ("wait",))

    def test_start_after_completion_returns_already_started(self):
        self.start()
        resume_approval(self.graph, self.sid, self.action_id, True)

        again = start_approval(self.graph, self.sid, self.action_id)

        self.assertFalse(again["ok"])
        self.assertEqual(again["error"]["code"], "APPROVAL_ALREADY_STARTED")

    def test_resume_before_start(self):
        result = resume_approval(self.graph, self.sid, self.action_id, True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "APPROVAL_NOT_STARTED")
        self.assertEqual(self.load_ticket()["status"], "待处理")

    def test_resume_twice_returns_not_waiting(self):
        self.start()
        first = resume_approval(self.graph, self.sid, self.action_id, True)
        self.assertTrue(first["ok"])

        second = resume_approval(self.graph, self.sid, self.action_id, True)

        self.assertFalse(second["ok"])
        self.assertEqual(second["error"]["code"], "APPROVAL_NOT_WAITING")
        # 重复提交没有再次执行修改
        self.assertEqual(self.load_ticket()["status"], "处理中")
        self.assertEqual(self.load_action()["status"], "executed")

    def test_resume_with_non_bool_is_rejected(self):
        self.start()

        result = resume_approval(
            self.graph, self.sid, self.action_id, cast(Any, 1)
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "INVALID_ARGUMENT")
        # 参数错误没有破坏等待中的审批
        self.assertEqual(self.graph.get_state(self.config).next, ("wait",))
        self.assertEqual(self.load_ticket()["status"], "待处理")

    def test_empty_arguments_are_rejected(self):
        started = start_approval(self.graph, "  ", self.action_id)
        resumed = resume_approval(self.graph, self.sid, "  ", True)

        self.assertEqual(started["error"]["code"], "INVALID_ARGUMENT")
        self.assertEqual(resumed["error"]["code"], "INVALID_ARGUMENT")

    def test_action_from_another_session_is_rejected(self):
        other_sid = db.create_session()

        started = start_approval(self.graph, other_sid, self.action_id)
        resumed = resume_approval(self.graph, other_sid, self.action_id, True)

        self.assertEqual(started["error"]["code"], "ACTION_NOT_FOUND")
        self.assertEqual(resumed["error"]["code"], "ACTION_NOT_FOUND")

    def test_non_pending_action_cannot_start(self):
        # 还没发起审批，action 就已经被别处拒绝
        self.assertTrue(reject_proposed_action(self.sid, self.action_id)["ok"])

        result = start_approval(self.graph, self.sid, self.action_id)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "ACTION_NOT_PENDING")

    def test_external_reject_then_resume_approve_fails(self):
        self.start()
        self.assertTrue(reject_proposed_action(self.sid, self.action_id)["ok"])

        result = resume_approval(self.graph, self.sid, self.action_id, True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "ACTION_REJECTED")
        self.assertEqual(self.load_ticket()["status"], "待处理")

    def test_external_reject_then_resume_reject_is_idempotent(self):
        self.start()
        self.assertTrue(reject_proposed_action(self.sid, self.action_id)["ok"])

        result = resume_approval(self.graph, self.sid, self.action_id, False)

        # 当前实现允许重复拒绝：结果仍是 rejected
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["status"], "rejected")
        self.assertEqual(self.load_ticket()["status"], "待处理")

    def test_ticket_changed_while_waiting_marks_action_stale(self):
        self.start()

        # 等待审批期间，另一个操作抢先完成了同一个状态流转
        changed = db.UpdateTicketStatus("T109", "待处理", "处理中")
        self.assertTrue(changed)

        result = resume_approval(self.graph, self.sid, self.action_id, True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "ACTION_STALE")
        self.assertEqual(self.load_action()["status"], "stale")
        self.assertEqual(self.load_ticket()["status"], "处理中")
        self.assertEqual(self.graph.get_state(self.config).next, ())


if __name__ == "__main__":
    unittest.main()
