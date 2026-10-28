import json
import os
import tempfile
import unittest
import sqlite3
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import db
import agent_graph as agent
from app import app
from session_service import chat_session
from tool_runner import execute_tool


def tool_message(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }],
    }


class DBTestCase(unittest.TestCase):
    """读取数据库后先断言存在，避免 Pylance 的 Optional 下标报错。"""

    def load_session(self, session_id):
        session = db.load_session(session_id)
        self.assertIsNotNone(session, f"会话不存在：{session_id}")
        assert session is not None
        return session

    def load_ticket(self, ticket_id):
        ticket = db.SelectTicket(ticket_id)
        self.assertIsNotNone(ticket, f"工单不存在：{ticket_id}")
        assert ticket is not None
        return ticket

    def load_action(self, action_id):
        action = db.load_proposed_action(action_id)
        self.assertIsNotNone(action, f"建议不存在：{action_id}")
        assert action is not None
        return action


class QueryLoopTests(DBTestCase):
    def setUp(self):
        self.original_path = db.DB_PATH
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        db.DB_PATH = Path(self.directory.name) / "test.db"
        db.init_db()
        self.sid = db.create_session()
        db.InsertTicket("T109", "待处理", "技术支持", "无法登录")

    def tearDown(self):
        db.DB_PATH = self.original_path
        self.directory.cleanup()

    def proposal(self):
        return tool_message(
            "propose_status_change",
            {"ticket_id": "T109", "new_status": "处理中"},
            "propose_1",
        )

    def assert_no_commit(self):
        self.assertEqual(self.load_session(self.sid)["version"], 0)
        self.assertEqual(self.load_session(self.sid)["messages"], [])
        self.assertEqual(self.load_ticket("T109")["status"], "待处理")

    def test_query_then_propose_then_answer(self):
        replies = [
            tool_message("get_ticket", {"ticket_id": "T109"}, "query_1"),
            self.proposal(),
            {"role": "assistant", "content": "建议改为处理中，等待确认。"},
        ]
        snapshots = []

        def send(messages, allow_tools=True, allow_proposals=False):
            self.assertTrue(allow_tools)
            self.assertTrue(allow_proposals)
            snapshots.append(deepcopy(messages))
            return replies.pop(0)

        with patch.object(agent, "send_messages", side_effect=send) as model:
            result = chat_session(self.sid, "请开始处理 T109")
        self.assertTrue(result["ok"])
        self.assertEqual(model.call_count, 3)
        self.assertEqual(
            [trace["name"] for trace in result["data"]["tool_results"]],
            ["get_ticket", "propose_status_change"],
        )
        action = result["data"]["actions"][0]
        self.assertEqual(self.load_action(action["action_id"]), action)
        self.assertEqual(action["status"], "pending")
        self.assertEqual(self.load_ticket("T109")["status"], "待处理")
        self.assertEqual(snapshots[1][-1]["tool_call_id"], "query_1")
        self.assertEqual(snapshots[2][-1]["tool_call_id"], "propose_1")

    def test_failed_proposal_returns_business_error_without_appending(self):
        cases = [
            ({"ticket_id": "missing", "new_status": "处理中"}, "TICKET_NOT_FOUND"),
            ({"ticket_id": "T109", "new_status": "待处理"}, "NO_STATUS_CHANGE"),
            ({"ticket_id": "T109", "new_status": "已完成"}, "INVALID_STATUS_TRANSITION"),
        ]
        for arguments, code in cases:
            with self.subTest(code=code):
                actions = []
                result = execute_tool(
                    json.dumps({"tool": "propose_status_change", "args": arguments}),
                    session_id=self.sid,
                    pending_actions=actions,
                )
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"]["code"], code)
                self.assertEqual(actions, [])
        self.assert_no_commit()

    def test_duplicate_proposal_reuses_the_persisted_id(self):
        first = self.proposal()
        second = deepcopy(first)
        second["tool_calls"][0]["id"] = "propose_2"
        with patch.object(agent, "send_messages", side_effect=[
            first, second, {"role": "assistant", "content": "等待用户确认。"},
        ]):
            result = chat_session(self.sid, "开始处理 T109")
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["data"]["actions"]), 1)
        action = result["data"]["actions"][0]
        for trace in result["data"]["tool_results"]:
            self.assertEqual(trace["result"]["data"]["action_id"], action["action_id"])
        self.assertEqual(self.load_action(action["action_id"]), action)
        self.assertEqual(self.load_ticket("T109")["status"], "待处理")

    def test_summary_failure_leaves_no_proposal(self):
        seen = []

        def send(messages, **kwargs):
            self.assertTrue(kwargs["allow_tools"])
            self.assertTrue(kwargs["allow_proposals"])
            if not seen:
                seen.append(True)
                return self.proposal()
            action = json.loads(messages[-1]["content"])["data"]
            seen.append(action["action_id"])
            raise TimeoutError("模拟总结失败")

        with patch.object(agent, "send_messages", side_effect=send):
            with self.assertRaises(TimeoutError):
                chat_session(self.sid, "开始处理 T109")
        self.assert_no_commit()
        self.assertIsNone(db.load_proposed_action(seen[-1]))

    def test_conflict_leaves_no_proposal(self):
        seen = []
        other_messages = [{"role": "user", "content": "其他请求"}]

        def send(messages, **kwargs):
            self.assertTrue(kwargs["allow_tools"])
            self.assertTrue(kwargs["allow_proposals"])
            if not seen:
                seen.append(True)
                return self.proposal()
            seen.append(json.loads(messages[-1]["content"])["data"]["action_id"])
            self.assertTrue(db.save_session(self.sid, other_messages, 0))
            return {"role": "assistant", "content": "等待确认"}

        with patch.object(agent, "send_messages", side_effect=send):
            result = chat_session(self.sid, "开始处理 T109")
        self.assertEqual(result["error"]["code"], "SESSION_CONFLICT")
        self.assertEqual(self.load_session(self.sid)["messages"], other_messages)
        self.assertIsNone(db.load_proposed_action(seen[-1]))

    def test_plain_text_tool_markup_is_not_executed_or_saved(self):
        message = {"role": "assistant", "content": "<｜｜DSML｜｜ calls>"}
        with patch.object(agent, "send_messages", return_value=message), patch.object(
            agent, "execute_tool"
        ) as execute:
            with self.assertRaises(agent.AgentResponseError) as caught:
                chat_session(self.sid, "开始处理 T109")
        self.assertEqual(caught.exception.code, "MODEL_PROTOCOL_ERROR")
        execute.assert_not_called()
        self.assert_no_commit()

    def test_round_limit_does_not_commit_pending_actions(self):
        action_ids = []

        def send(messages, **kwargs):
            self.assertTrue(kwargs["allow_tools"])
            self.assertTrue(kwargs["allow_proposals"])
            if len(messages) == 2:
                return self.proposal()
            if not action_ids:
                action_ids.append(json.loads(messages[-1]["content"])["data"]["action_id"])
            return tool_message("get_ticket", {"ticket_id": "T109"}, f"query_{len(messages)}")

        with patch.object(agent, "send_messages", side_effect=send) as model:
            with self.assertRaises(agent.AgentResponseError) as caught:
                chat_session(self.sid, "开始处理 T109")
        self.assertEqual(caught.exception.code, "MODEL_STEP_LIMIT")
        self.assertEqual(model.call_count, agent.MAX_MODEL_ROUNDS)
        self.assert_no_commit()
        self.assertIsNone(db.load_proposed_action(action_ids[0]))

    def test_tool_call_limit_stops_before_execution(self):
        message = {"role": "assistant", "tool_calls": [
            tool_message("get_ticket", {"ticket_id": "T109"}, str(i))["tool_calls"][0]
            for i in range(agent.MAX_TOOL_CALLS + 1)
        ]}
        with patch.object(agent, "send_messages", return_value=message), patch.object(
            agent, "execute_tool"
        ) as execute:
            with self.assertRaises(agent.AgentResponseError) as caught:
                chat_session(self.sid, "查询工单")
        self.assertEqual(caught.exception.code, "MODEL_STEP_LIMIT")
        execute.assert_not_called()
        self.assert_no_commit()


class QueryLoopAPITests(DBTestCase):
    def test_startup_creates_missing_tables_and_preserves_old_data(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "old.db"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("""
                    CREATE TABLE tickets (
                        ticket_id TEXT PRIMARY KEY NOT NULL,
                        status TEXT NOT NULL, owner TEXT NOT NULL, summary TEXT NOT NULL
                    )
                """)
                connection.execute("""
                    CREATE TABLE sessions (
                        session_id TEXT PRIMARY KEY NOT NULL,
                        messages TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 0
                    )
                """)
                connection.execute("INSERT INTO tickets VALUES (?,?,?,?)", (
                    "old-ticket", "待处理", "技术支持", "旧工单",
                ))
                connection.execute("INSERT INTO sessions VALUES (?,?,?)", (
                    "old-session", "[]", 2,
                ))
                connection.commit()
            with patch.object(db, "DB_PATH", path), TestClient(app):
                self.assertEqual(self.load_ticket("old-ticket")["summary"], "旧工单")
                self.assertEqual(self.load_session("old-session")["version"], 2)
                action_id = db.insert_proposed_action(
                    "old-session", "old-ticket", "待处理", "处理中",
                )
                self.assertEqual(self.load_action(action_id)["status"], "pending")

    def test_protocol_error_returns_502_for_both_routes(self):
        for target, url in [
            ("app.run_query", "/assistant/query"),
            ("app.chat_session", "/sessions/session-test/messages"),
        ]:
            with self.subTest(url=url), patch.dict(
                os.environ, {"DEEPSEEK_API_KEY": "test-only"}
            ), patch(
                target,
                side_effect=agent.AgentResponseError("MODEL_PROTOCOL_ERROR", "模型格式异常"),
            ), TestClient(app) as client:
                response = client.post(url, json={"question": "查询工单"})
            self.assertEqual(response.status_code, 502)
            self.assertEqual(response.json()["error"]["code"], "MODEL_PROTOCOL_ERROR")
