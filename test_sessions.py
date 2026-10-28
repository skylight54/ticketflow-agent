import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import db
from app import app
from session_service import chat_session


class SessionTestBase(unittest.TestCase):
    def setUp(self):
        self.original_db_path = db.DB_PATH
        self.temp_dir = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        )
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()


class SessionServiceTests(SessionTestBase):
    def test_blank_arguments_return_invalid_argument(self):
        session_id = db.create_session()

        cases = [
            ("", "查询 T103"),
            ("   ", "查询 T103"),
            (session_id, ""),
            (session_id, "   "),
        ]
        for sid, question in cases:
            with self.subTest(session_id=sid, question=question), patch(
                "session_service.run_query"
            ) as query:
                result = chat_session(sid, question)

                self.assertFalse(result["ok"])
                self.assertEqual(
                    result["error"]["code"],
                    "INVALID_ARGUMENT",
                )
                query.assert_not_called()

    def test_unknown_session_returns_not_found(self):
        with patch("session_service.run_query") as query:
            result = chat_session("not-exist", "查询 T103")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "SESSION_NOT_FOUND")
        query.assert_not_called()

    def test_successful_turn_saves_history_and_bumps_version(self):
        session_id = db.create_session()
        tool_results = [{"name": "get_ticket", "result": {"ok": True}}]
        expected_session_id = session_id

        def fake_run_query(question, history=None, session_id=None):
            self.assertEqual(session_id, expected_session_id)
            messages = list(history or []) + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "T103 处理中"},
            ]
            return {
                "answer": "T103 处理中",
                "messages": messages,
                "tool_results": tool_results,
                "pending_actions": []
            }

        with patch(
            "session_service.run_query",
            side_effect=fake_run_query,
        ) as query:
            result = chat_session(session_id, "  查询 T103  ")

        # 业务层会去掉首尾空白，并把读取到的历史传给 run_query
        query.assert_called_once_with("查询 T103", [],session_id=session_id,)

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["session_id"], session_id)
        self.assertEqual(result["data"]["version"], 1)
        self.assertEqual(result["data"]["answer"], "T103 处理中")
        self.assertEqual(result["data"]["tool_results"], tool_results)

        stored = db.load_session(session_id)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["version"], 1)
        self.assertEqual(len(stored["messages"]), 2)
        self.assertEqual(stored["messages"][0]["content"], "查询 T103")

    def test_history_survives_next_turn(self):
        session_id = db.create_session()
        first_messages = [
            {"role": "user", "content": "查询 T103"},
            {"role": "assistant", "content": "T103 处理中"},
        ]
        first_result = {
            "answer": "T103 处理中",
            "messages": first_messages,
            "tool_results": [],
            "pending_actions": []
        }

        with patch(
            "session_service.run_query",
            return_value=first_result,
        ):
            first = chat_session(session_id, "查询 T103")

        self.assertTrue(first["ok"])
        self.assertEqual(first["data"]["version"], 1)

        received_history = []
        received_questions = []
        expected_session_id = session_id

        def fake_second_turn(question, history=None, *, session_id=None):
            # chat_session 必须把当前会话 id 透传给 run_query
            self.assertEqual(session_id, expected_session_id)
            received_questions.append(question)
            received_history.append(list(history or []))
            return {
                "answer": "负责人是技术支持",
                "messages": list(history or []) + [
                    {"role": "assistant", "content": "负责人是技术支持"},
                ],
                "tool_results": [],
                "pending_actions": [],
            }

        with patch(
            "session_service.run_query",
            side_effect=fake_second_turn,
        ):
            second = chat_session(session_id, "负责人是谁")

        self.assertTrue(second["ok"])
        self.assertEqual(received_questions, ["负责人是谁"])
        self.assertEqual(received_history, [first_messages])
        self.assertEqual(second["data"]["version"], 2)

        stored = db.load_session(session_id)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["version"], 2)
        self.assertEqual(stored["messages"][0], first_messages[0])

    def test_version_conflict_does_not_overwrite_other_write(self):
        session_id = db.create_session()
        other_messages = [{"role": "user", "content": "其它请求"}]
        received_questions = []

        def fake_run_query(question, history=None, *, session_id=None):
            received_questions.append(question)
            assert session_id is not None
            # 模拟：本请求读取会话后、保存之前，另一个请求抢先写入
            db.save_session(session_id, other_messages, 0)
            return {
                "answer": "回答",
                "messages": list(history or []) + [
                    {"role": "assistant", "content": "回答"},
                ],
                "tool_results": [],
                "pending_actions": [],
            }

        with patch(
            "session_service.run_query",
            side_effect=fake_run_query,
        ):
            result = chat_session(session_id, "我的问题")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "SESSION_CONFLICT")
        self.assertEqual(received_questions, ["我的问题"])

        stored = db.load_session(session_id)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["version"], 1)
        self.assertEqual(stored["messages"], other_messages)

    def test_run_query_exception_propagates_without_saving(self):
        session_id = db.create_session()

        with patch(
            "session_service.run_query",
            side_effect=TimeoutError("测试超时"),
        ):
            with self.assertRaises(TimeoutError):
                chat_session(session_id, "继续查询")

        stored = db.load_session(session_id)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["version"], 0)
        self.assertEqual(stored["messages"], [])


class SessionAPITests(SessionTestBase):
    def test_create_session_returns_id_and_version_zero(self):
        response = self.client.post("/sessions")

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["data"]["version"], 0)

        stored = db.load_session(body["data"]["session_id"])
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["version"], 0)
        self.assertEqual(stored["messages"], [])

    def test_send_message_returns_service_result(self):
        chat_result = {
            "ok": True,
            "data": {
                "session_id": "session-1",
                "version": 1,
                "answer": "T103 处理中",
                "tool_results": [],
            },
            "error": None,
        }

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-only"},
        ), patch(
            "app.chat_session",
            return_value=chat_result,
        ) as chat:
            response = self.client.post(
                "/sessions/session-1/messages",
                json={"question": "查询 T103"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), chat_result)
        chat.assert_called_once_with("session-1", "查询 T103")

    def test_send_message_unknown_session_returns_404(self):
        not_found = {
            "ok": False,
            "data": None,
            "error": {
                "code": "SESSION_NOT_FOUND",
                "message": "会话不存在",
            },
        }

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-only"},
        ), patch(
            "app.chat_session",
            return_value=not_found,
        ):
            response = self.client.post(
                "/sessions/missing/messages",
                json={"question": "查询 T103"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), not_found)

    def test_send_message_without_api_key_returns_503(self):
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": ""},
        ), patch("app.chat_session") as chat:
            response = self.client.post(
                "/sessions/session-1/messages",
                json={"question": "查询 T103"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["error"]["code"],
            "MODEL_NOT_CONFIGURED",
        )
        chat.assert_not_called()

    def test_send_message_timeout_returns_504(self):
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-only"},
        ), patch(
            "app.chat_session",
            side_effect=TimeoutError("测试超时"),
        ):
            with self.assertLogs("tickets.api", level="WARNING") as logs:
                response = self.client.post(
                    "/sessions/session-1/messages",
                    json={"question": "查询 T103"},
                )

        # 原有响应检查
        self.assertEqual(response.status_code, 504)
        self.assertEqual(
            response.json()["error"]["code"],
            "MODEL_TIMEOUT",
        )

        # 新增日志检查
        request_id = response.headers["X-Request-ID"]
        text = "\n".join(logs.output)

        self.assertIn("session_turn_failed", text)
        self.assertIn("error_code=MODEL_TIMEOUT", text)
        self.assertIn(f"request_id={request_id}", text)

    def test_empty_question_returns_422(self):
        response = self.client.post(
            "/sessions/session-1/messages",
            json={"question": ""},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
