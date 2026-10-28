import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import app


class AssistantAPITests(unittest.TestCase):
    def test_success_calls_query_once(self):
        result = {
            "answer": "请提供工单编号。",
            "tool_results": [],
        }

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-only"},
        ), patch(
            "app.run_query",
            return_value=result,
        ) as query, TestClient(app) as client:
            response = client.post(
                "/assistant/query",
                json={"question": "  帮我查询工单  "},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"ok": True, "data": result, "error": None},
        )
        query.assert_called_once_with("帮我查询工单")
    def test_timeouterror_calls_query_once(self):
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-only"},
        ), patch(
            "app.run_query",
            side_effect=TimeoutError("测试超时"),
        ) as query, TestClient(app) as client:
            response = client.post(
                "/assistant/query",
                json={"question": "帮我查询工单"},
            )

        self.assertEqual(response.status_code, 504)
        self.assertEqual(
            response.json(),
            {
                "ok": False,
                "data": None,
                "error": { "code": "MODEL_TIMEOUT", "message": "模型请求超时" },
            },
        )
        query.assert_called_once_with("帮我查询工单")