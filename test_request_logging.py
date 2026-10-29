import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import os

from fastapi.testclient import TestClient

import app as app_module
import db


class RequestLoggingTests(unittest.TestCase):
    def setUp(self):
        self.original_path = db.DB_PATH
        self.directory = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        )
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self._restore_db_path)

        root = Path(self.directory.name)

        db_patch = patch.object(db, "DB_PATH", root / "tickets.db")
        checkpoint_patch = patch.object(
            app_module,
            "CHECKPOINT_DB_PATH",
            root / "checkpoints.sqlite",
        )

        db_patch.start()
        self.addCleanup(db_patch.stop)

        checkpoint_patch.start()
        self.addCleanup(checkpoint_patch.stop)

    def _restore_db_path(self):
        db.DB_PATH = self.original_path

    def test_unhandled_exception_returns_safe_error(self):
        # assertLogs 同时捕获中间件的异常日志，避免测试输出整段 traceback
        with self.assertLogs("tickets.api", level="WARNING") as logs:
            with TestClient(app_module.app) as client:
                with patch(
                    "app.get_ticket",
                    side_effect=RuntimeError("internal-test-detail"),
                ):
                    response = client.get("/tickets/T_TEST")

        # 1. 未处理异常统一返回 500
        self.assertEqual(response.status_code, 500)

        body = response.json()

        # 2. 使用统一的内部错误码
        self.assertEqual(body["error"]["code"], "INTERNAL_ERROR")

        # 3. 响应头与正文引用同一个请求编号
        request_id = response.headers.get("X-Request-ID")
        self.assertTrue(request_id)
        self.assertEqual(body["error"]["request_id"], request_id)

        # 4. 不把异常详情泄漏给调用方
        self.assertNotIn("internal-test-detail", response.text)

        # 异常写入日志，方便按 request_id 排查
        self.assertIn("request_failed", logs.output[0])

    def test_health_without_model_key(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            with TestClient(app_module.app) as client:
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["data"],
            {"service": "ticketflow-agent", "status": "alive"},
        )
        self.assertTrue(response.headers.get("X-Request-ID"))
if __name__ == "__main__":
    unittest.main()
