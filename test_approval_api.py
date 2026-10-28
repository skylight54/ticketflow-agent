import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as app_module
import db


class ApprovalAPITests(unittest.TestCase):
    TICKET_ID = "T106"

    def setUp(self):
        self.original_path = db.DB_PATH
        self.directory = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        )
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self._restore_db_path)

        root = Path(self.directory.name)
        self.checkpoint_path = root / "checkpoints.sqlite"

        db_patch = patch.object(db, "DB_PATH", root / "tickets.db")
        checkpoint_patch = patch.object(
            app_module,
            "CHECKPOINT_DB_PATH",
            self.checkpoint_path,
        )

        db_patch.start()
        self.addCleanup(db_patch.stop)

        checkpoint_patch.start()
        self.addCleanup(checkpoint_patch.stop)

    def _restore_db_path(self):
        db.DB_PATH = self.original_path

    # ---- 请求封装：只负责发请求和断言成功，业务断言留在测试里 ----

    def create_ticket(self, client):
        response = client.post(
            "/tickets",
            json={
                "ticket_id": self.TICKET_ID,
                "owner": "test",
                "summary": "无法登录",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertTrue(response.json()["ok"])

    def create_session(self, client):
        response = client.post("/sessions")
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["data"]["session_id"]

    def create_action(self, client, session_id):
        response = client.post(
            f"/sessions/{session_id}/actions",
            json={
                "ticket_id": self.TICKET_ID,
                "new_status": "处理中",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertTrue(response.json()["ok"])
        return response.json()["data"]["action_id"]

    def start_approval(self, client, session_id, action_id):
        return client.post(
            f"/sessions/{session_id}/actions/{action_id}/approval/start"
        )

    def resume_approval(self, client, session_id, action_id, approved):
        return client.post(
            f"/sessions/{session_id}/actions/{action_id}/approval/resume",
            json={"approved": approved},
        )

    def ticket_status(self, client):
        response = client.get(f"/tickets/{self.TICKET_ID}")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["data"]["status"]

    def prepare_started_approval(self, client):
        self.create_ticket(client)
        session_id = self.create_session(client)
        action_id = self.create_action(client, session_id)
        response = self.start_approval(client, session_id, action_id)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["ok"])
        return session_id, action_id

    # ---- 测试 ----

    def test_resume_after_app_restart(self):
        with TestClient(app_module.app) as client:
            session_id, action_id = self.prepare_started_approval(client)

            self.assertEqual(self.ticket_status(client), "待处理")
            self.assertTrue(self.checkpoint_path.exists())

        # 新的 TestClient 会重新执行 lifespan，相当于应用重启
        with TestClient(app_module.app) as client:
            response = self.resume_approval(
                client, session_id, action_id, True
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["ok"])
            self.assertEqual(response.json()["data"]["status"], "executed")

            self.assertEqual(self.ticket_status(client), "处理中")

            again = self.resume_approval(
                client, session_id, action_id, True
            )
            self.assertEqual(again.status_code, 409)
            self.assertEqual(
                again.json()["error"]["code"],
                "APPROVAL_NOT_WAITING",
            )

    def test_reject_after_app_restart(self):
        with TestClient(app_module.app) as client:
            session_id, action_id = self.prepare_started_approval(client)

        with TestClient(app_module.app) as client:
            response = self.resume_approval(
                client, session_id, action_id, False
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["ok"])
            self.assertEqual(response.json()["data"]["status"], "rejected")
            self.assertEqual(self.ticket_status(client), "待处理")

    def test_start_returns_waiting_question(self):
        with TestClient(app_module.app) as client:
            self.create_ticket(client)
            session_id = self.create_session(client)
            action_id = self.create_action(client, session_id)

            response = self.start_approval(client, session_id, action_id)

            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertTrue(body["ok"])
            self.assertEqual(body["data"]["status"], "waiting")
            self.assertEqual(len(body["data"]["interrupts"]), 1)
            self.assertEqual(
                body["data"]["interrupts"][0]["question"],
                "是否同意将工单T106由待处理改为处理中？",
            )
            self.assertEqual(self.ticket_status(client), "待处理")

    def test_start_twice_returns_409(self):
        with TestClient(app_module.app) as client:
            session_id, action_id = self.prepare_started_approval(client)

            response = self.start_approval(client, session_id, action_id)

            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(
                response.json()["error"]["code"],
                "APPROVAL_ALREADY_STARTED",
            )

    def test_resume_before_start_returns_409(self):
        with TestClient(app_module.app) as client:
            self.create_ticket(client)
            session_id = self.create_session(client)
            action_id = self.create_action(client, session_id)

            response = self.resume_approval(
                client, session_id, action_id, True
            )

            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(
                response.json()["error"]["code"],
                "APPROVAL_NOT_STARTED",
            )
            self.assertEqual(self.ticket_status(client), "待处理")

    def test_non_bool_decision_returns_422(self):
        with TestClient(app_module.app) as client:
            session_id, action_id = self.prepare_started_approval(client)

            response = client.post(
                f"/sessions/{session_id}/actions/{action_id}"
                "/approval/resume",
                json={"approved": 1},
            )

            # strict=True 的 bool 不接受 1
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(self.ticket_status(client), "待处理")

    def test_action_from_another_session_returns_404(self):
        with TestClient(app_module.app) as client:
            self.create_ticket(client)
            owner_session = self.create_session(client)
            action_id = self.create_action(client, owner_session)
            other_session = self.create_session(client)

            started = self.start_approval(
                client, other_session, action_id
            )
            resumed = self.resume_approval(
                client, other_session, action_id, True
            )

            self.assertEqual(started.status_code, 404, started.text)
            self.assertEqual(
                started.json()["error"]["code"], "ACTION_NOT_FOUND"
            )
            self.assertEqual(resumed.status_code, 404, resumed.text)
            self.assertEqual(
                resumed.json()["error"]["code"], "ACTION_NOT_FOUND"
            )

    def test_ticket_changed_while_waiting_returns_409(self):
        with TestClient(app_module.app) as client:
            session_id, action_id = self.prepare_started_approval(client)

            changed = client.patch(
                f"/tickets/{self.TICKET_ID}/status",
                json={"status": "处理中"},
            )
            self.assertEqual(changed.status_code, 200, changed.text)

            response = self.resume_approval(
                client, session_id, action_id, True
            )

            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(
                response.json()["error"]["code"], "ACTION_STALE"
            )
            self.assertEqual(self.ticket_status(client), "处理中")


if __name__ == "__main__":
    unittest.main()
