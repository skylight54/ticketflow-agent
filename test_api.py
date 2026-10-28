import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import db
from app import app


class TicketAPITests(unittest.TestCase):
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

    def test_create_and_query(self):
        response = self.client.post(
            "/tickets",
            json={
                "ticket_id": "T001",
                "owner": "技术支持",
                "summary": "无法登录",
            },
        )
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.json()["ok"])

        response = self.client.get("/tickets/T001")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["data"]["summary"],
            "无法登录",
        )
    def test_duplicate_returns_409(self):
        response = self.client.post(
            "/tickets",
            json={
                "ticket_id": "T002",
                "owner": "技术支持",
                "summary": "无法登录",
            },
        )
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.json()["ok"])

        # Try to create a duplicate ticket
        response = self.client.post(
            "/tickets",
            json={
                "ticket_id": "T002",
                "owner": "技术支持",
                "summary": "另一个问题",
            },
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["error"]["code"],
            "TICKET_ALREADY_EXISTS",
        )

    def test_missing_summary_returns_422(self):
        response = self.client.post(
            "/tickets",
            json={
                "ticket_id": "T003",
                "owner": "技术支持",
                # Missing summary
            },
        )
        self.assertEqual(response.status_code, 422)
        response = self.client.get("/tickets/T003")
        self.assertEqual(response.status_code, 404)

    def test_patch_status(self):
        response = self.client.post(
            "/tickets",
            json={
                "ticket_id": "T004",
                "owner": "技术支持",
                "summary": "无法登录",
            },
        )
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.json()["ok"])

        # Update status to a valid value
        response = self.client.patch(
            "/tickets/T004/status",
            json={"status": "处理中"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

        # Query the ticket to ensure the status was updated
        response = self.client.get("/tickets/T004")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["data"]["status"],
            "处理中",
        )

    def test_invalid_status_transition_returns_409(self):
        response = self.client.post(
            "/tickets",
            json={
                "ticket_id": "T005",
                "owner": "技术支持",
                "summary": "无法登录",
            },
        )
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.json()["ok"])

        # Try to change status to an invalid value
        response = self.client.patch(
            "/tickets/T005/status",
            json={"status": "已完成"},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["error"]["code"],
            "INVALID_STATUS_TRANSITION",
        )