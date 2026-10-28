import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import db
from ticket import create_ticket, get_ticket, update_ticket_status


class TicketTests(unittest.TestCase):
    def setUp(self):
        self.original_db_path = db.DB_PATH
        self.temp_dir = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        )
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_create_and_query(self):
        created = create_ticket("T001", "技术支持", "无法登录")
        self.assertTrue(created["ok"])

        queried = get_ticket("T001")
        self.assertTrue(queried["ok"])
        self.assertEqual(queried["data"]["status"], "待处理")
        self.assertEqual(queried["data"]["summary"], "无法登录")

    def test_duplicate_preserves_original(self):
        created = create_ticket("T002", "技术支持", "无法登录")
        self.assertTrue(created["ok"])
        # Try to create a duplicate ticket
        duplicate = create_ticket("T002", "技术支持", "另一个问题")
        self.assertFalse(duplicate["ok"])
        self.assertEqual(duplicate["error"]["code"], "TICKET_ALREADY_EXISTS")
        # Ensure the original ticket is still there
        queried = get_ticket("T002")
        self.assertTrue(queried["ok"])
        self.assertEqual(queried["data"]["summary"], "无法登录")

    def test_invalid_transition_preserves_status(self):
        created = create_ticket("T003", "技术支持", "无法登录")
        self.assertTrue(created["ok"])
        # Try to change status to an invalid value
        invalid_update = update_ticket_status("T003", "已完成")
        self.assertFalse(invalid_update["ok"])
        self.assertEqual(invalid_update["error"]["code"], "INVALID_STATUS_TRANSITION")
        # Ensure the status is still the original
        queried = get_ticket("T003")
        self.assertTrue(queried["ok"])
        self.assertEqual(queried["data"]["status"], "待处理")

    def test_conflict_preserves_status(self):
        created = create_ticket("T004", "技术支持", "无法登录")
        self.assertTrue(created["ok"])

        with patch(
            "ticket.UpdateTicketStatus",
            return_value=False,
        ) as mock_update:
            result = update_ticket_status("T004", "处理中")
            mock_update.assert_called_once_with(
                "T004", "待处理", "处理中"
            )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["error"]["code"],
            "TICKET_CONFLICT",
        )

        queried = get_ticket("T004")
        self.assertTrue(queried["ok"])
        self.assertEqual(queried["data"]["status"], "待处理")

    def test_full_status_flow(self):
        created = create_ticket("T005", "技术支持", "无法登录")
        self.assertTrue(created["ok"])

        for status in ["处理中", "处理中", "已完成"]:
            result = update_ticket_status("T005", status)
            self.assertTrue(result["ok"])
            self.assertEqual(result["data"]["status"], status)

        rejected = update_ticket_status("T005", "处理中")
        self.assertFalse(rejected["ok"])
        self.assertEqual(
            rejected["error"]["code"],
            "INVALID_STATUS_TRANSITION",
        )

        queried = get_ticket("T005")
        self.assertTrue(queried["ok"])
        self.assertEqual(queried["data"]["status"], "已完成")