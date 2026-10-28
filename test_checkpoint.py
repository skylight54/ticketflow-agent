import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "checkpoint_demo.py"


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "checkpoints.sqlite"

    def run_demo(self, *args):
        return subprocess.run(
            [
                sys.executable,
                "-X", "utf8",
                str(SCRIPT),
                *args,
                "--db-path", str(self.db_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )

    def test_resume_after_process_exit(self):
        started = self.run_demo(
            "start", "--thread-id", "approval-001"
        )
        self.assertEqual(started.returncode, 0, started.stderr)
        self.assertIn("__interrupt__", started.stdout)

        resumed = self.run_demo(
            "resume",
            "--thread-id", "approval-001",
            "--decision", "y",
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("模拟执行完成", resumed.stdout)
        self.assertIn("'approved': True", resumed.stdout)

    def test_reject_after_start(self):
        started = self.run_demo(
            "start", "--thread-id", "approval-001"
        )
        self.assertEqual(started.returncode, 0, started.stderr)
        self.assertIn("__interrupt__", started.stdout)

        resumed = self.run_demo(
            "resume",
            "--thread-id", "approval-001",
            "--decision", "n",
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("'approved': False", resumed.stdout)
        self.assertIn("已取消", resumed.stdout)
        self.assertNotIn("模拟执行完成", resumed.stdout)

    def test_resume_without_start_reports_error(self):
        resumed = self.run_demo(
            "resume",
            "--thread-id", "missing",
            "--decision", "y",
        )

        # 非 0 退出码：子进程里的 SystemExit 被当成失败
        self.assertNotEqual(resumed.returncode, 0)
        # SystemExit 的提示写到 stderr，stdout 为空
        self.assertEqual(resumed.stdout, "")
        self.assertIn("找不到该 thread_id", resumed.stderr)

    def test_repeated_resume_reports_error(self):
        started = self.run_demo(
            "start", "--thread-id", "approval-001"
        )
        self.assertEqual(started.returncode, 0, started.stderr)
        self.assertIn("__interrupt__", started.stdout)

        resumed = self.run_demo(
            "resume",
            "--thread-id", "approval-001",
            "--decision", "y",
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertIn("模拟执行完成", resumed.stdout)
        self.assertIn("'approved': True", resumed.stdout)

        resumed = self.run_demo(
            "resume",
            "--thread-id", "approval-001",
            "--decision", "y",
        )

        self.assertNotEqual(resumed.returncode,0)
        self.assertEqual(resumed.stdout, "")
        self.assertIn("没有等待确认的中断", resumed.stderr)

if __name__ == "__main__":
    unittest.main()