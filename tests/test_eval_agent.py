import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from evaluations import eval_agent
from evaluations.eval_agent import score_case, summarize_results


def make_case(**overrides):
    case = {
        "expected_tool": "get_ticket",
        "expected_ticket_id": "T_TEST",
        "expected_error": None,
        "expected_answer_fields": ["ticket_id", "status", "owner"],
    }
    case.update(overrides)
    return case


def make_tool_result(**overrides):
    result = {
        "ok": True,
        "data": {
            "ticket_id": "T_TEST",
            "status": "待处理",
            "owner": "技术支持",
        },
        "error": None,
    }
    result.update(overrides)
    return result


def make_trace(**overrides):
    trace = {
        "name": "get_ticket",
        "arguments": {"ticket_id": "T_TEST"},
        "result": make_tool_result(),
    }
    trace.update(overrides)
    return trace


def make_result(**overrides):
    result = {
        "answer": "工单 T_TEST 当前状态为待处理，负责人是技术支持。",
        "tool_results": [make_trace()],
    }
    result.update(overrides)
    return result


def make_summary_entry(passed, elapsed_seconds):
    return {
        "passed": passed,
        "elapsed_seconds": elapsed_seconds,
    }


class EvalAgentScoreCaseTests(unittest.TestCase):
    """score_case 只依赖传入的 case/result，不访问模型或数据库。"""

    def test_success_case_passes(self):
        self.assertEqual(score_case(make_case(), make_result()), [])

    def test_empty_answer_is_reported(self):
        reasons = score_case(make_case(), make_result(answer=""))

        self.assertIn("最终回答为空", reasons)

    def test_none_answer_is_reported_without_crashing(self):
        # 键存在但值为 None 时，不能对 None 做 in 运算
        reasons = score_case(make_case(), make_result(answer=None))

        self.assertIn("最终回答为空", reasons)

    def test_no_tool_expected_and_no_call_passes(self):
        case = make_case(
            expected_tool=None,
            expected_ticket_id=None,
            expected_answer_fields=[],
        )
        result = {"answer": "请提供工单编号。", "tool_results": []}

        self.assertEqual(score_case(case, result), [])

    def test_no_tool_expected_but_tool_called(self):
        case = make_case(
            expected_tool=None,
            expected_ticket_id=None,
            expected_answer_fields=[],
        )

        reasons = score_case(case, make_result())

        self.assertEqual(reasons, ["不应调用工具"])

    def test_missing_tool_call_is_reported(self):
        reasons = score_case(make_case(), make_result(tool_results=[]))

        self.assertEqual(reasons, ["预期恰好一次工具调用，实际 0 次"])

    def test_multiple_tool_calls_are_reported(self):
        result = make_result(tool_results=[make_trace(), make_trace()])

        reasons = score_case(make_case(), result)

        self.assertEqual(reasons, ["预期恰好一次工具调用，实际 2 次"])

    def test_wrong_tool_name_is_reported(self):
        result = make_result(
            tool_results=[make_trace(name="propose_status_change")]
        )

        reasons = score_case(make_case(), result)

        self.assertEqual(
            reasons,
            ["工具名称不匹配，预期 get_ticket，实际 propose_status_change"],
        )

    def test_non_dict_arguments_are_reported(self):
        result = make_result(tool_results=[make_trace(arguments="T_TEST")])

        reasons = score_case(make_case(), result)

        self.assertEqual(reasons, ["工具参数不是字典"])

    def test_ticket_id_mismatch_is_reported(self):
        result = make_result(
            tool_results=[make_trace(arguments={"ticket_id": "T_OTHER"})]
        )

        reasons = score_case(make_case(), result)

        self.assertEqual(
            reasons,
            ["工具参数 ticket_id 不匹配，预期 T_TEST，实际 T_OTHER"],
        )

    def test_non_dict_tool_result_is_reported(self):
        result = make_result(tool_results=[make_trace(result="ok")])

        reasons = score_case(make_case(), result)

        self.assertEqual(reasons, ["工具结果不是字典"])

    def test_expected_success_but_tool_failed(self):
        result = make_result(
            tool_results=[make_trace(result=make_tool_result(ok=False))]
        )

        reasons = score_case(make_case(), result)

        self.assertEqual(reasons, ["预期工具执行成功"])

    def test_expected_success_but_error_present(self):
        tool_result = make_tool_result(
            error={"code": "SOME_ERROR", "message": "不应出现"}
        )
        result = make_result(tool_results=[make_trace(result=tool_result)])

        reasons = score_case(make_case(), result)

        self.assertEqual(reasons, ["成功结果不应包含错误"])

    def test_success_without_ticket_data(self):
        result = make_result(
            tool_results=[make_trace(result=make_tool_result(data=None))]
        )

        reasons = score_case(make_case(), result)

        self.assertEqual(reasons, ["成功结果缺少工单数据"])

    def test_success_ticket_id_mismatch(self):
        data = {
            "ticket_id": "T_OTHER",
            "status": "待处理",
            "owner": "技术支持",
        }
        result = make_result(
            tool_results=[make_trace(result=make_tool_result(data=data))]
        )

        reasons = score_case(make_case(), result)

        self.assertIn("返回的工单编号不匹配", reasons)

    def test_answer_missing_field_is_reported(self):
        result = make_result(answer="工单 T_TEST 当前状态为待处理。")

        reasons = score_case(make_case(), result)

        self.assertEqual(
            reasons,
            ["回答未包含工具返回的 owner：技术支持"],
        )

    def test_empty_field_value_is_reported(self):
        data = {"ticket_id": "T_TEST", "status": "待处理", "owner": ""}
        result = make_result(
            tool_results=[make_trace(result=make_tool_result(data=data))]
        )

        reasons = score_case(make_case(), result)

        self.assertEqual(reasons, ["工具结果缺少有效字段：owner"])

    def test_expected_error_matches(self):
        case = make_case(expected_error="TICKET_NOT_FOUND")
        tool_result = make_tool_result(
            ok=False,
            data=None,
            error={"code": "TICKET_NOT_FOUND", "message": "工单编号不存在"},
        )
        result = make_result(
            answer="未找到工单 T_TEST。",
            tool_results=[make_trace(result=tool_result)],
        )

        self.assertEqual(score_case(case, result), [])

    def test_expected_error_but_tool_succeeds(self):
        case = make_case(expected_error="TICKET_NOT_FOUND")

        reasons = score_case(case, make_result())

        self.assertEqual(
            reasons,
            [
                "预期工具执行失败",
                "工具错误码不匹配，预期 TICKET_NOT_FOUND，实际 None",
            ],
        )

    def test_error_code_mismatch_is_reported(self):
        case = make_case(expected_error="TICKET_NOT_FOUND")
        tool_result = make_tool_result(
            ok=False,
            data=None,
            error={"code": "OTHER_ERROR", "message": "别的错误"},
        )
        result = make_result(
            answer="工单 T_TEST 处理失败。",
            tool_results=[make_trace(result=tool_result)],
        )

        reasons = score_case(case, result)

        self.assertEqual(
            reasons,
            ["工具错误码不匹配，预期 TICKET_NOT_FOUND，实际 OTHER_ERROR"],
        )


class EvalAgentMainTests(unittest.TestCase):
    def write_cases(self, directory, cases):
        path = Path(directory) / "cases.json"
        path.write_text(
            json.dumps(cases, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def test_main_writes_passed_result(self):
        case = make_case(id="success", question="查询 T_TEST")
        with tempfile.TemporaryDirectory() as directory:
            cases_path = self.write_cases(directory, [case])
            results_path = Path(directory) / "results.json"

            with (
                patch.object(eval_agent, "CASES_PATH", cases_path),
                patch.object(eval_agent, "RESULTS_PATH", results_path),
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}),
                patch.object(
                    eval_agent,
                    "run_query",
                    return_value=make_result(),
                ) as query,
                redirect_stdout(StringIO()) as output,
            ):
                eval_agent.main()

            query.assert_called_once_with("查询 T_TEST", history=None)

            data = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data), 1)
            self.assertTrue(data[0]["passed"])
            self.assertEqual(data[0]["reasons"], [])
            self.assertEqual(data[0]["case_id"], "success")
            self.assertEqual(
                data[0]["answer"],
                "工单 T_TEST 当前状态为待处理，负责人是技术支持。",
            )
            self.assertIn("通过 1/1", output.getvalue())

    def test_main_records_runtime_exception(self):
        case = make_case(id="timeout", question="查询 T_TEST")
        with tempfile.TemporaryDirectory() as directory:
            cases_path = self.write_cases(directory, [case])
            results_path = Path(directory) / "results.json"

            with (
                patch.object(eval_agent, "CASES_PATH", cases_path),
                patch.object(eval_agent, "RESULTS_PATH", results_path),
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}),
                patch.object(
                    eval_agent,
                    "run_query",
                    side_effect=TimeoutError("模型超时"),
                ),
                redirect_stdout(StringIO()) as output,
            ):
                eval_agent.main()

            data = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data), 1)
            self.assertFalse(data[0]["passed"])
            self.assertEqual(data[0]["case_id"], "timeout")
            self.assertIsNone(data[0]["answer"])
            self.assertEqual(data[0]["tool_results"], [])
            self.assertTrue(
                data[0]["reasons"][0].startswith("运行时异常: TimeoutError")
            )
            self.assertIn("通过 0/1", output.getvalue())

    def test_main_requires_api_key(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            with self.assertRaises(SystemExit):
                eval_agent.main()


class EvalAgentSummarizeTests(unittest.TestCase):
    """summarize_results 只做统计，不依赖文件或模型。"""

    def test_mixed_results(self):
        summary = summarize_results([
            make_summary_entry(True, 1.0),
            make_summary_entry(False, 2.0),
            make_summary_entry(True, 3.0),
        ])

        self.assertEqual(
            set(summary),
            {
                "total",
                "passed",
                "failed",
                "pass_rate",
                "avg_elapsed_seconds",
                "max_elapsed_seconds",
            },
        )
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertAlmostEqual(summary["pass_rate"], 2 / 3)
        self.assertAlmostEqual(summary["avg_elapsed_seconds"], 2.0)
        self.assertEqual(summary["max_elapsed_seconds"], 3.0)

    def test_all_passed(self):
        summary = summarize_results([
            make_summary_entry(True, 0.5),
            make_summary_entry(True, 1.5),
        ])

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertAlmostEqual(summary["avg_elapsed_seconds"], 1.0)
        self.assertEqual(summary["max_elapsed_seconds"], 1.5)

    def test_none_passed(self):
        summary = summarize_results([
            make_summary_entry(False, 0.25),
            make_summary_entry(False, 0.75),
        ])

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["pass_rate"], 0.0)
        self.assertAlmostEqual(summary["avg_elapsed_seconds"], 0.5)
        self.assertEqual(summary["max_elapsed_seconds"], 0.75)

    def test_single_result(self):
        summary = summarize_results([make_summary_entry(True, 2.5)])

        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertEqual(summary["avg_elapsed_seconds"], 2.5)
        self.assertEqual(summary["max_elapsed_seconds"], 2.5)

    def test_empty_results(self):
        # 没有用例时不能除零
        summary = summarize_results([])

        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["passed"], 0)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["pass_rate"], 0.0)
        self.assertEqual(summary["avg_elapsed_seconds"], 0.0)
        self.assertEqual(summary["max_elapsed_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
