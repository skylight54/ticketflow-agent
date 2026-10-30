import json
import os
from pathlib import Path
from time import perf_counter
import tempfile
from contextlib import contextmanager

import db
from agent_graph import run_graph_query as run_query

BASE_DIR = Path(__file__).resolve().parent
CASES_DIR = BASE_DIR / "cases"
RESULTS_DIR = BASE_DIR / "results"
CASES_PATH = CASES_DIR / "eval_cases.json"
RESULTS_PATH = RESULTS_DIR / "eval_results.json"

@contextmanager
def evaluation_database():
    original_path = db.DB_PATH

    with tempfile.TemporaryDirectory() as directory:
        try:
            db.DB_PATH = Path(directory) / "eval.db"
            db.init_db()

            db.InsertTicket(
                "T103",
                "待处理",
                "技术支持",
                "用户无法登录",
            )

            yield db.DB_PATH
        finally:
            db.DB_PATH = original_path

def _error_code(tool_result) -> str | None:
    if not isinstance(tool_result, dict):
        return None
    error = tool_result.get("error")
    if isinstance(error, dict):
        return error.get("code")
    return None


def score_case(case: dict, result: dict) -> list[str]:
    """返回失败原因；空列表表示自动检查通过。"""
    reasons = []
    if not result.get("answer"):
        reasons.append("最终回答为空")
    traces = result.get("tool_results") or []
    expected_tool = case.get("expected_tool")

    if expected_tool is None:
        if traces:
            reasons.append("不应调用工具")
        return reasons

    if len(traces) != 1:
        reasons.append(f"预期恰好一次工具调用，实际 {len(traces)} 次")
        return reasons

    trace = traces[0]

    if trace.get("name") != expected_tool:
        reasons.append(
            f"工具名称不匹配，预期 {expected_tool}，实际 {trace.get('name')}"
        )
        return reasons

    arguments = trace.get("arguments")
    if not isinstance(arguments, dict):
        reasons.append("工具参数不是字典")
        return reasons

    actual_ticket_id = arguments.get("ticket_id")
    if actual_ticket_id != case.get("expected_ticket_id"):
        reasons.append(
            "工具参数 ticket_id 不匹配，"
            f"预期 {case.get('expected_ticket_id')}，实际 {actual_ticket_id}"
        )

    tool_result = trace.get("result")
    expected_error = case.get("expected_error")

    if not isinstance(tool_result, dict):
        reasons.append("工具结果不是字典")
        return reasons

    if expected_error is None:
        if tool_result.get("ok") is not True:
            reasons.append("预期工具执行成功")

        if tool_result.get("error") is not None:
            reasons.append("成功结果不应包含错误")

        data = tool_result.get("data")
        if not isinstance(data, dict):
            reasons.append("成功结果缺少工单数据")
        elif data.get("ticket_id") != case["expected_ticket_id"]:
            reasons.append("返回的工单编号不匹配")
        if isinstance(data, dict):
            # answer 键存在但值为 None 时也要按空字符串处理
            answer = result.get("answer") or ""

            for field in case.get("expected_answer_fields", []):
                value = data.get(field)

                if not isinstance(value, str) or not value:
                    reasons.append(f"工具结果缺少有效字段：{field}")
                elif value not in answer:
                    reasons.append(f"回答未包含工具返回的 {field}：{value}")
    else:
        if tool_result.get("ok") is not False:
            reasons.append("预期工具执行失败")

        actual_error = _error_code(tool_result)
        if actual_error != expected_error:
            reasons.append(
                f"工具错误码不匹配，"
                f"预期 {expected_error}，实际 {actual_error}"
            )

    return reasons

def summarize_results(results: list[dict]) -> dict:
    """汇总评测结果；空列表返回全 0，避免除零。"""
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    elapsed_values = [item["elapsed_seconds"] for item in results]

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "avg_elapsed_seconds": (
            sum(elapsed_values) / total if total else 0.0
        ),
        "max_elapsed_seconds": max(elapsed_values, default=0.0),
    }

def evaluate_cases(cases: list[dict]) -> list[dict]:
    results = []

    for case in cases:
        start_time = perf_counter()
        try:
            result = run_query(case["question"], history=case.get("history"))
        except Exception as exc:
            elapsed = perf_counter() - start_time
            results.append({
                "case_id": case.get("id"),
                "passed": False,
                "reasons":  [
                    f"运行时异常: {type(exc).__name__}: {exc}"
                ],
                "elapsed_seconds": round(elapsed, 3),
                "answer": None,
                "tool_results": [],
            })
            continue

        elapsed = perf_counter() - start_time
        reasons = score_case(case, result)
        results.append({
            "case_id": case.get("id"),
            "passed": not reasons,
            "reasons": reasons,
            "elapsed_seconds": round(elapsed, 3),
            "answer": result.get("answer", ""),
            "tool_results": result.get("tool_results", []),
            "messages": result.get("messages", []),
        })
        
    return results


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("请先设置 DEEPSEEK_API_KEY 环境变量")

    with CASES_PATH.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    with evaluation_database():
        results = evaluate_cases(cases)
    
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary = summarize_results(results)
    print(f"通过 {summary['passed']}/{summary['total']}")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['case_id']} ({item['elapsed_seconds']:.2f}s)")
        for reason in item["reasons"]:
            print(f"  - {reason}")
    print(
        f"用例通过率：{summary['pass_rate']:.1%}"
        f"（{summary['passed']}/{summary['total']}）"
    )
    print(f"平均耗时：{summary['avg_elapsed_seconds']:.3f}s")
    print(f"最大耗时：{summary['max_elapsed_seconds']:.3f}s")

if __name__ == "__main__":
    main()
