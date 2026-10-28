import json
import os
from pathlib import Path
from time import perf_counter

from agent_graph import run_graph_query
from eval_agent import evaluation_database, summarize_results

BASE_DIR = Path(__file__).resolve().parent
CASES_PATH = BASE_DIR / "rag_cases.json"
RESULTS_PATH = BASE_DIR / "eval_rag_results.json"


def _source_ids(result: dict) -> set[str]:
    """从结果里取来源编号；结构异常时忽略，不在这里抛错。"""
    sources = result.get("sources") or []
    return {
        item["doc_id"]
        for item in sources
        if isinstance(item, dict) and isinstance(item.get("doc_id"), str)
    }


def _traces(tool_results: list | None, name: str) -> list[dict]:
    return [
        trace
        for trace in tool_results or []
        if isinstance(trace, dict) and trace.get("name") == name
    ]


def score_rag_case(case: dict, result: dict) -> list[str]:
    """返回失败原因；空列表表示通过。"""
    reasons = []

    # 1. 回答必须是非空字符串
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        reasons.append("最终回答为空")

    # 2. 至少一次 search_knowledge，且这一次必须执行成功
    search_traces = _traces(result.get("tool_results"), "search_knowledge")
    if not search_traces:
        reasons.append("没有调用 search_knowledge")
    elif not any(
        isinstance(trace.get("result"), dict)
        and trace["result"].get("ok") is True
        for trace in search_traces
    ):
        reasons.append("search_knowledge 未成功执行")

    # 3. 来源编号集合必须与预期一致（多一篇、少一篇都算失败）
    actual_ids = _source_ids(result)
    expected_ids = set(case.get("expected_source_ids") or [])
    if actual_ids != expected_ids:
        reasons.append(
            f"来源不匹配，预期 {sorted(expected_ids)}，实际 {sorted(actual_ids)}"
        )

    # 4. 问规范不应产生修改建议
    if result.get("pending_actions"):
        reasons.append("回答规范时产生了待确认建议")

    # 5. 也不应该调用 propose_status_change
    if _traces(result.get("tool_results"), "propose_status_change"):
        reasons.append("回答规范时调用了 propose_status_change")

    return reasons


def evaluate_cases(cases: list[dict]) -> list[dict]:
    results = []

    for case in cases:
        start_time = perf_counter()
        try:
            result = run_graph_query(case["question"])
        except Exception as exc:
            elapsed = perf_counter() - start_time
            results.append({
                "case_id": case.get("id"),
                "question": case.get("question"),
                "expected_source_ids": case.get("expected_source_ids", []),
                "passed": False,
                "reasons": [
                    f"运行时异常: {type(exc).__name__}: {exc}"
                ],
                "elapsed_seconds": round(elapsed, 3),
                "answer": None,
                "sources": [],
                "tool_results": [],
                "pending_actions": [],
            })
            continue

        elapsed = perf_counter() - start_time
        reasons = score_rag_case(case, result)
        results.append({
            "case_id": case.get("id"),
            "question": case.get("question"),
            "expected_source_ids": case.get("expected_source_ids", []),
            "passed": not reasons,
            "reasons": reasons,
            "elapsed_seconds": round(elapsed, 3),
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "tool_results": result.get("tool_results", []),
            "pending_actions": result.get("pending_actions", []),
            "messages": result.get("messages", []),
        })

    return results


def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("请先设置 DEEPSEEK_API_KEY 环境变量")

    with CASES_PATH.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    # 独立评测数据库，避免读写本地 tickets.db
    with evaluation_database():
        results = evaluate_cases(cases)

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary = summarize_results(results)
    print(f"通过 {summary['passed']}/{summary['total']}")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['case_id']} ({item['elapsed_seconds']:.2f}s)")
        print(
            "  来源："
            + (", ".join(item["expected_source_ids"]) or "无")
            + " → 实际："
            + (", ".join(sorted(_source_ids(item))) or "无")
        )
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
