import json

from ticket import get_ticket
from action_service import prepare_status_change
from knowledge_search import search_knowledge

TOOL_ARGUMENTS = {
    "get_ticket": {"ticket_id"},
    "propose_status_change": {"ticket_id", "new_status"},
    "search_knowledge": {"question"},
}

def tool_error(code: str, message: str) -> dict:
    return {
        "ok": False,
        "data": None,
        "error": {"code": code, "message": message},
    }

def execute_tool(
        call_json: str,
        *,
        session_id: str | None = None,
        pending_actions: list | None = None,
) -> dict:
    # 1. 解析 JSON
    # 2. 检查调用结构、工具名和参数
    # 3. 按工具分派：查询直接执行，改状态先准备建议
    try:
        call = json.loads(call_json)
    except (json.JSONDecodeError, TypeError):
        return tool_error("INVALID_TOOL_CALL", "无效的工具调用格式")

    if not isinstance(call, dict):
        return tool_error("INVALID_TOOL_CALL", "无效的工具调用格式")

    tool_name = call.get("tool")
    args = call.get("args")

    if not isinstance(tool_name, str) or not isinstance(args, dict):
        return tool_error("INVALID_TOOL_CALL", "无效的工具调用格式")

    expected_args = TOOL_ARGUMENTS.get(tool_name)
    if expected_args is None:
        return tool_error("UNKNOWN_TOOL", "未知的工具")

    if set(args) != expected_args:
        return tool_error(
            "INVALID_TOOL_ARGUMENTS",
            f"{tool_name} 的参数必须且只能包含："
            f"{', '.join(sorted(expected_args))}",
        )
    if not all(isinstance(value, str) for value in args.values()):
        return tool_error(
            "INVALID_TOOL_ARGUMENTS",
            "参数值必须是字符串",
        )

    if tool_name == "get_ticket":
        return get_ticket(args["ticket_id"])


    if tool_name == "search_knowledge":
        question = args["question"].strip()

        if not question:
            return tool_error(
                "INVALID_TOOL_ARGUMENTS",
                "question 不能为空",
            )

        return {
            "ok": True,
            "data": {
                "documents": search_knowledge(question, top_k=2),
            },
            "error": None,
        }

    if session_id is None or pending_actions is None:
        return tool_error(
            "TOOL_CONTEXT_REQUIRED",
            "提出建议需要会话上下文",
        )

    prepared = prepare_status_change(
        session_id,
        args["ticket_id"],
        args["new_status"],
    )
    if not prepared["ok"]:
        return prepared

    data = prepared["data"]
    key = (data["session_id"], data["ticket_id"], data["old_status"], data["new_status"])
    for existing in pending_actions:
        existing_key = (
            existing["session_id"], existing["ticket_id"],
            existing["old_status"], existing["new_status"],
        )
        if existing_key == key:
            # 同一轮重复提出相同建议，回传实际将保存的同一个编号。
            return {**prepared, "data": existing}
    pending_actions.append(data)

    return prepared


if __name__ == "__main__":
    # 测试用例
    test_call = json.dumps({
        "tool": "get_ticket",
        "args": {
            "ticket_id": "T103"
        }
    })
    result = execute_tool(test_call)
    print(result)
