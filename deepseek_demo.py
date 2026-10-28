import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from copy import deepcopy
from tool_runner import execute_tool, tool_error

MAX_MODEL_ROUNDS = 4
MAX_TOOL_CALLS = 8
DSML_MARKERS = ("<｜｜DSML｜｜", "<|DSML|")


class AgentResponseError(RuntimeError):
    """模型没有完成符合工具协议的有限执行流程。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket",
            "description": (
                "根据工单编号查询真实工单信息。"
                "用户未提供编号时，应先询问，不要猜测编号。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "工单编号，例如 T103",
                    }
                },
                "required": ["ticket_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "检索项目模拟知识库中的故障排查规范和工单流转规则。"
                "查询具体工单当前状态时使用 get_ticket。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "需要检索的规范问题",
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    }
]
PROPOSE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "propose_status_change",
        "description": (
            "提出工单状态变更建议，等待用户确认，不立即修改工单。"
            "需要明确的工单编号和目标状态。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "new_status": {
                    "type": "string",
                    "enum": ["待处理", "处理中", "已完成", "已取消"],
                },
            },
            "required": ["ticket_id", "new_status"],
            "additionalProperties": False,
        },
    },
}

def send_messages(
    messages: list[dict[str, Any]],
    allow_tools: bool = True,
    allow_proposals=False,
) -> dict:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请先设置 DEEPSEEK_API_KEY 环境变量")
    tools = list(TOOL_SCHEMAS)
    if allow_proposals:
        tools.append(PROPOSE_TOOL_SCHEMA)

    payload = {
        "model": "deepseek-flash",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto" if allow_tools else "none",
        "thinking": {"type": "disabled"},
        "max_tokens": 1200,
        "stream": False,
    }

    request = Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        result = json.load(response)

    choice = result["choices"][0]

    if choice.get("finish_reason") == "length":
        raise AgentResponseError(
            "MODEL_OUTPUT_TRUNCATED",
            "模型输出达到长度上限，本轮未提交",
        )

    return choice["message"]

def run_query(
        question: str,
        history: list | None = None,
        *,
        session_id: str | None = None
    ) -> dict:
    messages: list[dict[str, Any]] = []
    pending_actions = []
    if history:
        messages.extend(deepcopy(history))
    else:
        messages.append({
            "role": "system",
            "content": (
                "你是工单查询助手。查询具体工单必须使用工具，"
                "不得编造结果；缺少编号时先询问用户。"
                "可以提出工单状态变更建议，但工具成功仅表示等待用户确认，"
                "不代表工单已修改。展示建议编号、原状态和目标状态，"
                "提示用户通过独立的确认或拒绝操作处理。"
                "优先回答用户问题，简要概括，不整篇复述检索文档。"
            ),
        })    
    messages.append({
        "role": "user",
        "content": question
    })
    tool_results = []
    seen_call_ids = set()
    for round_index in range(MAX_MODEL_ROUNDS):
        message = send_messages(
            messages,
            allow_tools=True,
            allow_proposals=session_id is not None,
        )
        if not isinstance(message, dict):
            raise AgentResponseError("MODEL_PROTOCOL_ERROR", "模型消息格式异常")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise AgentResponseError("MODEL_PROTOCOL_ERROR", "模型消息内容格式异常")
        calls = message.get("tool_calls") or []
        if not calls:
            answer = message.get("content") or ""
            if not isinstance(answer, str) or not answer.strip():
                raise AgentResponseError("MODEL_PROTOCOL_ERROR", "模型未返回回答或工具调用")
            if any(marker in answer for marker in DSML_MARKERS):
                raise AgentResponseError(
                    "MODEL_PROTOCOL_ERROR",
                    "模型将工具调用写入了普通文本，本轮未提交，请重新发起请求",
                )
            messages.append({"role": "assistant", "content": answer})
            return {
                "answer": answer,
                "messages": messages,
                "tool_results": tool_results,
                "pending_actions": pending_actions,
            }

        if not isinstance(calls, list):
            raise AgentResponseError("MODEL_PROTOCOL_ERROR", "工具调用列表格式异常")
        if (
            round_index == MAX_MODEL_ROUNDS - 1
            or len(tool_results) + len(calls) > MAX_TOOL_CALLS
        ):
            raise AgentResponseError(
                "MODEL_STEP_LIMIT", "模型达到执行次数上限，本轮未提交"
            )

        normalized_calls = []
        for call in calls:
            if not isinstance(call, dict):
                raise AgentResponseError("MODEL_PROTOCOL_ERROR", "工具调用格式异常")
            function = call.get("function")
            call_id = call.get("id")
            if (
                call.get("type") != "function"
                or not isinstance(call_id, str)
                or not call_id
                or call_id in seen_call_ids
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not isinstance(function.get("arguments"), str)
            ):
                raise AgentResponseError("MODEL_PROTOCOL_ERROR", "工具调用字段异常")
            seen_call_ids.add(call_id)
            normalized_calls.append({
                "id": call_id,
                "type": "function",
                "function": function,
            })

        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": normalized_calls,
        })
        for call in normalized_calls:
            function = call["function"]
            arguments = None
            try:
                arguments = json.loads(function["arguments"])
            except json.JSONDecodeError:
                tool_result = tool_error(
                    "INVALID_TOOL_ARGUMENTS", "模型返回的参数不是合法 JSON"
                )
            else:
                tool_result = execute_tool(
                    json.dumps({
                        "tool": function["name"],
                        "args": arguments,
                    }, ensure_ascii=False),
                    session_id=session_id,
                    pending_actions=pending_actions,
                )
            tool_results.append({
                "name": function["name"],
                "arguments": arguments,
                "raw_arguments": function["arguments"],
                "result": tool_result,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

    # 正常情况下循环内一定会 return 或 raise；
    # 保留兜底，避免以后改循环边界时返回未定义变量。
    raise AgentResponseError(
        "MODEL_STEP_LIMIT", "模型达到执行次数上限，本轮未提交"
    )
        

def main():
    history = []
    while True:
        question = input("你：").strip()

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = run_query(question, history=history)
        history = result["messages"]

        print("工具记录：")
        print(json.dumps(
            result["tool_results"],
            ensure_ascii=False,
            indent=2,
        ))
        print("助手：", result["answer"])

if __name__ == "__main__":
    try:
        main()
    except AgentResponseError as exc:
        print(f"{exc.code}：{exc}")
    except HTTPError as exc:
        print(f"API 请求失败，HTTP 状态码：{exc.code}")
        print(exc.read().decode("utf-8", errors="replace"))
    except URLError as exc:
        print(f"网络连接失败：{exc.reason}")
    except (ValueError, TimeoutError) as exc:
        print(f"请求失败：{exc}")
