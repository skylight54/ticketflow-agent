from typing import NotRequired, TypedDict, Literal
from copy import deepcopy
from typing import Any
from langgraph.graph import StateGraph,START,END
import json

from tool_runner import execute_tool, tool_error
from deepseek_demo import (
    send_messages,
    AgentResponseError,
    MAX_MODEL_ROUNDS,
    MAX_TOOL_CALLS,
    DSML_MARKERS
)


class AgentState(TypedDict):
    messages: list
    session_id: str | None
    model_rounds: int
    tool_results: list
    pending_actions: list
    last_message: NotRequired[dict]
    answer: NotRequired[str]

def model_node(state: AgentState) -> dict:
    # 1. model_rounds 已达到上限：抛出 AgentResponseError
    # 2. 调用 send_messages
    # 3. 检查返回值是字典
    # 4. 返回 last_message 和递增后的 model_rounds
    if state["model_rounds"] >= MAX_MODEL_ROUNDS:
        raise AgentResponseError(
            "MODEL_STEP_LIMIT", "模型达到执行次数上限，本轮未提交")
    message = send_messages(
        deepcopy(state["messages"]),
        allow_tools=True,
        allow_proposals=state["session_id"] is not None,
    )
    if not isinstance(message,dict):
        raise AgentResponseError("MODEL_PROTOCOL_ERROR", "模型消息格式异常")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise AgentResponseError("MODEL_PROTOCOL_ERROR", "模型消息内容格式异常")
    return{
        "last_message":message,
        "model_rounds":state["model_rounds"]+1
    }

def route_after_model(
    state: AgentState,
) -> Literal["tools", "finish"]:
    # last_message 有 tool_calls → "tools"
    # 否则 → "finish"
    last_message = state.get('last_message') or {}
    if last_message.get('tool_calls'):
        return 'tools'
    return 'finish'

def tools_node(state: AgentState) -> dict:
    messages = deepcopy(state["messages"])
    tool_results = deepcopy(state["tool_results"])
    pending_actions = deepcopy(state["pending_actions"])

    last_message = state.get('last_message') or {}
    calls = last_message.get("tool_calls")

    # 校验调用格式与预算
    # 保存 assistant 的调用消息
    # 逐个解析参数、执行工具、记录结果
    # 返回更新后的三个完整列表

    # calls校验
    if not isinstance(calls, list) or not calls:
        raise AgentResponseError(
            "MODEL_PROTOCOL_ERROR",
            "工具调用必须是非空列表",
        )
    if (
        state["model_rounds"] >= MAX_MODEL_ROUNDS
        or len(tool_results) + len(calls) > MAX_TOOL_CALLS
    ):
        raise AgentResponseError(
            "MODEL_STEP_LIMIT", "模型达到执行次数上限，本轮未提交"
        )
    # call_id初始化：避免重复执行
    seen_call_ids = set()
    for message in messages:
        for previous_call in message.get("tool_calls") or []:
            seen_call_ids.add(previous_call["id"])

    # call校验合理调用，同时形成这轮对话tool_calls
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
        "content": last_message.get("content") or "",
        "tool_calls": normalized_calls,
    })
    # 执行每个调用
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
                session_id=state["session_id"],
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
    return{
        "messages": messages,
        "tool_results": tool_results,
        "pending_actions": pending_actions,
    }

def finish_node(state: AgentState) -> dict:

    last_message = state.get('last_message') or {}
    answer = last_message.get("content")

    # 请你完成校验：
    # 必须是非空字符串
    # 不能包含之前出现过的 DSML 工具协议标记
    # 不符合时抛出 MODEL_PROTOCOL_ERROR
    if not isinstance(answer, str) or not answer.strip():
        raise AgentResponseError("MODEL_PROTOCOL_ERROR", "模型未返回回答或工具调用")
    if any(marker in answer for marker in DSML_MARKERS):
        raise AgentResponseError(
            "MODEL_PROTOCOL_ERROR",
            "模型将工具调用写入了普通文本，本轮未提交，请重新发起请求",
        )
    
    messages = deepcopy(state["messages"])
    messages.append({
        "role": "assistant",
        "content": answer,
    })

    return {
        "answer": answer,
        "messages": messages,
    }

builder = StateGraph(AgentState)

builder.add_node("model", model_node)
builder.add_node("tools", tools_node)
builder.add_node("finish", finish_node)

builder.add_edge(START, "model")

builder.add_conditional_edges(
    "model",
    route_after_model,
    {
        "tools": "tools",
        "finish": "finish",
    },
)

builder.add_edge("tools", "model")
builder.add_edge("finish", END)

graph = builder.compile()

def run_graph_query(
    question: str,
    history: list | None = None,
    *,
    session_id: str | None = None,
) -> dict:
    messages: list[dict[str, Any]] = []
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
            ),
        })    
    messages.append({
        "role": "user",
        "content": question
    })
    initial_state: AgentState = {
        "messages": messages,
        "session_id": session_id,
        "model_rounds": 0,
        "tool_results": [],
        "pending_actions": [],
    }
    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "messages": final_state["messages"],
        "tool_results": final_state["tool_results"],
        "pending_actions": final_state["pending_actions"],
    }
