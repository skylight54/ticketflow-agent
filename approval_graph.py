from typing import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig 
from db import load_proposed_action

from action_service import (
    confirm_proposed_action,
    reject_proposed_action,
)

class ApprovalState(TypedDict):
    session_id: str
    action_id: str
    action: NotRequired[dict]
    approved: NotRequired[bool]
    result: NotRequired[dict]

def build_approval_graph(checkpointer):
    builder = StateGraph(ApprovalState)

    builder.add_node("load",load_action_node)
    builder.add_node("wait", wait_for_approval)
    builder.add_node("apply", apply_decision_node)

    builder.add_edge(START, "load")
    builder.add_edge("load", "wait")
    builder.add_edge("wait", "apply")
    builder.add_edge("apply",END)

    return builder.compile(checkpointer=checkpointer)

def load_action_node(state: ApprovalState) -> dict:
    action = load_proposed_action(state["action_id"])

    if action is None:
        raise ValueError("建议不存在")

    if action["session_id"] != state["session_id"]:
        raise ValueError("建议不属于当前会话")

    if action["status"] != "pending":
        raise ValueError("建议已处理，不能发起审批")

    return {"action": action}

def wait_for_approval(state: ApprovalState) -> dict:
    if "action" not in state:
        raise ValueError("缺少审批建议")
    action = state["action"]
    ticket_id = action["ticket_id"]
    old_status = action["old_status"]
    new_status = action["new_status"]
    decision = interrupt({
        **action,
        "question": f"是否同意将工单{ticket_id}由{old_status}改为{new_status}？",
    })

    if not isinstance(decision, bool):
        raise ValueError("确认结果必须是布尔值")

    return {"approved": decision}

def apply_decision_node(state: ApprovalState) -> dict:
    if "approved" not in state:
        raise ValueError("缺少确认结果")

    if state["approved"]:
        result = confirm_proposed_action(
            state["session_id"],
            state["action_id"],
        )
    else:
        result = reject_proposed_action(
            state["session_id"],
            state["action_id"],
        )

    return {"result": result}