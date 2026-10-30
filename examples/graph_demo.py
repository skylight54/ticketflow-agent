from typing import NotRequired, TypedDict

from langgraph.graph import StateGraph, START, END
from ticket import get_ticket


class TicketState(TypedDict):
    ticket_id: str
    result: NotRequired[dict]
    answer: NotRequired[str]


def fetch_ticket(state: TicketState) -> dict:
    return {
        "result": get_ticket(state["ticket_id"]),
    }


def format_success(state: TicketState) -> dict:
    result = state.get("result") or {}
    data = result["data"]

    return {
        "answer": (
            f"工单 {data['ticket_id']} 当前为 {data['status']}，"
            f"负责人是 {data['owner']}。"
        )
    }


def format_error(state: TicketState) -> dict:
    result = state.get("result") or {}
    error = result.get("error") or {}
    return {"answer": f"查询失败：{error.get('message', '未知错误')}"}


def route_after_fetch(state: TicketState) -> str:
    result = state.get("result") or {}
    return "success" if result.get("ok") else "error"


builder = StateGraph(TicketState)

builder.add_node("fetch_ticket", fetch_ticket)
builder.add_node("format_success", format_success)
builder.add_node("format_error",format_error)


builder.add_edge(START, "fetch_ticket")
builder.add_conditional_edges(
    "fetch_ticket",
    route_after_fetch,
    {
        "success": "format_success",
        "error": "format_error",
    },
)
builder.add_edge("format_success", END)
builder.add_edge("format_error", END)

graph = builder.compile()


if __name__ == "__main__":
    """result = graph.invoke({"ticket_id": "T108"})
    print(result)"""
    for update in graph.stream(
        {"ticket_id": "T108"},
        stream_mode="updates"):
        print(update)