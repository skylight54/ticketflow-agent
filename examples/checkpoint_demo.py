from typing import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig 

import argparse
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

class ApprovalState(TypedDict):
    ticket_id: str
    approved: NotRequired[bool]
    status: NotRequired[str]


def wait_for_approval(state: ApprovalState) -> dict:
    print("进入确认节点")

    decision = interrupt({
        "ticket_id": state["ticket_id"],
        "question": "是否同意将工单改为处理中？",
    })

    if not isinstance(decision, bool):
        raise ValueError("确认结果必须是布尔值")

    return {"approved": decision}


def finish(state: ApprovalState) -> dict:
    approved = state.get("approved", False)  # 给默认值
    return {
        "status": (
            "模拟执行完成" if approved else "已取消"
        )
    }

def build_approval_graph(checkpointer):
    builder = StateGraph(ApprovalState)

    builder.add_node("wait", wait_for_approval)
    builder.add_node("finish", finish)

    builder.add_edge(START, "wait")
    builder.add_edge("wait", "finish")
    builder.add_edge("finish", END)

    return builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["start", "resume"])
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--decision", choices=["y", "n"])
    parser.add_argument(
        "--db-path",
        default=str(Path(__file__).resolve().parents[1] / "checkpoints.sqlite"),
    )
    args = parser.parse_args()

    if args.mode == "resume" and args.decision is None:
        parser.error("resume 模式需要 --decision y 或 n")

    db_path = Path(args.db_path).resolve()

    config: RunnableConfig = {
        "configurable": {"thread_id": args.thread_id}
    }

    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = build_approval_graph(saver)
        snapshot = graph.get_state(config)

        if args.mode == "start":
            if snapshot.values:
                raise SystemExit("该 thread_id 已存在，请使用新编号")

            result = graph.invoke({"ticket_id": "T109"}, config)
        else:
            if not snapshot.values:
                raise SystemExit("找不到该 thread_id，请先 start")

            if not snapshot.interrupts:
                raise SystemExit("该流程没有等待确认的中断")

            result = graph.invoke(
                Command(resume=args.decision == "y"),
                config,
            )

        print("运行结果：", result)
        print("等待执行的节点：", graph.get_state(config).next)