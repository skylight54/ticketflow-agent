from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END


class DemoState(TypedDict):
    replaced: list[str]
    accumulated: Annotated[list[str], add]


def node_a(state: DemoState) -> dict:
    return {
        "replaced": ["A"],
        "accumulated": ["A"],
    }


def node_b(state: DemoState) -> dict:
    return {
        "replaced": ["B"],
        "accumulated": ["B"],
    }


builder = StateGraph(DemoState)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({
        "replaced": [],
        "accumulated": [],
    })
    print(result)