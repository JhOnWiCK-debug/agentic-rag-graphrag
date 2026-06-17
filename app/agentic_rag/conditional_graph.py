from typing import TypedDict

from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    answer: str


def reasoner(state):

    print("Reasoner Running")

    state["answer"] = "Wrong Answer"

    return state


def reflection(state):

    print("Reflection Running")

    if "Wrong" in state["answer"]:
        return "retry"

    return "finish"


workflow = StateGraph(GraphState)

workflow.add_node("reasoner", reasoner)

workflow.set_entry_point("reasoner")


workflow.add_conditional_edges(
    "reasoner",
    reflection,
    {
        "retry": "reasoner",
        "finish": END
    }
)

graph = workflow.compile()

result = graph.invoke(
    {
        "answer": ""
    }
)

print(result)