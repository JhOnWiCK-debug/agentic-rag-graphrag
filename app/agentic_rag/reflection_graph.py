from typing import TypedDict

from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    question: str
    answer: str
    reflection: str


def planner(state):
    print("Planner Running")
    return state


def reasoner(state):

    print("Reasoner Running")

    state["answer"] = (
        "Vishva's CGPA is 8.57"
    )

    return state


def reflection(state):

    print("Reflection Running")

    if "8.57" in state["answer"]:

        state["reflection"] = (
            "Answer looks correct."
        )

    else:

        state["reflection"] = (
            "Answer may be wrong."
        )

    return state


workflow = StateGraph(GraphState)

workflow.add_node("planner", planner)
workflow.add_node("reasoner", reasoner)
workflow.add_node("reflection", reflection)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "reasoner")
workflow.add_edge("reasoner", "reflection")
workflow.add_edge("reflection", END)

graph = workflow.compile()

result = graph.invoke(
    {
        "question": "What is Vishva's CGPA?",
        "answer": "",
        "reflection": ""
    }
)

print(result)