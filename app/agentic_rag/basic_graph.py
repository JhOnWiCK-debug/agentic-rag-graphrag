from typing import TypedDict
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    question: str
    answer: str

def planner(state):
    print("Planner Node Running")
    return state

def reasoner(state):
    print("Reasoner Node Running")

    state["answer"] = (
        f"You asked: {state['question']}"
    )

    return state

workflow = StateGraph(GraphState)

workflow.add_node("planner", planner)
workflow.add_node("reasoner", reasoner)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "reasoner")
workflow.add_edge("reasoner", END)

graph = workflow.compile()

result = graph.invoke(
    {
        "question": "What is my CGPA?",
        "answer": ""
    }
)

print(result)