from typing import TypedDict

from langgraph.graph import StateGraph, END

import chromadb
from sentence_transformers import SentenceTransformer


class GraphState(TypedDict):
    question: str
    plan: str
    context: str
    answer: str


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="./vector_store"
)

collection = client.get_collection(
    "resume"
 )


def planner(state):

    print("Planner Running")

    state["plan"] = (
        f"Find information needed to answer: "
        f"{state['question']}"
    )

    return state


def retriever(state):

    print("Retriever Running")

    query_embedding = model.encode(
        state["question"]
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    state["context"] = (
        results["documents"][0][0]
    )

    return state


def reasoner(state):

    print("Reasoner Running")

    state["answer"] = (
        f"Based on retrieved context:\n\n"
        f"{state['context'][:300]}"
    )

    return state


workflow = StateGraph(GraphState)

workflow.add_node("planner", planner)
workflow.add_node("retriever", retriever)
workflow.add_node("reasoner", reasoner)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "reasoner")
workflow.add_edge("reasoner", END)

graph = workflow.compile()

result = graph.invoke(
    {
        "question": "What is Vishva's CGPA?",
        "plan": "",
        "context": "",
        "answer": ""
    }
)

print("\nFINAL ANSWER:\n")
print(result["answer"])