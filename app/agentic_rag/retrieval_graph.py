from typing import TypedDict

from langgraph.graph import StateGraph, END

import chromadb
from sentence_transformers import SentenceTransformer


class GraphState(TypedDict):
    question: str
    context: str


# Load Model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Connect ChromaDB
client = chromadb.PersistentClient(
    path="./vector_store"
)

collection = client.get_collection(
    "resume"
)


def retriever(state):

    print("Retriever Node Running")

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


workflow = StateGraph(GraphState)

workflow.add_node(
    "retriever",
    retriever
)

workflow.set_entry_point(
    "retriever"
)

workflow.add_edge(
    "retriever",
    END
)

graph = workflow.compile()

result = graph.invoke(
    {
        "question": "What is Vishva's CGPA?",
        "context": ""
    }
)

print(result["context"])