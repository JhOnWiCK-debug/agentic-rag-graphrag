from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


# -------------------
# STATE
# -------------------

class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    reflection: str


# -------------------
# MODELS
# -------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatOllama(
    model="llama3:8b"
)

client = chromadb.PersistentClient(
    path="./vector_store"
)

collection = client.get_collection(
    "resume"
)


# -------------------
# NODES
# -------------------

def planner(state):

    print("\nPlanner Running")

    return state


def retriever(state):

    print("Retriever Running")

    query_embedding = embedding_model.encode(
        state["question"]
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n".join(
        results["documents"][0]
    )

    state["context"] = context

    return state


def reasoner(state):

    print("Reasoner Running")

    prompt = f"""
Answer only from the context.

Context:
{state['context']}

Question:
{state['question']}
"""

    response = llm.invoke(prompt)

    state["answer"] = response.content

    return state


def reflection(state):

    print("Reflection Running")

    if len(state["answer"]) > 20:
        state["reflection"] = "Answer looks good"

    else:
        state["reflection"] = "Answer may be weak"

    return state


# -------------------
# GRAPH
# -------------------

workflow = StateGraph(GraphState)

workflow.add_node(
    "planner",
    planner
)

workflow.add_node(
    "retriever",
    retriever
)

workflow.add_node(
    "reasoner",
    reasoner
)

workflow.add_node(
    "reflection",
    reflection
)

workflow.set_entry_point(
    "planner"
)

workflow.add_edge(
    "planner",
    "retriever"
)

workflow.add_edge(
    "retriever",
    "reasoner"
)

workflow.add_edge(
    "reasoner",
    "reflection"
)

workflow.add_edge(
    "reflection",
    END
)

graph = workflow.compile()


# -------------------
# RUN
# -------------------

question = input(
    "\nAsk a Question: "
)

result = graph.invoke(
    {
        "question": question,
        "context": "",
        "answer": "",
        "reflection": ""
    }
)

print("\nFINAL ANSWER:\n")

print(result["answer"])

print("\nREFLECTION:")

print(result["reflection"])