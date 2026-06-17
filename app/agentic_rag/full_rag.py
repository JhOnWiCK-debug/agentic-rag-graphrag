import chromadb

from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama


# Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load Llama 3
llm = ChatOllama(
    model="llama3:8b"
)

# Connect ChromaDB
client = chromadb.PersistentClient(
    path="./vector_store"
)

collection = client.get_collection(
    "resume"
)

while True:

    question = input("\nAsk a Question: ")

    if question.lower() == "exit":
        break

    # Convert question to embedding
    query_embedding = embedding_model.encode(
        question
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n".join(
        results["documents"][0]
    )

    prompt = f"""
Answer only from the provided context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)