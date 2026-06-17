import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="./vector_store"
)

collection = client.get_collection(
    "resume"
)

question = input("Ask a question: ")

query_embedding = model.encode(
    question
).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

context = "\n".join(
    results["documents"][0]
)

prompt = f"""
Answer the question using the context.

Context:
{context}

Question:
{question}
"""

print(prompt)