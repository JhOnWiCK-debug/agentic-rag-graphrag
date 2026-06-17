import chromadb
from sentence_transformers import SentenceTransformer

# Load model
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

# Question
query = "What is Vishva's CGPA?"

# Convert question into vector
query_embedding = model.encode(
    query
).tolist()

# Search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print(results["documents"][0])