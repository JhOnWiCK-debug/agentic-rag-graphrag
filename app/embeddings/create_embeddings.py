from sentence_transformers import SentenceTransformer

print("Loading Model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model Loaded")

text = "My CGPA is 8.57"

embedding = model.encode(text)

print(f"Vector Length: {len(embedding)}")

print(embedding[:10])