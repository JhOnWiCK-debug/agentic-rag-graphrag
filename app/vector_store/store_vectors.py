from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import chromadb

# Load PDF
loader = PyPDFLoader("data/Vishva_M_Resume_OnePage_pdf.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Create ChromaDB client
client = chromadb.PersistentClient(
    path="./vector_store"
)

collection = client.get_or_create_collection(
    name="resume"
)

# Store chunks
for i, chunk in enumerate(chunks):

    embedding = model.encode(
        chunk.page_content
    ).tolist()

    collection.add(
        ids=[str(i)],
        documents=[chunk.page_content],
        embeddings=[embedding]
    )

print(f"Stored {len(chunks)} chunks")