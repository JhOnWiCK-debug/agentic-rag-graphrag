from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("data/Vishva_M_Resume_OnePage_pdf.pdf")
documents = loader.load()

# Create splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split documents
chunks = splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

print("\nFIRST CHUNK:\n")
print(chunks[0].page_content)