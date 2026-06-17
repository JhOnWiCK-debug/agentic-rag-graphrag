from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3:8b"
)

response = llm.invoke(
    "What is machine learning?"
)

print(response.content)
