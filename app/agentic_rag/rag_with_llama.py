from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3:8b"
)

context = """
Vishva completed B.Sc Data Science.
CGPA: 8.57
"""

question = "What is Vishva's CGPA?"

prompt = f"""
Answer only using the context.

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print(response.content)
