<img width="1133" height="527" alt="demo png" src="https://github.com/user-attachments/assets/ed06118a-83f0-4e2b-9991-9a6ca0ee8e37" />
# Agentic RAG System with LangGraph and Local LLMs

## Overview

This project implements an Agentic Retrieval-Augmented Generation (RAG) system using LangGraph, ChromaDB, Sentence Transformers, Ollama, and Llama 3.

The system retrieves relevant information from documents, generates context-aware responses using a local Large Language Model (LLM), and evaluates the quality of generated answers through a Reflection Agent.

---

## Architecture

User Question
↓
Planner Agent
↓
Retriever Agent
↓
ChromaDB Vector Store
↓
Reasoner Agent (Llama 3 via Ollama)
↓
Reflection Agent
↓
Final Answer

---

## Features

* PDF Document Loading
* Intelligent Text Chunking
* Embedding Generation
* ChromaDB Vector Storage
* Semantic Similarity Search
* Local LLM Inference using Ollama
* LangGraph Workflow Orchestration
* Planner Agent
* Retriever Agent
* Reasoner Agent
* Reflection Agent
* End-to-End Agentic RAG Pipeline

---

## Tech Stack

* Python
* LangChain
* LangGraph
* ChromaDB
* Sentence Transformers
* Ollama
* Llama 3
* Hugging Face

---

## Project Structure

app/

├── ingestion/

│ ├── load_pdf.py

│ └── chunk_pdf.py

├── embeddings/

│ └── create_embeddings.py

├── vector_store/

│ └── store_vectors.py

├── retrieval/

│ └── search.py

├── rag/

│ └── rag_chat.py

└── agentic_rag/

├── basic_graph.py

├── retrieval_graph.py

├── planner_graph.py

├── reflection_graph.py

├── full_rag.py

└── agentic_rag.py

---

## Example

Question:

What programming languages does Vishva know?

Answer:

* Python
* Java
* C++
* JavaScript
* SQL

---

## Future Improvements

* GraphRAG using Neo4j
* Multi-Agent Collaboration
* Tool Calling Agents
* Web Search Integration
* Streamlit Deployment
* FastAPI Backend

---

## Author

Vishva M

B.Sc Data Science

Saveetha University
