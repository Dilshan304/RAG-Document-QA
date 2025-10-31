# RAG Document Q&A Chatbot

A **Streamlit-based Retrieval-Augmented Generation (RAG)** chatbot that answers questions from uploaded documents (PDF, DOCX, TXT).  
Uses **hybrid retrieval (semantic + BM25 with RRF fusion)** via **SentenceTransformer (all-mpnet-base-v2)** and powered by **Ollama Llama 3.1:8b**.

---

## Features

- Upload and analyze **PDF**, **DOCX**, or **TXT** files  
- **Smarter chunking** with overlap and sentence-aware splitting  
- **Hybrid search** with **RRF ranking** for better accuracy  
- Natural follow-ups handled via **LLM context reasoning**  
- **"Search deeper"** button to retrieve more context on demand  
- **No hallucinations** — answers strictly from document  
- Built with **Streamlit** for clean, interactive UI  
- **In-memory ChromaDB** for fast, isolated vector search  
- **Fully offline** (only Ollama required)

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Dilshan304/RAG-Document-QA.git
cd RAG-Document-QA
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, you can generate it:
```bash
pip freeze > requirements.txt
```

### 4. Install and run Ollama
Follow [Ollama’s installation guide](https://ollama.ai/download).

Then pull the required model and start the server:
```bash
ollama pull llama3.1:8b-instruct-q4_0
ollama serve
```

---

## Run the App

Start the chatbot interface:
```bash
streamlit run src/app.py
```

> **Note:** The app automatically starts the Ollama server if it’s not running.

---

## Usage

1. Open the Streamlit app (usually at [http://localhost:8501](http://localhost:8501))  
2. Upload a **PDF**, **DOCX**, or **TXT** file  
3. Ask any question related to the document  
4. Use "Not sure? Search deeper" for expanded results 

---

## Dependencies

- streamlit
- pypdf
- python-docx
- langchain-text-splitters
- sentence-transformers
- chromadb
- ollama-python
- rank-bm25
- numpy
- requests
---

## Author

**Dilshan**

---

## License

This project is licensed under the **MIT License**.

---