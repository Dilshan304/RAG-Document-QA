# RAG Document Q&A Chatbot

A **Streamlit-based Retrieval-Augmented Generation (RAG)** chatbot that answers your questions using uploaded documents (PDF, DOCX, or TXT).  
It uses **hybrid retrieval (semantic + keyword search)** with **SentenceTransformer** and **BM25**, powered by the **Ollama Qwen2.5-Coder:3b** model.

---

## Features

- Upload and analyze **PDF**, **DOCX**, or **TXT** files  
- Uses **SentenceTransformer (all-MiniLM-L6-v2)** for embeddings  
- Combines **semantic search** (vector) and **keyword search** (BM25) for higher accuracy  
- Chat interface with **conversation memory** (3 previous exchanges)  
- Answers are strictly limited to your document — **no hallucinations**  
- Built with **Streamlit** for a simple, interactive UI  
- Stores document embeddings in an **in-memory Chroma vector database**  
- Fully offline compatible (only Ollama required)

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
ollama pull qwen2.5-coder:3b
ollama serve
```

---

## Run the App

Start the chatbot interface:
```bash
streamlit run rag_pipeline.py
```

> **Note:** The app automatically starts the Ollama server if it’s not running.

---

## Usage

1. Open the Streamlit app (usually at [http://localhost:8501](http://localhost:8501))  
2. Upload a **PDF**, **DOCX**, or **TXT** file  
3. Ask any question related to the document  
4. View retrieved chunks and AI-generated answers  

---

## Dependencies

- `streamlit`  
- `pypdf`  
- `python-docx`  
- `langchain_text_splitters`  
- `sentence_transformers`  
- `chromadb`  
- `ollama-python`  
- `rank_bm25`  
- `requests`

---

## Author

**Dilshan**

---

## License

This project is licensed under the **MIT License**.

---