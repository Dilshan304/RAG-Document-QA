# RAG Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on PDF documents using embeddings and the Ollama Qwen2.5-Coder:3b model.

## Features
- Loads and chunks PDF documents (e.g., `test.pdf`).
- Generates embeddings with SentenceTransformer (`all-MiniLM-L6-v2`).
- Stores data in a Chroma vector database.
- Answers questions using the Ollama API.

## Setup
1. Clone the repo: `git clone https://github.com/Dilshan304/RAG-Document-QA.git`
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt` (create `requirements.txt` with `pip freeze > requirements.txt`)
5. Install Ollama and the `qwen2.5-coder:3b` model: Follow Ollama docs.
6. Run the server: `ollama serve`
7. Run the app: `python src/app.py`

## Usage
- Place PDFs in the `data/` folder.
- Ask questions via the script (e.g., "What is the main advice...").

## Dependencies
- pypdf
- langchain_text_splitters
- sentence_transformers
- chromadb
- ollama-python

## Author
Dilshan<a href="https://github.com/Dilshan304" target="_blank" rel="noopener noreferrer nofollow"></a>

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).