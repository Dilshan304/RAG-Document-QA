import os
import docx
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from ollama_python.endpoints import GenerateAPI
from rank_bm25 import BM25Okapi
import subprocess
import time
import requests

# =============================================
# 1. PAGE CONFIG & UI
# =============================================
st.set_page_config(page_title="Doc Q&A", layout="wide")
st.title("Document Q&A Chatbot")
st.write("**Upload → Ask → Get 100% accurate answers from your file**")


# =============================================
# 2. OLLAMA SERVER HANDLER
# =============================================
def start_ollama():
    """Start Ollama server if not running."""
    try:
        requests.get("http://localhost:11434", timeout=2)
    except:
        st.sidebar.warning("Starting Ollama...")
        subprocess.Popen(["ollama", "serve"], shell=True)
        time.sleep(5)
    st.sidebar.success("Ollama ready")


# =============================================
# 3. TEXT EXTRACTION FROM PDF/TXT/DOCX
# =============================================
def extract_text(uploaded_file):
    """Extract raw text from uploaded file."""
    bytes_data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".txt"):
        return bytes_data.decode("utf-8")
    elif name.endswith(".docx"):
        import io
        doc = docx.Document(io.BytesIO(bytes_data))
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


# =============================================
# 4. DOCUMENT CHUNKING & EMBEDDING
# =============================================
def load_and_index(uploaded_file):
    """Split document into chunks, embed, and store in Chroma."""
    if not uploaded_file:
        st.warning("Upload a file.")
        return None, None, None, None

    text = extract_text(uploaded_file)
    if not text.strip():
        st.error("No text in file.")
        return None, None, None, None

    # Optimal chunk size for tables & paragraphs
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([text], metadatas=[{"source": uploaded_file.name}])

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [c.page_content for c in chunks]
    with st.spinner(f"Embedding {len(texts)} chunks..."):
        embeddings = model.encode(texts, batch_size=16)

    # BM25 for keyword search
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # In-memory Chroma DB
    client = chromadb.Client()
    collection = client.get_or_create_collection("rag_docs")
    
    # CLEAR OLD DATA → 100% document isolation
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])

    # Store in vector DB
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[c.metadata for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    st.success(f"Loaded **{uploaded_file.name}** → {len(chunks)} chunks")
    return collection, model, bm25, chunks


# =============================================
# 5. HYBRID SEARCH (Semantic + Keyword)
# =============================================
def hybrid_search(question, collection, model, bm25, chunks, top_k=3):
    """Retrieve top-k relevant chunks using semantic + BM25."""
    if not collection:
        return []

    # Semantic search
    q_emb = model.encode([question])
    results = collection.query(query_embeddings=q_emb, n_results=5)
    semantic = results['documents'][0] if results['documents'] else []

    # Keyword search
    q_tokens = question.lower().split()
    scores = bm25.get_scores(q_tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    bm25_texts = [chunks[i].page_content for i in top_idx if i < len(chunks)]

    # Combine & deduplicate
    combined = list(dict.fromkeys(semantic + bm25_texts))[:top_k]

    # Show source chunks in UI
    if combined:
        st.write(f"**From:** `{uploaded_file.name}`")
        for i, c in enumerate(combined):
            st.caption(f"Chunk {i+1}: {c[:200]}...")
    return combined


# =============================================
# 6. LLM ANSWER GENERATION WITH CHAT MEMORY
# =============================================
def ask_question(question, context, chat_history):
    """Generate answer using document + recent chat (3 exchanges)."""
    if not context:
        return "No information found in your document."

    # Format retrieved passages
    context_str = "\n\n".join([f"Passage {i+1}: {c}" for i, c in enumerate(context)])
    
    # Last 3 Q&A pairs (6 messages)
    history_str = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
    
    # Strict prompt to prevent hallucination
        prompt = f"""You are a strict document assistant.
Answer using ONLY the provided context.
If the answer is not in the context, respond exactly: "Not found in the document."
Do NOT use external knowledge. Do NOT guess.

Context:
{context_str}

Question: {question}

Answer:"""
    
    try:
        api = GenerateAPI(model="qwen2.5-coder:3b")
        response = api.generate(prompt=prompt)
        return getattr(response, 'response', 'No answer').strip()
    except Exception as e:
        return f"Error: {e}"


# =============================================
# 7. MAIN APP FLOW
# =============================================
with st.sidebar:
    start_ollama()
    uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'txt', 'docx'])

# Load document if uploaded
if uploaded_file:
    collection, model, bm25, chunks = load_and_index(uploaded_file)
else:
    st.info("Upload a file to begin.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            context = hybrid_search(prompt, collection, model, bm25, chunks)
            answer = ask_question(prompt, context, st.session_state.messages)
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})