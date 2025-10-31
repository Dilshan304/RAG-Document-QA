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
import numpy as np

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
    try:
        requests.get("http://localhost:11434", timeout=2)
    except:
        st.sidebar.warning("Starting Ollama...")
        subprocess.Popen(["ollama", "serve"], shell=True)
        time.sleep(5)
    st.sidebar.success("Ollama ready")


# =============================================
# 3. TEXT EXTRACTION
# =============================================
def extract_text(uploaded_file):
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
# 4. CHUNKING & INDEXING
# =============================================
def load_and_index(uploaded_file):
    if not uploaded_file:
        st.warning("Upload a file.")
        return None, None, None, None

    text = extract_text(uploaded_file)
    if not text.strip():
        st.error("No text in file.")
        return None, None, None, None

    # ---------- smarter chunking ----------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=600,
        separators=["\n\n", "\n", ". ", "! ", "? "],
        keep_separator=True
    )
    docs = splitter.create_documents([text])
    metadatas = [
        {"source": uploaded_file.name, "section": f"Section {i//4 + 1}"}
        for i in range(len(docs))
    ]
    for d, m in zip(docs, metadatas):
        d.metadata = m

    # ---------- embeddings ----------
    model = SentenceTransformer('all-mpnet-base-v2')
    texts = [d.page_content for d in docs]
    with st.spinner(f"Embedding {len(texts)} chunks..."):
        embeddings = model.encode(texts, batch_size=16)

    # ---------- BM25 ----------
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # ---------- Chroma ----------
    client = chromadb.Client()
    collection = client.get_or_create_collection("rag_docs")
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[d.metadata for d in docs],
        ids=[f"chunk_{i}" for i in range(len(docs))]
    )

    st.success(f"Loaded **{uploaded_file.name}** → {len(docs)} chunks")
    return collection, model, bm25, docs


# =============================================
# 5. HYBRID SEARCH + RRF
# =============================================
def hybrid_search(question, collection, model, bm25, chunks, top_k=7):
    if not collection or not question:  # FIXED: Guard against empty question
        st.error("No question or document loaded for search.")
        return []

    # ---- semantic ----
    q_emb = model.encode([question])
    sem = collection.query(query_embeddings=q_emb, n_results=12)
    sem_docs = sem['documents'][0]
    sem_dist = sem['distances'][0]
    if sem_dist:
        max_d = max(sem_dist)
        sem_scores = [1 - (d / max_d) for d in sem_dist]
    else:
        sem_scores = []
    sem_rank = {doc: sc for doc, sc in zip(sem_docs, sem_scores)}

    # ---- BM25 ----
    q_tokens = question.lower().split()
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_idx = np.argsort(bm25_scores)[::-1][:12]
    bm25_docs = [chunks[i].page_content for i in bm25_idx if i < len(chunks)]
    bm25_norm = [bm25_scores[i] for i in bm25_idx]
    if bm25_norm and max(bm25_norm) > 0:
        bm25_norm = [s / max(bm25_norm) for s in bm25_norm]
    bm25_rank = {doc: sc for doc, sc in zip(bm25_docs, bm25_norm)}

    # ---- RRF ----
    all_docs = set(sem_rank) | set(bm25_rank)
    rrf = {}
    k = 60
    for doc in all_docs:
        sr = list(sem_rank).index(doc) + 1 if doc in sem_rank else float('inf')
        br = list(bm25_rank).index(doc) + 1 if doc in bm25_rank else float('inf')
        rrf[doc] = (1 / (k + sr)) + (1 / (k + br))

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    combined = [doc for doc, _ in ranked][:top_k]

    # UI: show chunks
    if combined:
        st.write(f"**From:** `{uploaded_file.name}`")
        for i, c in enumerate(combined):
            st.caption(f"Chunk {i+1}: {c[:200]}...")

    return combined


# =============================================
# 6. LLM + MEMORY (UPDATED PROMPT FOR DOUBT HANDLING)
# =============================================
def ask_question(question, context, chat_history):
    if not context or not question:  # FIXED: Guard against empty inputs
        return "No question or context available."

    # ---- format context ----
    context_str = "\n\n".join([f"Passage {i+1}: {c}" for i, c in enumerate(context)])

    # ---- recent 3 exchanges (6 msgs) ----
    recent = chat_history[-6:]
    hist = ""
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        hist += f"{role}: {m['content']}\n"

    # ---- UPDATED PROMPT: Handles doubt implicitly ----
    prompt = f"""You are a direct document assistant.
Answer using ONLY the provided context and history.
Quote the exact phrase or value from the context when possible.
If the answer cannot be derived, say exactly "Not found in the document."
Be concise - no explanations.

For follow-ups like "Are you sure?" or "Is that right?": 
- Re-review the context and history.
- Confirm by quoting additional relevant passages if available.
- If still uncertain, say "Based on the document, I confirm [answer], but here's more context: [quote]."
- Do not hallucinate or add external info.

Recent History (if any):
{hist}

Context:
{context_str}

Question: {question}

Answer:"""

    try:
        api = GenerateAPI(model="llama3.1:8b-instruct-q4_0")
        resp = api.generate(prompt=prompt, options={"temperature": 0.0})
        full = getattr(resp, 'response', 'No answer').strip()

        if "Answer:" in full:
            ans = full.split("Answer:")[-1].strip()
        else:
            lines = [l.strip() for l in full.split('\n') if l.strip()]
            ans = lines[-1] if lines else "Not found in the document."
        ans = ans.rstrip('.')
        return ans
    except Exception as e:
        return f"Error: {e}"


# =============================================
# 7. MAIN FLOW
# =============================================
with st.sidebar:
    start_ollama()
    uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'txt', 'docx'])

# ---- CACHE DOCUMENT ONCE ----
if uploaded_file:
    if ("cached_file" not in st.session_state or 
        st.session_state.cached_file != uploaded_file.name):
        collection, model, bm25, chunks = load_and_index(uploaded_file)
        st.session_state.collection = collection
        st.session_state.model = model
        st.session_state.bm25 = bm25
        st.session_state.chunks = chunks
        st.session_state.cached_file = uploaded_file.name
    else:
        collection = st.session_state.collection
        model = st.session_state.model
        bm25 = st.session_state.bm25
        chunks = st.session_state.chunks
else:
    st.info("Upload a file to begin.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "deep_search_trigger" not in st.session_state:
    st.session_state.deep_search_trigger = False

# ---- display history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---- new input ----
if prompt := st.chat_input("Ask about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_prompt = prompt  # FIXED: Store prompt for deep search
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # Always use top_k=7 for consistency
        ctx = hybrid_search(prompt, st.session_state.collection, st.session_state.model, 
                           st.session_state.bm25, st.session_state.chunks, top_k=7)
        answer = ask_question(prompt, ctx, st.session_state.messages)

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# FIXED: Deep search check moved outside 'if prompt' (runs every rerun)
if st.session_state.deep_search_trigger and 'last_prompt' in st.session_state:
    with st.chat_message("assistant"):
        with st.spinner("Searching with more context..."):
            deep_prompt = st.session_state.last_prompt
            deep_ctx = hybrid_search(deep_prompt, st.session_state.collection, st.session_state.model, 
                                   st.session_state.bm25, st.session_state.chunks, top_k=12)
            deep_answer = ask_question(deep_prompt, deep_ctx, st.session_state.messages)
        st.markdown("**Deep Search Result:**")
        st.write(deep_answer)
        st.session_state.messages.append({"role": "assistant", "content": f"Deep Search: {deep_answer}"})
    st.session_state.deep_search_trigger = False  # Reset

# Button to trigger deep search (only show if there's a last prompt)
if 'last_prompt' in st.session_state and st.button("Not sure? Search deeper"):
    st.session_state.deep_search_trigger = True
    st.rerun()

# ---- clear button ----
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    if 'last_prompt' in st.session_state:
        del st.session_state.last_prompt
    st.rerun()