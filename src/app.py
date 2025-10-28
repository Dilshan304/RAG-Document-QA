import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from ollama_python.endpoints import GenerateAPI
import docx   # <-- NEW

# Step 1: Load and chunk documents
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(project_root, "data")
documents = []

def extract_text(file_path):
    """Return full text from PDF, TXT or DOCX."""
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return "".join(page.extract_text() or "" for page in reader.pages)
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.lower().endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

# ---------- NEW LOADING LOOP ----------
try:
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        if file.lower().endswith((".pdf", ".txt", ".docx")):
            text = extract_text(file_path)
            if text.strip():
                documents.append({"content": text, "metadata": {"source": file}})
            else:
                print(f"No text extracted from {file}")
    if not documents:
        print("No supported files with text found in data/ folder.")
except FileNotFoundError:
    print(f"Error: data/ folder not found at {data_folder}. Please create it and add files.")
    exit(1)
# ----------------------------------------

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for doc in documents:
    split_docs = text_splitter.create_documents([doc["content"]], metadatas=[doc["metadata"]])
    chunks.extend(split_docs)

print(f"Loaded {len(documents)} files, split into {len(chunks)} chunks.")
print("Sample chunk:", chunks[0].page_content if chunks else "No chunks")

# Step 2: Generate embeddings and store in Chroma
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts)
print("Generated embeddings for chunks.")

client_db = chromadb.Client(Settings(persist_directory=os.path.abspath(os.path.join(project_root, "chroma_db")), is_persistent=True, anonymized_telemetry=False))
collection = client_db.get_or_create_collection(name="rag_documents")
collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=[chunk.metadata for chunk in chunks],
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print(f"Stored {len(chunks)} chunks in Chroma.")

# Step 3: Q&A with Ollama
def ask_question(question):
    query_embedding = model.encode([question])
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    retrieved_texts = results['documents'][0]
    prompt = f"Based on this context: {retrieved_texts}\n\nQuestion: {question}\nAnswer:"
    api = GenerateAPI(model="qwen2.5-coder:3b")
    response = api.generate(prompt=prompt)
    return getattr(response, 'response', 'No response available')

# Test the Q&A
question = "for designcrowd, can i upload other's work?"
answer = ask_question(question)
print(f"Question: {question}")
print(f"Answer: {answer}")