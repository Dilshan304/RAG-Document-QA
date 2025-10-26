import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer  # For embeddings
import chromadb  # For vector store
from chromadb.config import Settings

# Step 1: Load and chunk documents
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(project_root, "data")
documents = []
try:
    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, file)
            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                if text.strip():
                    documents.append({"content": text, "metadata": {"source": file}})
                else:
                    print(f"No text extracted from {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
    if not documents:
        print("No PDFs with extractable text found in data/ folder.")
except FileNotFoundError:
    print(f"Error: data/ folder not found at {data_folder}. Please create it and add PDFs.")
    exit(1)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for doc in documents:
    split_docs = text_splitter.create_documents([doc["content"]], metadatas=[doc["metadata"]])
    chunks.extend(split_docs)

print(f"Loaded {len(documents)} PDFs, split into {len(chunks)} chunks.")
print("Sample chunk:", chunks[0].page_content if chunks else "No chunks")

# Step 2: Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts)
print("Generated embeddings for chunks.")

# Step 3: Store in Chroma
client = chromadb.Client(Settings(persist_directory=os.path.join(project_root, "chroma_db"), anonymized_telemetry=False))
collection = client.get_or_create_collection(name="rag_documents")
collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=[chunk.metadata for chunk in chunks],
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print(f"Stored {len(chunks)} chunks in Chroma.")

# Optional: Test retrieval (add this to verify)
query = "What is the main advice in the document?"
query_embedding = model.encode([query])
results = collection.query(query_embeddings=query_embedding, n_results=3)
print("Sample retrieval results:", results['documents'])