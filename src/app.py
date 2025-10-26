import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load documents from data folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(project_root, "data")
print(f"Looking for data folder at: {data_folder}")
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

# Step 2: Chunk the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = []
for doc in documents:
    split_docs = text_splitter.create_documents(
        [doc["content"]], metadatas=[doc["metadata"]]
    )
    chunks.extend(split_docs)

# Step 3: Test output
print(f"Loaded {len(documents)} PDFs, split into {len(chunks)} chunks.")
print("Sample chunk:", chunks[0].page_content if chunks else "No chunks")