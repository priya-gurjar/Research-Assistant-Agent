import os
import pypdf
import tiktoken
import chromadb
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Load a local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def preprocess_text(text: str) -> str:
    """Basic preprocessing: normalize spaces and remove excessive newlines."""
    return " ".join(text.split())

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Splits text into overlapping chunks of a given size."""
    tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI tokenizer
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i : i + chunk_size]
        chunks.append(tokenizer.decode(chunk))
    return chunks


def embed_text(text: str) -> List[float]:
    """Generates embeddings using a local model."""
    return embedding_model.encode(text).tolist()

def store_in_chroma(chunks: List[Dict[str, str]]):
    """Stores text chunks and embeddings in ChromaDB."""
    client = chromadb.PersistentClient(path="/Users/priyagurjar/Desktop/Machine learning/Research Assistant Agent /data/chroma_db")
    collection = client.get_or_create_collection(name="research_papers")
    
    for doc in chunks:
        embedding = embed_text(doc["text"])
        collection.add(
            ids=[f"{doc['filename']}_chunk_{doc['chunk_id']}"],
            embeddings=[embedding],
            metadatas=[{"filename": doc["filename"], "chunk_id": doc["chunk_id"]}],
            documents=[doc["text"]]
        )

def ingest_pdfs(pdf_folder: str, chunk_size: int = 512, overlap: int = 50):
    """Loads, processes, chunks, and stores PDFs."""
    processed_docs = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            raw_text = extract_text_from_pdf(pdf_path)
            cleaned_text = preprocess_text(raw_text)
            chunks = chunk_text(cleaned_text, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                processed_docs.append({
                    "filename": filename,
                    "chunk_id": i,
                    "text": chunk
                })
    store_in_chroma(processed_docs)
    # print("✅ Successfully stored in ChromaDB!")

# Example usage
if __name__ == "__main__":
    pdf_folder = "/Users/priyagurjar/Desktop/Machine learning/Research Assistant Agent /data/raw_papers"
    ingest_pdfs(pdf_folder)
