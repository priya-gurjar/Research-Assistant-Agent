import chromadb
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict
CHROMA_DB_PATH = os.path.join(os.getcwd(), "data/chroma_db")
# Load the sentence transformer model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_query(query: str) -> List[float]:
    """Generates an embedding for the given query."""
    return embedding_model.encode(query).tolist()

def retrieve_from_chroma(query: str, top_k: int = 5):
    """Retrieves top-k most relevant document chunks from ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name="research_papers")
    
    query_embedding = embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    return results

def retrieve_documents(query: str, top_k: int = 3) -> List[Dict]:
    """Retrieves relevant document chunks and formats them for LLM input."""
    results = retrieve_from_chroma(query, top_k)
    
    # Check if results are empty to prevent indexing errors
    if not results.get("documents") or not results["documents"][0]:
        print("⚠️ No relevant documents found!")
        return []  # Return an empty list to prevent errors

    retrieved_docs = []
    for i in range(len(results["documents"][0])):  # Iterate over retrieved chunks
        retrieved_docs.append({
            "text": results["documents"][0][i],  # Extract text
            "metadata": results["metadatas"][0][i]  # Extract metadata
        })

    return retrieved_docs


# Example usage
if __name__ == "__main__":
    query_text = "What are the main findings regarding parental sharing?"
    retrieved_docs = retrieve_documents(query_text)
    for i, doc in enumerate(retrieved_docs):
        print(f"Rank {i+1}: {doc['text']}")
        print(f"Metadata: {doc['metadata']}")
        print("-" * 80)

