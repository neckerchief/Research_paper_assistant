from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import json
from pathlib import Path

class VectorStore:
    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_directory: str = "data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store with embeddings.
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Where to persist the database
            embedding_model: SentenceTransformer model name
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Research paper chunks with embeddings"}
        )
        
        print(f"✓ Vector store initialized")
        print(f"  - Collection: {collection_name}")
        print(f"  - Embedding model: {embedding_model}")
        print(f"  - Current documents: {self.collection.count()}")
    
    def add_documents(self, chunks: List[Dict]):
        """
        Add document chunks to vector store.
        
        Args:
            chunks: List of chunk dictionaries
        """
        print(f"\nAdding {len(chunks)} chunks to vector store...")
        
        # Prepare data for ChromaDB
        documents = [chunk['text'] for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Prepare metadata (ChromaDB needs simple types)
        metadatas = []
        for chunk in chunks:
            metadata = {
                'arxiv_id': chunk.get('arxiv_id', ''),
                'title': chunk.get('title', ''),
                'authors': ', '.join(chunk.get('authors', [])),
                'published': chunk.get('published', ''),
                'categories': ', '.join(chunk.get('categories', []))
            }
            metadatas.append(metadata)
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        
        # Add to ChromaDB
        print("Adding to database...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(chunks)} chunks to vector store")
        print(f"  Total documents: {self.collection.count()}")
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
        
        Returns:
            Dictionary with results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results

# Example usage
if __name__ == "__main__":
    # Load processed chunks
    with open("data/processed/chunks.json", 'r') as f:
        chunks = json.load(f)
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Add documents
    vector_store.add_documents(chunks)
    
    # Test search
    query = "What is quantum entanglement?"
    results = vector_store.search(query, n_results=3)
    
    print(f"\nSearch results for: '{query}'")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n--- Result {i+1} ---")
        print(doc[:200] + "...")