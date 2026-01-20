"""
Main script to run the complete Phase 1 pipeline.
"""
from src.download_papers import ArxivDownloader
from src.process_documents import DocumentProcessor
from src.embeddings import VectorStore
from src.rag_pipeline import RAGPipeline

def main():
    print("="*60)
    print("Research Paper Q&A System - Phase 1")
    print("="*60)
    
    # Step 1: Download papers
    print("\n[1/4] Downloading papers from ArXiv...")
    downloader = ArxivDownloader()
    papers = downloader.download_papers(
        query="quantum mechanics",
        max_results=20,  # Start small for testing
        category="quant-ph"
    )
    
    # Step 2: Process documents
    print("\n[2/4] Processing documents and creating chunks...")
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    chunks = processor.process_papers(papers)
    
    # Step 3: Create vector store
    print("\n[3/4] Creating vector embeddings...")
    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    
    # Step 4: Test RAG pipeline
    print("\n[4/4] Testing RAG pipeline...")
    rag = RAGPipeline(vector_store, use_openai=False)
    
    # Interactive Q&A
    print("\n" + "="*60)
    print("Ready for questions! (type 'quit' to exit)")
    print("="*60)
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        result = rag.ask(query, n_results=3, verbose=True)

if __name__ == "__main__":
    main()