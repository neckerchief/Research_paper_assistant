from src.embeddings import VectorStore
from typing import List, Dict
import os
from dotenv import load_dotenv

# If using OpenAI (optional - comment out if using only local models)
# from openai import OpenAI

class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        use_openai: bool = False
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: Initialized VectorStore instance
            use_openai: Whether to use OpenAI API (requires API key)
        """
        self.vector_store = vector_store
        self.use_openai = use_openai
        
        if use_openai:
            load_dotenv()
            # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            pass  # Uncomment above if using OpenAI
    
    def retrieve_context(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve relevant chunks for query."""
        results = self.vector_store.search(query, n_results=n_results)
        
        # Format results
        context_chunks = []
        for i in range(len(results['documents'][0])):
            chunk = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            }
            context_chunks.append(chunk)
        
        return context_chunks
    
    def build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Build prompt with retrieved context."""
        # Build context string
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            meta = chunk['metadata']
            context_parts.append(
                f"[Source {i}: {meta['title']} ({meta['arxiv_id']})]\n"
                f"{chunk['text']}\n"
            )
        
        context_str = "\n---\n".join(context_parts)
        
        # Build full prompt
        prompt = f"""You are a helpful research assistant. Answer the question based on the provided context from research papers.

Context from research papers:
{context_str}

Question: {query}

Instructions:
- Answer based on the context provided above
- If the context doesn't contain enough information, say so
- Cite which source(s) you used (by source number)
- Be precise and scientific in your answer

Answer:"""
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer using LLM.
        
        For now, this is a placeholder - you can:
        1. Use OpenAI API (requires API key)
        2. Use a local model (Ollama, llama.cpp)
        3. Just return the prompt for manual testing
        """
        if self.use_openai:
            # Uncomment if you have OpenAI API key
            # response = self.client.chat.completions.create(
            #     model="gpt-3.5-turbo",
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=0.7,
            #     max_tokens=500
            # )
            # return response.choices[0].message.content
            return "[OpenAI integration not configured]"
        else:
            # For Phase 1, just return the prompt
            # You can manually test this or integrate local LLM later
            return "[LLM response would go here - see prompt above]"
    
    def ask(self, query: str, n_results: int = 5, verbose: bool = True) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Args:
            query: User question
            n_results: Number of chunks to retrieve
            verbose: Whether to print intermediate steps
        
        Returns:
            Dictionary with answer and sources
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}\n")
        
        # Step 1: Retrieve relevant chunks
        if verbose:
            print("Step 1: Retrieving relevant context...")
        
        context_chunks = self.retrieve_context(query, n_results)
        
        if verbose:
            print(f"✓ Retrieved {len(context_chunks)} relevant chunks\n")
            for i, chunk in enumerate(context_chunks, 1):
                meta = chunk['metadata']
                print(f"  {i}. {meta['title'][:50]}... (distance: {chunk['distance']:.3f})")
        
        # Step 2: Build prompt
        if verbose:
            print("\nStep 2: Building prompt...")
        
        prompt = self.build_prompt(query, context_chunks)
        
        if verbose:
            print(f"✓ Prompt built ({len(prompt)} chars)\n")
        
        # Step 3: Generate answer
        if verbose:
            print("Step 3: Generating answer...\n")
        
        answer = self.generate_answer(prompt)
        
        # Return result
        result = {
            'query': query,
            'answer': answer,
            'prompt': prompt,
            'sources': context_chunks
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print("PROMPT:")
            print(f"{'='*60}")
            print(prompt)
            print(f"\n{'='*60}")
            print("ANSWER:")
            print(f"{'='*60}")
            print(answer)
            print(f"{'='*60}\n")
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize vector store
    vector_store = VectorStore()
    
    # Initialize RAG pipeline
    rag = RAGPipeline(vector_store, use_openai=False)
    
    # Ask questions
    questions = [
        "What is quantum entanglement?",
        "Explain the concept of wave-particle duality",
        "What are the main predictions of general relativity?"
    ]
    
    for question in questions:
        result = rag.ask(question, n_results=3)
        input("\nPress Enter for next question...")