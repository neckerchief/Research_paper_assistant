import pypdf
from pathlib import Path
from typing import List, Dict
import json
import re

class DocumentProcessor:
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return self._clean_text(text)
        
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove weird characters but keep scientific symbols
        text = re.sub(r'[^\w\s\-.,;:()[\]{}=+\-*/^<>≥≤≈∼°αβγδεζηθικλμνξοπρστυφχψω∫∂∇]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk end position
            end = start + self.chunk_size
            
            # If not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the target end position
                search_start = max(start, end - 100)
                search_text = text[search_start:end + 100]
                
                # Find last sentence ending
                sentence_endings = [m.end() for m in re.finditer(r'[.!?]\s', search_text)]
                
                if sentence_endings:
                    # Adjust end to last sentence boundary
                    end = search_start + sentence_endings[-1]
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = {
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end,
                }
                
                # Add metadata if provided
                if metadata:
                    chunk.update(metadata)
                
                chunks.append(chunk)
            
            # Move start position (with overlap)
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_papers(
        self, 
        papers_metadata: List[Dict],
        output_path: str = "data/processed/chunks.json"
    ) -> List[Dict]:
        """
        Process all papers and create chunks.
        
        Args:
            papers_metadata: List of paper metadata dicts (from downloader)
            output_path: Where to save processed chunks
        
        Returns:
            List of all chunks
        """
        all_chunks = []
        
        for paper in papers_metadata:
            print(f"Processing: {paper['title']}")
            
            # Extract text
            text = self.extract_text_from_pdf(paper['pdf_path'])
            
            if not text:
                print(f"  ⚠ No text extracted, skipping")
                continue
            
            # Create metadata for chunks
            chunk_metadata = {
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'authors': paper['authors'],
                'published': paper['published'],
                'categories': paper['categories']
            }
            
            # Chunk the text
            chunks = self.chunk_text(text, chunk_metadata)
            all_chunks.extend(chunks)
            
            print(f"  ✓ Created {len(chunks)} chunks")
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2)
        
        print(f"\n✓ Processed {len(all_chunks)} total chunks")
        print(f"✓ Saved to {output_path}")
        
        return all_chunks

# Example usage
if __name__ == "__main__":
    # Load papers metadata (from download step)
    import json
    
    # You'd load this from your download script
    # For now, let's assume you have papers in data/raw_papers/
    from download_papers import ArxivDownloader
    
    downloader = ArxivDownloader()
    papers = downloader.download_papers("quantum mechanics", max_results=10)
    
    # Process documents
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    chunks = processor.process_papers(papers)