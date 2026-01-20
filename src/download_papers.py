import arxiv
import os
from pathlib import Path
from typing import List
import time

class ArxivDownloader:
    def __init__(self, output_dir: str = "data/raw_papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_papers(
        self, 
        query: str, 
        max_results: int = 50,
        category: str = None
    ) -> List[dict]:
        """
        Download papers from ArXiv based on search query.
        
        Args:
            query: Search query (e.g., "quantum field theory")
            max_results: Maximum number of papers to download
            category: ArXiv category (e.g., "hep-th", "astro-ph")
        
        Returns:
            List of metadata dicts for downloaded papers
        """
        # Build search query
        if category:
            search_query = f"cat:{category} AND {query}"
        else:
            search_query = query
        
        print(f"Searching ArXiv for: {search_query}")
        
        # Search ArXiv
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers_metadata = []
        
        for result in search.results():
            try:
                # Extract ArXiv ID from URL
                arxiv_id = result.entry_id.split('/')[-1]
                
                # Create filename
                filename = f"{arxiv_id.replace('.', '_')}.pdf"
                filepath = self.output_dir / filename
                
                # Skip if already downloaded
                if filepath.exists():
                    print(f"Already downloaded: {result.title}")
                    papers_metadata.append(self._extract_metadata(result, filepath))
                    continue
                
                # Download PDF
                print(f"Downloading: {result.title}")
                result.download_pdf(dirpath=str(self.output_dir), filename=filename)
                
                # Store metadata
                papers_metadata.append(self._extract_metadata(result, filepath))
                
                # Be nice to ArXiv servers
                time.sleep(3)
                
            except Exception as e:
                print(f"Error downloading {result.title}: {e}")
                continue
        
        print(f"\nDownloaded {len(papers_metadata)} papers to {self.output_dir}")
        return papers_metadata
    
    def _extract_metadata(self, result, filepath: Path) -> dict:
        """Extract metadata from ArXiv result."""
        return {
            'arxiv_id': result.entry_id.split('/')[-1],
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'abstract': result.summary,
            'published': result.published.isoformat(),
            'categories': result.categories,
            'pdf_path': str(filepath)
        }

# Example usage
if __name__ == "__main__":
    downloader = ArxivDownloader()
    
    # Example: Download quantum field theory papers
    papers = downloader.download_papers(
        query="quantum field theory",
        max_results=50,
        category="hep-th"  # High Energy Physics - Theory
    )
    
    # Or download from multiple areas
    # papers = downloader.download_papers(
    #     query="machine learning",
    #     max_results=30,
    #     category="cs.LG"
    # )
    
    print(f"\nTotal papers downloaded: {len(papers)}")