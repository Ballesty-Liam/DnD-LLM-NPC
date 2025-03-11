"""
Process D&D source material from markdown format into chunks for embedding and retrieval.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import markdown
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter


class SourceProcessor:
    """Process D&D source material into retrievable chunks."""

    def __init__(
            self,
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            raw_dir: str = "data/raw",
            processed_dir: str = "data/processed"
    ):
        """
        Initialize the processor.

        Args:
            chunk_size: Target size of each text chunk
            chunk_overlap: Overlap between chunks to maintain context
            raw_dir: Directory containing raw markdown files
            processed_dir: Directory to store processed chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n", ". ", ", ", " ", ""]
        )

    def _convert_md_to_text(self, md_content: str) -> str:
        """Convert markdown to plain text while preserving structure."""
        # Convert markdown to HTML
        html = markdown.markdown(md_content)

        # Use BeautifulSoup to extract text
        soup = BeautifulSoup(html, "html.parser")

        # Get text while preserving some structure
        text = soup.get_text()

        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text

    def _extract_metadata(self, md_content: str) -> Dict[str, Any]:
        """Extract metadata from markdown headers."""
        metadata = {}

        # Extract title if present (assuming first h1 is title)
        title_match = re.search(r'# (.*?)(\n|$)', md_content)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        # Extract section info
        section_match = re.search(r'## (.*?)(\n|$)', md_content)
        if section_match:
            metadata["section"] = section_match.group(1).strip()

        return metadata

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single markdown file into chunks with metadata.

        Args:
            file_path: Path to the markdown file

        Returns:
            List of dictionaries with text chunks and metadata
        """
        path = Path(file_path)

        # Read the file
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract metadata
        metadata = self._extract_metadata(content)
        metadata["source"] = path.name

        # Convert to plain text
        text = self._convert_md_to_text(content)

        # Split into chunks
        chunks = self.text_splitter.split_text(text)

        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_id": i
                }
            }
            documents.append(doc)

        return documents

    def process_directory(self, dir_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process all markdown files in a directory.

        Args:
            dir_path: Directory containing markdown files (defaults to self.raw_dir)

        Returns:
            List of all processed chunks with metadata
        """
        dir_path = Path(dir_path) if dir_path else self.raw_dir
        all_documents = []

        # Process each markdown file
        for file_path in dir_path.glob("**/*.md"):
            documents = self.process_file(str(file_path))
            all_documents.extend(documents)

        return all_documents

    def save_processed_chunks(self, documents: List[Dict[str, Any]], output_file: str = "chunks.jsonl"):
        """
        Save processed chunks to jsonl file.

        Args:
            documents: List of document chunks
            output_file: Name of output file
        """
        import json

        output_path = self.processed_dir / output_file

        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')

        print(f"Saved {len(documents)} chunks to {output_path}")

    def process_and_save(self, dir_path: Optional[str] = None, output_file: str = "chunks.jsonl"):
        """
        Process all markdown files and save the results.

        Args:
            dir_path: Directory containing markdown files
            output_file: Name of output file
        """
        documents = self.process_directory(dir_path)
        self.save_processed_chunks(documents, output_file)
        return documents


if __name__ == "__main__":
    processor = SourceProcessor()
    processor.process_and_save()