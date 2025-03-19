"""
Generate and manage embeddings for D&D source material.
"""
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Manage text embeddings for efficient retrieval."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",  # Lightweight model, works without GPU
        processed_dir: str = "data/processed",
        collection_name: str = "radiant_citadel"
    ):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the sentence-transformers model to use
            processed_dir: Directory with processed text chunks
            collection_name: Name for the ChromaDB collection
        """
        self.model_name = model_name
        self.processed_dir = Path(processed_dir)
        self.collection_name = collection_name

        # Set up embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(os.path.join(processed_dir, "vectordb"))

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Loaded existing collection '{collection_name}' with {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Created new collection '{collection_name}'")

    def load_documents(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load processed document chunks from a jsonl file.

        Args:
            file_path: Path to the jsonl file

        Returns:
            List of document dictionaries
        """
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line))
        return documents

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector database.

        Args:
            documents: List of document dictionaries with content and metadata
        """
        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        # Add to collection in batches (to avoid memory issues with large collections)
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

        print(f"Added {len(documents)} documents to collection")

    def process_and_embed(self, chunks_file: str = "chunks.jsonl"):
        """
        Load processed chunks and add them to the vector database.

        Args:
            chunks_file: Name of the jsonl file with chunks
        """
        file_path = self.processed_dir / chunks_file
        documents = self.load_documents(file_path)
        self.add_documents(documents)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector database to find relevant passages.

        Args:
            query_text: The query text
            n_results: Number of results to return
            where_filter: Optional filter to apply to metadata

        Returns:
            Dictionary with query results
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )

        return results


if __name__ == "__main__":
    # Example usage
    manager = EmbeddingManager()
    manager.process_and_embed()

    # Test query
    results = manager.query("Tell me about the Radiant Citadel")
    for i, doc in enumerate(results["documents"][0]):
        print(f"Result {i+1}:")
        print(doc[:100] + "...\n")