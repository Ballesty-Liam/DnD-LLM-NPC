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
        index_name: str = "radiant_citadel"
    ):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the sentence-transformers model to use
            processed_dir: Directory with processed text chunks
            index_name: Name for the FAISS index
        """
        self.model_name = model_name
        self.processed_dir = Path(processed_dir)
        self.index_name = index_name

        # Ensure directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sentence transformer model
        self.model = SentenceTransformer(model_name)

        # Path to save the index
        self.index_path = self.processed_dir / f"{index_name}.faiss"
        self.metadata_path = self.processed_dir / f"{index_name}_metadata.pkl"
        self.documents_path = self.processed_dir / f"{index_name}_documents.pkl"

        # Initialize or load index
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(str(self.index_path))

            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)

            print(f"Loaded existing index '{index_name}' with {len(self.metadata)} documents")
        else:
            # Create a new index
            self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
            self.metadata = []
            self.documents = []
            print(f"Created new index '{index_name}'")

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
        # Extract texts and metadata
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        # Create embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Add to FAISS index
        start_idx = len(self.metadata)
        self.index.add(np.array(embeddings).astype('float32'))

        # Store metadata and documents
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            self.metadata.append({
                "id": start_idx + i,
                "metadata": meta
            })
            self.documents.append(text)

        # Save index and metadata
        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)

        print(f"Added {len(documents)} documents to index")

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
        # Encode the query
        query_embedding = self.model.encode([query_text])[0].reshape(1, -1).astype('float32')

        # Search the index
        distances, indices = self.index.search(query_embedding, n_results)

        # Prepare results
        results = {
            "documents": [[self.documents[idx] for idx in indices[0]]],
            "metadatas": [[self.metadata[idx]["metadata"] for idx in indices[0]]],
            "distances": [distances[0].tolist()]
        }

        # Apply filter if provided
        if where_filter:
            filtered_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            for i, metadata in enumerate(results["metadatas"][0]):
                # Check if metadata matches filter
                matches = True
                for key, value in where_filter.items():
                    if key not in metadata or metadata[key] != value:
                        matches = False
                        break

                if matches:
                    filtered_results["documents"][0].append(results["documents"][0][i])
                    filtered_results["metadatas"][0].append(metadata)
                    filtered_results["distances"][0].append(results["distances"][0][i])

            results = filtered_results

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