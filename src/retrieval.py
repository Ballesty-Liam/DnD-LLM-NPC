"""
Retrieve relevant D&D lore for answering user queries.
"""
import os
from typing import Dict, List, Any, Optional

from .embeddings import EmbeddingManager


class LoreRetriever:
    """Retrieve relevant lore from D&D sourcebooks."""

    def __init__(
        self,
        results_count: int = 5,
        embeddings_dir: str = "data/processed"
    ):
        """
        Initialize the lore retriever.

        Args:
            results_count: Number of results to return for each query
            embeddings_dir: Directory where embeddings are stored
        """
        self.results_count = results_count
        self.embeddings_dir = embeddings_dir

        # Initialize embedding manager
        self.embedding_manager = self._init_embeddings()

    def _init_embeddings(self) -> Optional[EmbeddingManager]:
        """
        Initialize the embedding manager if data exists.

        Returns:
            Initialized EmbeddingManager or None
        """
        # Check if embeddings exist
        expected_files = [
            "radiant_citadel.faiss",
            "radiant_citadel_metadata.pkl",
            "radiant_citadel_documents.pkl"
        ]

        embeddings_exist = all(
            os.path.exists(os.path.join(self.embeddings_dir, f))
            for f in expected_files
        )

        if embeddings_exist:
            return EmbeddingManager(processed_dir=self.embeddings_dir)
        else:
            print(f"No embeddings found at {self.embeddings_dir}. Cannot retrieve lore.")
            return None

    def query_lore(self, query_text: str, n_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Query the vector database to find relevant passages.

        Args:
            query_text: The query text
            n_results: Number of results to return

        Returns:
            Query results dictionary
        """
        if not self.embedding_manager:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        n_results = n_results or self.results_count
        return self.embedding_manager.query(query_text, n_results=n_results)

    def get_condensed_context(self, query_text: str, detailed: bool = False, verbose: bool = False) -> str:
        """
        Get a condensed context from multiple relevant chunks.

        Args:
            query_text: The query text
            detailed: Whether to include more detailed results
            verbose: Whether to print diagnostic information

        Returns:
            Condensed context text
        """
        # Use more results when detailed is requested
        n_results = self.results_count * 2 if detailed else self.results_count

        # Expand query with related terms to improve retrieval
        expanded_query = self._expand_query(query_text, verbose)

        # Get results for both original and expanded queries
        original_results = self.query_lore(query_text, n_results)
        expanded_results = self.query_lore(expanded_query, n_results)

        # Combine and deduplicate results
        combined_documents = []
        seen_docs = set()

        # Add original results first (higher priority)
        for doc in original_results["documents"][0]:
            if doc not in seen_docs:
                combined_documents.append(doc)
                seen_docs.add(doc)

        # Add expanded results if not already included
        for doc in expanded_results["documents"][0]:
            if doc not in seen_docs:
                combined_documents.append(doc)
                seen_docs.add(doc)

        # Limit to reasonable size
        max_docs = n_results * 2
        combined_documents = combined_documents[:max_docs]

        # Format context with document separation for clarity
        if not combined_documents:
            return "No relevant information found in the Radiant Citadel records."

        # Format with clear section separators
        context_sections = []

        for i, doc in enumerate(combined_documents):
            section_title = f"KNOWLEDGE SECTION {i+1}"
            context_sections.append(f"{section_title}:\n{doc.strip()}")

        return "\n\n".join(context_sections)

    def _expand_query(self, query_text: str, verbose: bool = False) -> str:
        """
        Expand query with D&D related terms to improve retrieval.

        Args:
            query_text: Original query text
            verbose: Whether to print diagnostic information

        Returns:
            Expanded query text
        """
        # Extract key terms from query
        import re
        words = re.findall(r'\b[A-Za-z]+\b', query_text)

        # Only add D&D related terms if they seem relevant to the query
        dnd_terms = []

        if any(term in query_text.lower() for term in ["citadel", "radiant", "city"]):
            dnd_terms.extend(["Radiant Citadel", "Dawn Incarnates", "Concord"])

        if any(term in query_text.lower() for term in ["market", "shop", "buy", "sell", "trade"]):
            dnd_terms.extend(["Amaranthine Market", "merchants", "goods"])

        if any(term in query_text.lower() for term in ["founder", "history", "ancient", "origin"]):
            dnd_terms.extend(["Founders", "Dawn Incarnates", "histories"])

        # Only add expansion if we found relevant terms
        if dnd_terms:
            expanded = f"{query_text} {' '.join(dnd_terms)}"
            if verbose:
                print(f"Expanded query: {expanded}")
            return expanded

        return query_text