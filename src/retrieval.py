"""
Handle semantic search and retrieval of relevant D&D lore.
"""
from typing import List, Dict, Any, Optional

from .embeddings import EmbeddingManager


class LoreRetriever:
    """Retrieve relevant D&D lore based on semantic similarity."""

    def __init__(
            self,
            embedding_manager: Optional[EmbeddingManager] = None,
            default_n_results: int = 5
    ):
        """
        Initialize the lore retriever.

        Args:
            embedding_manager: Manager for embeddings and vector search
            default_n_results: Default number of results to retrieve
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.default_n_results = default_n_results

    def retrieve(
            self,
            query: str,
            n_results: Optional[int] = None,
            where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant lore based on the query.

        Args:
            query: The query text
            n_results: Number of results to return
            where_filter: Optional filter for metadata

        Returns:
            List of relevant lore passages with metadata
        """
        n = n_results or self.default_n_results

        # Query the vector database
        results = self.embedding_manager.query(
            query_text=query,
            n_results=n,
            where_filter=where_filter
        )

        # Format the results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if i < len(results["metadatas"][0]) else {},
                "score": results["distances"][0][i] if i < len(results["distances"][0]) else None
            })

        return formatted_results

    def get_condensed_context(
            self,
            query: str,
            n_results: Optional[int] = None,
            where_filter: Optional[Dict[str, Any]] = None,
            max_tokens: int = 2000
    ) -> str:
        """
        Get condensed context relevant to the query.

        Args:
            query: The query text
            n_results: Number of results to retrieve
            where_filter: Optional filter for metadata
            max_tokens: Maximum tokens to include in context

        Returns:
            String with concatenated relevant passages
        """
        results = self.retrieve(query, n_results, where_filter)

        # Format as a single context string
        context_parts = []
        current_length = 0
        target_length = max_tokens * 4  # Approximate chars to tokens

        for result in results:
            content = result["content"]
            metadata = result["metadata"]

            # Format with source information
            source_info = f"[Source: {metadata.get('title', 'Unknown')}]"
            if "section" in metadata:
                source_info += f", Section: {metadata['section']}"

            formatted_text = f"{content}\n{source_info}\n\n"

            # Add if we're under the limit
            if current_length + len(formatted_text) <= target_length:
                context_parts.append(formatted_text)
                current_length += len(formatted_text)
            else:
                # Truncate the last part if needed
                remaining = target_length - current_length
                if remaining > 100:  # Only add if a meaningful amount can be included
                    truncated = formatted_text[:remaining] + "...\n"
                    context_parts.append(truncated)
                break

        return "".join(context_parts)


if __name__ == "__main__":
    # Example usage
    retriever = LoreRetriever()
    results = retriever.retrieve("Tell me about the Dawn Incarnates")

    for i, result in enumerate(results):
        print(f"Result {i + 1}:")
        print(f"Content: {result['content'][:100]}...")
        print(f"Source: {result['metadata'].get('title', 'Unknown')}")
        print(f"Score: {result['score']}")
        print()

    # Get condensed context
    context = retriever.get_condensed_context("What is the history of the Radiant Citadel?")
    print("Condensed Context:")
    print(context[:500] + "...\n")