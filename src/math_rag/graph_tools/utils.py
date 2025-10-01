"""
Common utilities for graph tools.

This module provides shared functionality for graph-based retrieval tools,
including vector store creation, document formatting, and other common operations.
"""

import logging
from typing import List

import coloredlogs
from langchain_core.documents import Document

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def format_document(doc: Document, index: int) -> str:
    """
    Format a single Document into a readable string representation.

    This function is used by all retriever tools to maintain consistent
    document formatting across the system.

    Args:
        doc: Document instance (must have `page_content` and optional `metadata`)
        index: 1-based index of the document for display

    Returns:
        Formatted string for the single document
    """
    allowed_metadata = {"number", "type", "title"}

    # Separate lists for content and metadata; we'll concatenate at the end.
    content_parts: List[str] = []
    metadata_parts: List[str] = []

    content_parts.append(f"\n\n===== Document {index} =====\n")
    content_parts.append(f"CONTENT:\n {doc.page_content}\n")
    content_parts.append(f"{doc.metadata.get('text_nl')}\n\n")

    # metadata may be missing or empty; only keep a small whitelist of keys
    metadata = getattr(doc, "metadata", None)
    if metadata:
        # Single-pass over metadata: collect allowed keys in the order they appear
        for key, value in metadata.items():
            if key.lower() in allowed_metadata:
                metadata_parts.append(f"  - {key}: {value}\n")

    if metadata_parts:
        metadata_parts.insert(0, "METADATA:\n")

    parts = content_parts + metadata_parts
    parts.append("-" * 40)
    return "".join(parts)


def format_retrieval_results(docs: List[Document], search_type: str) -> str:
    """
    Format a list of retrieved documents into a readable string.

    Args:
        docs: List of retrieved documents
        search_type: Description of the search method used (e.g., "hybrid search")

    Returns:
        Formatted string representation of all documents
    """
    result_parts: List[str] = [
        f"\nRetrieved {len(docs)} documents using {search_type}:\n"
    ]

    for i, doc in enumerate(docs, 1):
        result_parts.append(format_document(doc, i))

    return "".join(result_parts)


def get_pathrag_query() -> str:
    """
    Get the PathRAG Cypher query for graph traversal retrieval.

    This query extends search results to include connected nodes via CITES
    relationships, providing a more comprehensive view of related content.

    Returns:
        PathRAG Cypher query string
    """
    return """
    // This query extends the search result to include connected nodes via CITES
    WITH node, score

    // Get connected nodes via CITES relationships (both directions)
    OPTIONAL MATCH (node)-[:CITES]->(citedNode)
    OPTIONAL MATCH (referencingNode)-[:CITES]->(node)

    // Collect all connected nodes
    WITH node, score,
         collect(DISTINCT citedNode) +
         collect(DISTINCT referencingNode) as connectedNodes

    // Return the original node plus connected nodes
    WITH node, score, connectedNodes
    UNWIND ([node] + connectedNodes) as resultNode

    // Filter null nodes and calculate final score: original nodes keep their score,
    // connected nodes get 0.1
    WITH resultNode,
         CASE
            WHEN resultNode = node THEN score
            ELSE 0.1
         END as finalScore
    WHERE resultNode IS NOT NULL

    // Return required columns: text, score, metadata
    RETURN resultNode.text as text,
           finalScore as score,
           resultNode {.*, text: Null} as metadata
    ORDER BY finalScore DESC
    """
