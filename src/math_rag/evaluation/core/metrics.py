"""
Retrieval quality metrics for RAG evaluation using Pydantic models.

This module provides functions to evaluate retrieval quality by comparing
retrieved documents against gold standard references, with type-safe
Pydantic model integration.
"""

import logging
import re
from typing import Dict, List, Set

from math_rag.evaluation.core.models import RetrievalResult

logger = logging.getLogger(__name__)


def extract_document_id(doc) -> str:
    """Extract document ID from retrieved document metadata."""
    # Handle Document object with metadata
    if hasattr(doc, "metadata") and doc.metadata:
        return doc.metadata.get("number", "")
    return ""


def parse_gold_sources(source_string: str) -> Set[str]:
    """Parse gold standard sources from dataset source field."""
    if not source_string:
        return set()
    pattern = r"_(\d+\.\d+\.\d+)$"
    matches = re.findall(pattern, source_string)
    if matches:
        return set(matches)
    return set()


def calculate_precision(retrieved_ids: List[str], gold_ids: Set[str]) -> float:
    """
    Calculate precision for retrieved documents.

    Args:
        retrieved_ids: List of retrieved document IDs
        gold_ids: Set of gold standard document IDs

    Returns:
        Precision score (0.0 to 1.0)
    """
    retrieved_set = set(retrieved_ids)
    retrieved_set.discard("")  # Remove empty IDs

    if not retrieved_set:
        return 0.0

    relevant_retrieved = retrieved_set.intersection(gold_ids)
    return len(relevant_retrieved) / len(retrieved_set)


def calculate_recall(retrieved_ids: List[str], gold_ids: Set[str]) -> float:
    """
    Calculate recall for retrieved documents.

    Args:
        retrieved_ids: List of retrieved document IDs
        gold_ids: Set of gold standard document IDs

    Returns:
        Recall score (0.0 to 1.0)
    """
    retrieved_set = set(retrieved_ids)
    retrieved_set.discard("")  # Remove empty IDs

    if not gold_ids:
        return 0.0

    relevant_retrieved = retrieved_set.intersection(gold_ids)
    return len(relevant_retrieved) / len(gold_ids)


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.

    Args:
        precision: Precision score
        recall: Recall score

    Returns:
        F1 score (0.0 to 1.0)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def identify_missing_documents(
    retrieved_ids: List[str], gold_ids: Set[str]
) -> List[str]:
    """
    Identify gold standard documents that were not retrieved.

    Args:
        retrieved_ids: List of retrieved document IDs
        gold_ids: Set of gold standard document IDs

    Returns:
        List of missing document IDs
    """
    retrieved_set = set(retrieved_ids)
    retrieved_set.discard("")  # Remove empty IDs
    return list(gold_ids - retrieved_set)


def identify_irrelevant_documents(
    retrieved_ids: List[str], gold_ids: Set[str]
) -> List[str]:
    """
    Identify retrieved documents that are not in gold standard.

    Args:
        retrieved_ids: List of retrieved document IDs
        gold_ids: Set of gold standard document IDs

    Returns:
        List of irrelevant document IDs
    """
    retrieved_set = set(retrieved_ids)
    retrieved_set.discard("")  # Remove empty IDs
    return list(retrieved_set - gold_ids)


def evaluate_retrieval_quality(
    retrieved_docs: List, gold_source: str
) -> RetrievalResult:
    """
    Evaluate retrieval quality for a single question using Pydantic model.

    Args:
        retrieved_docs: List of Document objects from retrieval
        gold_source: Source field from dataset (e.g., "topological_spaces_1.1.1")

    Returns:
        RetrievalResult with all metrics and analysis
    """
    # Extract document IDs from retrieved documents
    retrieved_ids = [extract_document_id(doc) for doc in retrieved_docs]

    # Parse gold standard sources
    gold_ids = parse_gold_sources(gold_source)

    # Calculate core metrics
    precision = calculate_precision(retrieved_ids, gold_ids)
    recall = calculate_recall(retrieved_ids, gold_ids)
    f1 = calculate_f1_score(precision, recall)

    # Count metrics
    retrieved_set = set(retrieved_ids)
    retrieved_set.discard("")  # Remove empty IDs
    relevant_retrieved = retrieved_set.intersection(gold_ids)

    # Identify missing and irrelevant documents
    missing_docs = identify_missing_documents(retrieved_ids, gold_ids)
    irrelevant_docs = identify_irrelevant_documents(retrieved_ids, gold_ids)

    return RetrievalResult(
        precision=precision,
        recall=recall,
        f1=f1,
        retrieved_count=len(retrieved_set),
        gold_count=len(gold_ids),
        relevant_retrieved_count=len(relevant_retrieved),
        missing_from_retrieval=missing_docs,
        irrelevant_retrieved=irrelevant_docs,
    )


# Legacy function for backward compatibility
def calculate_precision_recall_f1(
    retrieved_ids: List[str], gold_ids: Set[str]
) -> Dict[str, float]:
    """
    Legacy function for calculating precision, recall, and F1.

    DEPRECATED: Use individual calculate_* functions or evaluate_retrieval_quality
    instead.
    """
    logger.warning(
        "calculate_precision_recall_f1 is deprecated. "
        "Use evaluate_retrieval_quality instead."
    )

    # Convert to sets for intersection operations
    retrieved_set = set(retrieved_ids)
    retrieved_set.discard("")  # Remove empty IDs

    # Calculate metrics
    if not retrieved_set:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "retrieved_count": 0,
            "gold_count": len(gold_ids),
            "relevant_retrieved": 0,
        }

    if not gold_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "retrieved_count": len(retrieved_set),
            "gold_count": 0,
            "relevant_retrieved": 0,
        }

    # Calculate intersection
    relevant_retrieved = retrieved_set.intersection(gold_ids)

    precision = len(relevant_retrieved) / len(retrieved_set)
    recall = len(relevant_retrieved) / len(gold_ids)

    # F1 calculation with zero-division protection
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "retrieved_count": len(retrieved_set),
        "gold_count": len(gold_ids),
        "relevant_retrieved": len(relevant_retrieved),
    }
