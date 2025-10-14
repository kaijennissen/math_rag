"""
Core evaluation functionality for Math-RAG system.

This submodule contains the core evaluation logic including:
- Pydantic models for type-safe evaluation results
- Retrieval quality metrics and calculations
- Summary calculation functions
"""

from math_rag.evaluation.core.metrics import (
    calculate_f1_score,
    calculate_precision,
    calculate_recall,
    evaluate_retrieval_quality,
    identify_irrelevant_documents,
    identify_missing_documents,
)
from math_rag.evaluation.core.models import (
    EvaluationResult,
    EvaluationSummary,
    JudgeResult,
    JudgeSummary,
    RetrievalResult,
    RetrievalSummary,
)
from math_rag.evaluation.core.summary import (
    calculate_evaluation_summary,
    calculate_judge_summary,
    calculate_retrieval_summary,
)

__all__ = [
    # Pydantic models
    "EvaluationResult",
    "EvaluationSummary",
    "JudgeResult",
    "JudgeSummary",
    "RetrievalResult",
    "RetrievalSummary",
    # Retrieval metrics
    "calculate_f1_score",
    "calculate_precision",
    "calculate_recall",
    "evaluate_retrieval_quality",
    "identify_irrelevant_documents",
    "identify_missing_documents",
    # Summary calculation
    "calculate_evaluation_summary",
    "calculate_judge_summary",
    "calculate_retrieval_summary",
]
