"""
Evaluation package for Math-RAG system.

This package provides evaluation functionality including:
- Pydantic models for type-safe evaluation results
- Dataset loading and preprocessing
- RAG system evaluation pipeline with structured data
- Unified judge system for response assessment
- Retrieval quality metrics and analysis
- Summary calculation and output formatting
- Output formatting and statistics
"""

# Core Pydantic models and evaluation logic
from math_rag.evaluation.core import (
    EvaluationResult,
    EvaluationSummary,
    JudgeResult,
    JudgeSummary,
    RetrievalResult,
    RetrievalSummary,
    calculate_evaluation_summary,
    calculate_f1_score,
    calculate_judge_summary,
    calculate_precision,
    calculate_recall,
    calculate_retrieval_summary,
    evaluate_retrieval_quality,
    identify_irrelevant_documents,
    identify_missing_documents,
)

# DSPy evaluation components
from math_rag.evaluation.dspy import (
    AssessResponse,
    LLMJudge,
    SimpleQA,
    judge_response,
    run_evaluation,
    setup_rag_system,
)

# I/O operations
from math_rag.evaluation.io import (
    load_dataset,
    log_summary_results,
    write_results,
    write_summary,
)

__all__ = [
    # Pydantic models
    "EvaluationResult",
    "EvaluationSummary",
    "JudgeResult",
    "JudgeSummary",
    "RetrievalResult",
    "RetrievalSummary",
    # DSPy components
    "AssessResponse",
    "LLMJudge",
    "SimpleQA",
    "judge_response",
    # DSPy evaluation functions
    "run_evaluation",
    "setup_rag_system",
    # Core evaluation metrics
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
    # I/O operations
    "load_dataset",
    "log_summary_results",
    "write_results",
    "write_summary",
]
