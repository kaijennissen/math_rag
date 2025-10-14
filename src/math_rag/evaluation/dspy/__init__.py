"""
DSPy-specific evaluation functionality for Math-RAG system.

This submodule contains DSPy framework integration including:
- DSPy modules for question answering and response assessment
- LLM judge components for response evaluation
- Main evaluation pipeline runner
"""

from math_rag.evaluation.dspy.modules import (
    AssessResponse,
    LLMJudge,
    SimpleQA,
    judge_response,
)
from math_rag.evaluation.dspy.runner import run_evaluation, setup_rag_system

__all__ = [
    # DSPy modules and components
    "AssessResponse",
    "LLMJudge",
    "SimpleQA",
    "judge_response",
    # Main evaluation functions
    "run_evaluation",
    "setup_rag_system",
]
