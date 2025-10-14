"""
Output writer for evaluation results using Pydantic models.

This module provides functions to write evaluation results and summaries
to various output formats, with type-safe Pydantic model integration.
"""

import json
import logging
from pathlib import Path
from typing import List

from math_rag.evaluation.core.models import EvaluationResult, EvaluationSummary

logger = logging.getLogger(__name__)


def write_results(results: List[EvaluationResult], output_path: Path) -> None:
    """
    Write evaluation results to JSON file using Pydantic serialization.

    Args:
        results: List of EvaluationResult objects
        output_path: Path to write the JSON file
    """
    try:
        # Convert Pydantic models to dictionaries for JSON serialization
        results_data = [result.model_dump() for result in results]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results written to {output_path}")

    except Exception as e:
        logger.error(f"Error writing results to {output_path}: {e}")
        raise


def write_summary(summary: EvaluationSummary, output_path: Path) -> None:
    """
    Write evaluation summary to JSON file.

    Args:
        summary: EvaluationSummary object
        output_path: Path to write the summary JSON file
    """
    try:
        summary_data = summary.model_dump()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary written to {output_path}")

    except Exception as e:
        logger.error(f"Error writing summary to {output_path}: {e}")
        raise


def log_summary_results(summary: EvaluationSummary, logger_obj: logging.Logger) -> None:
    """
    Log evaluation summary results in a formatted way.

    Args:
        summary: EvaluationSummary object
        logger_obj: Logger instance to use for output
    """
    logger_obj.info(
        f"Evaluation complete: {summary.total_questions} questions processed"
    )
    logger_obj.info(
        f"Successful: {summary.successful_evaluations}, "
        f"Failed: {summary.failed_evaluations}"
    )

    # Report judge evaluation results
    if summary.judge_summary:
        judge = summary.judge_summary
        logger_obj.info("Judge evaluation results:")
        logger_obj.info(
            f"  Mathematical Correctness: {judge.avg_mathematical_correctness:.2f}/5"
        )
        logger_obj.info(f"  Context Relevance: {judge.avg_context_relevance:.2f}/5")
        logger_obj.info(f"  Context Utilization: {judge.avg_context_utilization:.2f}/5")
        logger_obj.info(f"  Completeness: {judge.avg_completeness:.2f}/5")
        logger_obj.info(f"  Overall Average: {judge.avg_overall_score:.2f}/5")
        logger_obj.info(f"  Pass Rate: {judge.pass_rate * 100:.1f}%")
    else:
        logger_obj.warning("No valid judge evaluation results")

    # Report retrieval metrics
    if summary.retrieval_summary:
        retrieval = summary.retrieval_summary
        logger_obj.info("Retrieval metrics:")
        logger_obj.info(f"  Precision@K: {retrieval.avg_precision:.3f}")
        logger_obj.info(f"  Recall@K: {retrieval.avg_recall:.3f}")
        logger_obj.info(f"  F1@K: {retrieval.avg_f1:.3f}")
        logger_obj.info(f"  Questions Evaluated: {retrieval.total_questions_evaluated}")
    else:
        logger_obj.warning("No valid retrieval metrics available")
