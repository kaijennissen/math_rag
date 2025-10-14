"""
Summary calculation functions for evaluation results.

This module provides functions to calculate comprehensive summaries and statistics
from evaluation results using Pydantic models for type safety.
"""

import logging
from typing import Dict, List

from math_rag.evaluation.core.models import (
    EvaluationResult,
    EvaluationSummary,
    JudgeSummary,
    RetrievalSummary,
)

logger = logging.getLogger(__name__)


def create_distribution(scores: List[int]) -> Dict[int, int]:
    """
    Create distribution dictionary from list of scores.

    Args:
        scores: List of integer scores (1-5)

    Returns:
        Dictionary mapping score -> count
    """
    distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for score in scores:
        if 1 <= score <= 5:
            distribution[score] += 1
    return distribution


def calculate_judge_summary(evaluation_results: List[EvaluationResult]) -> JudgeSummary:
    """
    Calculate summary statistics for judge evaluation results.

    Args:
        evaluation_results: List of evaluation results with judge data

    Returns:
        JudgeSummary with aggregated statistics
    """
    # Filter results with valid judge data
    judge_results = [
        result.judge_result
        for result in evaluation_results
        if result.judge_result is not None
    ]

    if not judge_results:
        logger.warning("No valid judge results found for summary calculation")
        return JudgeSummary(
            avg_mathematical_correctness=0.0,
            avg_context_relevance=0.0,
            avg_context_utilization=0.0,
            avg_completeness=0.0,
            avg_overall_score=0.0,
            pass_rate=0.0,
        )

    # Calculate average scores
    math_scores = [result.mathematical_correctness_score for result in judge_results]
    context_rel_scores = [result.context_relevance_score for result in judge_results]
    context_util_scores = [result.context_utilization_score for result in judge_results]
    completeness_scores = [result.completeness_score for result in judge_results]
    overall_scores = [result.average_score for result in judge_results]

    # Calculate pass rate
    pass_count = sum(1 for result in judge_results if result.overall_pass)
    pass_rate = pass_count / len(judge_results)

    # Create distributions
    math_dist = create_distribution(math_scores)
    context_rel_dist = create_distribution(context_rel_scores)
    context_util_dist = create_distribution(context_util_scores)
    completeness_dist = create_distribution(completeness_scores)

    return JudgeSummary(
        avg_mathematical_correctness=sum(math_scores) / len(math_scores),
        avg_context_relevance=sum(context_rel_scores) / len(context_rel_scores),
        avg_context_utilization=sum(context_util_scores) / len(context_util_scores),
        avg_completeness=sum(completeness_scores) / len(completeness_scores),
        avg_overall_score=sum(overall_scores) / len(overall_scores),
        pass_rate=pass_rate,
        mathematical_correctness_dist=math_dist,
        context_relevance_dist=context_rel_dist,
        context_utilization_dist=context_util_dist,
        completeness_dist=completeness_dist,
    )


def calculate_retrieval_summary(
    evaluation_results: List[EvaluationResult],
) -> RetrievalSummary:
    """
    Calculate summary statistics for retrieval evaluation results.

    Args:
        evaluation_results: List of evaluation results with retrieval data

    Returns:
        RetrievalSummary with aggregated statistics
    """
    # Filter results with valid retrieval data
    retrieval_results = [
        result.retrieval_result
        for result in evaluation_results
        if result.retrieval_result is not None
    ]

    if not retrieval_results:
        logger.warning("No valid retrieval results found for summary calculation")
        return RetrievalSummary(
            avg_precision=0.0,
            avg_recall=0.0,
            avg_f1=0.0,
            total_questions_evaluated=0,
        )

    # Calculate averages
    precisions = [result.precision for result in retrieval_results]
    recalls = [result.recall for result in retrieval_results]
    f1_scores = [result.f1 for result in retrieval_results]

    return RetrievalSummary(
        avg_precision=sum(precisions) / len(precisions),
        avg_recall=sum(recalls) / len(recalls),
        avg_f1=sum(f1_scores) / len(f1_scores),
        total_questions_evaluated=len(retrieval_results),
    )


def calculate_evaluation_summary(
    evaluation_results: List[EvaluationResult],
) -> EvaluationSummary:
    """
    Calculate complete evaluation summary from results.

    Args:
        evaluation_results: List of all evaluation results

    Returns:
        EvaluationSummary with complete statistics
    """
    total_questions = len(evaluation_results)

    # Count successful evaluations (those with both judge and retrieval results)
    successful_count = sum(
        1
        for result in evaluation_results
        if result.judge_result is not None and result.retrieval_result is not None
    )

    failed_count = total_questions - successful_count

    # Calculate sub-summaries
    judge_summary = calculate_judge_summary(evaluation_results)
    retrieval_summary = calculate_retrieval_summary(evaluation_results)

    return EvaluationSummary(
        total_questions=total_questions,
        successful_evaluations=successful_count,
        failed_evaluations=failed_count,
        judge_summary=judge_summary,
        retrieval_summary=retrieval_summary,
    )
