"""
Unit tests for Pydantic evaluation models.

Tests focus on core validation, edge cases, and critical model behavior.
"""

import pytest
from pydantic import ValidationError

from math_rag.evaluation.core.models import (
    EvaluationResult,
    EvaluationSummary,
    JudgeResult,
    JudgeSummary,
    RetrievalResult,
    RetrievalSummary,
)


class TestJudgeResult:
    """Test JudgeResult model validation and behavior."""

    def test_valid_judge_result(self):
        """Test creating a valid JudgeResult."""
        result = JudgeResult(
            mathematical_correctness_score=4,
            context_relevance_score=3,
            context_utilization_score=5,
            completeness_score=2,
            overall_pass=True,
            judgement_summary="Good mathematical reasoning",
        )

        assert result.mathematical_correctness_score == 4
        assert result.overall_pass is True
        assert result.average_score == 3.5  # (4+3+5+2)/4

    def test_score_validation_bounds(self):
        """Test that scores are properly bounded between 1 and 5."""
        with pytest.raises(ValidationError):
            JudgeResult(
                mathematical_correctness_score=0,  # Below minimum
                context_relevance_score=3,
                context_utilization_score=3,
                completeness_score=3,
                overall_pass=True,
                judgement_summary="Test",
            )

        with pytest.raises(ValidationError):
            JudgeResult(
                mathematical_correctness_score=6,  # Above maximum
                context_relevance_score=3,
                context_utilization_score=3,
                completeness_score=3,
                overall_pass=True,
                judgement_summary="Test",
            )

    def test_empty_lists_default(self):
        """Test that list fields default to empty lists."""
        result = JudgeResult(
            mathematical_correctness_score=3,
            context_relevance_score=3,
            context_utilization_score=3,
            completeness_score=3,
            overall_pass=True,
            judgement_summary="Test",
        )

        assert result.key_info_found == []
        assert result.hallucinations == []
        assert result.missing_info == []


class TestRetrievalResult:
    """Test RetrievalResult model validation and behavior."""

    def test_valid_retrieval_result(self):
        """Test creating a valid RetrievalResult."""
        result = RetrievalResult(
            precision=0.75,
            recall=0.5,
            f1=0.6,
            retrieved_count=10,
            gold_count=5,
            relevant_retrieved_count=3,
        )

        assert result.precision == 0.75
        assert result.recall == 0.5
        assert result.retrieved_count == 10

    def test_metric_bounds_validation(self):
        """Test that precision, recall, and F1 are bounded between 0 and 1."""
        with pytest.raises(ValidationError):
            RetrievalResult(
                precision=-0.1,  # Below minimum
                recall=0.5,
                f1=0.5,
                retrieved_count=10,
                gold_count=5,
                relevant_retrieved_count=3,
            )

    def test_relevant_count_validation(self):
        """Test validation of relevant_retrieved_count consistency."""
        # relevant_retrieved_count cannot exceed retrieved_count
        with pytest.raises(ValidationError):
            RetrievalResult(
                precision=0.5,
                recall=0.5,
                f1=0.5,
                retrieved_count=5,
                gold_count=10,
                relevant_retrieved_count=7,  # > retrieved_count
            )


class TestEvaluationResult:
    """Test EvaluationResult model composition and behavior."""

    def test_valid_evaluation_result(self):
        """Test creating a complete EvaluationResult."""
        judge_result = JudgeResult(
            mathematical_correctness_score=4,
            context_relevance_score=3,
            context_utilization_score=5,
            completeness_score=2,
            overall_pass=True,
            judgement_summary="Good answer",
        )

        retrieval_result = RetrievalResult(
            precision=0.8,
            recall=0.6,
            f1=0.69,
            retrieved_count=10,
            gold_count=5,
            relevant_retrieved_count=4,
        )

        result = EvaluationResult(
            question="What is topology?",
            ground_truth="Topology is...",
            predicted_answer="Topology studies...",
            retrieved_context="Context about topology...",
            judge_result=judge_result,
            retrieval_result=retrieval_result,
            difficulty=2,
            source="topology_book_1.1.1",
        )

        assert result.question == "What is topology?"
        assert result.judge_result == judge_result
        assert result.retrieval_result == retrieval_result

    def test_optional_fields(self):
        """Test that optional fields can be None."""
        result = EvaluationResult(
            question="Test question",
            ground_truth="Test truth",
            retrieved_context="Test context",
        )

        assert result.predicted_answer is None
        assert result.judge_result is None
        assert result.retrieval_result is None

    def test_predicted_answer_provided(self):
        """Test that predicted_answer can be provided."""
        result = EvaluationResult(
            question="Test question",
            ground_truth="Test truth",
            predicted_answer="Test prediction",
            retrieved_context="Test context",
        )

        assert result.predicted_answer == "Test prediction"


class TestEvaluationSummary:
    """Test EvaluationSummary model validation and hierarchy."""

    def test_valid_evaluation_summary(self):
        """Test creating a complete EvaluationSummary."""
        judge_summary = JudgeSummary(
            avg_mathematical_correctness=3.5,
            avg_context_relevance=4.0,
            avg_context_utilization=3.2,
            avg_completeness=2.8,
            avg_overall_score=3.375,
            pass_rate=0.75,
        )

        retrieval_summary = RetrievalSummary(
            avg_precision=0.8,
            avg_recall=0.6,
            avg_f1=0.69,
            total_questions_evaluated=50,
        )

        summary = EvaluationSummary(
            total_questions=100,
            successful_evaluations=90,
            failed_evaluations=10,
            judge_summary=judge_summary,
            retrieval_summary=retrieval_summary,
        )

        assert summary.total_questions == 100
        assert summary.successful_evaluations == 90
        assert summary.failed_evaluations == 10
