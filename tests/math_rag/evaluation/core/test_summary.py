"""
Unit tests for summary calculation functions.

Tests focus on core summary calculation logic and edge cases.
"""

from math_rag.evaluation.core.models import (
    EvaluationResult,
    JudgeResult,
    RetrievalResult,
)
from math_rag.evaluation.core.summary import (
    calculate_evaluation_summary,
    calculate_judge_summary,
    calculate_retrieval_summary,
)


class TestCalculateJudgeSummary:
    """Test judge summary calculation."""

    def create_judge_result(
        self,
        math_score=3,
        context_rel=3,
        context_util=3,
        completeness=3,
        overall_pass=True,
    ):
        """Helper to create JudgeResult."""
        return JudgeResult(
            mathematical_correctness_score=math_score,
            context_relevance_score=context_rel,
            context_utilization_score=context_util,
            completeness_score=completeness,
            overall_pass=overall_pass,
            judgement_summary="Test summary",
        )

    def create_evaluation_result(self, judge_result=None):
        """Helper to create EvaluationResult."""
        return EvaluationResult(
            question="Test question",
            ground_truth="Test truth",
            predicted_answer="Test answer",
            retrieved_context="Test context",
            judge_result=judge_result,
        )

    def test_empty_results(self):
        """Test summary calculation with empty results."""
        results = []
        summary = calculate_judge_summary(results)

        assert summary.avg_mathematical_correctness == 0.0
        assert summary.pass_rate == 0.0

    def test_no_valid_judge_results(self):
        """Test summary with evaluation results but no valid judge data."""
        results = [
            self.create_evaluation_result(judge_result=None),
            self.create_evaluation_result(judge_result=None),
        ]
        summary = calculate_judge_summary(results)

        assert summary.avg_mathematical_correctness == 0.0
        assert summary.pass_rate == 0.0

    def test_single_result(self):
        """Test summary calculation with single result."""
        judge_result = self.create_judge_result(
            math_score=4,
            context_rel=3,
            context_util=5,
            completeness=2,
            overall_pass=True,
        )
        results = [self.create_evaluation_result(judge_result)]

        summary = calculate_judge_summary(results)

        assert summary.avg_mathematical_correctness == 4.0
        assert summary.avg_overall_score == 3.5  # (4+3+5+2)/4
        assert summary.pass_rate == 1.0

    def test_multiple_results(self):
        """Test summary calculation with multiple results."""
        results = [
            self.create_evaluation_result(self.create_judge_result(4, 3, 5, 2, True)),
            self.create_evaluation_result(self.create_judge_result(2, 4, 3, 4, False)),
            self.create_evaluation_result(self.create_judge_result(5, 5, 4, 3, True)),
        ]

        summary = calculate_judge_summary(results)

        assert summary.avg_mathematical_correctness == (4 + 2 + 5) / 3
        assert summary.pass_rate == 2 / 3  # 2 out of 3 passed


class TestCalculateRetrievalSummary:
    """Test retrieval summary calculation."""

    def create_retrieval_result(self, precision=0.5, recall=0.5, f1=0.5):
        """Helper to create RetrievalResult."""
        return RetrievalResult(
            precision=precision,
            recall=recall,
            f1=f1,
            retrieved_count=10,
            gold_count=5,
            relevant_retrieved_count=3,
        )

    def create_evaluation_result(self, retrieval_result=None):
        """Helper to create EvaluationResult."""
        return EvaluationResult(
            question="Test question",
            ground_truth="Test truth",
            predicted_answer="Test answer",
            retrieved_context="Test context",
            retrieval_result=retrieval_result,
        )

    def test_empty_results(self):
        """Test summary calculation with empty results."""
        results = []
        summary = calculate_retrieval_summary(results)

        assert summary.avg_precision == 0.0
        assert summary.total_questions_evaluated == 0

    def test_single_result(self):
        """Test summary calculation with single result."""
        retrieval_result = self.create_retrieval_result(0.8, 0.6, 0.69)
        results = [self.create_evaluation_result(retrieval_result)]

        summary = calculate_retrieval_summary(results)

        assert summary.avg_precision == 0.8
        assert summary.avg_recall == 0.6
        assert summary.total_questions_evaluated == 1

    def test_multiple_results(self):
        """Test summary calculation with multiple results."""
        results = [
            self.create_evaluation_result(self.create_retrieval_result(0.8, 0.6, 0.69)),
            self.create_evaluation_result(self.create_retrieval_result(0.6, 0.8, 0.69)),
            self.create_evaluation_result(self.create_retrieval_result(1.0, 0.4, 0.57)),
        ]

        summary = calculate_retrieval_summary(results)

        assert summary.avg_precision == (0.8 + 0.6 + 1.0) / 3
        assert summary.total_questions_evaluated == 3


class TestCalculateEvaluationSummary:
    """Test complete evaluation summary calculation."""

    def create_complete_evaluation_result(
        self, judge_pass=True, include_judge=True, include_retrieval=True
    ):
        """Helper to create complete EvaluationResult."""
        judge_result = None
        if include_judge:
            judge_result = JudgeResult(
                mathematical_correctness_score=4,
                context_relevance_score=3,
                context_utilization_score=5,
                completeness_score=2,
                overall_pass=judge_pass,
                judgement_summary="Test summary",
            )

        retrieval_result = None
        if include_retrieval:
            retrieval_result = RetrievalResult(
                precision=0.8,
                recall=0.6,
                f1=0.69,
                retrieved_count=10,
                gold_count=5,
                relevant_retrieved_count=4,
            )

        return EvaluationResult(
            question="Test question",
            ground_truth="Test truth",
            predicted_answer="Test answer",
            retrieved_context="Test context",
            judge_result=judge_result,
            retrieval_result=retrieval_result,
        )

    def test_all_successful(self):
        """Test summary with all successful evaluations."""
        results = [
            self.create_complete_evaluation_result(True, True, True),
            self.create_complete_evaluation_result(False, True, True),
            self.create_complete_evaluation_result(True, True, True),
        ]

        summary = calculate_evaluation_summary(results)

        assert summary.total_questions == 3
        assert summary.successful_evaluations == 3  # All have both judge and retrieval
        assert summary.failed_evaluations == 0

    def test_partial_failures(self):
        """Test summary with some failed evaluations."""
        results = [
            self.create_complete_evaluation_result(True, True, True),  # Success
            self.create_complete_evaluation_result(True, False, True),  # Missing judge
            self.create_complete_evaluation_result(
                True, True, False
            ),  # Missing retrieval
        ]

        summary = calculate_evaluation_summary(results)

        assert summary.total_questions == 3
        assert summary.successful_evaluations == 1  # Only first has both
        assert summary.failed_evaluations == 2
