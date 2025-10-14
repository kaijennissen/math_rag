"""
Unit tests for DSPy evaluation components.

Tests focus on core judge functionality and Pydantic model integration.
"""

from unittest.mock import Mock, patch

import pytest

from math_rag.evaluation.core.models import JudgeResult
from math_rag.evaluation.dspy.modules import LLMJudge, judge_response


class TestJudgeResponse:
    """Test the judge_response function and Pydantic integration."""

    @patch("math_rag.evaluation.dspy.modules.LLMJudge")
    def test_judge_response_returns_pydantic_model(self, mock_judge_class):
        """Test that judge_response returns a JudgeResult Pydantic model."""
        # Setup mock DSPy result
        mock_dspy_result = Mock()
        mock_dspy_result.mathematical_correctness_score = 4
        mock_dspy_result.context_relevance_score = 3
        mock_dspy_result.context_utilization_score = 5
        mock_dspy_result.completeness_score = 2
        mock_dspy_result.key_info_found = ["definition", "theorem"]
        mock_dspy_result.hallucinations = []
        mock_dspy_result.missing_info = ["proof steps"]
        mock_dspy_result.overall_pass = True
        mock_dspy_result.judgement_summary = "Good mathematical reasoning"

        mock_judge_instance = Mock()
        mock_judge_instance.return_value = mock_dspy_result
        mock_judge_class.return_value = mock_judge_instance

        # Test the function
        result = judge_response(
            question="What is topology?",
            predicted_answer="Topology is a branch of mathematics",
            ground_truth="Topology studies spatial properties",
            context="Mathematical context about topology",
        )

        # Verify it returns a JudgeResult Pydantic model
        assert isinstance(result, JudgeResult)
        assert result.mathematical_correctness_score == 4
        assert result.context_relevance_score == 3
        assert result.overall_pass is True
        assert result.judgement_summary == "Good mathematical reasoning"

    @patch("math_rag.evaluation.dspy.modules.LLMJudge")
    def test_judge_response_handles_empty_context(self, mock_judge_class):
        """Test judge_response works with empty context."""
        mock_dspy_result = Mock()
        mock_dspy_result.mathematical_correctness_score = 3
        mock_dspy_result.context_relevance_score = 1
        mock_dspy_result.context_utilization_score = 1
        mock_dspy_result.completeness_score = 3
        mock_dspy_result.key_info_found = []
        mock_dspy_result.hallucinations = []
        mock_dspy_result.missing_info = []
        mock_dspy_result.overall_pass = False
        mock_dspy_result.judgement_summary = "Limited context provided"

        mock_judge_instance = Mock()
        mock_judge_instance.return_value = mock_dspy_result
        mock_judge_class.return_value = mock_judge_instance

        result = judge_response(
            question="Test question",
            predicted_answer="Test answer",
            ground_truth="Test truth",
            context="",
        )

        assert isinstance(result, JudgeResult)
        assert result.context_relevance_score == 1
        assert result.overall_pass is False

    @patch("math_rag.evaluation.dspy.modules.LLMJudge")
    def test_judge_response_error_handling(self, mock_judge_class):
        """Test judge_response handles DSPy errors gracefully."""
        mock_judge_instance = Mock()
        mock_judge_instance.side_effect = Exception("DSPy error")
        mock_judge_class.return_value = mock_judge_instance

        with pytest.raises(Exception):
            judge_response(
                question="Test question",
                predicted_answer="Test answer",
                ground_truth="Test truth",
                context="Test context",
            )


class TestLLMJudge:
    """Test the LLMJudge class integration."""

    @patch("math_rag.evaluation.dspy.modules.dspy.ChainOfThought")
    def test_llm_judge_initialization(self, mock_chain_of_thought):
        """Test that LLMJudge initializes correctly."""
        mock_chain_of_thought.return_value = Mock()

        judge = LLMJudge()

        assert hasattr(judge, "assess")
        mock_chain_of_thought.assert_called_once()

    @patch("math_rag.evaluation.dspy.modules.dspy.ChainOfThought")
    def test_llm_judge_forward_call(self, mock_chain_of_thought):
        """Test LLMJudge forward method calls assess correctly."""
        mock_assess = Mock()
        mock_chain_of_thought.return_value = mock_assess
        mock_result = Mock()
        mock_assess.return_value = mock_result

        judge = LLMJudge()
        result = judge.forward(
            question="What is calculus?",
            predicted_answer="Calculus studies change",
            context="Mathematical context",
        )

        assert result == mock_result
        mock_assess.assert_called_once_with(
            question="What is calculus?",
            predicted_answer="Calculus studies change",
            context="Mathematical context",
        )
