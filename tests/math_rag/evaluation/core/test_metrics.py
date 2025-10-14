"""
Unit tests for retrieval metrics functions.

Tests focus on core retrieval quality evaluation functionality
and edge cases that could cause runtime errors.
"""

from math_rag.evaluation.core.metrics import (
    calculate_f1_score,
    calculate_precision,
    calculate_recall,
    evaluate_retrieval_quality,
    parse_gold_sources,
)
from math_rag.evaluation.core.models import RetrievalResult


class TestParseGoldSources:
    """Test parsing of gold standard sources."""

    def test_valid_source_string(self):
        """Test parsing valid source with pattern."""
        source = "topological_spaces_1.2.3"
        result = parse_gold_sources(source)
        assert result == {"1.2.3"}

    def test_empty_source_string(self):
        """Test parsing empty source."""
        result = parse_gold_sources("")
        assert result == set()

    def test_none_source_string(self):
        """Test parsing None source."""
        result = parse_gold_sources(None)
        assert result == set()


class TestCalculatePrecision:
    """Test precision calculation edge cases."""

    def test_perfect_precision(self):
        """Test precision when all retrieved are relevant."""
        retrieved = ["1.1.1", "1.1.2"]
        gold = {"1.1.1", "1.1.2", "1.1.3"}
        result = calculate_precision(retrieved, gold)
        assert result == 1.0

    def test_zero_precision(self):
        """Test precision when no retrieved are relevant."""
        retrieved = ["2.1.1", "2.1.2"]
        gold = {"1.1.1", "1.1.2"}
        result = calculate_precision(retrieved, gold)
        assert result == 0.0

    def test_empty_retrieved(self):
        """Test precision with empty retrieved list."""
        retrieved = []
        gold = {"1.1.1", "1.1.2"}
        result = calculate_precision(retrieved, gold)
        assert result == 0.0


class TestCalculateRecall:
    """Test recall calculation edge cases."""

    def test_perfect_recall(self):
        """Test recall when all gold documents are retrieved."""
        retrieved = ["1.1.1", "1.1.2", "2.1.1"]
        gold = {"1.1.1", "1.1.2"}
        result = calculate_recall(retrieved, gold)
        assert result == 1.0

    def test_zero_recall(self):
        """Test recall when no gold documents are retrieved."""
        retrieved = ["2.1.1", "2.1.2"]
        gold = {"1.1.1", "1.1.2"}
        result = calculate_recall(retrieved, gold)
        assert result == 0.0

    def test_empty_gold(self):
        """Test recall with empty gold set."""
        retrieved = ["1.1.1", "1.1.2"]
        gold = set()
        result = calculate_recall(retrieved, gold)
        assert result == 0.0


class TestCalculateF1Score:
    """Test F1 score calculation."""

    def test_perfect_f1(self):
        """Test F1 with perfect precision and recall."""
        result = calculate_f1_score(1.0, 1.0)
        assert result == 1.0

    def test_zero_f1_both_zero(self):
        """Test F1 when both precision and recall are zero."""
        result = calculate_f1_score(0.0, 0.0)
        assert result == 0.0

    def test_balanced_f1(self):
        """Test F1 with balanced precision and recall."""
        precision = 0.6
        recall = 0.6
        expected = 2 * (precision * recall) / (precision + recall)
        result = calculate_f1_score(precision, recall)
        assert result == expected


class TestEvaluateRetrievalQuality:
    """Test complete retrieval quality evaluation."""

    def test_successful_evaluation(self):
        """Test complete evaluation with mock documents."""

        class MockDoc:
            def __init__(self, doc_id):
                self.metadata = {"number": doc_id}

        retrieved_docs = [MockDoc("1.1.1"), MockDoc("1.1.2"), MockDoc("2.1.1")]
        gold_source = "topology_1.1.1"

        result = evaluate_retrieval_quality(retrieved_docs, gold_source)

        assert isinstance(result, RetrievalResult)
        assert result.precision == 1 / 3  # 1 relevant out of 3 retrieved
        assert result.recall == 1.0  # 1 relevant out of 1 gold
        assert result.retrieved_count == 3
        assert result.gold_count == 1
        assert result.relevant_retrieved_count == 1

    def test_empty_retrieved_docs(self):
        """Test evaluation with no retrieved documents."""
        retrieved_docs = []
        gold_source = "topology_1.1.1"

        result = evaluate_retrieval_quality(retrieved_docs, gold_source)

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.retrieved_count == 0
        assert result.gold_count == 1
        assert result.relevant_retrieved_count == 0

    def test_no_gold_source(self):
        """Test evaluation with invalid gold source."""

        class MockDoc:
            def __init__(self, doc_id):
                self.metadata = {"number": doc_id}

        retrieved_docs = [MockDoc("1.1.1")]
        gold_source = "invalid_source"

        result = evaluate_retrieval_quality(retrieved_docs, gold_source)

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.gold_count == 0
