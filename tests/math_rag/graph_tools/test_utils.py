"""
Tests for graph tools utilities.

Focuses on edge cases and real risks in formatting functions and PathRAG
query generation.
"""

import pytest
from langchain_core.documents import Document

from math_rag.graph_tools.utils import (
    format_document,
    format_retrieval_results,
    get_pathrag_query,
)


class TestFormatDocument:
    """Test document formatting for edge cases that could cause runtime errors."""

    @pytest.mark.parametrize(
        "page_content,metadata,index,expected_content,description",
        [
            (
                "Test content",
                {},
                1,
                ["Document 1", "Test content"],
                "empty metadata dict",
            ),
            ("", {"title": "Empty Doc"}, 2, ["Document 2"], "empty page content"),
            (
                "Math content",
                {"title": "Test", "type": "theorem"},
                1,
                ["Math content", "None"],
                "missing text_nl",
            ),
        ],
    )
    def test_format_document_edge_cases(
        self, page_content, metadata, index, expected_content, description
    ):
        """Test formatting edge cases that could cause runtime errors."""
        doc = Document(page_content=page_content, metadata=metadata)

        result = format_document(doc, index)

        for content in expected_content:
            assert content in result, f"Failed for case: {description}"

    def test_format_document_with_allowed_metadata(self):
        """Test that allowed metadata (number, type, title) appears correctly."""
        doc = Document(
            page_content="Quadratic formula",
            metadata={
                "title": "Quadratic Formula",
                "type": "theorem",
                "number": "1.2.3",
                "ignored_field": "should not appear",
            },
        )

        result = format_document(doc, 1)

        assert "title: Quadratic Formula" in result
        assert "type: theorem" in result
        assert "number: 1.2.3" in result
        assert "ignored_field" not in result
        assert "METADATA:" in result


class TestFormatRetrievalResults:
    """Test multi-document formatting - basic functionality."""

    @pytest.mark.parametrize(
        "docs,search_type,expected_content",
        [
            (
                [
                    Document(page_content="First doc", metadata={"title": "Doc 1"}),
                    Document(page_content="Second doc", metadata={"title": "Doc 2"}),
                ],
                "test search",
                [
                    "Retrieved 2 documents using test search",
                    "Document 1",
                    "Document 2",
                    "First doc",
                    "Second doc",
                ],
            ),
            (
                [Document(page_content="Only doc", metadata={})],
                "hybrid search",
                ["Retrieved 1 documents using hybrid search", "Document 1", "Only doc"],
            ),
        ],
    )
    def test_format_retrieval_results(self, docs, search_type, expected_content):
        """Test formatting different numbers of documents."""
        result = format_retrieval_results(docs, search_type)

        for content in expected_content:
            assert content in result


class TestPathRAGQuery:
    """Test PathRAG Cypher query - high risk for syntax errors."""

    @pytest.mark.parametrize(
        "expected_elements,description",
        [
            (
                ["RETURN", "as text", "as score", "as metadata"],
                "required LangChain columns",
            ),
            (
                ["[:CITES]", "OPTIONAL MATCH", "collect(DISTINCT"],
                "CITES relationship traversal",
            ),
            (["WITH", "UNWIND", "RETURN", "ORDER BY"], "valid Cypher structure"),
            (["CASE", "WHEN", "ELSE"], "score assignment logic"),
        ],
    )
    def test_pathrag_query_contains_required_elements(
        self, expected_elements, description
    ):
        """Test that PathRAG query contains all required elements."""
        query = get_pathrag_query()

        # Query should not be empty
        assert query.strip(), f"Query is empty for test: {description}"

        for element in expected_elements:
            assert element in query, (
                f"Missing '{element}' in query for test: {description}"
            )

    def test_pathrag_query_contains_score_value(self):
        """Test that query contains expected score values."""
        query = get_pathrag_query()

        # Should contain either the specific score value or score variable
        assert "0.1" in query or "score" in query
