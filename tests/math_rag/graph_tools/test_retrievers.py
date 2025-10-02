"""
Tests for retriever tools integration.

Focuses on integration tests with mocked vector store to verify the BaseRetriever
pattern and both GraphRetrieverTool and PathRAGRetrieverTool work correctly.
"""

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from math_rag.graph_tools.retrievers import GraphRetrieverTool, PathRAGRetrieverTool


class TestRetrieverIntegration:
    """Integration tests with mocked vector store - the main risk area."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock Neo4j vector store."""
        return Mock()

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                page_content="Quadratic equations have the form ax² + bx + c = 0",
                metadata={
                    "title": "Quadratic Equations",
                    "type": "definition",
                    "number": "1.1",
                },
            ),
            Document(
                page_content="The discriminant determines the nature of roots",
                metadata={"title": "Discriminant", "type": "theorem"},
            ),
        ]

    def test_graph_retriever_happy_path(self, mock_vector_store, sample_documents):
        """Test GraphRetriever with successful retrieval."""
        # Setup mock
        mock_vector_store.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.9),
            (sample_documents[1], 0.7),
        ]

        # Test
        retriever = GraphRetrieverTool(vector_store=mock_vector_store)
        result = retriever.forward("quadratic equations", k=2)

        # Verify behavior, not implementation details
        assert "hybrid search" in result
        assert "Document 1" in result
        assert "Document 2" in result
        assert "Quadratic equations" in result

    def test_pathrag_retriever_returns_documents(
        self, mock_vector_store, sample_documents
    ):
        """Test PathRAGRetriever returns formatted documents."""
        # Setup mock
        mock_vector_store.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.8)
        ]

        # Test
        retriever = PathRAGRetrieverTool(vector_store=mock_vector_store)
        result = retriever.forward("derivatives", k=1)

        # Verify behavior
        assert "PathRAG" in result
        assert "CITES" in result
        assert "Document 1" in result

    def test_empty_results_handling(self, mock_vector_store):
        """Test handling when vector store returns no results."""
        # Setup mock to return empty results
        mock_vector_store.similarity_search_with_score.return_value = []

        # Test
        retriever = GraphRetrieverTool(vector_store=mock_vector_store)
        result = retriever.forward("nonexistent query", k=5)

        # Verify
        assert "No documents found for query: 'nonexistent query'" in result

    def test_vector_store_exception_handling(self, mock_vector_store):
        """Test handling when vector store throws exception."""
        # Setup mock to throw exception
        mock_vector_store.similarity_search_with_score.side_effect = Exception(
            "Neo4j connection failed"
        )

        # Test
        retriever = GraphRetrieverTool(vector_store=mock_vector_store)
        result = retriever.forward("test query", k=3)

        # Verify
        assert "Error retrieving documents:" in result
        assert "Neo4j connection failed" in result

    def test_default_k_behavior(self, mock_vector_store):
        """Test that default k behavior works when k is not specified."""
        mock_vector_store.similarity_search_with_score.return_value = []

        # Test that both retrievers work with default k
        graph_retriever = GraphRetrieverTool(vector_store=mock_vector_store)
        graph_result = graph_retriever.forward("test")  # No k specified

        pathrag_retriever = PathRAGRetrieverTool(vector_store=mock_vector_store)
        pathrag_result = pathrag_retriever.forward("test")  # No k specified

        # Verify both return expected "no documents found" message
        assert "No documents found" in graph_result
        assert "No documents found" in pathrag_result

    def test_input_validation(self, mock_vector_store):
        """Test that forward method validates input parameters."""
        retriever = GraphRetrieverTool(vector_store=mock_vector_store)

        # Test invalid query type
        with pytest.raises(AssertionError, match="must be a string"):
            retriever.forward(query=123, k=5)

        # Test invalid k type
        with pytest.raises(AssertionError, match="must be a positive integer"):
            retriever.forward(query="valid query", k="invalid")

        # Test invalid k value
        with pytest.raises(AssertionError, match="must be a positive integer"):
            retriever.forward(query="valid query", k=0)


class TestToolConfiguration:
    """Light tests to verify tool configuration works."""

    def test_tool_names_are_correct(self):
        """Test that tools have correct names for registration."""
        mock_store = Mock()

        graph_tool = GraphRetrieverTool(vector_store=mock_store)
        pathrag_tool = PathRAGRetrieverTool(vector_store=mock_store)

        assert graph_tool.name == "graph_retriever"
        assert pathrag_tool.name == "path_rag_retriever"

    def test_tools_can_be_instantiated(self):
        """Test that both tools can be created with mock vector store."""
        mock_store = Mock()

        # Should not raise exceptions
        graph_tool = GraphRetrieverTool(vector_store=mock_store)
        pathrag_tool = PathRAGRetrieverTool(vector_store=mock_store)

        assert graph_tool.vector_store == mock_store
        assert pathrag_tool.vector_store == mock_store

    def test_tool_inputs_structure(self):
        """Test that Tool inputs have the correct structure without checking specific values."""  # noqa: E501
        # For both retrievers
        for retriever_class in [GraphRetrieverTool, PathRAGRetrieverTool]:
            # Check that name is set
            assert retriever_class.name is not None
            assert isinstance(retriever_class.name, str)

            # Check inputs structure
            inputs = retriever_class.inputs

            # Check query field exists and is configured correctly
            assert "query" in inputs
            assert inputs["query"]["type"] == "string"

            # Check k field exists and has expected structure
            assert "k" in inputs
            assert "default" in inputs["k"]  # has a default (don't check the value)
            assert inputs["k"]["type"] == "integer"
            assert inputs["k"]["nullable"] is True
            assert "description" in inputs["k"]
