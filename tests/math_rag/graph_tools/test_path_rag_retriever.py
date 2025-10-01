"""
Tests for PathRAG retriever functionality.

This module tests the PathRAG retriever implementation with mocked database dependencies
to validate the core logic without requiring an actual Neo4j database connection.

The key insight is to use dependency injection and mocking to test the Cypher query
logic and retrieval flow without needing a real database.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from math_rag.graph_tools.path_rag_retriever import (
    PathRAGRetrieverTool,
    format_document,
)


class TestPathRAGRetrieverTool:
    """Test cases for PathRAGRetrieverTool with mocked dependencies.
    
    These tests demonstrate how to test the PathRAG retriever without requiring
    an actual Neo4j database by using dependency injection and mocking.
    """

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        mock = Mock(spec=Embeddings)
        mock.embed_query.return_value = [0.1] * 384  # Mock embedding vector
        return mock

    @pytest.fixture
    def retriever_config(self):
        """Standard configuration for the retriever."""
        return {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
            "vector_index_name": "test_vector_index",
            "keyword_index_name": "test_keyword_index",
            "embedding_node_property": "embedding",
        }

    @patch("math_rag.graph_tools.path_rag_retriever.Neo4jVector")
    def test_initialization_creates_vector_store_with_custom_query(
        self, mock_neo4j_vector, mock_embedding_model, retriever_config
    ):
        """Test that initialization creates vector store with custom retrieval query.
        
        This test validates that:
        1. Neo4jVector.from_existing_index is called with the correct parameters
        2. A custom retrieval query containing CITES logic is provided
        3. The query has the expected structure for PathRAG functionality
        """
        # Arrange
        mock_vector_store = Mock()
        mock_neo4j_vector.from_existing_index.return_value = mock_vector_store

        # Act
        retriever = PathRAGRetrieverTool(
            embedding_model=mock_embedding_model, **retriever_config
        )

        # Assert
        mock_neo4j_vector.from_existing_index.assert_called_once()
        call_kwargs = mock_neo4j_vector.from_existing_index.call_args[1]

        # Verify custom query is provided and contains CITES logic
        assert "retrieval_query" in call_kwargs
        custom_query = call_kwargs["retrieval_query"]
        assert custom_query != ""
        assert "CITES" in custom_query
        assert "OPTIONAL MATCH" in custom_query
        assert "collect(DISTINCT" in custom_query
        assert "UNWIND" in custom_query

        # Verify other parameters are passed correctly
        assert call_kwargs["index_name"] == "test_vector_index"
        assert call_kwargs["keyword_index_name"] == "test_keyword_index"
        assert call_kwargs["embedding_node_property"] == "embedding"
        assert call_kwargs["url"] == "bolt://localhost:7687"
        assert call_kwargs["username"] == "neo4j"
        assert call_kwargs["password"] == "password"

    def test_custom_retrieval_query_structure(self):
        """Test that the custom retrieval query has the expected Cypher structure.
        
        This test validates the Cypher query without needing a database by
        examining the query string directly for required components.
        """
        # Arrange & Act
        retriever = PathRAGRetrieverTool.__new__(PathRAGRetrieverTool)
        query = retriever._get_custom_retrieval_query()

        # Assert key components are present for PathRAG functionality
        required_elements = [
            # Basic structure
            "WITH node, score",
            
            # CITES relationship traversal (both directions)
            "OPTIONAL MATCH (node)-[:CITES]->(citedNode)",
            "OPTIONAL MATCH (referencingNode)-[:CITES]->(node)",
            
            # Collection of connected nodes
            "collect(DISTINCT citedNode)",
            "collect(DISTINCT referencingNode)",
            
            # Expansion to include connected nodes
            "UNWIND ([node] + connectedNodes)",
            
            # Score assignment logic
            "CASE",
            "WHEN resultNode = node THEN score",
            "ELSE 0.1",
            
            # Final result formatting
            "RETURN DISTINCT resultNode as node, finalScore as score",
            "ORDER BY finalScore DESC",
        ]

        for element in required_elements:
            assert element in query, f"Query missing required element: {element}"

        # Verify query is not empty and has reasonable length
        assert len(query) > 500, "Query should be substantial"

    @patch("math_rag.graph_tools.path_rag_retriever.Neo4jVector")
    def test_retrieve_method_calls_vector_store_correctly(
        self, mock_neo4j_vector, mock_embedding_model, retriever_config
    ):
        """Test that _retrieve method calls the vector store with correct parameters.
        
        This test validates the retrieval flow by mocking the vector store
        and verifying the correct method calls and parameter passing.
        """
        # Arrange
        mock_vector_store = Mock()
        mock_neo4j_vector.from_existing_index.return_value = mock_vector_store

        # Mock search results - simulate PathRAG returning both direct matches and connected nodes
        mock_docs = [
            Document(page_content="Direct match: Definition", metadata={"id": "1", "type": "definition"}),
            Document(page_content="Connected: Theorem citing definition", metadata={"id": "2", "type": "theorem"}),
        ]
        mock_vector_store.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.9),  # High score for direct match
            (mock_docs[1], 0.1),  # Low score for connected node (set by custom query)
        ]

        retriever = PathRAGRetrieverTool(
            embedding_model=mock_embedding_model, **retriever_config
        )

        # Act
        result_docs = retriever._retrieve("topology definition", k=5)

        # Assert
        # Verify vector store was called with correct parameters
        mock_vector_store.similarity_search_with_score.assert_called_once_with(
            "topology definition", k=5, threshold=0.0  # threshold=0.0 to include connected nodes
        )
        
        # Verify results are extracted correctly
        assert len(result_docs) == 2
        assert result_docs[0].page_content == "Direct match: Definition"
        assert result_docs[1].page_content == "Connected: Theorem citing definition"

    @patch("math_rag.graph_tools.path_rag_retriever.Neo4jVector")
    def test_forward_method_complete_flow(
        self, mock_neo4j_vector, mock_embedding_model, retriever_config
    ):
        """Test the complete forward method flow from query to formatted output.
        
        This test validates the entire PathRAG flow including:
        1. Input validation
        2. Vector store querying
        3. Result formatting
        4. Error handling
        """
        # Arrange
        mock_vector_store = Mock()
        mock_neo4j_vector.from_existing_index.return_value = mock_vector_store

        # Mock realistic mathematical content with PathRAG results
        mock_docs = [
            Document(
                page_content="A topology on a set X is a collection of subsets...",
                metadata={
                    "number": "1.2.3",
                    "type": "definition",
                    "title": "Topology Definition",
                    "text_nl": "Eine Topologie auf einer Menge X ist...",
                },
            ),
            Document(
                page_content="Every continuous function preserves topological properties...",
                metadata={
                    "number": "2.3.4",
                    "type": "theorem",
                    "title": "Continuity Theorem",
                },
            ),
        ]
        mock_vector_store.similarity_search_with_score.return_value = [
            (mock_docs[0], 0.8),  # Direct match from vector/keyword search
            (mock_docs[1], 0.1),  # Connected node via CITES relationship
        ]

        retriever = PathRAGRetrieverTool(
            embedding_model=mock_embedding_model, **retriever_config
        )

        # Act
        result = retriever.forward(query="topology definition", k=3)

        # Assert
        assert isinstance(result, str)
        assert "PathRAG retrieved 2 documents" in result
        assert "initial matches + connected via CITES" in result
        
        # Verify content from both direct matches and connected nodes
        assert "A topology on a set X is a collection" in result
        assert "Every continuous function preserves" in result
        assert "Topology Definition" in result
        assert "Eine Topologie auf einer Menge X ist" in result

    @patch("math_rag.graph_tools.path_rag_retriever.Neo4jVector")
    def test_forward_method_handles_empty_results(
        self, mock_neo4j_vector, mock_embedding_model, retriever_config
    ):
        """Test that forward method handles empty search results gracefully."""
        # Arrange
        mock_vector_store = Mock()
        mock_neo4j_vector.from_existing_index.return_value = mock_vector_store
        mock_vector_store.similarity_search_with_score.return_value = []

        retriever = PathRAGRetrieverTool(
            embedding_model=mock_embedding_model, **retriever_config
        )

        # Act
        result = retriever.forward(query="nonexistent concept", k=5)

        # Assert
        assert result == "No documents found for query: 'nonexistent concept'"

    @patch("math_rag.graph_tools.path_rag_retriever.Neo4jVector")
    def test_forward_method_handles_database_exceptions(
        self, mock_neo4j_vector, mock_embedding_model, retriever_config
    ):
        """Test that forward method handles database exceptions gracefully."""
        # Arrange
        mock_vector_store = Mock()
        mock_neo4j_vector.from_existing_index.return_value = mock_vector_store
        mock_vector_store.similarity_search_with_score.side_effect = Exception(
            "Neo4j connection timeout"
        )

        retriever = PathRAGRetrieverTool(
            embedding_model=mock_embedding_model, **retriever_config
        )

        # Act
        result = retriever.forward(query="test query", k=5)

        # Assert
        assert "Error retrieving documents: Neo4j connection timeout" in result

    def test_input_validation(self, mock_embedding_model, retriever_config):
        """Test that forward method validates input parameters correctly."""
        # Mock Neo4jVector to avoid actual database connection during init
        with patch("math_rag.graph_tools.path_rag_retriever.Neo4jVector"):
            retriever = PathRAGRetrieverTool(
                embedding_model=mock_embedding_model, **retriever_config
            )

            # Test invalid query type
            with pytest.raises(AssertionError, match="must be a string"):
                retriever.forward(query=123, k=5)

            # Test invalid k type
            with pytest.raises(AssertionError, match="must be a positive integer"):
                retriever.forward(query="valid query", k="invalid")

            # Test invalid k value
            with pytest.raises(AssertionError, match="must be a positive integer"):
                retriever.forward(query="valid query", k=0)

    def test_tool_interface_compliance(self):
        """Test that the tool implements the correct smolagents interface."""
        # Check class attributes that define the tool interface
        assert PathRAGRetrieverTool.name == "path_rag_retriever"
        assert PathRAGRetrieverTool.output_type == "string"

        # Check description contains key functionality terms
        description = PathRAGRetrieverTool.description.lower()
        assert "hybrid search" in description
        assert "cites" in description
        assert "connected" in description

        # Check input specification
        inputs = PathRAGRetrieverTool.inputs
        assert "query" in inputs
        assert "k" in inputs

        # Verify query input specification
        query_spec = inputs["query"]
        assert query_spec["type"] == "string"
        assert "mathematical" in query_spec["description"].lower()

        # Verify k input specification
        k_spec = inputs["k"]
        assert k_spec["type"] == "integer"
        assert k_spec["default"] == 10
        assert k_spec["nullable"] is True


class TestFormatDocument:
    """Test cases for the format_document helper function.
    
    These tests can run independently of the database since they only
    test document formatting logic.
    """

    def test_format_document_with_mathematical_metadata(self):
        """Test formatting a mathematical document with complete metadata."""
        doc = Document(
            page_content="Let X be a topological space. A function f: X → Y is continuous if...",
            metadata={
                "number": "2.3.4",
                "type": "theorem",
                "title": "Continuity Characterization Theorem",
                "text_nl": "Sei X ein topologischer Raum. Eine Funktion f: X → Y ist stetig, wenn...",
                "source_section": "2.3",  # Should be ignored
                "page_number": 42,        # Should be ignored
            },
        )

        result = format_document(doc, 1)

        # Check document structure
        assert "===== Document 1 =====" in result
        assert "Let X be a topological space" in result
        assert "Sei X ein topologischer Raum" in result
        assert "METADATA:" in result

        # Check allowed metadata is included
        assert "number: 2.3.4" in result
        assert "type: theorem" in result
        assert "title: Continuity Characterization Theorem" in result

        # Check non-allowed metadata is excluded
        assert "source_section" not in result
        assert "page_number" not in result

    def test_format_document_with_partial_metadata(self):
        """Test formatting with only some allowed metadata fields."""
        doc = Document(
            page_content="A metric space is a set equipped with a distance function.",
            metadata={"type": "definition", "title": "Metric Space"},
        )

        result = format_document(doc, 2)

        assert "===== Document 2 =====" in result
        assert "A metric space is a set equipped" in result
        assert "type: definition" in result
        assert "title: Metric Space" in result
        assert "METADATA:" in result

    def test_format_document_no_allowed_metadata(self):
        """Test formatting when no allowed metadata fields are present."""
        doc = Document(
            page_content="Mathematical content without relevant metadata.",
            metadata={"irrelevant_field": "ignored", "another_field": "also ignored"},
        )

        result = format_document(doc, 3)

        assert "===== Document 3 =====" in result
        assert "Mathematical content without relevant metadata." in result
        
        # Should not have METADATA section when no allowed metadata exists
        assert "METADATA:" not in result
        assert "irrelevant_field" not in result

    def test_format_document_empty_metadata(self):
        """Test formatting with empty metadata dictionary."""
        doc = Document(
            page_content="Content with no metadata at all.",
            metadata={}
        )

        result = format_document(doc, 4)

        assert "===== Document 4 =====" in result
        assert "Content with no metadata at all." in result
        assert "METADATA:" not in result

    def test_format_document_with_text_nl_only(self):
        """Test that text_nl appears even without METADATA section."""
        doc = Document(
            page_content="English mathematical content.",
            metadata={"text_nl": "German mathematical content.", "ignored": "value"}
        )

        result = format_document(doc, 5)

        assert "===== Document 5 =====" in result
        assert "English mathematical content." in result
        assert "German mathematical content." in result
        # text_nl appears in content area, not metadata area
        assert "METADATA:" not in result


# Test demonstrating the key insight about dependency injection and mocking
class TestDependencyInjectionApproach:
    """Demonstrate the software design insight about dependency injection and mocking.
    
    The key insight from the user's comment is that by using dependency injection 
    (accepting a driver or connection) and mocking, we can test the database-dependent
    code without requiring an actual database.
    """

    def test_mocking_enables_cypher_validation(self):
        """Demonstrate how mocking allows us to validate Cypher query logic.
        
        This test shows how we can verify that the custom Cypher query
        would be executed correctly without needing a real Neo4j database.
        """
        # The PathRAG retriever uses Neo4jVector internally, but by mocking it,
        # we can verify the custom query is constructed and passed correctly
        with patch("math_rag.graph_tools.path_rag_retriever.Neo4jVector") as mock_neo4j:
            mock_vector_store = Mock()
            mock_neo4j.from_existing_index.return_value = mock_vector_store
            
            mock_embedding = Mock()
            retriever = PathRAGRetrieverTool(
                embedding_model=mock_embedding,
                uri="bolt://localhost:7687",
                username="neo4j", 
                password="password",
                vector_index_name="vector_index",
                keyword_index_name="keyword_index",
                embedding_node_property="embedding"
            )
            
            # Extract the custom query that would be executed
            call_kwargs = mock_neo4j.from_existing_index.call_args[1]
            custom_query = call_kwargs["retrieval_query"]
            
            # Now we can validate the Cypher logic without executing it
            assert "OPTIONAL MATCH (node)-[:CITES]->(citedNode)" in custom_query
            assert "OPTIONAL MATCH (referencingNode)-[:CITES]->(node)" in custom_query
            assert "collect(DISTINCT citedNode)" in custom_query
            assert "collect(DISTINCT referencingNode)" in custom_query
            
            # This validates the PathRAG logic structure without needing Neo4j

    def test_mocking_enables_flow_validation(self):
        """Demonstrate how mocking allows us to test the complete retrieval flow.
        
        This shows how dependency injection and mocking enable testing
        the main application flow without external dependencies.
        """
        with patch("math_rag.graph_tools.path_rag_retriever.Neo4jVector") as mock_neo4j:
            # Mock the complete flow
            mock_vector_store = Mock()
            mock_neo4j.from_existing_index.return_value = mock_vector_store
            
            # Simulate PathRAG results: direct matches + connected nodes
            mock_vector_store.similarity_search_with_score.return_value = [
                (Document(page_content="Direct match", metadata={"score_type": "direct"}), 0.9),
                (Document(page_content="Connected via CITES", metadata={"score_type": "connected"}), 0.1),
            ]
            
            # Test the complete flow
            mock_embedding = Mock()
            retriever = PathRAGRetrieverTool(
                embedding_model=mock_embedding,
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password", 
                vector_index_name="vector_index",
                keyword_index_name="keyword_index",
                embedding_node_property="embedding"
            )
            
            result = retriever.forward("test query", k=5)
            
            # Validate the flow worked correctly
            assert "PathRAG retrieved 2 documents" in result
            assert "Direct match" in result
            assert "Connected via CITES" in result
            
            # Verify the vector store was called with PathRAG-specific parameters
            mock_vector_store.similarity_search_with_score.assert_called_with(
                "test query", k=5, threshold=0.0  # threshold=0.0 is PathRAG-specific
            )


if __name__ == "__main__":
    pytest.main([__file__])