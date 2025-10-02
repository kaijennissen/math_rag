import logging
from typing import List

import coloredlogs
from langchain_core.documents import Document
from langchain_neo4j import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import SearchType
from smolagents import Tool

from math_rag.graph_tools.utils import format_retrieval_results, get_pathrag_query

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class BaseRetriever(Tool):
    """Base class for all Neo4j retriever tools with common implementation."""

    # Class attributes to be defined by subclasses
    name: str = None
    description: str = None
    default_k: int = 20
    search_description: str = None
    threshold: float = 0.25

    def __init_subclass__(cls, **kwargs):
        """Automatically configure Tool inputs based on class attributes."""
        super().__init_subclass__(**kwargs)

        # Set up inputs based on class attributes
        cls.inputs = {
            "query": {
                "type": "string",
                "description": """The query to search for in the mathematical
                knowledge graph. This can be a question or a statement about
                a mathematical concept. This should be semantically close to
                your target documents. Use the affirmative form rather than
                a question. Use german as language.""",
            },
            "k": {
                "type": "integer",
                "description": (
                    f"Number of documents to retrieve. Default is {cls.default_k}."
                ),
                "default": cls.default_k,
                "nullable": True,
            },
        }
        cls.output_type = "string"

    def __init__(
        self,
        vector_store: Neo4jVector,
        **kwargs,
    ):
        """
        Initialize the retriever tool with a pre-configured vector store.

        Args:
            vector_store: Pre-configured Neo4jVector instance
            **kwargs: Additional arguments passed to Tool base class
        """
        # Use class attributes for Tool configuration
        kwargs.setdefault("name", self.name)
        kwargs.setdefault("description", self.description)
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def _retrieve(self, query: str, k: int) -> List[Document]:
        """
        Retrieve documents from Neo4j using the configured vector store.

        Args:
            query: The search query
            k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving {k} documents for query: '{query}'")

        # Perform search with similarity scores using class-defined threshold
        results = self.vector_store.similarity_search_with_score(
            query, k=k, threshold=self.threshold
        )

        # Extract just the documents from the (doc, score) tuples
        docs = [doc for doc, score in results]

        logger.info(f"Successfully retrieved {len(docs)} documents")
        return docs

    def forward(self, query: str, k: int = None) -> str:
        """
        Main entry point for retrieving documents.

        Args:
            query: The search query
            k: Number of documents to retrieve (default: class default_k)

        Returns:
            A formatted string with the retrieved documents
        """
        if k is None:
            k = self.default_k

        assert isinstance(query, str), "Your search query must be a string"
        assert isinstance(k, int) and k > 0, "k must be a positive integer"

        try:
            # Retrieve documents
            docs = self._retrieve(query, k)

            # Handle empty results here so _format can assume non-empty input
            if not docs:
                return f"No documents found for query: '{query}'"

            # Format and return results
            return format_retrieval_results(docs, self.search_description)

        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return f"Error retrieving documents: {str(e)}"


class GraphRetrieverTool(BaseRetriever):
    """A tool that retrieves documents using hybrid search."""

    name = "graph_retriever"
    description = (
        "Uses hybrid search (vector + keyword) to retrieve mathematical "
        "content from a Neo4j graph database."
    )
    default_k = 10
    search_description = "hybrid search"
    threshold = 0.5


class PathRAGRetrieverTool(BaseRetriever):
    """A tool that retrieves documents using hybrid search + connected nodes."""

    name = "path_rag_retriever"
    description = (
        "Uses hybrid search (vector + keyword) combined with graph traversal "
        "to retrieve mathematical content from a Neo4j graph database. "
        "Returns both matched nodes and connected nodes via "
        "CITES/REFERENCED relationships."
    )
    default_k = 10
    search_description = "PathRAG (initial matches + connected via CITES)"
    threshold = 0.5


def main(query: str, k: int = 5, use_pathrag: bool = False):
    """
    Example usage of the retriever tools.

    Args:
        query: The search query
        k: Number of results to return
        use_pathrag: Whether to use PathRAG retriever instead of hybrid
    """
    import os

    from dotenv import load_dotenv
    from langchain_huggingface import HuggingFaceEmbeddings

    load_dotenv()

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # Option 2: OpenAI (uncomment to use)
    # embedding_model = OpenAIEmbeddings(
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    #     model="text-embedding-3-small"
    # )
    index_name = "vector_index_text_nl_Embedding"
    keyword_index_name = "fulltext_index_AtomicItem"
    embedding_node_property = "text_nl_Embedding"

    if use_pathrag:
        # Create PathRAG vector store directly
        vector_store = Neo4jVector.from_existing_index(
            embedding_model,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            embedding_node_property=embedding_node_property,
            search_type=SearchType.HYBRID,
            retrieval_query=get_pathrag_query(),
        )
        retriever = PathRAGRetrieverTool(vector_store=vector_store)
    else:
        # Create hybrid vector store directly
        vector_store = Neo4jVector.from_existing_index(
            embedding_model,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            embedding_node_property=embedding_node_property,
            search_type=SearchType.HYBRID,
        )
        retriever = GraphRetrieverTool(vector_store=vector_store)

    # Perform retrieval
    results = retriever.forward(query=query, k=k)
    print(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j retriever CLI")
    parser.add_argument("-q", "--query", type=str, required=True, help="Search query")
    parser.add_argument(
        "-k", "--k", type=int, default=5, help="Number of results to retrieve"
    )
    parser.add_argument(
        "--pathrag", action="store_true", help="Use PathRAG retriever instead of hybrid"
    )
    args = parser.parse_args()

    main(query=args.query, k=args.k, use_pathrag=args.pathrag)
