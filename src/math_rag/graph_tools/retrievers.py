import logging
from typing import List

import coloredlogs
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import SearchType
from smolagents import Tool

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def format_document(doc: Document, index: int) -> str:
    """
    Module-level helper to format a single Document into a string.

    Keeping this as a plain function (not a method) makes it much easier to
    unit-test in isolation.

    Args:
        doc: Document instance (must have `page_content` and optional `metadata`)
        index: 1-based index of the document for display

    Returns:
        Formatted string for the single document
    """
    allowed_metadata = {"number", "type", "title"}

    # Separate lists for content and metadata; we'll concatenate at the end.
    content_parts: List[str] = []
    metadata_parts: List[str] = []

    content_parts.append(f"\n\n===== Document {index} =====\n")
    content_parts.append(f"CONTENT:\n {doc.page_content}\n")

    # metadata may be missing or empty; only keep a small whitelist of keys
    metadata = getattr(doc, "metadata", None)
    if metadata:
        # Single-pass over metadata: collect allowed keys in the order they appear
        for key, value in metadata.items():
            if key.lower() in allowed_metadata:
                metadata_parts.append(f"  - {key}: {value}\n")

    if metadata_parts:
        metadata_parts.insert(0, "METADATA:\n")

    parts = content_parts + metadata_parts
    parts.append("-" * 40)
    return "".join(parts)


class GraphRetrieverTool(Tool):
    """A tool that retrieves documents from a Neo4j graph database using hybrid search (vector + keyword)."""  # noqa: E501

    name = "graph_retriever"
    description = """Uses hybrid search (vector + keyword) to retrieve mathematical content from a Neo4j graph database."""  # noqa: E501
    inputs = {
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
            "description": "Number of documents to retrieve. Default is 20.",
            "default": 20,
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        embedding_model: Embeddings,
        uri: str,
        username: str,
        password: str,
        vector_index_name: str,
        keyword_index_name: str,
        embedding_node_property: str,
        **kwargs,
    ):
        """
        Initialize the GraphRetrieverTool with Neo4j configuration.

        Args:
            embedding_model: An initialized embedding model instance
                (e.g., HuggingFaceEmbeddings)
            uri: Neo4j database URI
            username: Neo4j username
            password: Neo4j password
            vector_index_name: Name of the vector index in Neo4j
            keyword_index_name: Name of the keyword/fulltext index in Neo4j
            embedding_node_property: Property name containing embeddings in Neo4j nodes
            **kwargs: Additional arguments passed to Tool base class
        """
        super().__init__(**kwargs)

        self.embedding_model = embedding_model
        self.uri = uri
        self.username = username
        self.password = password
        self.vector_index_name = vector_index_name
        self.keyword_index_name = keyword_index_name
        self.embedding_node_property = embedding_node_property

        # Initialize the Neo4j vector store once
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the Neo4j vector store with hybrid search capabilities."""
        logger.info(
            f"Initializing Neo4j vector store with "
            f"vector_index={self.vector_index_name}, "
            f"keyword_index={self.keyword_index_name}"
        )

        self.vector_store = Neo4jVector.from_existing_index(
            self.embedding_model,
            url=self.uri,
            username=self.username,
            password=self.password,
            index_name=self.vector_index_name,  # vector index
            keyword_index_name=self.keyword_index_name,  # fulltext index
            embedding_node_property=self.embedding_node_property,
            search_type=SearchType.HYBRID,
        )

    def _retrieve(self, query: str, k: int) -> List[Document]:
        """
        Retrieve documents from Neo4j using hybrid search.

        Args:
            query: The search query
            k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving {k} documents for query: '{query}'")

        # Perform hybrid search with similarity scores
        results = self.vector_store.similarity_search_with_score(
            query, k=k, threshold=0.25
        )

        # Extract just the documents from the (doc, score) tuples
        docs = [doc for doc, score in results]

        logger.info(f"Successfully retrieved {len(docs)} documents")
        return docs

    def _format(self, docs: List[Document], query: str) -> str:
        """
        Format retrieved documents into a readable string.

        Args:
            docs: List of retrieved documents (assumed non-empty)
            query: The original search query (for context in output)

        Returns:
            Formatted string representation of the documents
        """
        result_parts: List[str] = [
            f"\nRetrieved {len(docs)} documents using hybrid search:\n"
        ]

        for i, doc in enumerate(docs, 1):
            result_parts.append(format_document(doc, i))

        return "".join(result_parts)

    def forward(self, query: str, k: int = 20) -> str:
        """
        Main entry point for retrieving documents from Neo4j.

        Args:
            query: The search query
            k: Number of documents to retrieve (default: 20)

        Returns:
            A formatted string with the retrieved documents
        """
        assert isinstance(query, str), "Your search query must be a string"
        assert isinstance(k, int) and k > 0, "k must be a positive integer"

        try:
            # Retrieve documents
            docs = self._retrieve(query, k)

            # Handle empty results here so _format can assume non-empty input
            if not docs:
                return f"No documents found for query: '{query}'"

            # Format and return results
            return self._format(docs, query)

        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            return f"Error retrieving documents: {str(e)}"


def main(query: str, k: int = 5):
    """
    Example usage of the GraphRetrieverTool.

    Args:
        query: The search query
        k: Number of results to return
    """
    import os

    from dotenv import load_dotenv
    from langchain_huggingface import HuggingFaceEmbeddings

    load_dotenv()

    # Option 1: HuggingFace
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # Option 2: OpenAI (uncomment to use)
    # embedding_model = OpenAIEmbeddings(
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    #     model="text-embedding-3-small"
    # )

    # Create the retriever tool
    retriever = GraphRetrieverTool(
        embedding_model=embedding_model,
        uri=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        vector_index_name="vector_index_summary_Embedding",
        keyword_index_name="fulltext_index_AtomicItem",
        embedding_node_property="summaryEmbedding",
    )

    # Perform retrieval
    results = retriever.forward(query=query, k=k)
    print(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j hybrid retriever CLI")
    parser.add_argument("-q", "--query", type=str, required=True, help="Search query")
    parser.add_argument(
        "-k", "--k", type=int, default=5, help="Number of results to retrieve"
    )
    args = parser.parse_args()

    main(query=args.query, k=args.k)
