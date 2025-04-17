from langchain_neo4j import Neo4jVector
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain.retrievers import EnsembleRetriever
import logging
import coloredlogs
import os
from smolagents import Tool

load_dotenv()
# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4.1")

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)


class GraphRetrieverTool(Tool):
    """A tool that retrieves documents from a Neo4j graph database using hybrid search."""

    name = "graph_retriever"
    description = "Uses hybrid search (vector + keyword) to retrieve mathematical content from a Neo4j graph database."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to search for in the mathematical knowledge graph. This can be a question or a statement about a mathematical concept.",
        },
        "k": {
            "type": "integer",
            "description": "The number of documents to retrieve. Default is 5.",
            "default": 5,
            "nullable": True,
        },
        "node_type": {
            "type": "string",
            "description": "The type of node to search for. Can be 'Definition', 'Theorem', 'Lemma', 'Proof', etc. Default is to search all types.",
            "default": "Definition",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        """Initialize the GraphRetrieverTool with custom configuration."""
        super().__init__(**kwargs)

        self.embedding_provider = embedding_provider
        self.neo4j_url = os.getenv("NEO4J_URI")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")

        # Set default node type and index names
        self.default_node_type = "Definition"

    def _get_retriever(self, node_type: str):
        """Get a Neo4j Vector retriever for the specified node type."""
        index_name = f"text_vector_index_{node_type.lower()}"
        keyword_index_name = f"fulltext_index_{node_type.lower()}"

        logger.info(
            f"Creating hybrid retriever with index_name={index_name}, keyword_index_name={keyword_index_name}"
        )

        return Neo4jVector.from_existing_index(
            self.embedding_provider,
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type="hybrid",
        )

    def forward(self, query: str, k: int = 5, node_type: str = None) -> str:
        """
        Retrieve documents from Neo4j that match the query.

        Args:
            query: The search query
            k: Number of results to return
            node_type: Type of node to search (Definition, Theorem, etc.)

        Returns:
            A formatted string with the retrieved documents
        """
        assert isinstance(query, str), "Your search query must be a string"

        # Use default node type if none provided
        if not node_type:
            node_type = self.default_node_type

        try:
            # Get the appropriate retriever
            retriever = self._get_retriever(node_type)

            # Perform the search
            docs = retriever.similarity_search(query, k=k)

            # Format the results
            result_str = f"\nRetrieved {len(docs)} {node_type} documents:\n"

            for i, doc in enumerate(docs):
                result_str += f"\n\n===== Document {i + 1} =====\n"
                result_str += f"CONTENT: {doc.page_content}\n"

                # Add metadata if available
                if hasattr(doc, "metadata") and doc.metadata:
                    result_str += "METADATA:\n"
                    for key, value in doc.metadata.items():
                        result_str += f"  - {key}: {value}\n"

                result_str += "-" * 40

            return result_str

        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            return f"Error retrieving documents: {str(e)}"


def get_hybrid_retriever(node_type: str = "Definition", k: int = 5):
    """
    Helper function to get a preconfigured hybrid retriever.

    Args:
        node_type: The node type to search (e.g., "Definition", "Theorem")
        k: Number of results to return

    Returns:
        A configured GraphRetrieverTool
    """
    return GraphRetrieverTool(node_type=node_type, k=k)


# Example usage (commented out)
"""
# Create a retriever tool
retriever = get_hybrid_retriever(node_type="Definition")

# Test the retriever
results = retriever.forward(query="What is a topological space?", k=3)
print(results)
"""


def main(query: str, search_type: str, k: int, node_type: str):
    logger.info(
        f"Retrieving {k} {node_type if search_type != 'ensemble' else 'Definition, Theorem, Lemma'} documents for query: {query} using {search_type} search."
    )

    if search_type == "vector":
        index_name = f"vector_index_{node_type.lower()}"
        store = Neo4jVector.from_existing_index(
            embedding_provider,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name=index_name,
            retrieval_query=f"RETURN node.text AS text, score, node {{.*}} AS metadata",  # noqa: F541
        )
        results = store.similarity_search_with_score(query, k=k, threshold=0.25)

    elif search_type == "hybrid":
        index_name = f"vector_index_{node_type.lower()}"
        keyword_index_name = f"fulltext_index_{node_type.lower()}"
        store = Neo4jVector.from_existing_index(
            embedding_provider,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type="hybrid",
            retrieval_query=f"RETURN node.text AS text, score, node {{.*}} AS metadata",  # noqa: F541
        )
        results = store.similarity_search_with_score(query, k=k, threshold=0.25)

    elif search_type == "ensemble":
        ensemble_node_types = [
            "Definition",
            "Theorem",
            "Lemma",
            "Corollary",
            "Remark",
            "Introduction",
        ]
        vector_retrievers = []
        logger.info(
            f"Creating ensemble retriever for node types: {ensemble_node_types}"
        )
        for nt in ensemble_node_types:
            index_name = f"vector_index_{nt.lower()}"
            try:
                retriever = Neo4jVector.from_existing_index(
                    embedding_provider,
                    url=os.getenv("NEO4J_URI"),
                    username=os.getenv("NEO4J_USERNAME"),
                    password=os.getenv("NEO4J_PASSWORD"),
                    index_name=index_name,
                    retrieval_query=f"RETURN node.text AS text, score, node {{.*}} AS metadata",  # noqa: F541
                ).as_retriever(search_kwargs={"k": k})
                vector_retrievers.append(retriever)
                logger.info(
                    f"Added retriever for node type: {nt} with index {index_name}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not create retriever for node type {nt} (index: {index_name}): {e}"
                )

        if not vector_retrievers:
            logger.error("No vector retrievers could be initialized for the ensemble.")

            results = []
        else:
            # Using default weights (equal weighting)
            ensemble_retriever = EnsembleRetriever(
                retrievers=vector_retrievers,
                weights=[1.0 / len(vector_retrievers)] * len(vector_retrievers),
            )
            # EnsembleRetriever uses invoke, not similarity_search
            results = ensemble_retriever.invoke(query, k=k)
            # Note: EnsembleRetriever might not respect 'k' directly in the same way.
            # The number of results depends on how it combines outputs.
            # We might need to slice the results list if a specific 'k' is strictly required.
            if len(results) > k:
                results = results[:k]

    else:
        raise ValueError(f"Invalid search type: {search_type}")

    if not results:
        logger.warning("No results found for query: %s", query)

    for i, (result, score) in enumerate(results, start=1):
        print(f"Result {i}:")
        print(f"{result.page_content}")
        if score is not None:
            print(f"Score: {score:.4f}")
        if hasattr(result, "metadata") and result.metadata:
            print("Metadata:")
            for key, value in result.metadata.items():
                if "embedding" not in key:
                    print(f"  - {key}: {value}")
        print("=" * 140)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j vector/hybrid retriever CLI")
    parser.add_argument("-q", "--query", type=str, required=True, help="Search query")
    parser.add_argument(
        "-s",
        "--search-type",
        type=str,
        choices=["vector", "hybrid", "ensemble"],
        default="vector",
        help="Search type: vector, hybrid, or ensemble",
    )
    parser.add_argument(
        "-k", "--k", type=int, default=5, help="Number of results to retrieve"
    )
    parser.add_argument(
        "-n",
        "--node-type",
        type=str,
        default="Definition",
        help="Node type (Definition, Theorem, etc.) - Ignored for ensemble search.",
    )
    args = parser.parse_args()

    main(
        query=args.query,
        search_type=args.search_type,
        k=args.k,
        node_type=args.node_type,
    )
