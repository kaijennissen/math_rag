import logging
import os

import coloredlogs
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import SearchType
from langchain_openai import OpenAIEmbeddings
from smolagents import Tool

load_dotenv()
# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Model configuration - same as in add_e5_embeddings.py
DEFAULT_MODEL = "E5 Multilingual"
MODEL_CONFIGS = {
    "OpenAI": {
        "type": "openai",
        "model_name": "text-embedding-3-small",
        "dimensions": 1536,
    },
    "E5 Multilingual": {
        "type": "huggingface",
        "model_name": "intfloat/multilingual-e5-large",
        "dimensions": 1024,
    },
    "MXBAI German": {
        "type": "huggingface",
        "model_name": "mixedbread-ai/deepset-mxbai-embed-de-large-v1",
        "dimensions": 1024,
    },
}


def get_embedding_provider(model_name: str = DEFAULT_MODEL):
    """
    Get the embedding model instance based on the model name.

    Args:
        model_name: Name of the embedding model to use

    Returns:
        An initialized embedding model instance
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: "
            f"{', '.join(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_name]

    if config["type"] == "huggingface":
        logger.info(
            f"Initializing HuggingFace embeddings model: {config['model_name']}"
        )
        return HuggingFaceEmbeddings(model_name=config["model_name"])
    elif config["type"] == "openai":
        logger.info(f"Initializing OpenAI embeddings model: {config['model_name']}")
        return OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"), model=config["model_name"]
        )
    else:
        raise ValueError(f"Unsupported model type: {config['type']}")


class GraphRetrieverTool(Tool):
    """A tool that retrieves documents from a Neo4j graph database
    using hybrid search."""

    name = "graph_retriever"
    description = """Uses hybrid search (vector + keyword) to
    retrieve mathematical content from a Neo4j graph database."""
    inputs = {
        "query": {
            "type": "string",
            "description": """The query to search for in the mathematical
            knowledge graph. This can be a question or a statement about
            a mathematical concept. This should be semantically close to
            your target documents. Use the affirmative form rather than
            a question. Use german as language.""",
        },
        "model": {
            "type": "string",
            "description": f"The embedding model to use. Available options: "
            f"{', '.join(MODEL_CONFIGS.keys())}. "
            f"Default is {DEFAULT_MODEL}.",
            "default": DEFAULT_MODEL,
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, embedding_model: str = DEFAULT_MODEL, k: int = 20, **kwargs):
        """Initialize the GraphRetrieverTool with custom configuration."""
        super().__init__(**kwargs)

        self.embedding_model = embedding_model
        self.neo4j_url = os.getenv("NEO4J_URI")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.k = k

    def _get_retriever(self, model_name: str = None):
        """
        Get a Neo4j Vector retriever with the specified embedding model.

        Args:
            model_name: Name of the embedding model to use

        Returns:
            Neo4jVector retriever instance
        """
        # Use the instance model if none is specified
        if model_name is None:
            model_name = self.embedding_model

        # Get the embedding provider
        embedding_provider = get_embedding_provider(model_name)

        # Determine the correct index name based on model
        index_name = "vector_index_summaryEmbedding"
        embedding_property = "summaryEmbedding"

        keyword_index_name = "fulltext_index_AtomicItem"

        logger.info(
            f"Creating hybrid retriever with model={model_name}, "
            f"index_name={index_name}, keyword_index_name={keyword_index_name}"
        )

        return Neo4jVector.from_existing_index(
            embedding_provider,
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name=index_name,  # name of the vector index
            keyword_index_name=keyword_index_name,  # name of the fulltext index
            embedding_node_property=embedding_property,
            search_type=SearchType.VECTOR,
        )

    def forward(self, query: str, model: str = None) -> str:
        """
        Retrieve documents from Neo4j that match the query using the specified model.

        Args:
            query: The search query
            model: The embedding model to use (defaults to the instance model)

        Returns:
            A formatted string with the retrieved documents
        """
        assert isinstance(query, str), "Your search query must be a string"

        # Use default model if none provided
        if model is None:
            model = self.embedding_model

        try:
            # Get the appropriate retriever with the specified model
            retriever = self._get_retriever(model)

            # Perform the search
            docs = retriever.similarity_search(query, k=self.k)

            # Format the results
            result_str = (
                f"\nRetrieved {len(docs)} documents using {model} embeddings:\n"
            )

            for i, doc in enumerate(docs):
                result_str += f"\n\n===== Document {i + 1} =====\n"
                result_str += f"CONTENT: {doc.page_content}\n"

                # Add metadata if available
                if hasattr(doc, "metadata") and doc.metadata:
                    result_str += "METADATA:\n"
                    for key, value in doc.metadata.items():
                        if (
                            "embedding" not in key.lower()
                        ):  # Skip embedding vectors in output
                            result_str += f"  - {key}: {value}\n"

                result_str += "-" * 40

            return result_str

        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            return f"Error retrieving documents: {str(e)}"


def get_hybrid_retriever(model: str = DEFAULT_MODEL, k: int = 5):
    """
    Helper function to get a preconfigured hybrid retriever.

    Args:
        model: The embedding model to use
        k: Number of results to return

    Returns:
        A configured GraphRetrieverTool
    """
    return GraphRetrieverTool(embedding_model=model, k=k)


def main(query: str, search_type: str, model: str = DEFAULT_MODEL, k: int = 5):
    """
    Main function to test the retriever.

    Args:
        query: The search query
        search_type: Type of search (vector or hybrid)
        model: Embedding model to use
        k: Number of results to return
    """
    logger.info(
        f"Retrieving {k} documents for query: '{query}' "
        f"using {search_type} search with {model} embeddings."
    )

    embedding_provider = get_embedding_provider(model)

    # Determine the correct index and property names based on model
    index_name = "vector_index_AtomicItem"
    embedding_property = "textEmbedding"

    if search_type == "vector":
        store = Neo4jVector.from_existing_index(
            embedding_provider,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name=index_name,
            embedding_node_property=embedding_property,
        )
        results = store.similarity_search_with_score(query, k=k, threshold=0.25)

    elif search_type == "hybrid":
        keyword_index_name = "fulltext_index_AtomicItem"
        store = Neo4jVector.from_existing_index(
            embedding_provider,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name=index_name,
            embedding_node_property=embedding_property,
            keyword_index_name=keyword_index_name,
            search_type=SearchType.HYBRID,
        )
        results = store.similarity_search_with_score(query, k=k, threshold=0.25)
    else:
        logger.error("Invalid search type. Use 'vector' or 'hybrid'.")
        return

    if not results:
        logger.warning("No results found for query: %s", query)
        return

    for i, (result, score) in enumerate(results, start=1):
        print(f"Result {i}:")
        print(f"{result.page_content}")
        if score is not None:
            print(f"Score: {score:.4f}")
        if hasattr(result, "metadata") and result.metadata:
            print("Metadata:")
            for key, value in result.metadata.items():
                if "Embedding" not in key:
                    print(f"  - {key}: {value}")
        print("=" * 140)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Neo4j vector/hybrid retriever CLI with multiple embedding models"
    )
    parser.add_argument("-q", "--query", type=str, required=True, help="Search query")
    parser.add_argument(
        "-s",
        "--search-type",
        type=str,
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Search type: vector or hybrid",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default=DEFAULT_MODEL,
        help=f"Embedding model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-k", "--k", type=int, default=5, help="Number of results to retrieve"
    )
    args = parser.parse_args()

    main(
        query=args.query,
        search_type=args.search_type,
        model=args.model,
        k=args.k,
    )
