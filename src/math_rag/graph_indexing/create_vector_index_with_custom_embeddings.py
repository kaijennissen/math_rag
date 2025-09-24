"""
Script to create embeddings and vector indexes in Neo4j using different embedding
models.

This script uses the Neo4jVector.from_existing_graph method which efficiently:
1. Generates embeddings for all nodes
2. Stores them in the specified node property
3. Creates a vector index automatically

Supported embedding models:
- E5 Multilingual (default) - best for academic German content
- MXBAI German - alternative for German content
- OpenAI - original embedding model

Enhanced Usage Examples:
- Default usage: python add_embeddings.py
- Different model: python add_embeddings.py -m "OpenAI"
- Custom node label: python add_embeddings.py --label MyNode
- Custom text properties: python add_embeddings.py --text-properties text title \
  description
- Custom embedding property: python add_embeddings.py --embedding-property \
  myCustomEmbedding
- Single property embedding: python add_embeddings.py --text-properties title \
  --embedding-property titleEmbedding
- Test with custom query: python add_embeddings.py -t -q "topology definition"
"""

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase

from math_rag.graph_indexing.utils import (
    count_nodes_without_property,
    ensure_atomic_unit_label,
    verify_nodes,
    verify_property_populated,
)

# Load environment variables
load_dotenv()

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding model configuration
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

# Direct Neo4j driver for verification
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def get_embedding_model(model_name):
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
    else:
        raise ValueError(f"Unsupported model type: {config['type']}")


def add_embeddings_with_neo4j_vector(
    model_name: str,
    label: str = "AtomicUnit",
    text_properties: list = None,
    embedding_property: str = None,
    index_name: str = None,
):
    """
    Create embeddings and vector index for nodes using Neo4jVector.from_existing_graph.
    This method handles both embedding creation and index creation in one step.

    Args:
        model_name: Name of the embedding model to use
        label: Node label to process (default: AtomicUnit)
        text_properties: List of properties to use for embedding calculation
                        (default: ["text", "title"])
        embedding_property: Property name to store embeddings
                           (default: auto-generated from text_properties)
        index_name: Vector index name (default: auto-generated)
    """

    if embedding_property is None:
        # Auto-generate embedding property name based on source properties
        # Add "Embedding" to the list of property names and join with underscore
        prop_list = text_properties.copy() + ["Embedding"]
        embedding_property = "_".join(prop_list)

    if index_name is None:
        index_name = f"vector_index_{embedding_property}"

    logger.info(f"Creating embeddings and vector index using {model_name}...")
    logger.info(
        f"Target: {label} nodes, properties: {text_properties} -> {embedding_property}"
    )

    # Initialize the embedding model
    embedding_model = get_embedding_model(model_name)

    # Count nodes that need embeddings
    node_count = count_nodes_without_property(
        driver, label, text_properties, embedding_property
    )
    logger.info(f"Found {node_count} {label} nodes without embeddings")

    if node_count == 0:
        logger.info("No nodes need embeddings. Skipping.")
        return

    # Use Neo4jVector.from_existing_graph to add embeddings
    start_time = time.time()

    logger.info(f"Creating Neo4jVector from existing graph with index {index_name}...")
    Neo4jVector.from_existing_graph(
        embedding=embedding_model,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=index_name,
        node_label=label,
        text_node_properties=text_properties,
        embedding_node_property=embedding_property,
    )

    total_duration = time.time() - start_time
    logger.info(
        f"Successfully added embeddings to {label} nodes in {total_duration:.2f}s"
    )

    # Verify all nodes have embeddings now
    remaining = verify_property_populated(
        driver, label, text_properties, embedding_property
    )

    if remaining > 0:
        logger.warning(f"There are still {remaining} {label} nodes without embeddings")
    elif remaining == 0:
        logger.info(f"All {label} nodes have been successfully embedded")


def test_vector_search(
    query="Topologie", model_name=DEFAULT_MODEL, label="AtomicUnit", index_name=None
):
    """
    Test vector search with a sample query using the specified embedding model.

    Args:
        query: The query string
        model_name: Name of the embedding model to use
        label: Node label to search (default: AtomicUnit)
        index_name: Vector index name (default: auto-generated)
    """
    if index_name is None:
        index_name = f"vector_index_{label}"

    # Add extra debug logging for the query
    logger.info(f"Testing vector search on {label} nodes with index {index_name}")
    logger.info(f"Raw query input: '{query}'")
    logger.info(f"Query type: {type(query)}")
    logger.info(f"Query length: {len(query)}")

    # Initialize embedding model
    embedding_model = get_embedding_model(model_name)

    # Generate embedding for the query
    query_embedding = embedding_model.embed_query(query)

    # Perform vector search
    with driver.session() as session:
        search_result = session.run(
            f"""
            CALL db.index.vector.queryNodes(
              '{index_name}',
              5,
              $embedding
            )
            YIELD node, score
            RETURN
              score,
              node.text AS text,
              node.type AS type,
              node.title AS title
            ORDER BY score DESC
            """,
            {"embedding": query_embedding},
        )

        results = list(search_result)

        logger.info(f"Found {len(results)} results for {label} nodes" + "=" * 80)
        for i, record in enumerate(results):
            logger.info(f"Result {i + 1} (Score: {record['score']:.4f}):")
            if (node_type := record.get("type")) is not None:
                logger.info(f"Type: {node_type}")
            if (title := record.get("title")) is not None:
                logger.info(f"Title: {title}")
            # Show a preview of the text
            text_preview = record["text"]
            if len(text_preview) > 200:
                text_preview = text_preview[:200] + "..."
            logger.info(f"Text: {text_preview}")
            logger.info("-" * 40)


def main(
    model_name: str,
    test: bool,
    query: str,
    text_properties: list,
    label: str = "AtomicUnit",
    embedding_property: str = None,
) -> None:
    """Main function to create embeddings and vector index."""

    # Ensure AtomicUnit label
    logger.info("Ensuring AtomicUnit label...")
    ensure_atomic_unit_label(driver)

    # Verify Neo4j connection and nodes
    exists, count = verify_nodes(driver, label)
    if not exists:
        logger.error(
            f"No {label} nodes found. Make sure your graph is properly populated."
        )
        return

    # Add embeddings to nodes
    add_embeddings_with_neo4j_vector(
        model_name=model_name,
        label=label,
        text_properties=text_properties,
        embedding_property=embedding_property,
    )

    logger.info(
        f"Embeddings and vector index created successfully for {label} nodes "
        f"using {model_name}. Properties: {text_properties} -> "
        f"{embedding_property or 'auto-generated'}. "
        f"The system is now ready for retrieval."
    )

    # Test vector search if requested
    if test:
        logger.info(f"Query for testing: '{query}'")
        logger.info(f"Sending to test_vector_search: '{query}'")
        test_vector_search(query=query, model_name=model_name, label=label)

    logger.info("Process completed.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create embeddings and vector index for Neo4j nodes"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODEL_CONFIGS.keys()),
        help=f"Embedding model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Test vector search after creating embeddings and vector index",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default="Was ist ein $T_{4}$-Raum?",
        help="Query to use for testing (simple queries only)",
    )
    parser.add_argument(
        "-f",
        "--query-file",
        type=str,
        help="Path to a file containing the query text (recommended for LaTeX "
        "expressions)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="AtomicUnit",
        help="Node label to process (default: AtomicUnit)",
    )
    parser.add_argument(
        "--text-properties",
        nargs="+",
        default=["text", "title"],
        help="Properties to use for embedding calculation (default: text title)",
    )
    parser.add_argument(
        "--embedding-property",
        type=str,
        help="Property name to store embeddings (default: auto-generated from text "
        "properties)",
    )

    args = parser.parse_args()

    # Handle query from file if provided
    query = args.query
    if args.query_file:
        try:
            with open(args.query_file, "r") as f:
                query = f.read().strip()
            logger.info(f"Read query from file: {args.query_file}")
        except Exception as e:
            logger.error(f"Error reading query file: {e}")
            sys.exit(1)

    main(
        test=args.test,
        query=query,
        model_name=args.model,
        label=args.label,
        text_properties=args.text_properties,
        embedding_property=args.embedding_property,
    )
