#!/usr/bin/env python
"""
Script to create embeddings and vector indexes in Neo4j using different embedding models.

This script uses the Neo4jVector.from_existing_graph method which efficiently:
1. Generates embeddings for all nodes
2. Stores them in the specified node property
3. Creates a vector index automatically

Supported embedding models:
- E5 Multilingual (default) - best for academic German content
- MXBAI German - alternative for German content
- OpenAI - original embedding model

"""

import os
import logging
import time
import argparse
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("add_embeddings.log"), logging.StreamHandler()],
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


def verify_nodes_exist():
    """Verify that AtomicUnit nodes exist in the database."""
    logger.info("Verifying AtomicUnit nodes...")
    try:
        with driver.session() as session:
            result = session.run("""
            MATCH (n:AtomicUnit)
            RETURN count(n) AS count
            """)
            count = result.single()["count"]
            logger.info(f"Found {count} AtomicUnit nodes in the database")
            return count > 0
    except Exception as e:
        logger.error(f"Error verifying nodes: {e}")
        return False


def ensure_atomic_unit_label():
    """Ensure that all content nodes have the AtomicUnit label."""
    logger.info("Ensuring all content nodes have the AtomicUnit label...")
    try:
        with driver.session() as session:
            result = session.run("""
            MATCH (n:Introduction|Definition|Corollary|Theorem|Lemma|Proof|Example|Exercise|Remark)
            WHERE NOT n:AtomicUnit
            WITH count(n) AS missingLabel
            MATCH (n:Introduction|Definition|Corollary|Theorem|Lemma|Proof|Example|Exercise|Remark)
            WHERE NOT n:AtomicUnit
            SET n:AtomicUnit
            RETURN missingLabel, count(n) AS updated
            """)
            record = result.single()
            if record and record["missingLabel"] > 0:
                logger.info(f"Added AtomicUnit label to {record['updated']} nodes")
            else:
                logger.info("All content nodes already have the AtomicUnit label")
            return True
    except Exception as e:
        logger.error(f"Error ensuring AtomicUnit label: {e}")
        return False


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
            f"Unknown model: {model_name}. Available models: {', '.join(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_name]

    if config["type"] == "huggingface":
        logger.info(
            f"Initializing HuggingFace embeddings model: {config['model_name']}"
        )
        return HuggingFaceEmbeddings(model_name=config["model_name"])
    else:
        raise ValueError(f"Unsupported model type: {config['type']}")


def add_embeddings_with_neo4j_vector(model_name):
    """
    Create embeddings and vector index for AtomicUnit nodes using Neo4jVector.from_existing_graph.
    This method handles both embedding creation and index creation in one step.

    Args:
        model_name: Name of the embedding model to use
    """
    logger.info(f"Creating embeddings and vector index using {model_name}...")

    try:
        # Initialize the embedding model
        embedding_model = get_embedding_model(model_name)

        # Count nodes without embeddings
        embedding_property = (
            "textEmbedding2"
            if model_name in ["E5 Multilingual", "MXBAI German"]
            else "textEmbedding"
        )

        with driver.session() as session:
            count_result = session.run(f"""
            MATCH (n:AtomicUnit)
            WHERE n.text IS NOT NULL AND n.{embedding_property} IS NULL
            RETURN count(n) as count
            """)
            node_count = count_result.single()["count"]

        logger.info(f"Found {node_count} nodes without embeddings")

        if node_count == 0:
            logger.info("No nodes need embeddings. Skipping.")
            return True

        # Use Neo4jVector.from_existing_graph to add embeddings
        start_time = time.time()

        # Get configuration for the selected model
        embedding_property = "textEmbedding"
        index_name = "vector_index_AtomicUnit"

        logger.info(
            f"Creating Neo4jVector from existing graph with index {index_name}..."
        )
        vector_store = Neo4jVector.from_existing_graph(
            embedding=embedding_model,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name=index_name,
            node_label="AtomicUnit",
            text_node_properties=["text", "title"],
            embedding_node_property=embedding_property,
        )

        total_duration = time.time() - start_time
        logger.info(f"Successfully added embeddings to nodes in {total_duration:.2f}s")

        # Verify all nodes have embeddings now
        with driver.session() as session:
            verification_result = session.run(f"""
            MATCH (n:AtomicUnit)
            WHERE n.text IS NOT NULL AND n.{embedding_property} IS NULL
            RETURN count(n) as remaining
            """)
            remaining = verification_result.single()["remaining"]

        if remaining > 0:
            logger.warning(f"There are still {remaining} nodes without embeddings")
        else:
            logger.info("All nodes have been successfully embedded")

        return vector_store is not None

    except Exception as e:
        logger.error(f"Failed to add embeddings: {e}")
        return False


def test_vector_search(query="Topologie", model_name=DEFAULT_MODEL):
    """
    Test vector search with a sample query using the specified embedding model.

    Args:
        query: The query string
        model_name: Name of the embedding model to use
    """
    # Add extra debug logging for the query
    logger.info(f"Raw query input: '{query}'")
    logger.info(f"Query type: {type(query)}")
    logger.info(f"Query length: {len(query)}")

    try:
        # Initialize embedding model
        embedding_model = get_embedding_model(model_name)

        # Get configuration for the selected model
        index_name = "vector_index_AtomicUnit"

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

            logger.info(f"Found {len(results)} results" + "=" * 80)
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

            return len(results) > 0
    except Exception as e:
        logger.error(f"Error testing vector search: {e}")
        return False


def main(model_name: str, test: bool, query: str) -> None:
    """Main function to create embeddings and vector index."""
    # Verify Neo4j connection and nodes
    if not verify_nodes_exist():
        logger.error(
            "No AtomicUnit nodes found. Make sure your graph is properly populated."
        )
        return

    # Ensure all content nodes have the AtomicUnit label
    ensure_atomic_unit_label()

    # Add embeddings to nodes
    add_embeddings_with_neo4j_vector(model_name)

    logger.info(
        f"Embeddings and vector index created successfully using {model_name}. The system is now ready for retrieval."
    )

    # Test vector search if requested
    if test:
        logger.info(f"Query for testing: '{query}'")
        logger.info(f"Sending to test_vector_search: '{query}'")
        test_vector_search(query=query, model_name=model_name)

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
        help="Path to a file containing the query text (recommended for LaTeX expressions)",
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

    main(test=args.test, query=query, model_name=args.model)
