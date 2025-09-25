"""
Script to create vector indexes in Neo4j using OpenAI embeddings for similarity search.

This script handles:
1. Creating the AtomicItem label on all content nodes
2. Creating vector indexes for embedding-based similarity search using OpenAI embeddings
3. Testing the vector index with sample queries

Note: This script specifically uses OpenAI embeddings (text-embedding-3-small).
For custom embedding models, use create_vector_index_with_custom_embeddings.py instead.
"""

import argparse
import logging
import os

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

from math_rag.graph_indexing.utils import (
    drop_index_if_exists,
    ensure_atomic_unit_label,
    verify_nodes,
)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_vector_index(
    driver: GraphDatabase.driver,
    label: str = "AtomicItem",
    property_name: str = "textEmbedding",
    dimensions: int = 1536,
    similarity_function: str = "cosine",
    index_name: str = None,
):
    """Create or recreate a vector index for specified nodes and property.

    Args:
        driver: Neo4j driver instance
        label: Node label to index (default: AtomicItem)
        property_name: Property containing vector embeddings (default: textEmbedding)
        dimensions: Vector dimensions (default: 1536 for OpenAI text-embedding-3-small)
        similarity_function: Similarity function (default: cosine)
        index_name: Custom index name (default: vector_index_{label})
    """
    if index_name is None:
        index_name = f"vector_index_{label}"

    # Verify nodes have the required property
    exists, count = verify_nodes(driver, label, property_name)
    if not exists:
        logger.error(f"No {label} nodes with {property_name} property found.")
        return

    # Drop existing index
    drop_index_if_exists(driver, index_name)

    # Create new vector index
    logger.info(f"Creating vector index {index_name} on {label}.{property_name}...")
    with driver.session() as session:
        session.run(
            f"""
            CREATE VECTOR INDEX {index_name}
            FOR (n:{label}) ON n.{property_name}
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: '{similarity_function}'
                }}
            }}
            """
        )
    logger.info(f"Successfully created vector index {index_name}")


def test_vector_search(driver: Driver, query: str = "Topologie"):
    """Test vector search with a sample query using OpenAI embeddings."""
    logger.info(f"Testing vector search with query: '{query}'")

    # Import OpenAI embeddings here to avoid circular imports
    from langchain_openai import OpenAIEmbeddings

    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(
        openai_api_key=openai_api_key, model="text-embedding-3-small"
    )

    # Generate embedding for the query using LangChain
    query_embedding = embedding_model.embed_query(query)

    # Perform vector search
    with driver.session() as session:
        search_result = session.run(
            """
            CALL db.index.vector.queryNodes(
              'vector_index_AtomicItem',
              5,
              $embedding
            )
            YIELD node, score
            RETURN
              score,
              node.text AS text,
              node.type AS type,
              node.title AS title,
              node.identifier AS identifier
            ORDER BY score DESC
            """,
            {"embedding": query_embedding},
        )

        results = list(search_result)

        logger.info(f"Found {len(results)} results")

        for i, record in enumerate(results):
            logger.info(f"Result {i + 1} (Score: {record['score']:.4f}):")
            if "identifier" in record and record["identifier"]:
                logger.info(f"Identifier: {record['identifier']}")
            if "type" in record and record["type"]:
                logger.info(f"Type: {record['type']}")
            if "title" in record and record["title"]:
                logger.info(f"Title: {record['title']}")

            # Show a preview of the text
            text_preview = record["text"]
            if len(text_preview) > 200:
                text_preview = text_preview[:200] + "..."
            logger.info(f"Text: {text_preview}")
            logger.info("-" * 40)


def main(
    test: bool = False,
    query: str = "",
    label: str = "AtomicItem",
    property_name: str = "textEmbedding",
):
    """Main function to create and test vector index."""

    # Load environment variables
    load_dotenv()

    # Get Neo4j connection details
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    # Create driver
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

    try:
        # Ensure AtomicItem label
        logger.info("Ensuring AtomicItem label...")
        ensure_atomic_unit_label(driver)

        # Create vector index
        logger.info(f"Creating vector index for {label}.{property_name}...")
        create_vector_index(driver, label=label, property_name=property_name)
        logger.info("Vector index created successfully.")

        # Test if requested
        if test:
            test_vector_search(driver, query if query else "Topologie")

        logger.info("Vector index operations completed.")
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and manage Neo4j vector indexes with OpenAI embeddings"
    )
    parser.add_argument("--test", action="store_true", help="Test index after creation")
    parser.add_argument(
        "--query",
        type=str,
        default="Was ist die definition eines T_{4}-Raums?",
        help="Query for testing vector search",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="AtomicItem",
        help="Node label to create index for (default: AtomicItem)",
    )
    parser.add_argument(
        "--property",
        type=str,
        default="textEmbedding",
        help="Property name for vector index (default: textEmbedding)",
    )

    args = parser.parse_args()

    main(
        test=args.test,
        query=args.query,
        label=args.label,
        property_name=args.property,
    )
