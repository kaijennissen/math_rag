"""
Script to create fulltext indexes in Neo4j for keyword search in the knowledge graph.

This script handles:
1. Creating the AtomicUnit label on all content nodes
2. Creating fulltext indexes for keyword search
3. Testing the fulltext index with sample queries
"""

import argparse
import logging
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

from math_rag.graph_indexing.utils import (
    drop_index_if_exists,
    ensure_atomic_unit_label,
)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_fulltext_index(
    driver: GraphDatabase.driver,
    label: str = "AtomicUnit",
    properties: list = None,
    index_name: str = None,
):
    """Create a fulltext index for specified nodes and properties.

    Args:
        driver: Neo4j driver instance
        label: Node label to index (default: AtomicUnit)
        properties: List of properties to index (default: ["text", "title", "proof"])
        index_name: Custom index name (default: fulltext_index_{label})
    """
    if properties is None:
        properties = ["text", "title", "proof"]

    if index_name is None:
        index_name = f"fulltext_index_{label}"

    property_list = ", ".join([f"n.{prop}" for prop in properties])

    logger.info(
        f"Creating fulltext index {index_name} on {label} properties: {properties}"
    )

    # Drop existing index
    drop_index_if_exists(driver, index_name)

    # Create new fulltext index
    with driver.session() as session:
        session.run(
            f"""
            CREATE FULLTEXT INDEX {index_name}
            FOR (n:{label}) ON EACH [{property_list}]
            """
        )
    logger.info(f"Successfully created fulltext index {index_name}")


def test_fulltext_search(driver: GraphDatabase.driver, query: str = "Umgebungsbasis"):
    """Test fulltext search with a sample query."""
    logger.info(f"Testing fulltext search with query: '{query}'")

    with driver.session() as session:
        search_result = session.run(
            """
            CALL db.index.fulltext.queryNodes(
              "fulltext_index_AtomicUnit",
              $query,
              {limit: 5}
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
            {"query": query},
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
    label: str = "AtomicUnit",
    properties: list = [],
):
    """Main function to create and test fulltext index."""

    # Load environment variables
    load_dotenv()

    # Get Neo4j connection details
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    # Create driver
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

    try:
        # Ensure AtomicUnit label
        logger.info("Ensuring AtomicUnit label...")
        ensure_atomic_unit_label(driver)

        # Set default properties if not provided
        if not properties:
            properties = ["text", "title", "proof"]

        # Create fulltext index
        logger.info(
            f"Creating fulltext index for {label} on properties {properties}..."
        )
        create_fulltext_index(driver, label=label, properties=properties)
        logger.info("Fulltext index created successfully.")

        # Test if requested
        if test:
            test_fulltext_search(driver, query if query else "topologisch")

        logger.info("Fulltext index operations completed.")
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and manage Neo4j fulltext indexes"
    )
    parser.add_argument("--test", action="store_true", help="Test index after creation")
    parser.add_argument(
        "--query",
        type=str,
        default="topologisch",
        help="Query for testing fulltext search",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="AtomicUnit",
        help="Node label to create index for (default: AtomicUnit)",
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=["text", "title", "proof", "summary"],
        help="Properties for fulltext index (default: text title proof summary)",
    )

    args = parser.parse_args()

    main(
        test=args.test,
        query=args.query,
        label=args.label,
        properties=args.properties,
    )
