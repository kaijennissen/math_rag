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

# Direct Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def ensure_atomic_unit_label():
    """Ensure that all content nodes have the AtomicUnit label."""
    logger.info("Ensuring all content nodes have the AtomicUnit label...")
    try:
        with driver.session() as session:
            result = session.run(
                """
            MATCH (n:Introduction|Definition|Corollary|Theorem|Lemma|Proof|Example|
                  Exercise|Remark)
            WHERE NOT n:AtomicUnit
            WITH count(n) AS missingLabel
            MATCH (n:Introduction|Definition|Corollary|Theorem|Lemma|Proof|Example|
                  Exercise|Remark)
            WHERE NOT n:AtomicUnit
            SET n:AtomicUnit
            RETURN missingLabel, count(n) AS updated
            """
            )
            record = result.single()
            if record and record["missingLabel"] > 0:
                logger.info(f"Added AtomicUnit label to {record['updated']} nodes")
            else:
                logger.info("All content nodes already have the AtomicUnit label")
            return True
    except Exception as e:
        logger.error(f"Error ensuring AtomicUnit label: {e}")
        return False


def drop_index_if_exists(index_name: str) -> bool:
    """Drop an index if it exists."""
    try:
        logger.info(f"Dropping existing index {index_name} if it exists...")
        with driver.session() as session:
            session.run(f"DROP INDEX {index_name} IF EXISTS")
        return True
    except Exception as e:
        logger.warning(f"Error dropping index {index_name}: {e}")
        return False


def create_fulltext_index(
    label: str = "AtomicUnit", properties: list = None, index_name: str = None
) -> bool:
    """Create a fulltext index for specified nodes and properties.

    Args:
        label: Node label to index (default: AtomicUnit)
        properties: List of properties to index (default: ["text", "title", "proof"])
        index_name: Custom index name (default: fulltext_index_{label})

    Returns:
        True if successful, False otherwise
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
    if not drop_index_if_exists(index_name):
        logger.warning("Failed to drop existing index, but continuing...")

    # Create new fulltext index
    try:
        with driver.session() as session:
            session.run(
                f"""
            CREATE FULLTEXT INDEX {index_name}
            FOR (n:{label}) ON EACH [{property_list}]
            """
            )
        logger.info(f"Successfully created fulltext index {index_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create fulltext index {index_name}: {e}")
        return False


def test_fulltext_search(query="topologisch"):
    """Test fulltext search with a sample query."""
    logger.info(f"Testing fulltext search with query: '{query}'")

    try:
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

            return len(results) > 0
    except Exception as e:
        logger.error(f"Error testing fulltext search: {e}")
        return False


def main(
    test: bool = False,
    query: str = "",
    label: str = "AtomicUnit",
    properties: list = None,
):
    """Main function to create and test fulltext index."""

    # Ensure AtomicUnit label
    logger.info("Ensuring AtomicUnit label...")
    ensure_atomic_unit_label()

    # Set default properties if not provided
    if properties is None:
        properties = ["text", "title", "proof"]

    # Create fulltext index
    logger.info(f"Creating fulltext index for {label} on properties {properties}...")
    success = create_fulltext_index(label=label, properties=properties)
    if success:
        logger.info("Fulltext index created successfully.")
        if test and query:
            test_fulltext_search(query)
    else:
        logger.error("Failed to create fulltext index.")

    # If only testing was requested
    if test and not query:
        test_fulltext_search()

    logger.info("Fulltext index operations completed.")


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
