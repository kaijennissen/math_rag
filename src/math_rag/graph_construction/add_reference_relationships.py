"""
Script to add reference relationships to the knowledge graph based on extracted PDF references.
This script loads reference tuples from a pickle file and creates REFERENCES relationships
between atomic units in the Neo4j graph.
"""

import logging
import os
import pickle

import coloredlogs
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

from math_rag.core import ROOT

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DOCUMENT_NAME = "topological_spaces"


def create_reference_relationship(
    graph: Neo4jGraph, document_name: str, source_number: str, target_number: str
):
    """
    Create a REFERENCES relationship between two atomic units.

    Args:
        graph: Neo4j graph instance
        document_name: Name of the document
        source_number: Number of the source atomic unit (e.g., "1.2.3")
        target_number: Number of the target atomic unit (e.g., "2.1.4")
    """
    source_id = f"{document_name}_{source_number}"
    target_id = f"{document_name}_{target_number}"

    # Create REFERENCES relationship from source to target
    graph.query(
        """
        MATCH (source WHERE source.id = $source_id)
        MATCH (target WHERE target.id = $target_id)
        MERGE (source)-[:REFERENCED_IN]->(target)
        """,
        {"source_id": source_id, "target_id": target_id},
    )

    logger.info(f"Created REFERENCES relationship: {source_number} -> {target_number}")


def main(document_name: str = DOCUMENT_NAME):
    """
    Main function to load reference tuples and create relationships in the graph.

    Args:
        document_name: Name of the document to process
    """
    logger.info("Starting reference relationship creation.")

    # Load reference tuples from pickle file
    pickle_path = ROOT / "data" / "reference_tuples.pkl"
    # Neo4j connection details
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )

    if not pickle_path.exists():
        logger.error(f"Reference tuples file not found at {pickle_path}")
        logger.error("Please run infer_refs.py first to generate the reference tuples.")
        return

    with open(pickle_path, "rb") as f:
        reference_tuples = pickle.load(f)

    logger.info(f"Loaded {len(reference_tuples)} reference tuples from {pickle_path}")

    # Create relationships for each tuple
    relationships_created = 0
    # reverse the tuples as this is how links in the pdf are stored

    for target_number, source_number in reference_tuples:
        try:
            create_reference_relationship(
                graph=graph,
                document_name=document_name,
                source_number=source_number,
                target_number=target_number,
            )
            relationships_created += 1
        except Exception as e:
            logger.warning(
                f"Failed to create relationship {source_number} -> {target_number}: {e}"
            )

    logger.info(
        f"Successfully created {relationships_created} REFERENCES relationships."
    )
    logger.info("Reference relationship creation completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add reference relationships to the knowledge graph"
    )
    parser.add_argument(
        "--document-name",
        default=DOCUMENT_NAME,
        help=f"Name of the document to process (default: {DOCUMENT_NAME})",
    )

    args = parser.parse_args()
    main(document_name=args.document_name)
