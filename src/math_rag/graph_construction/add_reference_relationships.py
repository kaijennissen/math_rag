"""
Script to add reference relationships to the knowledge graph based on extracted PDF
references.
This script loads reference tuples from a pickle file and creates REFERENCES
relationships between atomic units in the Neo4j graph.
"""

import logging
import pickle
from pathlib import Path

import coloredlogs
from langchain_neo4j import Neo4jGraph

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


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
        MERGE (target)-[:CITES]->(source)
        """,
        {"source_id": source_id, "target_id": target_id},
    )

    logger.info(f"Created REFERENCES relationship: {source_number} -> {target_number}")


def load_reference_tuples(pickle_path: Path) -> list:
    """
    Load reference tuples from pickle file.

    Args:
        pickle_path: Path to the pickle file

    Returns:
        List of reference tuples
    """

    try:
        with open(pickle_path, "rb") as f:
            reference_tuples: list = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""Reference tuples file not found at {pickle_path}.
            Please run infer_refs.py first to generate the reference tuples."""
        )

    logger.info(f"Loaded {len(reference_tuples)} reference tuples from {pickle_path}")
    return reference_tuples


def add_references_to_graph(
    graph: Neo4jGraph, document_name: str, reference_tuples: list
) -> int:
    """
    Add reference relationships to the graph.

    Args:
        graph: Neo4j graph instance
        document_name: Name of the document
        reference_tuples: List of (target, source) tuples

    Returns:
        Number of relationships created
    """
    relationships_created = 0

    # Reverse the tuples as this is how links in the pdf are stored
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

    return relationships_created
