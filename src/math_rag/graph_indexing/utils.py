"""
Utility functions for graph indexing operations.

This module contains common functions used across different indexing scripts
to reduce code duplication and improve maintainability.
"""

import logging
from typing import Optional, Tuple

from neo4j import GraphDatabase

# Configure module logger
logger = logging.getLogger(__name__)


def ensure_atomic_unit_label(driver: GraphDatabase.driver) -> None:
    """
    Ensure that all content nodes have the AtomicItem label.

    This function adds the AtomicItem label to all content nodes
    (Introduction, Definition, Corollary, Theorem, Lemma, Proof, Example,
    Exercise, Remark) that don't already have it.

    Args:
        driver: Neo4j driver instance
    """
    logger.info("Ensuring all content nodes have the AtomicItem label...")
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:Introduction|Definition|Corollary|Theorem|Lemma|Proof|Example|
                  Exercise|Remark)
            WHERE NOT n:AtomicItem
            WITH count(n) AS missingLabel
            MATCH (n:Introduction|Definition|Corollary|Theorem|Lemma|Proof|Example|
                  Exercise|Remark)
            WHERE NOT n:AtomicItem
            SET n:AtomicItem
            RETURN missingLabel, count(n) AS updated
            """
        )
        record = result.single()
        if record and record["missingLabel"] > 0:
            logger.info(f"Added AtomicItem label to {record['updated']} nodes")
        else:
            logger.info("All content nodes already have the AtomicItem label")


def drop_index_if_exists(driver: GraphDatabase.driver, index_name: str) -> bool:
    """
    Drop an index if it exists.

    This function attempts to drop an index with the given name. If the index
    doesn't exist or an error occurs, it logs a warning but returns True to
    allow the calling code to continue.

    Args:
        driver: Neo4j driver instance
        index_name: Name of the index to drop

    Returns:
        True if successful or index doesn't exist, False on critical errors
    """
    try:
        logger.info(f"Dropping existing index {index_name} if it exists...")
        with driver.session() as session:
            session.run(f"DROP INDEX {index_name} IF EXISTS")
        return True
    except Exception as e:
        logger.warning(f"Error dropping index {index_name}: {e}")
        return False


def verify_nodes(
    driver: GraphDatabase.driver, label: str, property_name: Optional[str] = None
) -> Tuple[bool, int]:
    """
    Verify that nodes exist in the database, optionally checking for a
    specific property.

    This function consolidates the functionality of verify_nodes_exist() and
    verify_vector_property_exists() from the original scripts.

    Args:
        driver: Neo4j driver instance
        label: Node label to check
        property_name: Optional property name to verify exists on nodes

    Returns:
        Tuple of (exists: bool, count: int) where:
        - exists: True if nodes exist (and have the property if specified)
        - count: Number of nodes found
    """
    if property_name:
        logger.info(f"Verifying {property_name} property exists on {label} nodes...")
        query = f"""
            MATCH (n:{label})
            WHERE n.{property_name} IS NOT NULL
            RETURN count(n) as count
        """
    else:
        logger.info(f"Verifying {label} nodes...")
        query = f"""
            MATCH (n:{label})
            RETURN count(n) as count
        """

    with driver.session() as session:
        result = session.run(query)
        count = result.single()["count"]

    if property_name:
        if count == 0:
            logger.warning(f"No {label} nodes with {property_name} property found.")
            return False, 0
        else:
            logger.info(f"Found {count} {label} nodes with {property_name} property.")
            return True, count
    else:
        logger.info(f"Found {count} {label} nodes in the database")
        return count > 0, count


def count_nodes_without_property(
    driver: GraphDatabase.driver,
    label: str,
    text_properties: list,
    target_property: str,
) -> int:
    """
    Count nodes that have at least one text property but are missing the
    target property.

    This is useful for determining how many nodes need processing
    (e.g., embedding generation).

    Args:
        driver: Neo4j driver instance
        label: Node label to check
        text_properties: List of text properties to check for existence
        target_property: The property that should be added

    Returns:
        Count of nodes that need processing
    """
    # Build condition to check if any text property exists
    text_conditions = " OR ".join([f"n.{prop} IS NOT NULL" for prop in text_properties])

    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (n:{label})
            WHERE ({text_conditions}) AND n.{target_property} IS NULL
            RETURN count(n) as count
            """
        )
        return result.single()["count"]


def verify_property_populated(
    driver: GraphDatabase.driver,
    label: str,
    text_properties: list,
    target_property: str,
) -> int:
    """
    Verify how many nodes with text properties are still missing the target property.

    This is useful for verifying that a processing step (like embedding generation)
    has been completed successfully.

    Args:
        driver: Neo4j driver instance
        label: Node label to check
        text_properties: List of text properties that should exist
        target_property: The property that should have been added

    Returns:
        Count of nodes still missing the target property
    """
    # Build condition to check if any text property exists
    text_conditions = " OR ".join([f"n.{prop} IS NOT NULL" for prop in text_properties])

    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (n:{label})
            WHERE ({text_conditions}) AND n.{target_property} IS NULL
            RETURN count(n) as remaining
            """
        )
        return result.single()["remaining"]
