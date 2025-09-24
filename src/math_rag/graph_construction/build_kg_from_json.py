"""
Module for building a knowledge graph from atomic units extracted from
mathematical text.
The graph is stored in Neo4j and represents the document structure and relationships.

This script handles:
1. Creating document, section, and subsection nodes
2. Creating relationships between nodes (PART_OF, NEXT, PREVIOUS)
3. Adding atomic units from JSON files to the graph
4. Creating a fulltext index for keyword search

"""

import json
import logging
import os
from pathlib import Path

import coloredlogs
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase

from math_rag.core import AtomicItem
from math_rag.data_processing import SectionHeaders

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="WARNING",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Default paths and names
DOCS_PATH = Path("docs/atomic_units/")
SECTION_HEADERS_PATH = Path("docs/section_headers.yaml")
ATOMIC_UNITS_PATH = Path("docs/atomic_units")
DOCUMENT_NAME = "topological_spaces"


def create_section_node(
    graph: Neo4jGraph, document_name: str, section_number: int, title: str
):
    """Create section node and link to document."""
    graph.query(
        """
        MERGE (d:Document {id: $document_id})
        MERGE (s:Section {id: $section_id, number: $section_number})
        SET s.title = $title
        MERGE (s)-[:PART_OF]->(d)
        """,
        {
            "document_id": document_name,
            "section_id": f"{document_name}_{section_number}",
            "section_number": section_number,
            "title": title,
        },
    )
    logger.info(f"Created section node for Section {section_number}")


def create_subsection_node(
    graph: Neo4jGraph,
    document_name: str,
    section_number: int,
    subsection_number: int,
    title: str,
):
    """Create subsection node and link to section."""
    graph.query(
        """
        MERGE (s1:Section {id: $section_id, number: $section_number})
        MERGE (s2:Subsection {id: $subsection_id, number: $subsection_number})
        SET s2.title = $title
        MERGE (s2)-[:PART_OF]->(s1)
        """,
        {
            "document_id": document_name,
            "section_id": f"{document_name}_{section_number}",
            "section_number": section_number,
            "subsection_id": f"{document_name}_{subsection_number}",
            "subsection_number": subsection_number,
            "title": title,
        },
    )
    logger.info(f"Created subsection node for Subsection {subsection_number}")


def add_chunk_to_graph(graph: Neo4jGraph, document_name: str, chunk: AtomicItem):
    """Add an atomic unit chunk to the graph."""
    subsection_number = chunk.get_subsection_number()
    subsubsection_number = chunk.get_full_number()
    logger.info(f"Adding subsubsection: {subsubsection_number}")
    sanitized_label = chunk.type

    # Construct the MERGE query dynamically with the sanitized label
    query = f"""
    MATCH (sub:Subsection {{id: $subsection_id}})
    MERGE (c:{sanitized_label} {{id: $subsubsection_id, number: $subsubsection_number}})
    SET c.text = $text,
        c.type = $type,
        c.title = $title,
        c.proof = $proof,
        c.textEmbedding = Null
    MERGE (c)-[:PART_OF]->(sub)
    """

    graph.query(
        query,
        {
            "subsection_id": f"{document_name}_{subsection_number}",
            "subsubsection_id": f"{document_name}_{subsubsection_number}",
            "subsubsection_number": subsubsection_number,
            "text": chunk.text,
            "type": chunk.type,
            "title": chunk.identifier,
            "proof": chunk.proof,
        },
    )


def create_next_relationship(
    graph: Neo4jGraph,
    document_name: str,
    node_type: str,  # "section" or "subsection"
    current_number,
    next_number,
):
    """Create NEXT/PREVIOUS relationships between nodes."""
    if node_type == "section":
        current_id = f"{document_name}_{current_number}"
        next_id = f"{document_name}_{next_number}"
        label = "Section"
    elif node_type == "subsection":
        current_id = f"{document_name}_{current_number}"
        next_id = f"{document_name}_{next_number}"
        label = "Subsection"
    else:
        raise ValueError("node_type must be 'section' or 'subsection'")

    graph.query(
        f"""
        MATCH (a:{label} {{id: $current_id}})
        MATCH (b:{label} {{id: $next_id}})
        MERGE (a)-[:NEXT]->(b)
        MERGE (a)<-[:PREVIOUS]-(b)
        """,
        {"current_id": current_id, "next_id": next_id},
    )


def create_previous_relationship_atomic_units(
    graph: Neo4jGraph,
    document_name: str,
    current_number,
    next_number,
):
    """Create NEXT/PREVIOUS relationships between atomic units."""
    current_id = f"{document_name}_{current_number}"
    next_id = f"{document_name}_{next_number}"

    graph.query(
        """
        MATCH (a WHERE a.id = $current_id)
        MATCH (b WHERE b.id= $next_id)
        MERGE (a)-[:NEXT]->(b)
        MERGE (a)<-[:PREVIOUS]-(b)
        """,
        {"current_id": current_id, "next_id": next_id},
    )


def ensure_atomic_unit_label(driver):
    """Ensure that all content nodes have the AtomicItem label."""
    logger.info("Ensuring all content nodes have the AtomicItem label...")
    try:
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
            return True
    except Exception as e:
        logger.error(f"Error ensuring AtomicItem label: {e}")
        return False


def build_knowledge_graph(
    graph: Neo4jGraph,
    driver,
    document_name: str,
    section_headers_path: Path,
    atomic_units_path: Path,
):
    """
    Build the complete knowledge graph.
    Creates document, section, and subsection nodes and their relationships.
    Adds atomic units from JSON files and establishes their relationships.
    """
    logger.info("Starting knowledge graph construction.")
    section_headers = SectionHeaders(section_headers_path)

    # First, create the document node
    logger.info(f"Creating document node for '{document_name}'")
    graph.query(
        """
        MERGE (d:Document {id: $document_id})
        SET d.title = $title
        """,
        {
            "document_id": document_name,
            "title": f"Mathematical document: {document_name}",
        },
    )

    # Step 1: Create nodes and PART_OF relationships
    logger.info("Creating section and subsection nodes with PART_OF relationships...")
    for section in section_headers.all_sections():
        logger.info(f"Creating section node: {section.number} '{section.title}'")
        create_section_node(
            graph=graph,
            document_name=document_name,
            section_number=section.number,
            title=section.title,
        )
        for subsection in section.subsections:
            logger.info(
                f"  Creating subsection node: {subsection.number} '{subsection.title}' "
                f"(parent section: {section.number})"
            )
            create_subsection_node(
                graph=graph,
                document_name=document_name,
                section_number=section.number,
                subsection_number=subsection.number,
                title=subsection.title,
            )

    # Step 2: Create NEXT relationships
    logger.info("Creating NEXT/PREVIOUS relationships for sections...")
    sorted_sections = sorted(section_headers.all_sections(), key=lambda x: x.number)
    for current, next in zip(sorted_sections[:-1], sorted_sections[1:]):
        logger.info(
            f"  Linking section {current.number} -> {next.number} (NEXT/PREVIOUS)"
        )
        create_next_relationship(
            graph=graph,
            document_name=document_name,
            node_type="section",
            current_number=current.number,
            next_number=next.number,
        )

    logger.info("Creating NEXT/PREVIOUS relationships for subsections...")
    for section in section_headers.all_sections():
        sorted_subs = sorted(
            section_headers.all_subsections(section.number), key=lambda x: x.number
        )
        for current, next in zip(sorted_subs[:-1], sorted_subs[1:]):
            logger.info(
                f"  Linking subsection {current.number} -> {next.number} "
                f"(NEXT/PREVIOUS) in section {section.number}"
            )
            create_next_relationship(
                graph,
                document_name=document_name,
                node_type="subsection",
                current_number=current.number,
                next_number=next.number,
            )

    # Add atomic unit chunks from docs/atomic_units
    logger.info("Adding atomic unit chunks from docs/atomic_units...")

    for json_file in atomic_units_path.glob("subsection_*_*_units.json"):
        logger.info(f"  Processing file: {json_file.name}")
        with open(json_file, "r") as f:
            data = json.load(f)
        atomic_units = [AtomicItem.from_dict(unit) for unit in data.get("chunks")]

        for unit in atomic_units:
            add_chunk_to_graph(graph=graph, document_name=document_name, chunk=unit)

        for current, next in zip(atomic_units[:-1], atomic_units[1:]):
            create_previous_relationship_atomic_units(
                graph,
                document_name=document_name,
                current_number=current.get_full_number(),
                next_number=next.get_full_number(),
            )

    logger.info("All atomic unit chunks have been added to the graph.")

    # Ensure all content nodes have the AtomicItem label
    ensure_atomic_unit_label(driver)

    logger.info(
        "Knowledge graph construction completed. "
        "Run create_embeddings_and_vector_index.py to add embeddings and create "
        "vector index."
    )


def main():
    """
    Main function to build the knowledge graph.
    Creates all necessary resources and builds the graph.
    """

    load_dotenv()
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
    )

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

    try:
        build_knowledge_graph(
            graph=graph,
            driver=driver,
            document_name=DOCUMENT_NAME,
            section_headers_path=SECTION_HEADERS_PATH,
            atomic_units_path=ATOMIC_UNITS_PATH,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
