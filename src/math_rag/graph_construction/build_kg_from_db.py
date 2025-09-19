"""
Build Neo4j knowledge graph from SQLite database instead of files.
"""

import logging
import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase
from sqlmodel import select

from math_rag.core import AtomicUnit
from math_rag.core.db_models import AtomicUnitDB, DatabaseManager
from math_rag.core.project_root import ROOT
from math_rag.data_processing import SectionHeaders

# Load environment variables
load_dotenv()

# Default database path
DB_PATH = ROOT / "data" / "atomic_units.sqlite"
SECTION_HEADERS_PATH = ROOT / "docs" / "section_headers.yaml"
DOCUMENT_NAME = "topological_spaces"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_hierarchy_from_file(graph: Neo4jGraph):
    """
    Main function to build the knowledge graph.
    Creates document, section, and subsection nodes and their relationships.
    Adds atomic units from JSON files and establishes their relationships.
    Finally, creates a fulltext index for keyword search.
    """
    logger.info("Starting knowledge graph construction.")
    section_headers = SectionHeaders(SECTION_HEADERS_PATH)

    # First, create the document node
    logger.info(f"Creating document node for '{DOCUMENT_NAME}'")
    graph.query(
        """
        MERGE (d:Document {id: $document_id})
        SET d.title = $title
        """,
        {
            "document_id": DOCUMENT_NAME,
            "title": f"Mathematical document: {DOCUMENT_NAME}",
        },
    )

    # Step 1: Create nodes and PART_OF relationships
    logger.info("Creating section and subsection nodes with PART_OF relationships...")
    for section in section_headers.all_sections():
        logger.info(f"Creating section node: {section.number} '{section.title}'")
        create_section_node(
            graph=graph,
            document_name=DOCUMENT_NAME,
            section_number=section.number,
            title=section.title,
        )
        for subsection in section.subsections:
            logger.info(
                f"  Creating subsection node: {subsection.number} '{subsection.title}' (parent section: {section.number})"
            )
            create_subsection_node(
                graph=graph,
                document_name=DOCUMENT_NAME,
                section_number=section.number,
                subsection_number=subsection.number,
                title=subsection.title,
            )
            # Add PART_OF relationship: subsection -> section

    # Step 2: Create NEXT relationships
    logger.info("Creating NEXT/PREVIOUS relationships for sections...")
    sorted_sections = sorted(section_headers.all_sections(), key=lambda x: x.number)
    for current, next in zip(sorted_sections[:-1], sorted_sections[1:]):
        logger.info(
            f"  Linking section {current.number} -> {next.number} (NEXT/PREVIOUS)"
        )
        create_next_relationship(
            graph=graph,
            document_name=DOCUMENT_NAME,
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
                f"  Linking subsection {current.number} -> {next.number} (NEXT/PREVIOUS) in section {section.number}"
            )
            create_next_relationship(
                graph,
                document_name=DOCUMENT_NAME,
                node_type="subsection",
                current_number=current.number,
                next_number=next.number,
            )
    logger.info("Knowledge graph construction completed.")


# Create nodes for each section to establish hierarchy
def create_section_node(
    graph: Neo4jGraph, document_name: str, section_number: int, title: str
):
    # Create section node
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
    # Create subsection node
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


def create_next_relationship(
    graph: Neo4jGraph,
    document_name: str,
    node_type: str,  # "section" or "subsection"
    current_number,
    next_number,
):
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


# def create_atomic_unit_nodes(session, unit_data: List[Dict[str, Any]]) -> None:
#     """
#     Create atomic unit nodes in Neo4j from unit data.

#     Args:
#         session: Neo4j session
#         unit_data: List of atomic unit dictionaries
#     """
#     for unit in unit_data:
#         document_name = DOCUMENT_NAME
#         subsection_number = f"{unit['section']}.{unit['subsection']}"
#         subsubsection_number = (
#             f"{unit['section']}.{unit['subsection']}.{unit['subsubsection']}"
#         )
#         sanitized_label = unit["type"]

#         # Create the subsection ID that matches what create_document_hierarchy creates
#         subsection_id = f"{document_name}_{subsection_number}"

#         # First, verify the subsection exists (this helps debug missing connections)
#         check_query = """
#         MATCH (sub:Subsection {id: $subsection_id})
#         RETURN sub.id as id
#         """
#         check_result = session.run(check_query, {"subsection_id": subsection_id})
#         if not check_result.single():
#             logger.warning(
#                 f"Subsection {subsection_id} not found for atomic unit {subsubsection_number}"
#             )
#             continue

#         # Construct the MERGE query dynamically with the sanitized label
#         query = f"""
#         MATCH (sub:Subsection {{id: $subsection_id}})
#         MERGE (c:{sanitized_label} {{id: $subsubsection_id, number: $subsubsection_number}})
#         SET c.text = $text,
#             c.type = $type,
#             c.title = $title,
#             c.proof = $proof,
#             c.textEmbedding = Null
#         MERGE (c)-[:PART_OF]->(sub)
#         """

#         # Add summary if available
#         if "summary" in unit and unit["summary"]:
#             query = query.replace(
#                 "c.textEmbedding = Null", "c.textEmbedding = Null, c.summary = $summary"
#             )

#         session.run(
#             query,
#             {
#                 "subsection_id": subsection_id,
#                 "subsubsection_id": f"{document_name}_{subsubsection_number}",
#                 "subsubsection_number": subsubsection_number,
#                 "text": unit["text"],
#                 "type": unit["type"],
#                 "title": unit.get("identifier", ""),
#                 "proof": unit.get("proof"),
#                 "summary": unit.get("summary"),
#             },
#         )


def clear_neo4j_database(driver) -> None:
    """Clear all data from the Neo4j database."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("Cleared Neo4j database")


def add_atomic_unit_to_graph(
    graph: Neo4jGraph, document_name: str, atomic_unit: AtomicUnit
):
    subsection_number = atomic_unit.get_subsection_number()
    subsubsection_number = atomic_unit.get_full_number()
    logger.info(f"Adding subsubsection: {subsubsection_number}")
    sanitized_label = atomic_unit.type

    # Construct the MERGE query dynamically with the sanitized label
    query = f"""
    MATCH (sub:Subsection {{id: $subsection_id}})
    MERGE (c:{sanitized_label} {{id: $subsubsection_id, number: $subsubsection_number}})
    SET c.text = $text,
        c.type = $type,
        c.title = $title,
        c.proof = $proof,
        c.summary = $summary
    MERGE (c)-[:PART_OF]->(sub)
    """

    graph.query(
        query,
        {
            "subsection_id": f"{document_name}_{subsection_number}",
            "subsubsection_id": f"{document_name}_{subsubsection_number}",
            "subsubsection_number": subsubsection_number,
            "text": atomic_unit.text,
            "type": atomic_unit.type,
            "title": atomic_unit.identifier,
            "proof": atomic_unit.proof,
            "summary": atomic_unit.summary,
        },
    )


def create_next_relationship_atomic_units(
    graph: Neo4jGraph,
    document_name: str,
    current_number,
    next_number,
):
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


def create_atomic_units_from_list(graph: Neo4jGraph, units: list[AtomicUnit]) -> None:
    """
    Create atomic unit nodes in Neo4j from SQLite database.

    Args:
        graph: Neo4j graph instance
        units: List of AtomicUnit objects
    """
    logger.info(f"Creating {len(units)} atomic unit nodes in Neo4j")

    # Create atomic unit nodes
    for atomic_unit in units:
        add_atomic_unit_to_graph(
            graph=graph, document_name=DOCUMENT_NAME, atomic_unit=atomic_unit
        )

    logger.info(f"Created {len(units)} atomic unit nodes in Neo4j")


def get_atomic_units_from_db(db_manager: DatabaseManager) -> list[AtomicUnit]:
    """
    Retrieve atomic units from SQLite database.

    Args:
        db_manager: Database manager instance

    Returns:
        List of AtomicUnit objects
    """
    with db_manager.get_session() as session:
        db_rows = session.exec(select(AtomicUnitDB)).all()
        units = [row.to_core_atomic_unit() for row in db_rows]
    return units


def create_atomic_units_from_db(db_manager: DatabaseManager, graph: Neo4jGraph) -> None:
    atomic_units = get_atomic_units_from_db(db_manager)
    create_atomic_units_from_list(graph=graph, units=atomic_units)


def create_atomic_unit_relationships(
    graph: Neo4jGraph, units: list[AtomicUnit]
) -> None:
    """
    Create NEXT/PREVIOUS relationships between atomic units within each subsection.
    Groups units by subsection and sorts by subsubsection number.
    """
    logger.info(f"Creating relationships for {len(units)} atomic units")

    # Group units by subsection (e.g., "1.1", "1.2", "2.1")
    units_by_subsection = defaultdict(list)

    for unit in units:
        subsection_key = f"{unit.section}.{unit.subsection}"
        units_by_subsection[subsection_key].append(unit)

    # Create NEXT/PREVIOUS relationships within each subsection
    total_relationships = 0
    for subsection_key, subsection_units in units_by_subsection.items():
        if len(subsection_units) < 2:
            continue  # Need at least 2 units to create relationships

        # Sort by subsubsection number (they should already be sorted from SQL ORDER BY)
        sorted_units = sorted(subsection_units, key=lambda u: u.subsubsection)

        logger.info(
            f"Creating {len(sorted_units) - 1} relationships in subsection {subsection_key}"
        )

        # Create relationships between consecutive units
        for current, next_unit in zip(sorted_units[:-1], sorted_units[1:]):
            create_next_relationship_atomic_units(
                graph=graph,
                document_name=DOCUMENT_NAME,
                current_number=f"{current.section}.{current.subsection}.{current.subsubsection}",
                next_number=f"{next_unit.section}.{next_unit.subsection}.{next_unit.subsubsection}",
            )
            total_relationships += 1

    logger.info(
        f"Created {total_relationships} NEXT/PREVIOUS relationships between atomic units"
    )


def ensure_atomic_unit_label(driver):
    """Ensure that all content nodes have the AtomicUnit label."""
    logger.info("Ensuring all content nodes have the AtomicUnit label...")
    try:
        with driver.session() as session:
            result = session.run(
                """
            MATCH (n:Introduction|Definition|Corollary|Theorem|Lemma|Proof|Example|Exercise|Remark)
            WHERE NOT n:AtomicUnit
            WITH count(n) AS missingLabel
            MATCH (n:Introduction|Definition|Corollary|Theorem|Lemma|Proof|Example|Exercise|Remark)
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


def build_knowledge_graph_from_sqlite(
    db_manager: DatabaseManager,
    graph: Neo4jGraph,
    clear_first: bool = True,
) -> None:
    """
    Build complete knowledge graph from SQLite database.

    Args:
        db_manager: Database manager instance
        clear_first: Whether to clear the database before building
    """

    # Create Neo4j connection
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        if clear_first:
            clear_neo4j_database(driver)

        # Create document structure from file
        create_hierarchy_from_file(graph)

        # Create atomic unit nodes
        atomic_units = get_atomic_units_from_db(db_manager)
        create_atomic_units_from_list(graph=graph, units=atomic_units)

        ensure_atomic_unit_label(driver)

        # Create atomic unit relationships
        create_atomic_unit_relationships(graph=graph, units=atomic_units)

        logger.info("Knowledge graph build complete")

    finally:
        driver.close()


def main(clear_first: bool = True, db_path: str = None):
    """Main entry point for CLI usage.

    Args:
        clear_first: Whether to clear Neo4j database before building
        db_path: Path to SQLite database (defaults to default path)
    """
    # Use default path if none provided
    if db_path is None:
        db_path = DB_PATH

    # Create database manager
    db_manager = DatabaseManager(db_path)
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    build_knowledge_graph_from_sqlite(
        graph=graph,
        db_manager=db_manager,
        clear_first=clear_first,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Neo4j knowledge graph from SQLite database"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear Neo4j database before building",
    )
    parser.add_argument(
        "--db-path", help=f"Path to SQLite database (default: {DB_PATH})"
    )

    args = parser.parse_args()
    main(clear_first=args.clear, db_path=args.db_path)
