import os
from pathlib import Path
import logging
import coloredlogs
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from section_headers import SectionHeaders
import json

load_dotenv()
# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DOCS_PATH = Path("docs/atomic_units/")
SECTION_HEADERS_PATH = "docs/section_headers.yaml"
DOCUMENT_NAME = "Topologische Räume"

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4.5-preview"
)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)


def link_previous_section(graph: Neo4jGraph, document_name: str, section_number: int):
    previous_section_id = f"{document_name}.section_{section_number - 1}"
    if section_number > 1 and graph.exists_node(previous_section_id):
        graph.query(
            """
            MATCH (s1:Section {id: $previous_section_id}),
                  (s2:Section {id: $section_id})
            MERGE (s1)-[:NEXT]->(s2)
            """,
            {
                "section_id": f"{document_name}.section_{section_number}",
                "previous_section_id": previous_section_id,
            },
        )


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
            "section_id": f"{document_name}.section_{section_number}",
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
            "section_id": f"{document_name}.section_{section_number}",
            "section_number": section_number,
            "subsection_id": f"{document_name}.subsection_{subsection_number}",
            "subsection_number": subsection_number,
            "title": title,
        },
    )
    logger.info(f"Created subsection node for Subsection {subsection_number}")


def add_chunk_to_graph(graph: Neo4jGraph, document_name: str, chunk: dict):
    section_id = chunk.get("section")
    subsection_id = chunk.get("subsection")
    subsubsection_id = chunk.get("subsubsection")
    subsubsection_number = ".".join(
        [str(section_id), str(subsection_id), str(subsubsection_id)]
    )
    logger.info(f"Adding subsubsection: {subsubsection_number}")

    chunk_type = chunk.get("type", "ContentUnit")  # Default label
    # Sanitize for Neo4j label (e.g., capitalize, no spaces)
    sanitized_label = (
        chunk_type.replace(" ", "_").capitalize() if chunk_type else "ContentUnit"
    )

    # Construct the MERGE query dynamically with the sanitized label
    query = f"""
    MATCH (sub:Subsection {{id: $subsection_id}})
    MERGE (c:{sanitized_label} {{id: $subsubsection_id, number: $subsubsection_number}})
    SET c.text = $text,
        c.type = $type,
        c.title = $title,
        c.proof = $proof
    MERGE (c)-[:PART_OF]->(sub)
    """

    graph.query(
        query,
        {
            "subsection_id": f"{document_name}.subsection_{section_id}.{subsection_id}",
            "subsubsection_id": f"{document_name}.subsection_{section_id}.{subsection_id}.{subsubsection_id}",
            "subsubsection_number": subsubsection_number,
            "text": chunk.get("text"),
            "type": chunk.get("type"),
            "title": chunk.get("identifier"),
            "proof": chunk.get("proof"),
        },
    )


def create_next_relationship(
    graph: Neo4jGraph,
    document_name: str,
    node_type: str,  # "section" or "subsection"
    current_number,
    next_number,
    section_number=None,  # Only needed for subsections
):
    if node_type == "section":
        current_id = f"{document_name}.section_{current_number}"
        next_id = f"{document_name}.section_{next_number}"
        label = "Section"
    elif node_type == "subsection":
        current_id = f"{document_name}.subsection_{current_number}"
        next_id = f"{document_name}.subsection_{next_number}"
        label = "Subsection"
    else:
        raise ValueError("node_type must be 'section' or 'subsection'")

    graph.query(
        f"""
        MATCH (a:{label} {{id: $current_id}})
        MATCH (b:{label} {{id: $next_id}})
        MERGE (a)-[:NEXT]->(b)
        MERGE (a)<-[:PREVIOUS]->(b)
        """,
        {"current_id": current_id, "next_id": next_id},
    )


def main():
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
                section_number=section.number,
            )
    logger.info("Knowledge graph construction completed.")

    # Add atomic unit chunks from docs/atomic_units
    logger.info("Adding atomic unit chunks from docs/atomic_units...")
    atomic_units_path = Path("docs/atomic_units")
    for json_file in atomic_units_path.glob("subsection_*_*_units.json"):
        logger.info(f"  Processing file: {json_file.name}")
        with open(json_file, "r") as f:
            data = json.load(f)

        for chunk in data.get("chunks", []):
            add_chunk_to_graph(graph=graph, document_name=DOCUMENT_NAME, chunk=chunk)

    logger.info("All atomic unit chunks have been added to the graph.")


if __name__ == "__main__":
    main()
