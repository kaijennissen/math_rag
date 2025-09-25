"""
CLI for building the complete Knowledge Graph with all indices.

This module provides a simple interface to build the entire knowledge graph including:
1. Graph structure from database
2. Reference relationships
3. Fulltext index
4. Vector index
"""

import logging
import sys
from pathlib import Path

from langchain_neo4j import Neo4jGraph
from neo4j import Driver, GraphDatabase

from math_rag.config.settings import KnowledgeGraphSettings
from math_rag.core.db_models import DatabaseManager
from math_rag.graph_construction.add_reference_relationships import (
    add_references_to_graph,
    load_reference_tuples,
)
from math_rag.graph_construction.build_kg_from_db import (
    build_knowledge_graph_from_sqlite,
    clear_neo4j_database,
)
from math_rag.graph_indexing.create_fulltext_index import (
    create_fulltext_index as create_fulltext_index_impl,
)
from math_rag.graph_indexing.create_vector_index_with_custom_embeddings import (
    add_embeddings_with_neo4j_vector,
    get_embedding_model,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_complete_knowledge_graph(
    db_manager: DatabaseManager,
    graph: Neo4jGraph,
    driver: Driver,
    document_name: str,
    reference_tuples_path: Path,
    clear_first: bool = True,
):
    """
    Orchestrate the complete knowledge graph build.

    Args:
        db_manager: Database manager instance
        graph: Neo4j graph instance
        driver: Neo4j driver instance
        document_name: Name of the document to process
        clear_first: Whether to clear existing graph data

    Returns:
        bool: True if successful, False otherwise
    """

    # Phase 1: Build graph structure and add atomic items
    logger.info("=== Phase 1: Building Knowledge Graph Structure ===")
    if clear_first:
        clear_neo4j_database(driver)

    build_knowledge_graph_from_sqlite(
        db_manager=db_manager,
        graph=graph,
        driver=driver,
        document_name=document_name,
    )
    logger.info("✓ Graph structure and atomic items added successfully")

    # Phase 2: Add reference relationships
    logger.info("=== Phase 2: Adding Reference Relationships ===")
    reference_tuples = load_reference_tuples(reference_tuples_path)
    if reference_tuples:
        relationships_created = add_references_to_graph(
            graph=graph,
            document_name=document_name,
            reference_tuples=reference_tuples,
        )
        logger.info(f"✓ Added {relationships_created} reference relationships")
    else:
        logger.warning("No reference tuples found, skipping references")

    # Phase 3: Create fulltext index
    logger.info("=== Phase 3: Creating Fulltext Index ===")
    create_fulltext_index_impl(
        driver=driver,
        label="AtomicItem",
        properties=["text", "title", "proof", "summary"],
    )
    logger.info("✓ Fulltext index created successfully")

    # Phase 4: Create vector index with embeddings
    logger.info("=== Phase 4: Creating Vector Index with Embeddings ===")
    # Use E5 Multilingual model by default (best for German academic content)
    embedding_model = get_embedding_model("E5 Multilingual")
    add_embeddings_with_neo4j_vector(
        driver=driver,
        embedding_model=embedding_model,
        label="AtomicItem",
        index_name="vector_index_summary_Embedding",
        text_properties=["summary"],
        embedding_property="summaryEmbedding",
    )
    logger.info("✓ Vector index and embeddings created successfully")

    logger.info("=== ✓ Complete knowledge graph build finished successfully ===")


def main(
    db_path: Path,
    document_name: str,
    clear: bool,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    reference_tuples_path: Path,
):
    """Main entry point for the CLI.

    Args:
        db_path: Path to SQLite database
        document_name: Name of the document to process
        clear: Whether to clear existing graph data
        neo4j_uri: Neo4j connection URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        reference_tuples_path: Path to reference tuples pickle file
    """

    # Create all resources
    logger.info("Creating database connections and resources...")
    db_manager = DatabaseManager(db_path)
    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
    )
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

    try:
        logger.info("Starting complete knowledge graph build")
        logger.info(f"  Database: {db_path}")
        logger.info(f"  Document: {document_name}")

        success = build_complete_knowledge_graph(
            db_manager=db_manager,
            graph=graph,
            driver=driver,
            document_name=document_name,
            reference_tuples_path=reference_tuples_path,
            clear_first=clear,
        )

        if not success:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        logger.info("Closing database connections...")
        driver.close()


if __name__ == "__main__":
    settings = KnowledgeGraphSettings()
    main(
        db_path=settings.db_path,
        document_name=settings.document_name,
        clear=settings.clear,
        neo4j_uri=settings.neo4j_uri,
        neo4j_username=settings.neo4j_username,
        neo4j_password=settings.neo4j_password,
        reference_tuples_path=settings.reference_tuples_path,
    )
