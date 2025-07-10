"""
CLI for Knowledge Graph construction and indexing operations.

This module provides unified commands to:
1. Build the knowledge graph from database
2. Add reference relationships
3. Create fulltext and vector indexes
"""

import argparse
import logging
import sys

from math_rag.graph_construction.build_kg_from_db import main as build_kg_main
from math_rag.graph_construction.add_reference_relationships import (
    main as add_refs_main,
)
from math_rag.graph_indexing.create_fulltext_index import main as fulltext_main
from math_rag.graph_indexing.create_vector_index_with_custom_embeddings import (
    main as custom_main,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_graph_command(clear_first: bool, db_path: str, document_name: str):
    """Build the knowledge graph from the database."""
    logger.info("Starting knowledge graph construction...")

    try:
        # Build knowledge graph
        logger.info("Building knowledge graph from SQLite database...")
        build_kg_main(clear_first=clear_first, db_path=db_path)

        # Add reference relationships
        logger.info("Adding reference relationships...")
        add_refs_main(document_name=document_name)

        logger.info("Knowledge graph construction completed")
        return True

    except Exception as e:
        logger.error(f"Failed to build knowledge graph: {e}")
        return False


def create_indexes_command(create_fulltext: bool, create_vector: bool, model: str):
    """Create fulltext and vector indexes."""
    logger.info("Starting index creation...")

    success = True

    try:
        # Create fulltext index if requested
        if create_fulltext:
            logger.info("Creating fulltext index...")
            success &= create_fulltext_index_func()

        # Create vector index if requested
        if create_vector and success:
            logger.info(f"Creating vector index with model: {model}")
            success &= create_vector_index_func(model)

        if success:
            logger.info("Index creation completed")
        else:
            logger.error("Index creation failed")

        return success

    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        return False


def create_fulltext_index_func():
    """Create fulltext index using existing main function."""
    try:
        fulltext_main(
            test=False,
            query="",
            label="AtomicUnit",
            properties=["text", "title", "proof", "summary"],
        )

        logger.info("Fulltext index created successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to create fulltext index: {e}")
        return False


def create_vector_index_func(model: str):
    """Create vector index with custom embeddings."""
    try:
        custom_main(
            model_name=model,
            test=False,
            query="",
            text_properties=["text", "title"],
            label="AtomicUnit",
            embedding_property=None,  # Will be auto-generated
        )

        logger.info(f"Vector index created successfully with {model}")
        return True

    except Exception as e:
        logger.error(f"Failed to create vector index: {e}")
        return False


def build_complete_command(
    clear_first: bool, db_path: str, document_name: str, model: str
):
    """Build complete knowledge graph with indexes."""
    logger.info("Starting complete knowledge graph build...")

    # Step 1: Build knowledge graph
    logger.info("=== Phase 1: Building Knowledge Graph ===")
    if not build_graph_command(clear_first, db_path, document_name):
        logger.error("Failed to build knowledge graph")
        return False

    # Step 2: Create indexes
    logger.info("=== Phase 2: Creating Indexes ===")
    # Always create both fulltext and vector indexes for build-all
    if not create_indexes_command(
        create_fulltext=True, create_vector=True, model=model
    ):
        logger.error("Failed to create indexes")
        return False

    logger.info("Complete knowledge graph build finished")
    return True


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Construction and Indexing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build complete knowledge graph with all indexes
  python -m math_rag.cli.kg_cli build-all --model "E5 Multilingual"

  # Build only the graph structure
  python -m math_rag.cli.kg_cli build-graph

  # Create only fulltext index
  python -m math_rag.cli.kg_cli create-indexes --fulltext

  # Create only vector index with custom model
  python -m math_rag.cli.kg_cli create-indexes --vector --model "MXBAI German"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Build graph command
    build_parser = subparsers.add_parser(
        "build-graph", help="Build knowledge graph from database"
    )
    build_parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing graph data before building",
    )
    build_parser.add_argument(
        "--db-path", help="Path to SQLite database (uses default if not specified)"
    )
    build_parser.add_argument(
        "--document-name",
        default="topological_spaces",
        help="Name of the document to process (default: topological_spaces)",
    )

    # Create indexes command
    index_parser = subparsers.add_parser(
        "create-indexes", help="Create fulltext and/or vector indexes"
    )
    index_parser.add_argument(
        "--fulltext", action="store_true", help="Create fulltext index"
    )
    index_parser.add_argument(
        "--vector", action="store_true", help="Create vector index"
    )
    index_parser.add_argument(
        "--model",
        default="E5 Multilingual",
        help="Embedding model for vector index (default: E5 Multilingual, options: 'E5 Multilingual', 'MXBAI German')",
    )

    # Build complete command
    complete_parser = subparsers.add_parser(
        "build-all", help="Build complete knowledge graph with indexes"
    )
    complete_parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing graph data before building",
    )
    complete_parser.add_argument(
        "--db-path", help="Path to SQLite database (uses default if not specified)"
    )
    complete_parser.add_argument(
        "--document-name",
        default="topological_spaces",
        help="Name of the document to process (default: topological_spaces)",
    )
    complete_parser.add_argument(
        "--model",
        default="E5 Multilingual",
        help="Embedding model for vector index (default: E5 Multilingual)",
    )
    # No need for set_defaults since build-all always creates both indexes

    # Parse args
    args = parser.parse_args()

    # Execute command
    if args.command == "build-graph":
        success = build_graph_command(
            clear_first=not args.no_clear,
            db_path=args.db_path,
            document_name=args.document_name,
        )
    elif args.command == "create-indexes":
        if not (args.fulltext or args.vector):
            logger.error("Must specify --fulltext and/or --vector")
            sys.exit(1)
        success = create_indexes_command(
            create_fulltext=args.fulltext, create_vector=args.vector, model=args.model
        )
    elif args.command == "build-all":
        success = build_complete_command(
            clear_first=not args.no_clear,
            db_path=args.db_path,
            document_name=args.document_name,
            model=args.model,
        )
    else:
        parser.print_help()
        sys.exit(1)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
