"""
Standalone script to add text embeddings to existing nodes in Neo4j database
using Neo4j's GenAI module with efficient batch processing.

Workflow for building the knowledge graph and enabling search:

1. Build graph structure:
   python -m src.math_rag.build_knowledge_graph

2. Add embeddings to nodes:
   python add_embeddings.py

3. Create indexes for search:
   python create_indexes.py --all --test
"""

import os
import logging
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("add_embeddings.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Direct Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def configure_openai_connection():
    """Configure OpenAI connection in Neo4j using stored credentials."""
    logger.info("Configuring OpenAI connection in Neo4j...")
    try:
        # Configure the OpenAI connection in Neo4j
        with driver.session() as session:
            session.run(
                """
            CALL genai.config.openai.set({
              apiKey: $api_key,
              embeddingModel: 'text-embedding-3-small'
            })
            """,
                {"api_key": OPENAI_API_KEY},
            )
        logger.info("OpenAI connection configured successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to configure OpenAI connection: {e}")
        return False


def verify_nodes_exist():
    """Verify that AtomicUnit nodes exist in the database."""
    logger.info("Verifying AtomicUnit nodes...")
    try:
        with driver.session() as session:
            result = session.run("""
            MATCH (n:AtomicUnit)
            RETURN count(n) AS count
            """)
            count = result.single()["count"]
            logger.info(f"Found {count} AtomicUnit nodes in the database")
            return count > 0
    except Exception as e:
        logger.error(f"Error verifying nodes: {e}")
        return False


def ensure_atomic_unit_label():
    """Ensure that all content nodes have the AtomicUnit label."""
    logger.info("Ensuring all content nodes have the AtomicUnit label...")
    try:
        with driver.session() as session:
            result = session.run("""
            MATCH (n:Introduction|Definition|Corollary|Theorem|Lemma|Proof|Example|Exercise|Remark)
            WHERE NOT n:AtomicUnit
            WITH count(n) AS missingLabel
            MATCH (n:Introduction|Definition|Corollary|Theorem|Lemma|Proof|Example|Exercise|Remark)
            WHERE NOT n:AtomicUnit
            SET n:AtomicUnit
            RETURN missingLabel, count(n) AS updated
            """)
            record = result.single()
            if record and record["missingLabel"] > 0:
                logger.info(f"Added AtomicUnit label to {record['updated']} nodes")
            else:
                logger.info("All content nodes already have the AtomicUnit label")
            return True
    except Exception as e:
        logger.error(f"Error ensuring AtomicUnit label: {e}")
        return False


def add_embeddings_batch(batch_size=20):
    """
    Add embeddings to AtomicUnit nodes using Neo4j's GenAI batch encoding with the encodeBatch function.
    This implementation uses the efficient batch processing approach.

    Args:
        batch_size: Number of nodes to process in each batch
    """
    logger.info(f"Adding embeddings to AtomicUnit nodes in batches of {batch_size}...")

    try:
        # Count nodes without embeddings
        with driver.session() as session:
            count_result = session.run("""
            MATCH (n:AtomicUnit)
            WHERE n.text IS NOT NULL AND n.textEmbedding IS NULL
            RETURN count(n) as count
            """)
            node_count = count_result.single()["count"]

        logger.info(f"Found {node_count} nodes without embeddings")

        if node_count == 0:
            logger.info("No nodes need embeddings. Skipping.")
            return True

        # Process in batches
        processed = 0
        start_time = time.time()

        while processed < node_count:
            batch_start_time = time.time()

            # Use the efficient single-query batch processing approach
            try:
                with driver.session() as session:
                    # Process a batch of nodes with the efficient approach
                    update_result = session.run(
                        """
                        MATCH (n:AtomicUnit WHERE n.text IS NOT NULL AND n.textEmbedding IS NULL)
                        WITH n
                        ORDER BY size(n.text) ASC  // Process shorter texts first
                        LIMIT $batch_size
                        WITH collect(n) AS nodesList
                        WITH nodesList, [node IN nodesList | node.text] AS batch
                        CALL genai.vector.encodeBatch(batch, 'OpenAI', { token: $token }) YIELD index, vector
                        WITH nodesList, index, vector
                        CALL db.create.setNodeVectorProperty(nodesList[index], 'textEmbedding', vector)
                        WITH count(*) as updated, size(nodesList) as batch_size
                        RETURN updated, batch_size
                        """,
                        {"batch_size": batch_size, "token": OPENAI_API_KEY},
                    )

                    result = update_result.single()
                    batch_size_actual = result["batch_size"]

                    # Exit loop if no nodes were found/processed
                    if batch_size_actual == 0:
                        break

                    processed += batch_size_actual

                    batch_duration = time.time() - batch_start_time
                    logger.info(
                        f"Processed {batch_size_actual} nodes in {batch_duration:.2f}s"
                    )
                    logger.info(
                        f"Progress: {processed}/{node_count} nodes ({(processed / node_count) * 100:.1f}%)"
                    )

                    # Add a short delay to avoid rate limiting
                    if batch_size > 10:
                        time.sleep(1)

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # If batch fails, try with a smaller batch size
                if batch_size > 5:
                    batch_size = batch_size // 2
                    logger.info(f"Reducing batch size to {batch_size}")
                else:
                    raise

        total_duration = time.time() - start_time
        logger.info(
            f"Successfully added embeddings to {processed} nodes in {total_duration:.2f}s"
        )

        # Verify all nodes have embeddings now
        with driver.session() as session:
            verification_result = session.run("""
            MATCH (n:AtomicUnit)
            WHERE n.text IS NOT NULL AND n.textEmbedding IS NULL
            RETURN count(n) as remaining
            """)
            remaining = verification_result.single()["remaining"]

        if remaining > 0:
            logger.warning(f"There are still {remaining} nodes without embeddings")
        else:
            logger.info("All nodes have been successfully embedded")

        return True
    except Exception as e:
        logger.error(f"Failed to add embeddings: {e}")
        return False


def test_vector_search(query="Topologie"):
    """Test vector search with a sample query using LangChain for the embedding."""
    # Add extra debug logging for the query
    logger.info(f"Raw query input: '{query}'")
    logger.info(f"Query type: {type(query)}")
    logger.info(f"Query length: {len(query)}")

    # For debugging only, show exact character codes for the first 20 chars
    if len(query) > 0:
        char_codes = ", ".join(f"{c}({ord(c)})" for c in query[:20])
        logger.info(f"First 20 characters: {char_codes}")

    try:
        # Import OpenAI embeddings here to avoid circular imports
        from langchain_openai import OpenAIEmbeddings

        # Initialize embedding model
        embedding_model = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"
        )

        # Generate embedding for the query using LangChain
        query_embedding = embedding_model.embed_query(query)

        # Perform vector search
        with driver.session() as session:
            search_result = session.run(
                """
            CALL db.index.vector.queryNodes(
              'vector_index_AtomicUnit',
              5,
              $embedding
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
                {"embedding": query_embedding},
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
        logger.error(f"Error testing vector search: {e}")
        return False


def main():
    """Main function to add embeddings to nodes."""
    import argparse
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Add embeddings to Neo4j nodes")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for processing nodes (default: 20)",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Test vector search after adding embeddings (requires that indexes have been created with create_indexes.py)",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default="Was ist ein topologischer Raum?",
        help="Query to use for testing (simple queries only)",
    )
    parser.add_argument(
        "-f",
        "--query-file",
        type=str,
        help="Path to a file containing the query text (recommended for LaTeX expressions)",
    )

    args = parser.parse_args()

    # Handle query from file if provided
    query = args.query
    if args.query_file:
        try:
            with open(args.query_file, "r") as f:
                query = f.read().strip()
            logger.info(f"Read query from file: {args.query_file}")
        except Exception as e:
            logger.error(f"Error reading query file: {e}")
            sys.exit(1)

    # Debug logging for query
    if args.test:
        logger.info(f"Query for testing: '{query}'")

    logger.info("Starting embedding generation process...")

    # Verify Neo4j connection and nodes
    if not verify_nodes_exist():
        logger.error(
            "No AtomicUnit nodes found. Make sure your graph is properly populated."
        )
        return

    # Ensure all content nodes have the AtomicUnit label
    ensure_atomic_unit_label()

    # Add embeddings to nodes
    add_embeddings_batch(batch_size=args.batch_size)

    # Note: Index creation has been moved to create_indexes.py
    logger.info(
        "Embeddings added successfully. Run create_indexes.py to create search indexes."
    )

    # Test vector search if requested
    if args.test:
        # Log the final query that will be sent
        logger.info(f"Sending to test_vector_search: '{query}'")
        test_vector_search(query)

    logger.info("Process completed.")


if __name__ == "__main__":
    main()
