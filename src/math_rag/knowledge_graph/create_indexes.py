"""
Standalone script to create and manage Neo4j indexes (vector and fulltext)
for efficient searching in the knowledge graph.

This script handles:
1. Creating the AtomicUnit label on all content nodes
2. Creating vector indexes for embedding-based similarity search
3. Creating fulltext indexes for keyword search
4. Testing the indexes with sample queries

Workflow for building the knowledge graph and enabling search:

1. Build graph structure:
   python -m src/math_rag/build_knowledge_graph

2. Add embeddings to nodes:
   python add_embeddings.py

3. Create indexes for search:
   python create_indexes.py --all --test

Each script can be run independently as needed:
- To recreate just the vector index: python create_indexes.py --vector
- To recreate just the fulltext index: python create_indexes.py --fulltext
- To test search without recreating indexes: python create_indexes.py --test
"""

import os
import logging
import argparse
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("create_indexes.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Direct Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def ensure_atomic_unit_label():
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


def verify_vector_property_exists(label: str, property_name: str) -> bool:
    """Verify that nodes have the required vector property."""
    try:
        logger.info(f"Verifying {property_name} property exists on {label} nodes...")
        with driver.session() as session:
            result = session.run(
                f"""
            MATCH (n:{label})
            WHERE n.{property_name} IS NOT NULL
            RETURN count(n) as count
            """
            )
            node_count = result.single()["count"]

        if node_count == 0:
            logger.warning(f"No {label} nodes with {property_name} property found.")
            return False
        else:
            logger.info(
                f"Found {node_count} {label} nodes with {property_name} property."
            )
            return True
    except Exception as e:
        logger.error(f"Error verifying {property_name} property: {e}")
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


def create_vector_index(
    label: str = "AtomicUnit",
    property_name: str = "textEmbedding",
    dimensions: int = 1536,
    similarity_function: str = "cosine",
    index_name: str = None,
) -> bool:
    """Create or recreate a vector index for specified nodes and property.

    Args:
        label: Node label to index (default: AtomicUnit)
        property_name: Property containing vector embeddings (default: textEmbedding)
        dimensions: Vector dimensions (default: 1536)
        similarity_function: Similarity function (default: cosine)
        index_name: Custom index name (default: vector_index_{label})

    Returns:
        True if successful, False otherwise
    """
    if index_name is None:
        index_name = f"vector_index_{label}"

    # Verify nodes have the required property
    if not verify_vector_property_exists(label, property_name):
        return False

    # Drop existing index
    if not drop_index_if_exists(index_name):
        logger.warning("Failed to drop existing index, but continuing...")

    # Create new vector index
    try:
        logger.info(f"Creating vector index {index_name} on {label}.{property_name}...")
        with driver.session() as session:
            session.run(
                f"""
            CREATE VECTOR INDEX {index_name}
            FOR (n:{label}) ON n.{property_name}
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: '{similarity_function}'
                }}
            }}
            """
            )
        logger.info(f"Successfully created vector index {index_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create vector index {index_name}: {e}")
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


def test_vector_search(query="Topologie"):
    """Test vector search with a sample query using LangChain for the embedding."""
    # Add extra debug logging for the query
    logger.info(f"Testing vector search with query: '{query}'")

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
    vector: bool = False,
    fulltext: bool = False,
    all_indexes: bool = False,
    test: bool = False,
    vector_query: str = "",
    fulltext_query: str = "",
    label: str = "AtomicUnit",
    vector_property: str = "textEmbedding",
    fulltext_properties: list = None,
):
    """Main function to create and test indexes."""

    # Ensure AtomicUnit label
    logger.info("Ensuring AtomicUnit label...")
    ensure_atomic_unit_label()

    # Set default fulltext properties if not provided
    if fulltext_properties is None:
        fulltext_properties = ["text", "title", "proof"]

    # Determine what to create
    create_vector = vector or all_indexes
    create_fulltext = fulltext or all_indexes

    # Create vector index if requested
    if create_vector:
        logger.info(f"Creating vector index for {label}.{vector_property}...")
        success = create_vector_index(label=label, property_name=vector_property)
        if success:
            logger.info("Vector index created successfully.")
            if test and vector_query:
                test_vector_search(vector_query)
        else:
            logger.error("Failed to create vector index.")

    # Create fulltext index if requested
    if create_fulltext:
        logger.info(
            f"Creating fulltext index for {label} on properties {fulltext_properties}..."
        )
        success = create_fulltext_index(label=label, properties=fulltext_properties)
        if success:
            logger.info("Fulltext index created successfully.")
            if test and fulltext_query:
                test_fulltext_search(fulltext_query)
        else:
            logger.error("Failed to create fulltext index.")

    # If only testing was requested
    if test and not (create_vector or create_fulltext):
        if vector_query:
            test_vector_search(vector_query)
        if fulltext_query:
            test_fulltext_search(fulltext_query)

    logger.info("Index operations completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and manage Neo4j indexes")
    parser.add_argument(
        "--vector",
        action="store_true",
        help="Create vector index for similarity search",
    )
    parser.add_argument(
        "--fulltext",
        action="store_true",
        help="Create fulltext index for keyword search",
    )
    parser.add_argument(
        "--all", action="store_true", help="Create both vector and fulltext indexes"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test indexes after creation"
    )
    parser.add_argument(
        "--vector-query",
        type=str,
        default="Was ist die definition eines T_{4}-Raums?",
        help="Query for testing vector search",
    )
    parser.add_argument(
        "--fulltext-query",
        type=str,
        default="topologisch",
        help="Query for testing fulltext search",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="AtomicUnit",
        help="Node label to create indexes for (default: AtomicUnit)",
    )
    parser.add_argument(
        "--vector-property",
        type=str,
        default="textEmbedding",
        help="Property name for vector index (default: textEmbedding)",
    )
    parser.add_argument(
        "--fulltext-properties",
        nargs="+",
        default=["text", "title", "proof", "summary"],
        help="Properties for fulltext index (default: text title proof summary)",
    )

    args = parser.parse_args()

    main(
        vector=args.vector,
        fulltext=args.fulltext,
        all_indexes=args.all,
        test=args.test,
        vector_query=args.vector_query,
        fulltext_query=args.fulltext_query,
        label=args.label,
        vector_property=args.vector_property,
        fulltext_properties=args.fulltext_properties,
    )
