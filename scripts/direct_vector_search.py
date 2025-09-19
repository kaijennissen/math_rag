#!/usr/bin/env python
"""
Direct Neo4j vector search using Cypher queries.
"""

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
INDEX_NAME = "vector_index_AtomicUnit"

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize embedding model
embedding_model = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
)

# Search query - modify this to test different queries
query_text = "Wie lautet die Definition für einen T_4-Raum?"


def get_node_properties():
    """Get the properties of a sample node to understand the schema."""
    cypher_query = """
    MATCH (n)
    WHERE n.text IS NOT NULL
    RETURN n LIMIT 1
    """

    with driver.session() as session:
        result = session.run(cypher_query)
        for record in result:
            node = record["n"]
            print("Node properties:")
            for key in node.keys():
                print(f"  - {key}: {type(node[key]).__name__}")
            return node.keys()
    return []


def check_vector_index():
    """Check if the vector index exists and what properties it has."""
    cypher_query = """
    SHOW INDEXES
    YIELD name, type, labelsOrTypes, properties
    WHERE type = 'VECTOR' AND name = $index_name
    RETURN name, labelsOrTypes, properties
    """

    with driver.session() as session:
        result = session.run(cypher_query, index_name=INDEX_NAME)
        for record in result:
            print(f"Vector Index: {record['name']}")
            print(f"Labels: {record['labelsOrTypes']}")
            print(f"Properties: {record['properties']}")
            return True
    print(f"Index '{INDEX_NAME}' not found!")
    return False


def run_vector_search(query_text, k=10):
    """
    Run a direct vector search using Cypher.

    Args:
        query_text: The text query to search for
        k: Number of results to return

    Returns:
        List of search results
    """
    # Check if index exists
    print("Checking vector index...")
    check_vector_index()

    # Generate embedding for the query
    print("Generating embedding...")
    query_embedding = embedding_model.embed_query(query_text)

    # Create a simpler Cypher query with only guaranteed properties
    cypher_query = f"""
    CALL db.index.vector.queryNodes(
      '{INDEX_NAME}',
      {k},
      $queryVector
    )
    YIELD node, score
    RETURN
      score,
      node.text AS text
    ORDER BY score DESC
    """

    print("Executing vector search query...")
    with driver.session() as session:
        result = session.run(cypher_query, queryVector=query_embedding)
        return [record.data() for record in result]


def main():
    print(f"Running vector search for query: {query_text}")
    print("-" * 80)

    results = run_vector_search(query_text)

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} results\n")

    for i, result in enumerate(results, 1):
        print(f"Result {i} (Score: {result['score']:.4f}):")

        # Print text content
        print("\nText:")
        print(result["text"])

        print("-" * 80)


if __name__ == "__main__":
    main()
