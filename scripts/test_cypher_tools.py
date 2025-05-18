#!/usr/bin/env python3
"""
Test script for the Cypher tools and meta-agent.
"""

import logging
import coloredlogs
from math_rag.cypher_tools import (
    SchemaInfoTool,
    CypherQueryGeneratorTool,
    CypherExecutorTool,
)
from math_rag.graph_meta_agent import query_graph_structure

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def test_individual_tools():
    """Test each Cypher tool individually."""
    print("\n===== Testing Individual Cypher Tools =====\n")

    # Test schema info tool
    print("Testing SchemaInfoTool...")
    schema_tool = SchemaInfoTool()
    schema_info = schema_tool.forward()
    print(f"Schema Info:\n{schema_info}\n")

    # Test query generator tool
    print("Testing CypherQueryGeneratorTool...")
    query_tool = CypherQueryGeneratorTool()
    question = "How many Definition nodes are in the graph?"
    generated_query = query_tool.forward(question=question, schema_info=schema_info)
    print(f"Question: {question}")
    print(f"Generated Query: {generated_query}\n")

    # Test executor tool with the generated query
    print("Testing CypherExecutorTool...")
    executor_tool = CypherExecutorTool()
    results = executor_tool.forward(query=generated_query)
    print(f"Query Results:\n{results}\n")


def test_meta_agent():
    """Test the graph meta agent with different questions."""
    print("\n===== Testing Graph Meta Agent =====\n")

    # Test simple count question
    question = "How many Definition nodes are in the graph?"
    print(f"Question: {question}")
    response = query_graph_structure(question)
    print(f"Response:\n{response}\n")

    # Test distribution question
    question = "What's the distribution of different node types in the graph?"
    print(f"Question: {question}")
    response = query_graph_structure(question)
    print(f"Response:\n{response}\n")

    # Test relationship question
    question = "Show me the most connected nodes in the graph"
    print(f"Question: {question}")
    response = query_graph_structure(question)
    print(f"Response:\n{response}\n")

    # Test schema question
    question = "What node labels and relationship types exist in the graph?"
    print(f"Question: {question}")
    response = query_graph_structure(question)
    print(f"Response:\n{response}\n")


if __name__ == "__main__":
    print("Testing Cypher tools and meta-agent...")

    try:
        # First test individual tools
        test_individual_tools()

        # Then test the meta agent
        test_meta_agent()

    except Exception as e:
        logger.error(f"Error during testing: {e}")
