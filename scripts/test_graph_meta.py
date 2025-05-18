#!/usr/bin/env python3
"""
Test file for the graph meta-questioning capability.
This demonstrates both direct use of the meta-agent and use via the main RAG agent.
"""

import logging
import coloredlogs
from math_rag.graph_meta_agent import query_graph_structure
from smolagents_rag import setup_rag_chat

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def test_direct_meta_agent():
    """Test direct use of the graph meta agent."""
    print("\n===== Testing Direct Meta Agent =====\n")

    # Test simple node count question
    question = "How many Definition nodes are in the graph?"
    print(f"Question: {question}")
    response = query_graph_structure(question)
    print(f"Response:\n{response}\n")

    # Test graph structure question
    question = "What node labels exist in the graph?"
    print(f"Question: {question}")
    response = query_graph_structure(question)
    print(f"Response:\n{response}\n")

    # Test relationship question
    question = "Show me the most connected nodes in the graph"
    print(f"Question: {question}")
    response = query_graph_structure(question)
    print(f"Response:\n{response}\n")


def test_main_agent_delegation():
    """Test the main agent's ability to delegate to the meta agent."""
    print("\n===== Testing Main Agent Delegation =====\n")

    # Set up the main RAG agent
    agent = setup_rag_chat()

    # Test meta-question
    question = "What's the distribution of different node types in the graph?"
    print(f"Question: {question}")
    response = agent.run(question)
    print(f"Response:\n{response}\n")

    # Test content question (should use graph_retriever)
    question = "What is a topological space?"
    print(f"Question: {question}")
    response = agent.run(question)
    print(f"Response:\n{response}\n")


if __name__ == "__main__":
    # Test the direct meta agent first
    test_direct_meta_agent()

    # Then test the main agent's delegation ability
    test_main_agent_delegation()
