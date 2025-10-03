"""Agent system module for RAG-based mathematical question answering."""

from math_rag.rag_agents.agents import setup_rag_chat
from math_rag.rag_agents.dspy_agent import setup_dspy_rag_chat
from math_rag.rag_agents.langgraph_agent import setup_langgraph_rag_chat

__all__ = ["setup_rag_chat", "setup_dspy_rag_chat", "setup_langgraph_rag_chat"]
