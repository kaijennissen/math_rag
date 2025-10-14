"""
Streamlit Math-RAG Chat Interface - POC Version

Simple web interface for the Math-RAG system.
"""

import streamlit as st
from components import render_chat_interface, render_info_section

from math_rag.config import RagChatSettings, settings_provider
from math_rag.rag_agents.agents import setup_rag_chat


@st.cache_resource
def initialize_agent():
    """Initialize the Math-RAG agent system."""
    try:
        settings = settings_provider.get_settings(RagChatSettings)
        return setup_rag_chat(
            openai_api_key=settings.openai_api_key,
            neo4j_uri=settings.neo4j_uri,
            neo4j_username=settings.neo4j_username,
            neo4j_password=settings.neo4j_password,
            neo4j_database=settings.neo4j_database,
            agent_config_path=settings.agent_config_path,
            model_id=settings.model_id,
            api_base=settings.api_base,
            huggingface_api_key=settings.huggingface_api_key,
        )
    except Exception as e:
        st.error(f"Failed to initialize Math-RAG system: {e}")
        return None, None


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Math-RAG Chat", page_icon="🧮", layout="wide")

    # Info section
    render_info_section()

    # Initialize agent
    agent, mcp_client = initialize_agent()

    if agent is None:
        st.error("❌ System initialization failed. Check your .env configuration.")
        st.stop()

    st.success("✅ Math-RAG system ready!")
    try:
        # Chat interface
        render_chat_interface(agent)
    finally:
        mcp_client.disconnect()


if __name__ == "__main__":
    main()
