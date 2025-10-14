"""All UI components for the Streamlit Math-RAG interface."""

from typing import Dict, List

import streamlit as st
from smolagents import CodeAgent


def render_info_section():
    """Render project information and system architecture."""
    st.title("🧮 Math-RAG: Mathematical Document Q&A")

    with st.expander("ℹ️ What is Math-RAG?", expanded=False):
        st.markdown(
            """
        **Math-RAG** is a Knowledge Graph-based Question-Answering system for
        mathematical documents.

        **What you can ask:**
        - Definitions of mathematical concepts
        - Explanations of theorems and proofs
        - Relationships between mathematical entities
        - References to specific theorems or definitions

        **Example Questions:**
        - "What is the definition of a topological space?"
        - "Which theorems depend on the axiom of choice?"
        - "Show me all definitions related to continuity"
        """
        )

    with st.expander("🏗️ System Architecture", expanded=False):
        st.markdown("**Multi-Agent System:**")
        st.code(
            """
                         +-----------------+
                         |  Manager Agent  |
                         |  (graph_agent)  |
                         +-----------------+
                                 |
                  _______________|______________
                 |                              |
      +-----------------------+              +----------------+
      | graph_retriever_agent |              |  cypher_agent  |
      +-----------------------+              +----------------+
                  |                             |           |
         GraphRetrieverTool          CypherExecutorTool     |
                                                       SchemaInfoTool
        """,
            language="text",
        )


def initialize_chat_history() -> List[Dict[str, str]]:
    """Initialize chat history in session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "👋 Welcome! Ask me about mathematical concepts,"
                    "definitions, or theorems."
                ),
            }
        ]
    return st.session_state.chat_history


def render_chat_interface(agent: CodeAgent):
    """Render the main chat interface."""
    # Clear chat button
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

    # Display chat history
    chat_history = initialize_chat_history()
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask a mathematical question..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get and display assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner("🤔 Thinking..."):
                    response = agent.run(user_input)
                st.markdown(response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                error_msg = f"Error processing your question: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )

        st.rerun()
