import logging

import streamlit as st

from rag_chat.rag_chatbot import create_rag_chatbot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


st.set_page_config(layout="wide", page_title="RAG Chatbot", page_icon="🚀")


def chatbot_page():
    st.title("🤖 Chatbot 📝")

    if "chatbot" not in st.session_state:
        chatbot = create_rag_chatbot()
        st.session_state["chatbot"] = chatbot
    else:
        chatbot = st.session_state.get("chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if my_question := st.chat_input(
        "Ask me a question about your data",
    ):
        with st.spinner("Answering your question..."):
            # Running the graph with the user's question
            inputs = {"question": my_question}
            result = None
            try:
                result = chatbot.invoke(inputs, stream_mode="values")
                with st.chat_message("assistant"):
                    st.markdown(result["generation"])

            except Exception as e:
                logging.error(f"Error occurred: {e}")
                with st.chat_message("assistant"):
                    st.markdown(
                        "An error occurred while processing your request. Please try again later."
                    )


if __name__ == "__main__":
    chatbot_page()
