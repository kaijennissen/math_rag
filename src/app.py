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

    if "chatbot" not in st.session_state:
        chatbot = create_rag_chatbot()
        st.session_state["chatbot"] = chatbot
    else:
        chatbot = st.session_state.get("chatbot")

    st.title("Chatbot")
    user_input = st.text_area("Ask a question:", height=100)

    if user_input:
        with st.spinner("Answering your question..."):
            # Running the graph with the user's question
            inputs = {"question": user_input}
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
