import argparse
import glob
import logging
import os
import pprint
import random
from typing import Dict, List, TypedDict

import pandas as pd
import rich
import streamlit as st
from dotenv import load_dotenv

from rag_chatbot import create_rag_chatbot

st.set_page_config(layout="wide", page_title="RAG Chatbot", page_icon="🚀")


def chatbot_page():

    chatbot = create_rag_chatbot()

    st.title("Chatbot")
    user_input = st.text_area("Ask a question:", height=100)

    if user_input:
        with st.spinner("Answering your question..."):
            # Running the graph with the user's question
            inputs = {"question": user_input}

            try:
                for output in chatbot.stream(inputs):
                    for key, value in output.items():
                        if "generation" in value:
                            result = value["generation"]

            except Exception as e:
                st.error(f"Error occurred: {e}")

        if result:
            st.write(f"\nAnswer: {result}\n")
        else:
            print("\nSorry, I couldn't find an answer to that question.\n")


if __name__ == "__main__":
    chatbot_page()
