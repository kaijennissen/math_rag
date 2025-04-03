import streamlit as st
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
import tiktoken

# Page configuration
st.set_page_config(
    page_title="Chunking - RAG Tutorial",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Chunking")

# Select chunker type
chunker_type = st.selectbox(
    "Select Text Splitter",
    ["CharacterTextSplitter", "RecursiveCharacterTextSplitter", "TokenTextSplitter"],
)

# Parameters section based on selected splitter
with st.expander("Chunker Parameters"):
    if chunker_type == "CharacterTextSplitter":
        chunk_size = st.slider("Chunk Size", 100, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
        separator = st.text_input("Separator", "\n\n")

    elif chunker_type == "RecursiveCharacterTextSplitter":
        chunk_size = st.slider("Chunk Size", 100, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
        separators = st.text_area("Separators (one per line)", "\n\n\n\n\n").split("\n")

    elif chunker_type == "TokenTextSplitter":
        chunk_size = st.slider("Chunk Size (tokens)", 100, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap (tokens)", 0, 500, 200)
        encoding_name = st.selectbox(
            "Encoding", ["cl100k_base", "p50k_base", "r50k_base"]
        )

# Explanations section
with st.expander("How it works"):
    if chunker_type == "CharacterTextSplitter":
        st.markdown("""
        **CharacterTextSplitter** splits text based on a character separator. It's the simplest form of chunking.

        **Parameters:**
        - **Chunk Size**: Maximum size of each chunk in characters
        - **Chunk Overlap**: Number of characters to overlap between chunks
        - **Separator**: Character(s) to split on

        **Best used when**: You have text with natural separators like paragraphs or line breaks.
        """)

    elif chunker_type == "RecursiveCharacterTextSplitter":
        st.markdown("""
        **RecursiveCharacterTextSplitter** splits text by trying a list of separators in order.

        **Parameters:**
        - **Chunk Size**: Maximum size of each chunk in characters
        - **Chunk Overlap**: Number of characters to overlap between chunks
        - **Separators**: List of separators to try in order (e.g., ["\\n\\n", "\\n", ". ", " "])

        **Best used when**: You want more control over how text is split, preserving semantic structure.
        This splitter first tries to split on double newlines, then single newlines, and so on.
        """)

    elif chunker_type == "TokenTextSplitter":
        st.markdown("""
        **TokenTextSplitter** splits text based on token count rather than characters.

        **Parameters:**
        - **Chunk Size**: Maximum size of each chunk in tokens
        - **Chunk Overlap**: Number of tokens to overlap between chunks
        - **Encoding**: The tokenizer encoding to use

        **Best used when**: You want to ensure chunks stay within token limits for your model.
        """)

# Text input
st.subheader("Try it out")
input_text = st.text_area(
    "Paste your text here",
    height=200,
    value="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n\nUt enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.\n\nDuis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.\n\nExcepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
)

# Process and display chunks
if st.button("Split Text"):
    try:
        chunks = []

        if chunker_type == "CharacterTextSplitter":
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator
            )
            chunks = splitter.split_text(input_text)

        elif chunker_type == "RecursiveCharacterTextSplitter":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
            chunks = splitter.split_text(input_text)

        elif chunker_type == "TokenTextSplitter":
            splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name=encoding_name,
            )
            chunks = splitter.split_text(input_text)

        # Display chunks with stats
        st.subheader(f"Output: {len(chunks)} chunks")

        encoding = tiktoken.get_encoding("cl100k_base")

        for i, chunk in enumerate(chunks):
            with st.expander(f"Chunk {i + 1}"):
                st.text(chunk)
                token_count = len(encoding.encode(chunk))
                st.caption(f"Characters: {len(chunk)} | Tokens: {token_count}")

    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
