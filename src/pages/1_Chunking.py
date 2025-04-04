import streamlit as st
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# from llama_index.core.node_parser import (
#     MarkdownNodeParser,
#     SentenceSplitter,
#     SentenceWindowNodeParser,
# )
import tiktoken

# Page configuration
st.set_page_config(
    page_title="Chunking - RAG Tutorial",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Chunking")
default_text = """
# Title: The Wonders of Procrastination

##Introduction: Procrastination is often seen as the enemy of productivity, but let's be honest—it's an art form. Why rush to complete tasks when you can savor the sweet thrill of doing absolutely nothing? After all, deadlines are just suggestions, right?

## Section 1: The Benefits of Procrastination

Enhanced Creativity: When you procrastinate, your mind is free to wander and explore new ideas. Who knows? You might come up with the next big innovation while avoiding that report.
Stress Management: By putting off tasks, you avoid the immediate stress of tackling them. Sure, the stress will come back tenfold later, but that's a problem for future you.
Improved Decision Making: Taking your time means you can weigh all your options carefully. Or, you might just forget about the decision altogether—problem solved!

## Section 2: Techniques for Effective Procrastination

The Art of Distraction: Master the skill of finding anything and everything to do except the task at hand. Cleaning your room, organizing your files, or even binge-watching a new series can be surprisingly productive.
The Last-Minute Rush: Embrace the adrenaline rush that comes with completing tasks at the eleventh hour. It's amazing how much you can accomplish when you're racing against the clock.
The "I'll Do It Tomorrow" Strategy: Perfect the habit of convincing yourself that tomorrow is the ideal day to start. Spoiler alert: tomorrow never comes.

## Section 3: Real-Life Applications

Workplace: Procrastination can lead to unexpected team bonding moments as everyone scrambles to meet the deadline together.
Education: Students have long known the secret power of procrastination. Those late-night study sessions can be surprisingly effective (or at least memorable).
Personal Life: Why tackle chores immediately when you can enjoy a leisurely day and deal with them later? Life is too short to be constantly productive.

## Conclusion: Procrastination may not be the most efficient way to get things done, but it certainly adds a touch of excitement to the mundane. So, the next time you find yourself putting off a task, remember—you're not lazy, you're just embracing the art of procrastination.
"""

parrot_icon = "\U0001f99c"  # 🦜
llama_icon = "\U0001f999"  # 🦙

# Select chunker type
splitter_type = st.selectbox(
    "Select Text Splitter",
    [
        f"{parrot_icon} CharacterTextSplitter",
        f"{parrot_icon} RecursiveCharacterTextSplitter",
        f"{parrot_icon} TokenTextSplitter",
        # f"{llama_icon} MarkdownNodeParser",
        # f"{llama_icon} SentenceWindowNodeParser",
    ],
)

# Create a dictionary mapping displayed names to internal chunker types
SPLITTER_MAPPING = {
    f"{parrot_icon} CharacterTextSplitter": "CharacterTextSplitter",
    f"{parrot_icon} RecursiveCharacterTextSplitter": "RecursiveCharacterTextSplitter",
    f"{parrot_icon} TokenTextSplitter": "TokenTextSplitter",
    # f"{llama_icon} MarkdownNodeParser": "MarkdownNodeParser",
    # f"{llama_icon} SentenceWindowNodeParser": "SentenceWindowNodeParser",
}


# Fetch the internal chunker type from the dictionary
splitter_type = SPLITTER_MAPPING[splitter_type]
# langgraph_splitter = True if parrot_icon in splitter_type else False
# llama_index_splitter = not langgraph_splitter

# Parameters section based on selected splitter
with st.expander("Chunker Parameters", expanded=True):
    separators_options = [
        ("\\n\\n", "\n\n"),  # Double newline
        ("\\n", "\n"),  # Single newline
        (".", "."),  # Period
        (",", ","),  # Comma
        ("' '", " "),  # Space
    ]
    if splitter_type == "CharacterTextSplitter":
        chunk_size = st.slider("Chunk Size", 100, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
        selected_separators = st.selectbox(
            "Separator",
            [option[0] for option in separators_options],
            index=0,
        )
        separators = [
            actual
            for display, actual in separators_options
            if display == selected_separators
        ][0]

    elif splitter_type == "RecursiveCharacterTextSplitter":
        chunk_size = st.slider("Chunk Size", 100, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
        selected_separators = st.multiselect(
            "Separators (in order of priority)",
            [option[0] for option in separators_options],
            default=["\\n\\n"],
        )
        separators = [
            next(actual for display, actual in separators_options if display == sep)
            for sep in selected_separators
        ]

    elif splitter_type == "TokenTextSplitter":
        chunk_size = st.slider("Chunk Size (tokens)", 100, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap (tokens)", 0, 500, 200)
        encoding_name = st.selectbox(
            "Encoding", ["cl100k_base", "p50k_base", "r50k_base"]
        )

# Explanations section
with st.expander("How it works"):
    if splitter_type == "CharacterTextSplitter":
        st.markdown(
            """
        **CharacterTextSplitter** splits text based on a character separator. It's the simplest form of chunking.

        **Parameters:**
        - **Chunk Size**: Maximum size of each chunk in characters
        - **Chunk Overlap**: Number of characters to overlap between chunks
        - **Separator**: Character(s) to split on

        **Best used when**: You have text with natural separators like paragraphs or line breaks.
        """
        )

    elif splitter_type == "RecursiveCharacterTextSplitter":
        st.markdown(
            """
        **RecursiveCharacterTextSplitter** splits text by trying a list of separators in order.

        **Parameters:**
        - **Chunk Size**: Maximum size of each chunk in characters
        - **Chunk Overlap**: Number of characters to overlap between chunks
        - **Separators**: List of separators to try in order (e.g., ["\\n\\n", "\\n", ". ", " "])

        **Best used when**: You want more control over how text is split, preserving semantic structure.
        This splitter first tries to split on double newlines, then single newlines, and so on.
        """
        )

    elif splitter_type == "TokenTextSplitter":
        st.markdown(
            """
        **TokenTextSplitter** splits text based on token count rather than characters.

        **Parameters:**
        - **Chunk Size**: Maximum size of each chunk in tokens
        - **Chunk Overlap**: Number of tokens to overlap between chunks
        - **Encoding**: The tokenizer encoding to use

        **Best used when**: You want to ensure chunks stay within token limits for your model.
        """
        )

# Text input
st.subheader("Try it out")
input_text = st.text_area(
    "Paste your text here",
    height=500,
    value=default_text,
)

# Process and display chunks
if st.button("Split Text"):
    try:
        if splitter_type == "CharacterTextSplitter":
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separators,
            )

        elif splitter_type == "RecursiveCharacterTextSplitter":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )

        elif splitter_type == "TokenTextSplitter":
            splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name=encoding_name,
            )
        chunks = splitter.split_text(input_text)

        # if llama_index_splitter:
        #     if splitter_type == "MarkdownNodeParser":
        #         splitter = MarkdownNodeParser()
        #     elif splitter_type == "SentenceSplitter":
        #         splitter = SentenceSplitter(
        #             chunk_size=chunk_size,
        #             chunk_overlap=chunk_overlap,
        #             separators=separators,
        #         )
        #     elif splitter_type == "SentenceWindowNodeParser":
        #         splitter = SentenceWindowNodeParser(
        #             chunk_size=chunk_size,
        #             chunk_overlap=chunk_overlap,
        #             separators=separators,
        #         )

        #     chunks = splitter.get_nodes_from_documents(input_text)
        # # Display chunks with stats

        encoding = tiktoken.get_encoding("cl100k_base")

        for i, chunk in enumerate(chunks, start=1):
            token_count = len(encoding.encode(chunk))
            st.caption(
                f"Chunk: {i}/{len(chunks)} | Characters: {len(chunk)} | Tokens: {token_count}"
            )
            st.text(chunk)

    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
