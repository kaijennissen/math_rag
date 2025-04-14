# RAG Pipeline Tutorial App

This Streamlit application provides an interactive tutorial on building RAG (Retrieval Augmented Generation) pipelines.

## Features

- **Home**: Introduction to RAG concepts
- **Chunking**: Interactive demonstration of different text chunking methods
- **Embeddings**: Overview of embedding models (placeholder)
- **Retrievers**: Overview of retrieval methods (placeholder)

## Running the App

1. Install dependencies:
   ```
   pip install -r requirements/dev.txt
   ```

2. Run the Streamlit app:
   ```
   cd /path/to/rag_chat
   streamlit run src/rag_tutorial_app.py
   ```

3. The app will open in your browser at http://localhost:8501

## App Structure

- `rag_tutorial_app.py`: Main application entry point
- `pages/1_Chunking.py`: Document chunking page with interactive demo
- `pages/2_Embeddings.py`: Information about embedding models (placeholder)
- `pages/3_Retrievers.py`: Information about retrieval methods (placeholder)

## How to Use

1. Navigate between pages using the sidebar
2. On the Chunking page:
   - Select a text splitter type
   - Adjust parameters
   - Paste your text
   - Click "Split Text" to see the results
