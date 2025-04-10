# 🧮 Math-RAG: Knowledge Graph-Based QA System for Mathematical Documents

## 🧭 Project Overview

This project implements a GraphRAG system for question-answering on mathematical documents. It combines the power of Knowledge Graphs with Retrieval-Augmented Generation to provide accurate answers to mathematical queries by capturing the hierarchical and relational structure of mathematical concepts.

Key features:
- 🧠 Knowledge Graph construction from mathematical PDFs with Neo4j
- 📊 Specialized processing for mathematical notation via MathPix API
- 🔄 Graph-enhanced retrieval combining structured and unstructured knowledge
- 🤖 Flexible LLM integration supporting OpenAI and local models via LangChain
- 🎯 Advanced query routing with multi-stage answer generation
- 📱 Streamlit web interface for chatting with your mathematical documents

## 🚧 Prerequisites

- Python 3.11+
- Neo4j database (can be run via Docker)
- API keys:
  - OpenAI API key (optional, can use local models)
  - MathPix API for processing mathematical notation in PDFs
- Docker and Docker Compose (for running Neo4j)

## 🎛 Project Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/math_rag.git
   cd math_rag
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements/dev.txt
   ```

4. Set up environment variables:
   Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key  # Optional if using local models
   MATHPIX_API_ID=your_mathpix_api_id
   MATHPIX_API_KEY=your_mathpix_api_key
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=password
   ```

5. Start the Neo4j database:
   ```
   docker-compose up -d
   ```

6. Process your mathematical documents:
   Place mathematical PDFs in the `./docs` folder and run the following workflows:

   ```
   # Complete workflow to create a knowledge graph
   python src/pdf_to_text.py path/to/your/document.pdf                # Step 1: Parse PDF with MathPix
   python src/section_splitter.py --input docs/processed/document.pkl  # Step 2: Split document into major sections
   python src/subsection_splitter.py --section 5                       # Step 3: Split sections into subsections
   python src/extract_atomic_units.py --section 5                      # Step 4: Extract definitions, theorems, etc. with LLM
   python src/build_knowledge_graph.py                                 # Step 5: Create the knowledge graph
   ```

   This workflow:
   - Processes PDFs using MathPix to preserve mathematical notation (Step 1)
   - Extracts and splits the document into major sections (Step 2)
   - Further splits sections into meaningful subsections (Step 3)
   - Uses LLM to identify atomic units like definitions and theorems (Step 4)
   - Creates a structured knowledge graph in Neo4j (Step 5)

   **Detailed Usage for Each Tool**:
   ```
   # Process a single PDF file
   python src/pdf_to_text.py /absolute/path/to/your/document.pdf

   # Split document into major sections
   python src/section_splitter.py --input docs/processed/document.pkl                # Process all sections
   python src/section_splitter.py --input docs/processed/document.pkl --section 5    # Process specific section

   # Split sections into subsections
   python src/subsection_splitter.py --section 5                  # Process one section
   python src/subsection_splitter.py --section 5 --section 6      # Process multiple sections

   # Extract atomic units from sections or specific subsections
   python src/extract_atomic_units.py --section 5                # Process all subsections in section 5
   python src/extract_atomic_units.py --subsection 5.1           # Process just subsection 5.1
   python src/extract_atomic_units.py --section 5 --section 6    # Process all subsections in sections 5 and 6
   python src/extract_atomic_units.py --section 5 --subsection 6.1 # Combine specific sections and subsections
   ```

   The tools have these resilient features:
   - Page-by-page processing with checkpoints (pdf_to_text.py)
   - Automatic retries with exponential backoff for API failures
   - Saves intermediate results to enable resume capability
   - Robust error handling to skip problematic content

7. Launch the Streamlit interface:
   ```
   python src/app.py
   ```

## 📦 Project Structure
```
math_rag/
│
├── config/
│   └── config.yaml               # Configuration file
│
├── docs/                         # Folder for storing mathematical PDFs
│
├── src/
│   ├── app.py                    # Streamlit interface
│   ├── pdf_to_text.py            # Parse PDF with MathPix
│   ├── section_splitter.py       # Split document into major sections
│   ├── subsection_splitter.py    # Split sections into subsections
│   ├── extract_atomic_units.py   # Extract definitions/theorems using LLM
│   ├── build_knowledge_graph.py  # Create knowledge graph from atomic units
│   ├── graph_creation.py         # Graph functions
│   ├── graph_rag.py              # Generic Graph RAG implementation
│   ├── graph_rag_math.py         # Math-specific Graph RAG implementation
│   └── rag_chat/                 # Core RAG implementation
│       ├── __init__.py
│       ├── config.py             # Config loading and management
│       ├── document_processing.py # PDF loading and processing
│       ├── embeddings.py         # Vector embeddings
│       ├── llm_utils.py          # LLM utilities
│       ├── project_root.py       # Project path utilities
│       ├── prompts.py            # System prompts
│       ├── rag_chatbot.py        # LangGraph RAG implementation
│       └── retrievers.py         # Retrieval methods
│
├── notebooks/                    # Jupyter notebooks
│
├── docker-compose.yml            # Docker setup for Neo4j
├── Makefile                      # Build utilities
└── README.md                     # Project documentation
```

## 🔄 Knowledge Graph Structure

The system creates a knowledge graph that captures the following elements from mathematical documents:

- Hierarchical structure: Sections, subsections, subsubsections
- Mathematical entities: Theorems, definitions, lemmas, propositions
- Relationships: Dependencies between mathematical concepts
- Proofs: Connected to their corresponding theorems

This structure enables more sophisticated retrieval than traditional vector-based approaches, allowing the system to answer complex mathematical questions that require understanding of mathematical relationships.

## 🚀 Using the Chat Interface

After launching the Streamlit app, you can:

1. Ask questions about the mathematical content in your documents
2. The system will:
   - Route your question to the appropriate retrieval method
   - Use the knowledge graph to find relevant mathematical concepts
   - Generate a comprehensive answer with proper mathematical notation
   - Verify the answer against the source material to prevent hallucinations

## 🛠️ Customization

The system can be customized through the `config/config.yaml` file:

- `llm_model`: The LLM to use (defaults to "llama3.1:8b" for local models)
- `use_openai`: Set to `true` to use OpenAI models instead of local models
- `docs_path`: Path to the documents directory
- `top_k`: Number of documents to retrieve
- `score_threshold`: Minimum similarity score for retrieval
- `temperature`: LLM temperature (higher values = more creative responses)

## 📚 References

- [Neo4j Graph Database](https://neo4j.com/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [MathPix API](https://mathpix.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🏆 Conclusion

This Math-RAG system demonstrates the power of combining Knowledge Graphs with LLM-based RAG for mathematical question answering. Its graph-based approach captures the complex relationships between mathematical concepts, enabling more precise and comprehensive answers to specialized mathematical queries.

## 🤝 Contributions

Contributions are welcome! Please feel free to submit a Pull Request with improvements, bug fixes, or new features.
