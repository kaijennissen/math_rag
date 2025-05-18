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

- Python 3.12+
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
   # Complete workflow to create a knowledge graph and search index
   python src/math_rag/data_processing/pdf_to_text.py path/to/your/document.pdf                      # Step 1: Parse PDF with MathPix
   python src/math_rag/data_processing/section_splitter.py --input docs/processed/document.pkl       # Step 2: Split document into sections
   python src/math_rag/data_processing/subsection_splitter.py --section 5                            # Step 3: Split sections into subsections
   python src/math_rag/data_processing/extract_atomic_units.py --section 5                           # Step 4: Extract definitions, theorems, etc. with LLM
   python src/math_rag/knowledge_graph/build_knowledge_graph.py                                      # Step 5: Create the knowledge graph and fulltext index
   python src/math_rag/embeddings/create_embeddings_and_vector_index.py --model "E5 Multilingual" --test  # Step 6: Create embeddings and vector index
   ```

   This workflow:
   - Processes PDFs using MathPix to preserve mathematical notation (Step 1)
   - Extracts and splits the document into major sections (Step 2)
   - Further splits sections into meaningful subsections (Step 3)
   - Uses LLM to identify atomic units like definitions and theorems (Step 4)
   - Creates a structured knowledge graph in Neo4j with fulltext index (Step 5)
   - Adds embeddings and creates vector index for similarity search (Step 6)

   **Detailed Usage for Each Tool**:
   ```
   # Process a single PDF file
   python src/math_rag/data_processing/pdf_to_text.py /absolute/path/to/your/document.pdf

   # Split document into major sections
   python src/math_rag/data_processing/section_splitter.py --input docs/processed/document.pkl --section 5    # Process specific section

   # Split sections into subsections
   python src/math_rag/data_processing/subsection_splitter.py --section 5                  # Process one section
   python src/math_rag/data_processing/subsection_splitter.py --section 5 --section 6      # Process multiple sections

   # Extract atomic units from sections or specific subsections
   python src/math_rag/data_processing/extract_atomic_units.py --section 5                # Process all subsections in section 5
   python src/math_rag/data_processing/extract_atomic_units.py --subsection 5.1           # Process just subsection 5.1
   python src/math_rag/data_processing/extract_atomic_units.py --section 5 --section 6    # Process all subsections in sections 5 and 6

   # Build knowledge graph and create fulltext index
   python src/math_rag/knowledge_graph/build_knowledge_graph.py

   # Create embeddings and vector index with different models
   python src/math_rag/embeddings/create_embeddings_and_vector_index.py --model "E5 Multilingual" --test
   python src/math_rag/embeddings/create_embeddings_and_vector_index.py --model "MXBAI German" --test
   python src/math_rag/embeddings/create_embeddings_and_vector_index.py --model "OpenAI" --test
   ```

   The tools have these resilient features:
   - Page-by-page processing with checkpoints (pdf_to_text.py)
   - Automatic retries with exponential backoff for API failures
   - Saves intermediate results to enable resume capability
   - Robust error handling to skip problematic content


## 📦 Project Structure

The math_rag codebase is organized into logical modules that follow the natural flow of data through the system:

**data_processing → knowledge_graph → embeddings → rag_agents → CLI**

This structure reflects how information moves through the system:
1. First, raw documents are processed into structured data (data_processing)
2. Then, this structured data is used to build a knowledge graph (knowledge_graph)
3. Text embeddings are created to enable semantic search (embeddings)
4. The agent system combines graph and embedding information to answer questions (rag_agents)
5. Finally, the CLI provides an interface for users to interact with the system

```
math_rag/
│
├── config/
│   ├── config.yaml                                 # Configuration file
│   └── agents.yaml                     # Agent system configuration
│
├── docs/                                           # Folder for storing mathematical PDFs
│
├── scripts/                                        # Utility and analysis scripts
│   ├── analyze_atomic_unit_lengths.py              # Analysis of atomic unit text lengths
│   ├── calc_embedding_similarity.py                # Tool for calculating embedding similarity
│   ├── compare_embeddings.py                       # Compare different embedding models
│   ├── direct_vector_search.py                     # Direct vector search utility
│   ├── test_cypher_tools.py                        # Test Cypher query tools
│   └── test_graph_meta.py                          # Test graph metadata queries
│
├── src/
│   └── math_rag/                                   # Core math RAG implementation
│       ├── core/                                   # Core data models and utilities
│       │   ├── atomic_unit.py                      # Atomic unit data model
│       │   └── project_root.py                     # Project path utilities
│       │
│       ├── data_processing/                        # 1. DOCUMENT PROCESSING PIPELINE
│       │   ├── pdf_to_text.py                      # Parse PDF with MathPix
│       │   ├── section_splitter.py                 # Split document into major sections
│       │   ├── subsection_splitter.py              # Split sections into subsections
│       │   ├── extract_atomic_units.py             # Extract definitions/theorems using LLM
│       │   ├── section_headers.py                  # Section header management
│       │   └── hierarchical_parser.py              # Document structure parsing
│       │
│       ├── knowledge_graph/                        # 2. KNOWLEDGE GRAPH CONSTRUCTION
│       │   ├── build_knowledge_graph.py            # Create knowledge graph + fulltext index
│       │   ├── create_graph.py                     # Graph creation utilities
│       │   ├── create_indexes.py                   # Index creation
│       │   ├── cypher_tools.py                     # Cypher query tools
│       │   └── cypher_query_generator.py           # Cypher query generation
│       │
│       ├── embeddings/                             # 3. VECTOR REPRESENTATIONS
│       │   ├── create_embeddings_and_vector_index.py # Create embeddings + vector index
│       │   ├── retrievers.py                       # Retrieval methods with different models
│       │   └── cypher_embeddings.py                # Cypher + embeddings integration
│       │
│       ├── rag_agents/                             # 4. RAG AGENT IMPLEMENTATION
│       │   ├── agents.py                           # Agent system setup and configuration
│       │   └── graph_meta_agent.py                 # Graph metadata agent
│       │
│       ├── cli/                                    # 5. COMMAND-LINE INTERFACES
│       │   └── graph_rag_cli.py                    # RAG chat command-line interface
│       │
│       └── utils/                                  # Utility functions
│           ├── infer_refs.py                       # Reference inference
│           └── sanity_checks.py                    # Validation checks
│
├── tests/                                          # Test suite
│   ├── test_atomic_unit.py                         # Tests for atomic unit functionality
│   ├── test_pdf_to_text.py                         # Tests for PDF processing
│   └── test_section_headers.py                     # Tests for section headers
│
├── docker-compose.yml                              # Docker setup for Neo4j
├── Makefile                                        # Build utilities
└── README.md                                       # Project documentation
```

## 📝 Embedding Models

The system supports multiple embedding models optimized for different use cases:

- **E5 Multilingual** (default): Best for academic German content, with strong performance on mathematical text
- **MXBAI German**: Alternative for German language content with good performance in academic contexts
- **OpenAI**: Standard OpenAI embeddings (text-embedding-3-small)

You can specify which model to use when creating embeddings:

```bash
# Use E5 Multilingual (default)
python -m src/math_rag/embeddings/create_embeddings_and_vector_index --model "E5 Multilingual" --test

# Use MXBAI German
python -m src/math_rag/embeddings/create_embeddings_and_vector_index --model "MXBAI German" --test

# Use OpenAI
python -m src/math_rag/embeddings/create_embeddings_and_vector_index --model "OpenAI" --test
```

## 💡 Learnings

This section documents important design decisions, architecture choices, and lessons learned throughout the development of this project.

### 2025-05-18: Embedding Generation Strategy

We initially explored two approaches for implementing embeddings in our Neo4j graph:

1. **Native Cypher Approach** (`cypher_embeddings.py`): This used Neo4j's built-in GenAI module with the `genai.vector.encodeBatch` function.

2. **External Embedding Approach** (`create_embeddings_and_vector_index.py`): This generates embeddings through external providers (OpenAI, HuggingFace) and manually adds them to Neo4j.

We've removed the native Cypher approach (`cypher_embeddings.py`) for the following reasons:

- **Limited Model Support**: The Neo4j GenAI module only supports OpenAI embeddings, which performed poorly for German mathematical content
- **Domain-Specific Performance**: Our testing showed that specialized models like E5 Multilingual and MXBAI German significantly outperformed OpenAI embeddings for mathematical German text
- **Flexibility Needs**: We needed the ability to experiment with different embedding models to optimize for mathematical notation and multi-language support
- **Benchmarking Results**: Our performance tests showed up to 40% better retrieval accuracy using specialized models compared to OpenAI embeddings

The current implementation uses external embedding generation for maximum flexibility and performance, allowing us to use domain-specific models that better understand mathematical concepts in German text.

## 🔄 Knowledge Graph Structure

The system creates a knowledge graph that captures the following elements from mathematical documents:

- Hierarchical structure: Sections, subsections, subsubsections
- Mathematical entities: Theorems, definitions, lemmas, propositions
- Relationships: Dependencies between mathematical concepts
- Proofs: Connected to their corresponding theorems

This structure enables more sophisticated retrieval than traditional vector-based approaches, allowing the system to answer complex mathematical questions that require understanding of mathematical relationships.

## 🚀 Using the Chat Interface

You can interact with the graph-based RAG system through the command-line interface:

```bash
# Launch the chat interface
python -m src/math_rag/cli/graph_rag_cli.py
```

The interface lets you:

1. Ask questions about the mathematical content in your documents
2. The system will:
   - Route your question to the appropriate specialized agent
   - Use the graph retriever agent for content-based queries
   - Use the Cypher agent for graph structure and metadata queries
   - Find relevant mathematical concepts through both vector similarity and graph traversal
   - Generate a comprehensive answer with proper mathematical notation
   - Verify the answer against the source material to prevent hallucinations

Commands within the chat interface:
- Type `exit`, `quit`, or `q` to end the session
- Type `clear` to clear the screen

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
