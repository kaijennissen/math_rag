#!/usr/bin/env python3
"""
DSPy-based single-agent approach for mathematical RAG system.

This module implements a simplified approach using DSPy for more deterministic
and reliable RAG responses without the complexity of multi-agent interactions.

Key features:
- Single agent design instead of multi-agent system
- Explicit document grading step to filter relevant context
- Programmatic prompting with DSPy for more reliable outputs
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

# Third-party imports
import dspy
import yaml
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import SearchType

# Project imports
from math_rag.graph_tools import GraphRetrieverTool, PathRAGRetrieverTool  # noqa: F401
from math_rag.graph_tools.utils import get_pathrag_query

# Configure logging
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DocumentGrader(dspy.Module):
    """Module for grading documents based on relevance to a query."""

    def __init__(self, threshold: float = 5.0):
        super().__init__()
        self.threshold = threshold
        # Use ChainOfThought for better reasoning when grading documents
        self.grade_doc = dspy.ChainOfThought(
            "question, document -> relevance_score: float, reasoning"
        )

    def forward(self, question: str, documents: List[str]) -> List[str]:
        """
        Grade documents based on relevance to the question.

        Args:
            question: The user's query
            documents: List of documents to grade

        Returns:
            List of relevant documents above threshold
        """
        graded_docs = []
        for doc in documents:
            try:
                # Use DSPy to predict relevance score
                result = self.grade_doc(question=question, document=doc)
                score = float(result.relevance_score)

                if score >= self.threshold:
                    graded_docs.append(doc)
                    logger.debug(
                        f"Document passed grading with score {score:.2f}: "
                        f"{result.reasoning}"
                    )
            except (ValueError, AttributeError) as e:
                # Handle case where score isn't a valid float
                logger.warning(f"Error grading document: {e}")
                continue

        logger.info(
            f"Document grading complete: {len(graded_docs)}/{len(documents)} "
            f"documents passed threshold {self.threshold}"
        )
        return graded_docs


class MathRAGModule(dspy.Module):
    """DSPy module for mathematical RAG following DSPy best practices."""

    def __init__(
        self,
        retriever_fn: Callable[[str], List[str]],
        document_grader: Optional[DocumentGrader] = None,
        max_context_docs: int = 20,
        use_query_rewriting: bool = True,
    ):
        """
        Initialize the Math RAG module.

        Args:
            retriever_fn: A callable that takes a query and returns list of passages
            document_grader: Optional document grading module
            max_context_docs: Maximum number of context documents to use
            use_query_rewriting: Whether to rewrite queries for better retrieval
        """
        super().__init__()
        self.retriever_fn = retriever_fn
        self.document_grader = document_grader
        self.max_context_docs = max_context_docs
        self.use_query_rewriting = use_query_rewriting

        # Query rewriter with domain-specific instructions for math content
        if self.use_query_rewriting:
            self.rewrite_query = dspy.ChainOfThought(
                dspy.Signature(
                    "question -> rewritten_query, reasoning",
                    instructions="""Rewrite the question to be better suited for
                    semantic search in a mathematical knowledge base. The rewritten
                    query should:
                    1. Be in German (the corpus language)
                    2. Use mathematical terminology precisely
                    3. Be phrased as a statement rather than a question
                    4. Include key mathematical concepts that should be matched
                    5. Be specific about the type of mathematical object
                       (Definition, Satz, Theorem, Beweis, Lemma, Korollar, etc.)

                    Example:
                    Question: "Was ist eine Gruppe?"
                    Rewritten: "Definition einer mathematischen Gruppe mit
                    Verknüpfung und Gruppeneigenschaften"

                    Question: "Wie hängen Differentiation und Integration zusammen?"
                    Rewritten: "Hauptsatz der Differential- und Integralrechnung
                    beschreibt Zusammenhang zwischen Ableitung und Integral"
                    """,
                )
            )

        self.generate_answer = dspy.ChainOfThought(
            "context: list[str], question -> answer, citations: list[str]"
        )

    def forward(self, question: str) -> dspy.Prediction:
        """
        Process a question through the complete RAG pipeline.

        Args:
            question: The user's question

        Returns:
            DSPy Prediction with answer and metadata
        """
        # Step 1: Optionally rewrite query for better retrieval
        if self.use_query_rewriting:
            rewrite_result = self.rewrite_query(question=question)
            search_query = rewrite_result.rewritten_query
            logger.info(f"Original question: {question}")
            logger.info(f"Rewritten query: {search_query}")
            logger.info(f"Rewriting reasoning: {rewrite_result.reasoning}")
        else:
            search_query = question

        # Step 2: Retrieve using (possibly rewritten) query
        logger.info(f"Retrieving documents for query: {search_query}")
        documents = self.retriever_fn(search_query)
        logger.info(f"Retrieved {len(documents)} initial documents")

        # Step 3: Grade documents for relevance (using ORIGINAL question)
        if self.document_grader:
            relevant_docs = self.document_grader(question, documents)
        else:
            relevant_docs = documents

        # Limit to top N most relevant documents
        context = relevant_docs[: self.max_context_docs]
        logger.info(f"Using {len(context)} documents for context")

        # Step 4: Generate final answer (using ORIGINAL question)
        result = self.generate_answer(
            context=context,
            question=question,
        )

        # Return DSPy Prediction object with metadata
        prediction = dspy.Prediction(
            answer=result.answer,
            citations=result.citations if hasattr(result, "citations") else [],
            context=context,
            reasoning=result.reasoning if hasattr(result, "reasoning") else "",
        )

        # Add rewritten query to metadata if used
        if self.use_query_rewriting:
            prediction.rewritten_query = search_query
            prediction.rewrite_reasoning = rewrite_result.reasoning

        return prediction


def create_retriever_function(retriever_tools: List[Any]) -> Callable[[str], List[str]]:
    """
    Create a DSPy-compatible retriever function from retriever tools.

    Args:
        retriever_tools: List of retriever tools
            (GraphRetrieverTool, PathRAGRetrieverTool)

    Returns:
        A callable function that takes a query and returns list of passages
    """

    def retrieve(query: str, k: int = 10) -> List[str]:
        """Retrieve documents from all retriever tools."""
        all_passages = []

        for tool in retriever_tools:
            try:
                # Call the tool's forward method with query and k
                result = tool.forward(query=query, k=k)

                # Result is a formatted string, parse it to extract passages
                if isinstance(result, str):
                    # Split by document separators and clean up
                    passages = [
                        p.strip()
                        for p in result.split("Document ")
                        if p.strip() and not p.startswith("===")
                    ]
                    all_passages.extend(passages)
                elif isinstance(result, list):
                    all_passages.extend(result)

            except Exception as e:
                logger.warning(f"Error retrieving from {tool.name}: {e}")
                continue

        logger.info(f"Retrieved total of {len(all_passages)} passages")
        return all_passages

    return retrieve


def setup_dspy_rag_chat(
    openai_api_key: str,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    neo4j_database: str,
    agent_config_path: Path,
    model_id: str = "gpt-4.1",
    api_base: str = "https://api.openai.com/v1",
    huggingface_api_key: Optional[str] = None,
    use_query_rewriting: bool = True,
) -> MathRAGModule:
    """
    Setup a DSPy-based RAG system for mathematical content.

    Args:
        openai_api_key: OpenAI API key
        neo4j_uri: Neo4j connection URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        agent_config_path: Path to agent configuration YAML file
        model_id: Model ID for the LLM
        api_base: API base URL for OpenAI
        huggingface_api_key: Optional HuggingFace API key
        use_query_rewriting: Whether to use query rewriting for better retrieval

    Returns:
        MathRAGModule: The configured DSPy RAG module
    """
    # Load agent configurations
    try:
        with open(agent_config_path, "r") as file:
            yaml.safe_load(file)  # Load config to validate file exists
    except FileNotFoundError:
        logger.error(f"Agent configuration file not found at {agent_config_path}")
        raise

    # Initialize DSPy with the OpenAI model
    lm = dspy.LM(model=f"openai/{model_id}", api_key=openai_api_key, api_base=api_base)
    dspy.settings.configure(lm=lm)

    # Setup embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # Setup vector stores
    # hybrid_vector_store = Neo4jVector.from_existing_index(
    #     embedding_model,
    #     url=neo4j_uri,
    #     username=neo4j_username,
    #     password=neo4j_password,
    #     index_name="vector_index_text_nl_Embedding",
    #     keyword_index_name="fulltext_index_AtomicItem",
    #     embedding_node_property="text_nl_Embedding",
    #     search_type=SearchType.HYBRID,
    # )

    pathrag_vector_store = Neo4jVector.from_existing_index(
        embedding_model,
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        index_name="vector_index_text_nl_Embedding",
        keyword_index_name="fulltext_index_AtomicItem",
        embedding_node_property="text_nl_Embedding",
        search_type=SearchType.HYBRID,
        retrieval_query=get_pathrag_query(),
    )

    # Initialize retriever tools
    # graph_retriever_tool = GraphRetrieverTool(vector_store=hybrid_vector_store)
    pathrag_retriever_tool = PathRAGRetrieverTool(vector_store=pathrag_vector_store)
    retriever_tools = [pathrag_retriever_tool]

    # Create DSPy-compatible retriever function
    retriever_fn = create_retriever_function(retriever_tools)

    # Create document grader
    # document_grader = DocumentGrader(threshold=1.0)

    # Create RAG module with query rewriting
    rag_module = MathRAGModule(
        retriever_fn=retriever_fn,
        # document_grader=document_grader,
        max_context_docs=5,
        use_query_rewriting=use_query_rewriting,
    )

    return rag_module


def run_rag_chat(rag_module: MathRAGModule, question: str) -> str:
    """
    Run the RAG chat with a given question.

    Args:
        rag_module: The DSPy RAG module
        question: User's question

    Returns:
        Answer string
    """
    result = rag_module(question)
    # Result is now a dspy.Prediction object
    return result.answer


def main():
    """Main function for command-line usage."""
    from math_rag.config.settings import RagChatSettings

    settings = RagChatSettings()
    rag_module = setup_dspy_rag_chat(
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

    question = (
        "Welche Definitionen für ein T_1 Raum gibt es?"
        "Bitte nenne alle Definitionen, bzw. äquivalente Definitionen inkl. der Nummer der Definition oder des Satzes."  # noqa: E501
        "Sollte innerhalb einer Definition auf einen anderen Satz oder eine andere Definition verwiesen, so sollte auch diese mit genannt werden."  # noqa: E501
    )
    answer = run_rag_chat(rag_module, question)
    print(f"\nQuestion: {question}\n")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
