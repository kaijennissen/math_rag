"""
Simplified evaluation runner for Math-RAG system using DSPy modules and Pydantic models.

This module provides a refactored evaluation pipeline that uses type-safe Pydantic
models for data handling and provides comprehensive evaluation metrics.
"""

import logging
from typing import List

import dspy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import SearchType

from math_rag.config.shared import Neo4jConfig, OpenAIConfig
from math_rag.evaluation.core.metrics import evaluate_retrieval_quality
from math_rag.evaluation.core.models import EvaluationResult
from math_rag.evaluation.dspy.modules import SimpleQA, judge_response

logger = logging.getLogger(__name__)


def setup_rag_system(neo4j_config: Neo4jConfig) -> SimpleQA:
    """
    Set up the RAG system with Neo4j vector store and embeddings.

    Args:
        neo4j_config: Neo4j configuration settings

    Returns:
        Configured SimpleQA system
    """
    # Create vector store
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vector_store = Neo4jVector.from_existing_index(
        embedding_model,
        url=str(neo4j_config.uri),
        username=neo4j_config.username,
        password=neo4j_config.password.get_secret_value(),
        index_name="vector_index_text_nl_Embedding",
        keyword_index_name="fulltext_index_AtomicItem",
        embedding_node_property="text_nl_Embedding",
        search_type=SearchType.HYBRID,
    )

    class TopologyQuestion(dspy.Signature):
        """Answer mathematical topology questions using retrieved context."""

        question: str = dspy.InputField(
            desc="A mathematical topology question to be answered"
        )
        context: str = dspy.InputField(
            desc="Relevant mathematical definitions, theorems, and explanations "
            "retrieved from topology textbooks and papers"
        )
        answer: str = dspy.OutputField(
            desc="A consise and precise answer to the topology question, "
            "citing relevant mathematical concepts"
        )
        reasoning: str = dspy.OutputField(
            desc="Step-by-step mathematical reasoning leading to the answer"
        )
        source_links: List[str] = dspy.OutputField(
            desc="List of source links for the answer"
        )

    rag_system = SimpleQA(
        model=dspy.ChainOfThought,
        signature=TopologyQuestion,
        vector_store=vector_store,
    )
    return rag_system


def run_evaluation(
    rag_system: SimpleQA, dataset: List[dict], openai_config: OpenAIConfig
) -> List[EvaluationResult]:
    """
    Run evaluation on provided dataset using Pydantic models.

    Args:
        rag_system: Configured RAG system
        dataset: List of question dictionaries with keys: question, answer,
            difficulty, source
        openai_config: OpenAI configuration for judge

    Returns:
        List of EvaluationResult objects for each question
    """
    results = []

    for i, item in enumerate(dataset, start=1):
        logger.info(f"Processing question {i}/{len(dataset)}")

        try:
            # Get RAG answer
            prediction = rag_system(item.question)
            retrieved_context = rag_system.get_last_context()
            retrieved_docs = rag_system.get_last_context_docs()

            # Evaluate retrieval quality
            retrieval_result = evaluate_retrieval_quality(
                retrieved_docs=retrieved_docs, gold_source=item.source
            )

            # Run unified judge with context evaluation (only if we have an answer)
            with dspy.context(
                lm=dspy.LM(
                    openai_config.model_id,
                    api_key=openai_config.api_key.get_secret_value(),
                    temperature=1.0,
                    max_tokens=20_000,
                )
            ):
                judge_result = judge_response(
                    question=item.question,
                    predicted_answer=prediction.answer,
                    ground_truth=item.answer,
                    context=retrieved_context,
                )

            # Create structured evaluation result
            evaluation_result = EvaluationResult(
                question=item.question,
                ground_truth=item.answer,
                difficulty=item.difficulty,
                source=item.source,
                predicted_answer=prediction.answer,
                retrieved_context=retrieved_context,
                judge_result=judge_result,
                retrieval_result=retrieval_result,
            )

        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")

            # Create result with error information
            evaluation_result = EvaluationResult(
                question=item.question,
                ground_truth=item.answer,
                difficulty=item.difficulty,
                source=item.source,
                predicted_answer=prediction.answer,
                retrieved_context=retrieved_context,
                judge_result=None,
                retrieval_result=None,
            )

        results.append(evaluation_result)

    return results
