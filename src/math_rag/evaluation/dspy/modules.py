import logging
from typing import List

import dspy
from langchain_neo4j import Neo4jVector

from math_rag.evaluation.core.models import JudgeResult
from math_rag.graph_tools import GraphRetrieverTool, PathRAGRetrieverTool  # noqa: F401
from math_rag.graph_tools.utils import format_retrieval_results

# Configure logging
logger = logging.getLogger(__name__)

# Configure DSPy with Ollama (can be overridden in settings)
try:
    lm = dspy.LM(
        "ollama_chat/deepseek-r1:latest",
        api_base="http://localhost:11434",
        api_key="",
        temperature=0.0,
        cache=True,
    )
    dspy.configure(lm=lm)
except Exception as e:
    logger.error(f"Failed to configure DSPy with Ollama: {e}")
    raise ValueError("Failed to configure DSPy with Ollama")


class AssessResponse(dspy.Signature):
    """Signature for comprehensive response assessment."""

    question: str = dspy.InputField(desc="The original question asked")
    predicted_answer: str = dspy.InputField(
        desc="The answer provided by the RAG system"
    )
    context: str = dspy.InputField(
        desc="The retrieved context provided to the RAG system"
    )

    # 1-5 scale scores
    mathematical_correctness_score: int = dspy.OutputField(
        desc="""Score between 1 and 5 measuring the mathematical accuracy of the answer:
            1 = Contains serious mathematical errors or incorrect formulas
            2 = Contains minor mathematical errors or imprecisions
            3 = Mostly mathematically correct with some small issues
            4 = Mathematically correct with proper reasoning
            5 = Mathematically rigorous and precise throughout"""
    )
    context_relevance_score: int = dspy.OutputField(
        desc="""Score between 1 and 5 measuring how relevant the retrieved context is to
        the question:
            1 = Retrieved context contains no relevant mathematical concepts
            2 = Retrieved context barely relevant to the question
            3 = Retrieved context somewhat relevant but missing key information
            4 = Retrieved context contains most needed mathematical concepts
            5 = Retrieved context perfectly matches the question's requirements"""
    )
    context_utilization_score: int = dspy.OutputField(
        desc="""Score between 1 and 5 measuring how well the answer utilizes the
        retrieved context:
            1 = Answer ignores relevant context entirely
            2 = Answer barely uses available context information
            3 = Answer uses some context information but misses key points
            4 = Answer effectively uses most relevant context information
            5 = Answer comprehensively integrates all useful context information"""
    )
    completeness_score: int = dspy.OutputField(
        desc="""Score between 1 and 5 measuring mathematical thoroughness of the
        response:
                1 = Very incomplete mathematical treatment
                2 = Missing several important mathematical steps or concepts
                3 = Covers basic mathematical approach but lacks detail
                4 = Mostly complete with good mathematical detail
                5 = Comprehensively addresses all mathematical aspects"""
    )

    # Analysis fields
    key_info_found: List[str] = dspy.OutputField(
        desc="List of key information points found in the answer"
    )
    hallucinations: List[str] = dspy.OutputField(
        desc="List of mathematical claims made without support from the retrieved context"  # noqa: E501
    )
    missing_info: List[str] = dspy.OutputField(
        desc="List of important information missing from the answer"
    )
    overall_pass: bool = dspy.OutputField(
        desc="Does this answer pass as acceptable? (True/False)"
    )
    judgement_summary: str = dspy.OutputField(
        desc="Brief summary explaining the assessment including context utilization"
    )


class LLMJudge(dspy.Module):
    """Unified judge for comprehensive response evaluation with context assessment."""

    def __init__(self):
        self.assess = dspy.ChainOfThought(AssessResponse)

    def forward(
        self,
        question: str,
        predicted_answer: str,
        context: str,
    ):
        """
        Assess the response quality and context utilization.

        Args:
            question: The original question
            predicted_answer: The answer provided by the RAG system
            context: The retrieved context provided to the RAG system
            expected_info: Expected information points for the answer

        Returns:
            AssessResponse with various evaluation metrics
        """
        return self.assess(
            question=question,
            predicted_answer=predicted_answer,
            context=context,
        )


def judge_response(
    question: str, predicted_answer: str, ground_truth: str, context: str = ""
) -> JudgeResult:
    """
    Judge a response using the unified LLM judge with context evaluation.

    Args:
        question: The original question
        predicted_answer: The answer provided by the RAG system (DSPy object or string)
        ground_truth: The expected correct answer
        context: The retrieved context provided to the RAG system (optional)

    Returns:
        JudgeResult with scores and analysis including context utilization
    """
    # Initialize judge
    judge = LLMJudge()

    # Get assessment including context evaluation
    dspy_result = judge(
        question=question,
        predicted_answer=predicted_answer,
        context=context,
    )

    # Convert DSPy result to Pydantic model
    return JudgeResult(
        mathematical_correctness_score=dspy_result.mathematical_correctness_score,
        context_relevance_score=dspy_result.context_relevance_score,
        context_utilization_score=dspy_result.context_utilization_score,
        completeness_score=dspy_result.completeness_score,
        key_info_found=dspy_result.key_info_found,
        hallucinations=dspy_result.hallucinations,
        missing_info=dspy_result.missing_info,
        overall_pass=dspy_result.overall_pass,
        judgement_summary=dspy_result.judgement_summary,
    )


# Model
class SimpleQA(dspy.Module):
    """DSPy module for question answering with optional vector store retrieval."""

    def __init__(
        self,
        model: dspy.Module,
        signature: dspy.Signature,
        vector_store: Neo4jVector,
    ):
        self.vector_store = vector_store
        self.qa = model(signature)
        self._last_context = ""
        self._last_context_docs = []

    def _get_context(self, question: str) -> tuple[List, str]:
        """Retrieve and format context for the question."""
        context_docs = (
            self.vector_store.similarity_search(query=question, k=10)
            if self.vector_store
            else []
        )
        # Format context using the same method as retrievers
        context = format_retrieval_results(context_docs, "hybrid search")

        # Store context for evaluation purposes
        self._last_context = context
        self._last_context_docs = context_docs

        return context_docs, context

    def forward(self, question: str) -> str:
        """Generate answer for the given question using DSPy Module pattern."""
        context_docs, context = self._get_context(question)
        result = self.qa(question=question, context=context)

        return result

    def get_last_context(self) -> str:
        """Get the context from the last forward call."""
        return self._last_context

    def get_last_context_docs(self) -> list:
        """Get the raw context docs from the last forward call."""
        return self._last_context_docs
