"""
Pydantic models for the Math-RAG evaluation system.

This module defines the core data structures used throughout the evaluation pipeline,
providing type safety, validation, and serialization capabilities.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class JudgeResult(BaseModel):
    """Result from LLM judge evaluation of a single response."""

    # Core scores (1-5 scale)
    mathematical_correctness_score: int = Field(
        ..., ge=1, le=5, description="Mathematical accuracy score (1-5)"
    )
    context_relevance_score: int = Field(
        ..., ge=1, le=5, description="Context relevance score (1-5)"
    )
    context_utilization_score: int = Field(
        ..., ge=1, le=5, description="Context utilization score (1-5)"
    )
    completeness_score: int = Field(
        ..., ge=1, le=5, description="Response completeness score (1-5)"
    )

    # Analysis fields
    key_info_found: List[str] = Field(
        default_factory=list, description="Key information points found"
    )
    hallucinations: List[str] = Field(
        default_factory=list, description="Unsupported claims made"
    )
    missing_info: List[str] = Field(
        default_factory=list, description="Important missing information"
    )
    overall_pass: bool = Field(..., description="Whether answer is acceptable")
    judgement_summary: str = Field(..., description="Summary of assessment")

    @property
    def average_score(self) -> float:
        """Calculate average of all numeric scores."""
        scores = [
            self.mathematical_correctness_score,
            self.context_relevance_score,
            self.context_utilization_score,
            self.completeness_score,
        ]
        return sum(scores) / len(scores)


class RetrievalResult(BaseModel):
    """Result from retrieval quality evaluation."""

    # Core metrics
    precision: float = Field(..., ge=0.0, le=1.0, description="Retrieval precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Retrieval recall")
    f1: float = Field(..., ge=0.0, le=1.0, description="F1 score")

    # Counts
    retrieved_count: int = Field(..., ge=0, description="Number of documents retrieved")
    gold_count: int = Field(..., ge=0, description="Number of gold standard documents")
    relevant_retrieved_count: int = Field(
        ..., ge=0, description="Number of relevant documents retrieved"
    )

    # Document lists
    missing_from_retrieval: List[str] = Field(
        default_factory=list, description="Gold documents not retrieved"
    )
    irrelevant_retrieved: List[str] = Field(
        default_factory=list, description="Retrieved documents not in gold standard"
    )

    @field_validator("relevant_retrieved_count")
    @classmethod
    def validate_relevant_count(cls, v, info):
        """Ensure relevant_retrieved_count is consistent with other metrics."""
        if (
            info.data.get("retrieved_count") is not None
            and v > info.data["retrieved_count"]
        ):
            raise ValueError("Relevant retrieved cannot exceed total retrieved")
        if info.data.get("gold_count") is not None and v > info.data["gold_count"]:
            raise ValueError("Relevant retrieved cannot exceed gold count")
        return v


class EvaluationResult(BaseModel):
    """Complete evaluation result for a single question-answer pair."""

    question: str = Field(..., description="Original question")
    ground_truth: str = Field(..., description="Expected answer")
    predicted_answer: Optional[str] = Field(None, description="RAG system answer")
    retrieved_context: str = Field(..., description="Retrieved context")
    difficulty: Optional[int] = Field(None, description="Question difficulty level")
    source: Optional[str] = Field(None, description="Question source")

    # Evaluation results
    judge_result: Optional[JudgeResult] = Field(
        None, description="LLM judge evaluation"
    )
    retrieval_result: Optional[RetrievalResult] = Field(
        None, description="Retrieval quality evaluation"
    )


class JudgeSummary(BaseModel):
    """Summary statistics for judge evaluation results."""

    # Score statistics
    avg_mathematical_correctness: float = Field(..., description="Average math score")
    avg_context_relevance: float = Field(..., description="Average context relevance")
    avg_context_utilization: float = Field(..., description="Average context usage")
    avg_completeness: float = Field(..., description="Average completeness")
    avg_overall_score: float = Field(..., description="Average of all scores")

    # Pass rate
    pass_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Proportion of passing responses"
    )

    # Score distributions (count by score value)
    mathematical_correctness_dist: dict = Field(
        default_factory=dict, description="Distribution of math scores"
    )
    context_relevance_dist: dict = Field(
        default_factory=dict, description="Distribution of context relevance scores"
    )
    context_utilization_dist: dict = Field(
        default_factory=dict, description="Distribution of context usage scores"
    )
    completeness_dist: dict = Field(
        default_factory=dict, description="Distribution of completeness scores"
    )


class RetrievalSummary(BaseModel):
    """Summary statistics for retrieval evaluation results."""

    avg_precision: float = Field(..., description="Average precision across questions")
    avg_recall: float = Field(..., description="Average recall across questions")
    avg_f1: float = Field(..., description="Average F1 score across questions")
    total_questions_evaluated: int = Field(
        ..., ge=0, description="Number of questions with retrieval evaluation"
    )


class EvaluationSummary(BaseModel):
    """Complete summary of evaluation run."""

    total_questions: int = Field(..., ge=0, description="Total questions evaluated")
    successful_evaluations: int = Field(
        ..., ge=0, description="Successfully completed evaluations"
    )
    failed_evaluations: int = Field(..., ge=0, description="Failed evaluations")

    judge_summary: Optional[JudgeSummary] = Field(
        None, description="Judge evaluation summary"
    )
    retrieval_summary: Optional[RetrievalSummary] = Field(
        None, description="Retrieval evaluation summary"
    )

    @field_validator("successful_evaluations")
    @classmethod
    def validate_successful_count(cls, v, info):
        """Ensure successful + failed = total."""
        total_questions = info.data.get("total_questions")
        failed_evaluations = info.data.get("failed_evaluations")
        if total_questions is not None and failed_evaluations is not None:
            expected_total = v + failed_evaluations
            if expected_total != total_questions:
                raise ValueError(
                    f"successful ({v}) + failed ({failed_evaluations}) "
                    f"!= total ({total_questions})"
                )
        return v
