# src/math_rag/core/llm_annotations.py
"""
SQLModel definitions for LLM-produced annotations of atomic units.

This POC table stores one annotation row per AtomicItemDB unit (upsert by `unit_id`).
It intentionally keeps fields simple (JSON as string for concepts) to avoid complex
migrations during the POC stage.
"""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class LLMAnnotationBase(SQLModel):
    """
    Base fields for an LLM annotation.

    Note: `concepts_json` stores a JSON-encoded list of strings (e.g. '["a","b"]')
    to keep the schema migration-free for the POC. Consider a proper JSON column
    or a related table if you need richer querying later.
    """

    id: int = Field(index=True, description="Foreign key to AtomicItemDB.id")
    text_nl: Optional[str] = Field(
        default=None, description="Natural-language rendering of the `text` field"
    )
    proof_nl: Optional[str] = Field(
        default=None, description="Natural-language rendering of the `proof` field"
    )
    concepts_json: Optional[str] = Field(
        default=None,
        description='JSON-encoded list of concept strings, e.g. \'["filter","ultrafilter"]\'', #noqa: E501
    )
    input_hash: Optional[str] = Field(
        default=None,
        description="sha256 hash of the original input fields + prompt_version for idempotency/audit", #noqa: E501
    )
    warnings: Optional[str] = Field(
        default=None,
        description="JSON-encoded list of warning strings from preprocessing or LLM (nullable)", # noqa: E501
    )


class LLMAnnotation(LLMAnnotationBase, table=True):
    """
    Table to persist annotations produced by the LLM.

    For the POC we allow simple insert-or-update logic:
      - When persisting, update the existing row for `unit_id` if present,
        otherwise insert a new row.
    """

    __tablename__ = "llm_annotations"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.now, nullable=False)

    class Config:
        """Pydantic/SQLModel config for nicer reprs if needed."""

        arbitrary_types_allowed = True
