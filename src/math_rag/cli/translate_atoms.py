#!/usr/bin/env python3
"""
POC CLI to translate (normalize) atomic units using OpenAI (gpt-4.1+) and store
high‑fidelity natural language renderings with minimal semantic drift.

Focus:
- Preserve original meaning precisely (especially subtle structural notation
  like underlined symbols denoting topological spaces).
- Remove only superficial LaTeX wrappers.
- Avoid hallucinated explanations, equivalences, or added claims.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import coloredlogs
import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sqlmodel import Session, create_engine, select

from math_rag.config import TranslateAtomsSettings, settings_provider

# Project imports
from math_rag.core.db_models import ROOT, AtomicItemDB
from math_rag.core.llm_annotations import LLMAnnotation
from math_rag.core.llm_processor import annotate_row

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger)


class LLMAnnotationOutput(BaseModel):
    """
    Structured output schema returned by the LLM.
    All natural-language fields must remain in the original (source) language.
    """

    text_nl: Optional[str] = Field(
        None,
        description="Natural language rendering of the main text (source language)",
    )
    proof_nl: Optional[str] = Field(
        None, description="Natural language rendering of the proof (source language)"
    )
    summary_nl: Optional[str] = Field(
        None, description="Natural language summary (source language, minimal drift)"
    )
    language: Optional[str] = Field(
        None, description="Detected source language (e.g. 'de' or 'German')"
    )
    concepts: List[str] = Field(
        default_factory=list,
        description="Minimal list of distinct concept tokens present in the source",
    )
    input_hash: str = Field(..., description="Echoed input hash for idempotency")
    warnings: Optional[List[str]] = Field(
        None,
        description="List of warning strings about ambiguity or preservation choices",
    )


def load_prompts(prompts_path) -> tuple[str, str]:
    """
    Load prompts from the required YAML file.

    The YAML must contain two top-level keys: `system` and `user`. The `user`
    prompt must include the placeholders: {id}, {input_hash}, {text}, {proof},
    {summary}. If the file is missing or invalid the function raises SystemExit.
    """
    p = prompts_path
    if not p.exists():
        raise SystemExit(f"Prompts file not found: {p}")

    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as e:
        raise SystemExit(f"Failed to read prompts file {p}: {e}")

    system = data.get("system")
    user = data.get("user")

    if not system or not user:
        raise SystemExit(
            f"Prompts file {p} must contain top-level `system` and `user` keys"
        )

    # Validate required placeholders in the user prompt
    required_placeholders = ["{id}", "{input_hash}", "{text}", "{proof}", "{summary}"]
    missing = [ph for ph in required_placeholders if ph not in user]
    if missing:
        raise SystemExit(
            "User prompt missing required placeholders: " + ", ".join(missing)
        )

    # Ensure the deprecated {prompt_version} placeholder is not present
    if "{prompt_version}" in user:
        raise SystemExit(
            "{prompt_version} placeholder is not allowed in the user prompt"
        )

    return system, user


def build_chain(
    system_prompt: str, user_prompt: str, model_name: str, openai_api_key: str
):
    """Create the composed prompt → structured LLM chain using supplied prompts."""
    chat_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", user_prompt)]
    )
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=model_name,
    )
    structured_llm = llm.with_structured_output(
        LLMAnnotationOutput, method="json_schema"
    )
    return chat_prompt | structured_llm


def upsert_annotation(session: Session, ann: LLMAnnotation) -> LLMAnnotation:
    """
    Upsert an LLMAnnotation by primary key (id) or unique semantic key if present.

    Current model uses `id` aligned with AtomicItemDB primary key.
    NOTE: If you later switch to a `unit_id` FK pattern, adjust this logic
    accordingly (and add uniqueness constraints).
    """
    existing = session.exec(
        select(LLMAnnotation).where(LLMAnnotation.id == ann.id)
    ).first()

    if existing and getattr(existing, "id", None) is not None:
        ann.id = existing.id

    merged = session.merge(ann)
    return merged


def main(
    limit: int,
    write_to_db: bool,
    batch_size: int,
    model_name: str,
    openai_api_key: str,
    prompts_path: Path,
):
    """Main driver: keyset pagination over AtomicItemDB and per-row annotation.

    This function accepts explicit named arguments (no settings object). It loads
    prompts, enforces the `limit` semantics (0 = all), and drives the annotation
    loop. All inputs are required; validation is strict and will raise SystemExit
    with an explanatory message if a required item is missing or malformed.
    """

    # Load prompts (required)
    system_prompt, user_prompt = load_prompts(prompts_path)

    # Respect `limit` semantics: 0 means all
    limit_val = limit if limit > 0 else 0

    db_path = ROOT / "data" / "atomic_units.sqlite"
    engine = create_engine(f"sqlite:///{str(db_path)}", echo=False)

    # Compute a deterministic prompt_version from the prompts content so that
    # input-hash behavior remains stable and auditable without an external
    # prompt-version concept.
    prompt_hash = hashlib.sha256(
        (system_prompt + user_prompt).encode("utf-8")
    ).hexdigest()
    prompt_version = f"sha256:{prompt_hash}"

    chain = build_chain(system_prompt, user_prompt, model_name, openai_api_key)

    processed = 0
    failures = 0
    last_id = 0

    while True:
        if limit_val and processed >= limit_val:
            break

        with Session(engine) as read_sess:
            stmt = (
                select(AtomicItemDB)
                .where(AtomicItemDB.id > last_id)
                .order_by(AtomicItemDB.id)
                .limit(batch_size)
            )
            batch = read_sess.exec(stmt).all()

        if not batch:
            break

        for row in batch:
            if limit_val and processed >= limit_val:
                break

            ann = annotate_row(row, chain, model_name, prompt_version)
            if ann is None:
                failures += 1
                last_id = row.id
                continue

            if write_to_db:
                try:
                    with Session(engine) as write_sess:
                        with write_sess.begin():
                            upsert_annotation(write_sess, ann)
                    processed += 1
                except Exception as e:
                    failures += 1
                    logger.warning("DB write failed for unit %s: %s", row.id, str(e))
            else:
                logger.debug(
                    "Dry-run output for unit %s: %s",
                    row.id,
                    (ann.text_nl or "<no text_nl>")[:500],
                )
                processed += 1

            last_id = row.id

    logger.info("Done. Processed: %d, Failures: %d", processed, failures)


if __name__ == "__main__":
    settings = settings_provider.get_settings(TranslateAtomsSettings)

    main(
        limit=settings.limit,
        write_to_db=settings.write_to_db,
        batch_size=settings.batch_size,
        model_name=settings.model_name,
        openai_api_key=settings.openai_api_key,
        prompts_path=settings.prompts_path,
    )
