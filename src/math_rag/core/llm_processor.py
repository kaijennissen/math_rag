# math_rag/src/math_rag/core/llm_processor.py
"""
Helper that annotates a single AtomicItemDB row using the structured LLM chain.

This module provides a single public function `annotate_row` which:
- Accepts a DB row (an `AtomicItemDB` instance), a LangChain-style `chain`
  (the object returned by `build_chain(...)` in translate_atoms), a `model_name`
  string and a `prompt_version` string.
- Preprocesses LaTeX, computes an input hash, calls the chain and validates the
  echoed hash, then builds and returns an `LLMAnnotation` instance.
- DOES NOT persist or commit anything to the DB. Persistence/upsert should be
  performed by the caller (this keeps responsibilities separated and makes the
  helper easy to test).

Design goals:
- Keep the function small and POC-friendly.
- Minimal error handling: on any LLM call error or validation failure, log a
  warning and return None so the caller can increment counters and decide what
  to do next.
"""

import json
import logging
from typing import Any, List, Optional

from math_rag.core.db_models import AtomicItemDB
from math_rag.core.llm_annotations import LLMAnnotation
from math_rag.core.llm_utils import compute_input_hash, preprocess_latex

logger = logging.getLogger(__name__)


def annotate_row(
    row: AtomicItemDB, chain: Any, model_name: str, prompt_version: str
) -> Optional[LLMAnnotation]:
    """
    Annotate a single AtomicItemDB row with the LLM.

    Args:
        row: AtomicItemDB instance (the DB row to annotate).
        chain: The structured LLM chain (e.g., ChatPromptTemplate | structured_llm).
               It must support `chain.invoke(dict)` and return an object with
               attributes like `.text_nl`, `.proof_nl`, `.summary_nl`,
               `.concepts`, `.input_hash`, and `.warnings`.
        model_name: Name of the LLM model used (stored in the annotation metadata).
        prompt_version: Prompt template / version identifier .

    Returns:
        LLMAnnotation instance on success (not persisted), or None on failure.
    """
    if row is None:
        logger.warning("annotate_row called with None row")
        return None

    # 1) Preprocess LaTeX fields (conservative cleaning)
    try:
        text_proc, text_warn = preprocess_latex(row.text)
        proof_proc, proof_warn = (
            preprocess_latex(row.proof) if getattr(row, "proof", None) else (None, None)
        )
        summary_proc, summary_warn = (
            preprocess_latex(row.summary)
            if getattr(row, "summary", None)
            else (None, None)
        )
    except Exception as e:
        logger.warning(
            "Preprocessing failed for unit %s: %s",
            getattr(row, "id", "<no-id>"),
            str(e),
        )
        return None

    preproc_warnings: List[str] = []
    for w in (text_warn, proof_warn, summary_warn):
        if w:
            preproc_warnings.append(w)

    # 2) Compute input hash (uses original fields to reflect exact DB content)
    try:
        input_hash = compute_input_hash(
            row.text, row.proof, row.summary, prompt_version
        )
    except Exception as e:
        logger.warning(
            "Failed to compute input_hash for unit %s: %s",
            getattr(row, "id", "<no-id>"),
            str(e),
        )
        return None

    # 3) Build invoke_vars dict for the chain
    invoke_vars = {
        "id": row.id,
        "prompt_version": prompt_version,
        "input_hash": input_hash,
        "text": text_proc or "",
        "proof": proof_proc or "",
        "summary": summary_proc or "",
    }

    # 4) Call the chain (centralized try/except)
    try:
        result = chain.invoke(invoke_vars)
    except Exception as e:
        logger.warning(
            "LLM invocation failed for unit %s: %s",
            getattr(row, "id", "<no-id>"),
            str(e),
        )
        return None

    # 5) Validate echoed input_hash
    try:
        model_hash = getattr(result, "input_hash", None)
    except Exception:
        model_hash = None

    if not model_hash or model_hash != input_hash:
        logger.warning(
            "Input-hash validation failed for unit %s: expected %s got %s",
            getattr(row, "id", "<no-id>"),
            input_hash,
            model_hash,
        )
        return None

    # 6) Serialize concepts and merge warnings
    try:
        concepts_json = json.dumps(
            getattr(result, "concepts", []) or [], ensure_ascii=False
        )
    except Exception:
        concepts_json = "[]"

    warnings_list: List[str] = []
    warnings_list.extend(preproc_warnings)
    try:
        rw = getattr(result, "warnings", None)
        if rw:
            # result.warnings might be list[str] or a single string
            if isinstance(rw, list):
                warnings_list.extend(rw)
            else:
                warnings_list.append(str(rw))
    except Exception:
        warnings_list.append("failed_to_read_model_warnings")

    # 7) Build the LLMAnnotation instance (do not persist here)
    try:
        ann = LLMAnnotation(
            id=row.id,
            text_nl=getattr(result, "text_nl", None),
            proof_nl=getattr(result, "proof_nl", None),
            concepts_json=concepts_json,
            input_hash=input_hash,
            warnings=json.dumps(warnings_list, ensure_ascii=False)
            if warnings_list
            else None,
        )
    except Exception as e:
        logger.warning(
            "Failed to build LLMAnnotation for unit %s: %s",
            getattr(row, "id", "<no-id>"),
            str(e),
        )
        return None

    return ann
