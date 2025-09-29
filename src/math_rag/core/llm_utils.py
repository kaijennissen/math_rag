"""
Utilities for LLM-driven processing.

Provides:
- compute_input_hash(...) : deterministic sha256 hash used for idempotency/audit.
- preprocess_latex(...)  : light-weight LaTeX pre-processing to remove trivial
  wrappers and normalize input for the LLM.

Design notes (POC):
- Keep preprocessing conservative: remove formatting wrappers but preserve math
  content. Do not try to expand or fully normalize LaTeX macros.
- Return warnings as a JSON-encoded list (string) to make it easy to persist in
  a DB field without introducing migration complexity.
"""

import hashlib
import json
import re
from typing import List, Optional, Tuple


def compute_input_hash(
    text: Optional[str],
    proof: Optional[str],
    summary: Optional[str],
    prompt_version: str,
) -> str:
    """
    Compute a stable sha256 hash for the concatenation of inputs and prompt_version.

    The hash is returned as a string prefixed with "sha256:" to make it explicit
    when stored in DB fields.

    Args:
        text: main text field (may be None)
        proof: proof field (may be None)
        summary: summary field (may be None)
        prompt_version: prompt template/version identifier

    Returns:
        A string like "sha256:<hex>"
    """
    parts = [text or "", proof or "", summary or "", prompt_version or ""]
    joined = "\n<<FIELD_SEPARATOR>>\n".join(parts)
    h = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def preprocess_latex(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Perform lightweight LaTeX cleanup to reduce trivial markup burden for the LLM.

    The function:
    - Normalizes repeated backslash paragraph-separators (\\) into newlines.
    - Strips inline/display math delimiters ($...$, $$...$$, \\(...\\), \\[...\\]) but
      retains the content inside.
    - Removes simple one-level wrappers like \\underline{...}, \\emph{...},
      \\textbf{...}, \\mathrm{...} by keeping their inner content.
    - Trims and normalizes whitespace.

    NOTE: This is intentionally conservative. Nested macros, macro definitions,
    and complex TeX constructs are left intact. If preprocessing makes any change
    a warning token is emitted so the caller can log or inspect it.

    Args:
        text: raw input string possibly containing LaTeX markup.

    Returns:
        Tuple of (processed_text_or_None, warnings_json_or_None)

        - processed_text_or_None: the preprocessed string or None if input was None.
        - warnings_json_or_None: JSON-encoded list of strings describing changes or
          issues (e.g. ["stripped_simple_wrappers"]), or None when no warnings.
    """
    if text is None:
        return None, None

    warnings: List[str] = []
    s = text

    # Normalize common paragraph separators (LaTeX often uses "\\" to force line breaks)
    # Convert double backslash sequences into single newline characters.
    if "\\\\" in s:
        s = s.replace("\\\\", "\n")
        warnings.append("normalized_double_backslashes")

    # Remove display and inline math delimiters but keep inner content.
    # Order matters: handle $$...$$ first.
    # Use DOTALL to allow newlines inside math regions.
    s_before = s
    s = re.sub(r"\$\$(.*?)\$\$", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\$(.*?)\$", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\\\((.*?)\\\)", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\\\[(.*?)\\\]", r"\1", s, flags=re.DOTALL)
    if s != s_before:
        warnings.append("stripped_math_delimiters")

    def _strip_simple_wrappers(inp: str) -> Tuple[str, bool]:
        pattern = re.compile(
            r"\\(?:underline|emph|textbf|mathrm)\{([^{}]*)\}", flags=re.DOTALL
        )
        new, n = pattern.subn(r"\1", inp)
        return new, n > 0

    s_after_wrappers, changed = _strip_simple_wrappers(s)
    if changed:
        warnings.append("stripped_simple_wrappers")
        s = s_after_wrappers

    # Collapse multiple spaces and normalize whitespace around newlines
    s = re.sub(r"[ \t]+", " ", s)
    # Remove trailing/leading whitespace on each line
    s = "\n".join(line.rstrip() for line in s.splitlines())
    s = s.strip()

    # If the processed text is empty (very unlikely), warn
    if len(s) == 0:
        warnings.append("resulting_text_empty")

    return s, (json.dumps(warnings, ensure_ascii=False) if warnings else None)
