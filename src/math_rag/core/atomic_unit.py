import re
import logging
from dataclasses import dataclass
from typing import Optional, Any

# Setup basic logging configuration
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Lookup table for German type names to English
GERMAN_TO_ENGLISH_TYPE = {
    "Satz": "Theorem",
    "Definition": "Definition",
    "Definitionen": "Definition",
    "Lemma": "Lemma",
    "Korollar": "Corollary",
    "Folgerung": "Corollary",
    "Folgerungen": "Corollary",
    "Beispiel": "Example",  # Singular
    "Beispiele": "Example",  # Singular
    "Bemerkung": "Remark",  # Singular
    "Bemerkungen": "Remark",  # Plural
    "Aufgabe": "Exercise",  # Singular
    "Aufgaben": "Exercise",  # Plural
    "Vorbemerkung": "Remark",
    "Proposition": "Proposition",
    "Propositionen": "Proposition",
    "Einleitung": "Introduction",
}


@dataclass(frozen=True)
class AtomicUnit:
    """Represents a single atomic unit of mathematical content."""

    section: int
    subsection: int
    subsubsection: int
    type: str  # English type name
    identifier: str  # e.g., "Satz 7.4.9"
    text: str
    section_title: Optional[str] = ""
    subsection_title: Optional[str] = ""
    proof: Optional[str] = None
    summary: Optional[str] = None

    def get_full_number(self) -> str:
        """Returns the full three-part number string (e.g., '7.4.9')."""
        return f"{self.section}.{self.subsection}{f'.{self.subsubsection}' if self.subsubsection else ''}"

    def get_subsection_number(self) -> str:
        """Returns the subsection number string (e.g., '7.4')."""
        return f"{self.section}.{self.subsection}"

    def __post_init__(self):
        """Performs consistency checks after initialization, logging warnings for mismatches."""
        # Skip identifier-based checks only if it's likely an intro chunk
        # (no identifier AND no subsubsection number).
        # If subsubsection exists, identifier checks should run even if identifier is missing
        # (which will likely trigger warnings as expected).
        if not self.identifier:
            return

        # --- Proceed with existing checks only if identifier is present OR subsubsection is present ---

        # 1. Check number consistency (only if identifier is actually present)
        if self.identifier:
            expected_number = self.get_full_number()
            try:
                match = re.search(r"(\d+\.\d+\.\d+)", self.identifier)
            except Exception:  # Keep broad exception for safety during regex
                logging.warning(
                    f"Regex error extracting number from identifier '{self.identifier}'. Skipping number check."
                )
                match = None  # Ensure match is None if regex fails

            if not match:
                # If we expected an identifier (because subsubsection is not None), warn here.
                if self.subsubsection is not None:
                    logging.warning(
                        f"Could not extract number from identifier '{self.identifier}' for subsubsection {self.get_full_number()}. Skipping number check."
                    )
            elif match.group(1) != expected_number:
                logging.warning(
                    f"Identifier number mismatch for '{self.identifier}'. "
                    f"Expected '{expected_number}', found '{match.group(1)}'."
                )
        elif self.subsubsection is not None:
            # Identifier is missing, but subsubsection is not None - this is suspicious.
            logging.warning(
                f"Missing identifier for content in subsubsection {self.get_full_number()}."
            )

        # 2. Check type consistency (only if identifier is actually present)
        if self.identifier:
            identifier_type_match = re.match(r"([a-zA-ZäöüÄÖÜß]+)", self.identifier)
            if not identifier_type_match:
                # If we expected an identifier (because subsubsection is not None), warn here.
                if self.subsubsection is not None:
                    logging.warning(
                        f"Could not extract type from identifier '{self.identifier}' for subsubsection {self.get_full_number()}. Skipping type check."
                    )
            else:
                german_type = identifier_type_match.group(1)
                expected_english_type = GERMAN_TO_ENGLISH_TYPE.get(
                    german_type
                )  # Removed default="Unknown"

                if expected_english_type is None:  # Explicit check for None is better
                    # Keep this as a warning - indicates missing config or unexpected type
                    logging.warning(
                        f"Unknown German type '{german_type}' found in identifier '{self.identifier}'. "
                        f"Please update GERMAN_TO_ENGLISH_TYPE map if this type is valid."
                    )
                elif expected_english_type.lower() != self.type.lower():
                    logging.warning(
                        f"Type mismatch for identifier '{self.identifier}'. "
                        f"Identifier implies '{expected_english_type}', but type field is '{self.type}'."
                    )
        # No specific type check needed if identifier is missing, number check covers the warning.

    @classmethod
    def from_dict(cls, data: dict):
        """Creates an AtomicUnit instance from a dictionary."""
        return cls(
            section=data.get("section"),
            subsection=data.get("subsection"),
            subsubsection=data.get("subsubsection"),
            type=data.get("type"),
            identifier=data.get("identifier"),
            text=data.get("text"),
            section_title=data.get("section_title"),
            subsection_title=data.get("subsection_title"),
            proof=data.get("proof"),
            summary=data.get("summary"),
        )

    @classmethod
    def from_db_row(cls, db_row: "Any") -> "AtomicUnit":
        """
        Creates an AtomicUnit instance from a database row (db_models.AtomicUnit).
        Args:
            db_row: An instance of db_models.AtomicUnit (SQLModel row).
        Returns:
            AtomicUnit: The core dataclass instance.
        """
        return cls(
            section=db_row.section,
            subsection=db_row.subsection,
            subsubsection=db_row.subsubsection,
            type=db_row.type,
            identifier=db_row.identifier or "",
            text=db_row.text,
            section_title=db_row.section_title,
            subsection_title=db_row.subsection_title,
            proof=db_row.proof,
            summary=db_row.summary,
        )
