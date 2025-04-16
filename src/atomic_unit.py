import re
import logging
from dataclasses import dataclass
from typing import Optional

# Setup basic logging configuration
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Lookup table for German type names to English
GERMAN_TO_ENGLISH_TYPE = {
    "Satz": "Theorem",
    "Definition": "Definition",
    "Lemma": "Lemma",
    "Korollar": "Corollary",
    "Beispiel": "Example",
    "Bemerkung": "Remark",
    "Aufgabe": "Exercise",
    "Vorbemerkung": "Remark",
    "Proposition": "Proposition",
    "Einleitung": "Introduction",
}


@dataclass(frozen=True)
class AtomicUnit:
    """Represents a single atomic unit of mathematical content."""

    section: int
    section_title: str
    subsection: int
    subsection_title: str
    subsubsection: int
    type: str  # English type name
    identifier: str  # e.g., "Satz 7.4.9"
    text: str
    proof: Optional[str] = None

    def get_full_number(self) -> str:
        """Returns the full three-part number string (e.g., '7.4.9')."""
        return f"{self.section}.{self.subsection}.{self.subsubsection}"

    def get_subsection_number(self) -> str:
        """Returns the subsection number string (e.g., '7.4')."""
        return f"{self.section}.{self.subsection}"

    def __post_init__(self):
        """Performs consistency checks after initialization, logging warnings for mismatches."""
        # 1. Check number consistency
        expected_number = self.get_full_number()
        match = re.search(r"(\d+\.\d+\.\d+)", self.identifier)
        if not match:
            logging.warning(
                f"Could not extract number from identifier '{self.identifier}'. Skipping number check."
            )
        elif match.group(1) != expected_number:
            logging.warning(
                f"Identifier number mismatch for '{self.identifier}'. "
                f"Expected '{expected_number}', found '{match.group(1)}'."
            )

        # 2. Check type consistency
        identifier_type_match = re.match(r"([a-zA-ZäöüÄÖÜß]+)", self.identifier)
        if not identifier_type_match:
            logging.warning(
                f"Could not extract type from identifier '{self.identifier}'. Skipping type check."
            )
        else:
            german_type = identifier_type_match.group(1)
            expected_english_type = GERMAN_TO_ENGLISH_TYPE.get(german_type, "Unknown")

            if not expected_english_type:
                # Keep this as an error - indicates missing config
                logging.warning(
                    f"Unknown German type '{german_type}' in identifier '{self.identifier}'. Please update GERMAN_TO_ENGLISH_TYPE map."
                )
            try:
                if expected_english_type.lower() != self.type.lower():
                    logging.warning(
                        f"Type mismatch for identifier '{self.identifier}'. "
                        f"Identifier implies '{expected_english_type}', but type field is '{self.type}'."
                    )
            except AttributeError:
                logging.warning(
                    f"Type field is missing in identifier '{self.identifier}'. "
                    f"Expected type '{expected_english_type}'."
                )

    @classmethod
    def from_dict(cls, data: dict):
        """Creates an AtomicUnit instance from a dictionary."""
        return cls(
            section=data.get("section"),
            section_title=data.get("section_title"),
            subsection=data.get("subsection"),
            subsection_title=data.get("subsection_title"),
            subsubsection=data.get("subsubsection"),
            type=data.get("type"),
            identifier=data.get("identifier"),
            text=data.get("text"),
            proof=data.get("proof"),
        )
