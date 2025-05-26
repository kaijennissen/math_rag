import re
import logging
from typing import Optional, Any
from pydantic import BaseModel, field_validator, ConfigDict

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


class AtomicUnit(BaseModel):
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

    # Pydantic v2 model configuration
    model_config = ConfigDict(
        frozen=True,  # Makes the model immutable
        extra="forbid",  # Forbid extra fields not defined in the model
    )

    def get_full_number(self) -> str:
        """Returns the full three-part number string (e.g., '7.4.9')."""
        return f"{self.section}.{self.subsection}.{self.subsubsection}"

    def get_subsection_number(self) -> str:
        """Returns the subsection number string (e.g., '7.4')."""
        return f"{self.section}.{self.subsection}"

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, v: str, values):
        """
        Validate the identifier against section numbers and type.

        Args:
            v: The identifier string to validate
            values: Dictionary containing other field values

        Returns:
            The validated identifier
        """
        if not v:
            return v

        # Get section numbers from model
        section = values.data.get("section")
        subsection = values.data.get("subsection")
        subsubsection = values.data.get("subsubsection")
        type_ = values.data.get("type")

        # 1. Check if identifier contains numbers that should match section numbers
        number_match = re.search(r"(\d+(?:\.\d+){2})", v)
        if number_match:
            expected_number = f"{section}.{subsection}.{subsubsection}"
            if number_match.group(1) != expected_number:
                logging.warning(
                    f"Section number mismatch in identifier. "
                    f"Expected {expected_number}, found {number_match.group(1)}"
                )

        # 2. Check if type matches the German/English type in identifier
        type_match = re.match(r"^([a-zA-ZäöüÄÖÜß]+)", v.strip())
        if type_match:
            german_type = type_match.group(1)
            expected_english_type = GERMAN_TO_ENGLISH_TYPE.get(german_type)

            if expected_english_type is None:
                logging.warning(f"Unknown type '{german_type}' in identifier")
            elif expected_english_type.lower() != type_.lower():
                logging.warning(
                    f"Type mismatch. Identifier suggests '{expected_english_type}', "
                    f"but type is '{type_}'"
                )

        return v

    @classmethod
    def from_dict(cls, data: dict) -> "AtomicUnit":
        """
        Creates an AtomicUnit instance from a dictionary.

        Args:
            data: Dictionary containing the atomic unit data

        Returns:
            An instance of AtomicUnit
        """
        return cls.model_validate(data)

    @classmethod
    def from_db_row(cls, db_row: Any) -> "AtomicUnit":
        """
        Creates an AtomicUnit instance from a database row (db_models.AtomicUnit).
        Args:
            db_row: An instance of db_models.AtomicUnit (SQLModel row).
        Returns:
            AtomicUnit: The Pydantic model instance.
        """
        return cls.model_validate(
            {
                "section": db_row.section,
                "subsection": db_row.subsection,
                "subsubsection": db_row.subsection,
                "type": db_row.type,
                "identifier": db_row.identifier or "",
                "text": db_row.text,
                "section_title": db_row.section_title or "",
                "subsection_title": db_row.subsection_title or "",
                "proof": db_row.proof,
                "summary": db_row.summary,
            }
        )


if __name__ == "__main__":
    import json
    from math_rag.core.project_root import ROOT

    # Get all JSON files in the atomic_units directory
    atomic_units_dir = ROOT / "docs/atomic_units"
    json_files = list(atomic_units_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))

            # Handle both list of chunks and dict with 'chunks' key
            chunks = data if isinstance(data, list) else data.get("chunks", [])
            if not chunks:
                print(f"  No chunks found in {json_file.name}")
                continue

            print(f"  Found {len(chunks)} chunks")
            for i, chunk in enumerate(chunks, 1):
                try:
                    unit = AtomicUnit.from_dict(chunk)
                    # print(f"    [OK] Chunk {i}/{len(chunks)}: {unit.identifier}")
                except Exception as e:
                    print(f"    [FAIL] Chunk {i}/{len(chunks)}: {e}")

        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")
