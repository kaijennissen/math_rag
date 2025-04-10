import argparse
import glob
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import coloredlogs
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import pickle

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

load_dotenv()

# Constants
DOCS_PATH = Path("docs")
SECTIONS_PATH = DOCS_PATH / "sections"
OUTPUT_PATH = DOCS_PATH / "atomic_units"


# Initialize the LLM (allow override via environment variable)
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="o3-mini")


class DocChunk(BaseModel):
    """An atomic segment of a mathematical document, containing a complete logical unit
    such as a theorem, definition, proof, or example, with its hierarchical position
    and classification to facilitate precise knowledge graph construction."""

    # Integer-based hierarchical structure
    section: int = Field(description="The main section number (e.g., 5)")
    section_title: Optional[str] = Field(
        default=None,
        description="The title of the section if applicable (e.g., 'Trennungsaxiome')",
    )
    subsection: Optional[int] = Field(
        default=None,
        description="The relative subsection number (e.g., for 5.2, this would be 2)",
    )
    subsection_title: Optional[str] = Field(
        default=None,
        description="The title of the subsection if applicable (e.g., 'T₀-Räume')",
    )
    subsubsection: Optional[int] = Field(
        default=None,
        description="The relative subsubsection number (e.g., for 5.2.3, this would be 3)",
    )
    type: str = Field(
        description="The type of the mathematical entity (Definition, Theorem, Note, Exercise, Example, Lemma, Proposition, Introduction, Remark, etc.)"
    )
    identifier: Optional[str] = Field(
        default=None,
        description="The identifier of the entity (e.g., 'Theorem 5.1.2', 'Definition 5.2.2')",
    )
    text: str = Field(description="The text contained in the chunk")
    proof: Optional[str] = Field(
        default=None, description="The proof associated with a theorem or proposition"
    )


class Chunks(BaseModel):
    """A collection of structured mathematical document chunks."""

    chunks: list[DocChunk] = Field(
        description="A list of DocChunk objects representing the extracted chunks"
    )


# Create a more explicit system prompt
system_prompt = """
You are a precise mathematical document parser. Your task is to parse German mathematical documents into
structured chunks that preserve their hierarchical organization.

YOU MUST RETURN A VALID JSON OBJECT that follows the schema EXACTLY. Do not include explanations or notes
outside the JSON structure.

Parsing guidelines:
1. Extract sections (like 5, 6), subsections (like 5.1, 5.2), and subsubsections (like 5.1.1, 5.1.2) and the respective titles (like 'Trennungsaxiome')
2. Identify mathematical entities: Satz (Theorem), Definition, Lemma, Aufgabe (Exercise), etc.
3. Maintain the hierarchical relationships between sections, subsections, and subsubsections. I.e. 5.1.2 is a subsubsection of 5.1 and 5.1 is a subsection of 5.
4. Include proofs with their associated theorems
5. CRITICALLY IMPORTANT: Preserve ALL LaTeX mathematical notation EXACTLY as it appears in the text. Do NOT modify, simplify, or escape ANY LaTeX code. All mathematical symbols, commands, and structures must be preserved perfectly.
6. In some cases, the title of a theorem is more verbose, like "Satz von ...". In such cases, use the verbose title as the identifier.
7. Subsections and mathematical entities have the same level of hierarchy, i.e. Theorem 3.1.2 is equivalent to subsubsection 3.1.2

Mathematical terminology in German:
- "Satz" = Theorem
- "Definition" = Definition
- "Lemma" = Lemma
- "Aufgabe" = Exercise
- "Bemerkung" = Remark
- "Beispiel" = Example
- "Vorbemerkung" = Remark
- "Proposition" = Proposition
- "Einleitung" = Introduction

Remember to:
- Use integer numbers for section numbers (5, 6, 7)
- Capture the full text of each entity WITH ALL MATHEMATICAL NOTATION INTACT
- Keep ALL LaTeX notation exactly as is, including all backslashes (\\) and special characters
- NEVER remove, simplify, or escape any LaTeX code - copy it verbatim
- Include proofs with their theorems, remarks, examples, etc.
- Set proper types for each mathematical entity
- Return your response as valid JSON that follows the schema exactly
"""

# Use a more structured prompt with example schema
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "human",
            """Please parse the following mathematical text into structured chunks following the schema:

{{
  "chunks": [
    {{
      "section": 5,
      "section_title": "Trennungsaxiome",
      "subsection": 1,
      "subsection_title": "R₀-Räume",
      "subsubsection": 1,
      "type": "Introduction",
      "identifier": "Vorbemerkung 5.1.1",
      "text": "The full text of this introduction...",
      "proof": null
    }},
    {{
      "section": 5,
      "section_title": "Trennungsaxiome",
      "subsection": 1,
      "subsection_title": "R₀-Räume",
      "subsubsection": 2,
      "type": "Theorem",
      "identifier": "Satz 5.1.2",
      "text": "Ist $f: \\underline{{X}} \\rightarrow \\underline{{Y}}$ surjektiv, stetig und abgeschlossen, ...",
      "proof": "Seien $y_{{1}}$ und $y_{{2}}$ Punkte von ..."
    }}
  ]
}}

Here's the text to parse:

{input}""",
        ),
    ]
)

# Set up the structured output chain
structured_llm = llm.with_structured_output(Chunks, method="json_schema")
chain = chat_prompt | structured_llm


def extract_atomic_units(content: str) -> Chunks:
    """
    Process content to extract atomic units

    Args:
        content: Document or subsection content

    Returns:
        Chunks: A collection of parsed document chunks
    """
    try:
        # Process the content
        chunk_result = chain.invoke({"input": content})
        logger.info(f"Successfully extracted {len(chunk_result.chunks)} atomic units")
        return chunk_result
    except Exception as e:
        logger.error(f"Error extracting atomic units: {str(e)}")
        return Chunks(chunks=[])


def get_section_files_to_process(
    sections: List[int], subsections: List[str]
) -> List[Path]:
    """
    Build a list of files to process based on section and subsection arguments

    Args:
        sections: List of section numbers
        subsections: List of subsection identifiers (e.g., ["5.1", "6.2"])

    Returns:
        List[Path]: List of files to process
    """
    files_to_process = []

    # Add files for section arguments
    if sections:
        for section_num in sections:
            # Simple glob for all files in this section
            pattern = f"section_{section_num}_*.md"
            section_files = glob.glob(str(SECTIONS_PATH / pattern))

            if section_files:
                files_to_process.extend([Path(f) for f in section_files])
                logger.info(
                    f"Found {len(section_files)} files for section {section_num}"
                )
            else:
                logger.warning(f"No files found for section {section_num}")

    # Add files for subsection arguments
    if subsections:
        for subsection_id in subsections:
            try:
                section, subsection = subsection_id.split(".")
                # Direct file pattern for this specific subsection
                pattern = f"section_{section}_{subsection}*.md"
                subsection_files = glob.glob(str(SECTIONS_PATH / pattern))

                if subsection_files:
                    # If multiple files match, just take the first one
                    if len(subsection_files) > 1:
                        logger.warning(
                            f"Multiple files found for subsection {subsection_id}, using first match"
                        )

                    files_to_process.append(Path(subsection_files[0]))
                    logger.info(
                        f"Found file for subsection {subsection_id}: {Path(subsection_files[0]).name}"
                    )
                else:
                    logger.warning(f"No file found for subsection {subsection_id}")
            except ValueError:
                logger.error(
                    f"Invalid subsection format: {subsection_id}. Expected format: '5.1'"
                )

    # Remove duplicates while preserving order
    unique_files = []
    seen = set()
    for file in files_to_process:
        if str(file) not in seen:
            seen.add(str(file))
            unique_files.append(file)

    logger.info(f"Total files to process: {len(unique_files)}")
    return unique_files


def process_file(file_path: Path) -> Optional[Chunks]:
    """
    Process a single file to extract atomic units

    Args:
        file_path: Path to the file to process

    Returns:
        Optional[Chunks]: Extracted atomic units if successful, None otherwise
    """
    logger.info(f"Processing file: {file_path.name}")

    # Extract section and subsection from filename
    match = re.match(r"section_(\d+)_(\d+)", file_path.stem)
    if not match:
        logger.error(f"Invalid file naming pattern: {file_path.name}")
        return None

    section_num = int(match.group(1))
    subsection_num = int(match.group(2))

    try:
        # Read the file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract atomic units
        result = extract_atomic_units(content)

        # Save the result as JSON
        output_file = OUTPUT_PATH / f"section_{section_num}_{subsection_num}_units.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        logger.info(f"Saved {len(result.chunks)} atomic units to {output_file}")

        # Also save as pickle for compatibility
        pickle_output = (
            OUTPUT_PATH / f"section_{section_num}_{subsection_num}_units.pkl"
        )
        with open(pickle_output, "wb") as f:
            pickle.dump(result, f)

        return result

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None


def combine_results_by_section(results: Dict[Path, Chunks]) -> None:
    """
    Combine results by section and save combined files

    Args:
        results: Dictionary mapping file paths to results
    """
    # Group results by section
    section_results: Dict[int, List[DocChunk]] = {}

    for file_path, chunks in results.items():
        if not chunks or not chunks.chunks:
            continue

        # Extract section number from filename
        match = re.match(r"section_(\d+)_", file_path.stem)
        if match:
            section_num = int(match.group(1))

            if section_num not in section_results:
                section_results[section_num] = []

            section_results[section_num].extend(chunks.chunks)

    # Save combined results for each section
    for section_num, chunks_list in section_results.items():
        if not chunks_list:
            continue

        combined_chunks = Chunks(chunks=chunks_list)

        # Save as JSON
        json_output = OUTPUT_PATH / f"section_{section_num}_all_units.json"
        with open(json_output, "w", encoding="utf-8") as f:
            f.write(combined_chunks.model_dump_json(indent=2))

        # Save as pickle
        pickle_output = OUTPUT_PATH / f"section_{section_num}_all_units.pkl"
        with open(pickle_output, "wb") as f:
            pickle.dump(combined_chunks, f)

        logger.info(
            f"Saved combined results for section {section_num} with {len(chunks_list)} units"
        )


def main(sections: List, subsections: List):
    OUTPUT_PATH.mkdir(exist_ok=True)
    # Get files to process
    files_to_process = get_section_files_to_process(sections, subsections)

    if not files_to_process:
        logger.error("No files found to process")
        return 1

    # Process each file
    results = {}
    for file_path in files_to_process:
        result = process_file(file_path)
        if result:
            results[file_path] = result

    logger.info("✅ Extraction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract atomic units from mathematical documents"
    )

    parser.add_argument(
        "--section",
        type=int,
        action="append",
        help="Process all subsections in this section (can be used multiple times)",
    )
    parser.add_argument(
        "--subsection",
        action="append",
        help='Process specific subsection (can be used multiple times, format: "5.1")',
    )

    args = parser.parse_args()

    # Check if at least one argument is provided
    if not args.section and not args.subsection:
        parser.error("At least one --section or --subsection argument is required")

    main(sections=args.section, subsections=args.subsection)
