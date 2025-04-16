import concurrent.futures
from tqdm import tqdm
import argparse
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

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
SUBSECTIONS_PATH = DOCS_PATH / "subsections"
OUTPUT_PATH = DOCS_PATH / "atomic_units"


# Initialize the LLM (allow override via environment variable)
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4.1")


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
        description="The 'german' identifier of the entity (e.g., 'Satz 5.1.2', 'Definition 5.2.2', 'Beispiel 5.3.2', 'Lemma 5.4.2')",
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
You are a precise mathematical document parser specializing in translating German mathematical documents into structured JSON. Your task is to analyze each document and create a hierarchical representation that maintains all mathematical content with perfect fidelity.

## OUTPUT FORMAT
Return ONLY a valid JSON object matching the schema exactly - no explanations, comments or markdown outside the JSON structure.

<<<<<<< HEAD
## PARSING RULES

### Hierarchical Structure:
- Extract sections (e.g., 5, 6), subsections (e.g., 5.1, 5.2), and subsubsections (e.g., 5.1.1, 5.1.2)
- Capture their respective titles (e.g., 'Trennungsaxiome')
- Maintain correct hierarchical relationships: 5.1.2 is a subsubsection of 5.1, which is a subsection of 5
- Subsections and mathematical entities have the same level of hierarchy (e.g., Theorem 3.1.2 is equivalent to subsubsection 3.1.2)
=======
Parsing guidelines:
1. Extract sections (like 5, 6), subsections (like 5.1, 5.2), and subsubsections (like 5.1.1, 5.1.2) and the respective titles (like 'Trennungsaxiome')
2. Identify mathematical entities: Satz (Theorem), Definition, Lemma, Aufgabe (Exercise), etc.
3. In case you cannot identify the mathematical entity, use 'Remark' as the type.
4. Introductions are only allowed at the beginning of a section, i.e. as the first chunk of a section.
5. Maintain the hierarchical relationships between sections, subsections, and subsubsections. I.e. 5.1.2 is a subsubsection of 5.1 and 5.1 is a subsection of 5.
6. Include proofs with their associated theorems
7. CRITICALLY IMPORTANT: Preserve ALL LaTeX mathematical notation EXACTLY as it appears in the text. Do NOT modify, simplify, or escape ANY LaTeX code. All mathematical symbols, commands, and structures must be preserved perfectly.
8. In some cases, the title of a theorem is more verbose, like "Satz von ...". In such cases, use the verbose title as the identifier.
9. Subsections and mathematical entities have the same level of hierarchy, i.e. Theorem 3.1.2 is equivalent to subsubsection 3.1.2
10. In some cases there are comments regarding the notation at the beginning of a section, subsection or subsubsection. Do not include these comments.

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
- "Folgerung" = Corollary
- "Korollar" = Corollary
>>>>>>> main

### Mathematical Entities:
- Identify mathematical entities by their German names: Satz, Definition, Lemma, Beispiel, etc.
- Convert to appropriate English types as specified in the terminology mapping
- Use 'Remark' as the type if unable to identify the mathematical entity type
- Use singular forms for all types, even when the German term is plural (e.g., 'Beispiele' → 'Example')
- For verbose theorem names like "Satz von Pythagoras", use the entire phrase as the identifier

### Content Rules:
- Preserve ALL LaTeX mathematical notation EXACTLY - this is CRITICAL
- Do not modify, simplify, or escape any LaTeX code
- Retain all backslashes (\\), special characters, and command structures exactly as they appear
- Include proofs with their associated theorems, lemmas, etc.
- Skip notation comments that appear at the beginning of sections/subsections
- Introductions ("Einleitung") are only allowed at the beginning of a section

### Special Cases:
- When multiple entities appear consecutively without clear separation, create separate chunks
- For content that doesn't match a standard pattern, use the most appropriate type based on context
- If no identifier is present but the content is clearly a mathematical entity, generate an appropriate identifier

## GERMAN TO ENGLISH TERMINOLOGY
- "Satz" → "Theorem"
- "Definition"/"Definitionen" → "Definition"
- "Lemma" → "Lemma"
- "Korollar"/"Folgerung"/"Folgerungen" → "Corollary"
- "Beispiel"/"Beispiele" → "Example"
- "Bemerkung"/"Bemerkungen" → "Remark"
- "Aufgabe"/"Aufgaben" → "Exercise"
- "Vorbemerkung" → "Remark"
- "Proposition"/"Propositionen" → "Proposition"
- "Einleitung" → "Introduction"

## EXAMPLE TRANSFORMATION

German Source:
```
5 Trennungsaxiome

5.1 R_{{0}}-Räume

Wir betrachten nun...

Satz 5.1.1. Sei (X,T) ein topologischer Raum. X ist ein R_{{0}}-Raum genau dann, wenn für alle $x,y \in X$ gilt: $\overline{{x}} = \overline{{y}} \iff x = y$.

Beweis. Sei X ein R_{{0}}-Raum und seien $x,y \in X$ mit $\overline{{x}} = \overline{{y}}$...
```

JSON Output:
```json
{{
  "chunks": [
    {{
      "section": 5,
      "section_title": "Trennungsaxiome",
      "subsection": 1,
      "subsection_title": "R_{{0}}-Räume",
      "subsubsection": 1,
      "type": "Theorem",
      "identifier": "Satz 5.1.1",
      "text": "Sei (X,T) ein topologischer Raum. X ist ein R_{{0}}-Raum genau dann, wenn für alle $x,y \\in X$ gilt: $\\overline{{x}} = \\overline{{y}} \\iff x = y$.",
      "proof": "Sei X ein R_{{0}}-Raum und seien $x,y \\in X$ mit $\\overline{{x}} = \\overline{{y}}$..."
    }}
  ]
}}
```
"""

# Create prompt template without redundant schema definition
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "human",
            """Parse the following German mathematical text into structured chunks according to the guidelines. Remember to preserve all LaTeX notation exactly as it appears in the text.

        Text to parse:

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


def get_subsection_files_to_process(
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
            # Use pathlib's glob method directly
            section_files = list(
                SUBSECTIONS_PATH.glob(f"subsection_{section_num}_*.md")
            )

            if section_files:
                files_to_process.extend(section_files)
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
                file_path = SUBSECTIONS_PATH / f"subsection_{section}_{subsection}.md"
                if file_path.exists():
                    files_to_process.append(file_path)
                    logger.info(
                        f"Found file for subsection {subsection_id}: {file_path.name}"
                    )
                else:
                    logger.warning(f"No file found for subsection {subsection_id}")
            except ValueError:
                logger.error(
                    f"Invalid subsection format: {subsection_id}. Expected format: '5.1'"
                )

    unique_files = list(set(files_to_process))

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
    match = re.match(r"subsection_(\d+)_(\d+)", file_path.stem)
    if not match:
        logger.error(f"Invalid file naming pattern: {file_path.name}")
        return None

    section_num = int(match.group(1))
    subsection_num = int(match.group(2))

    try:
        # Read the file content
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()

        # Extract atomic units
        result = extract_atomic_units(content)

        # Save the result as JSON
        output_file = (
            OUTPUT_PATH / f"subsection_{section_num}_{subsection_num}_units.json"
        )
        with output_file.open("w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        logger.info(f"Saved {len(result.chunks)} atomic units to {output_file}")

        # Also save as pickle for compatibility
        pickle_output = (
            OUTPUT_PATH / f"subsection_{section_num}_{subsection_num}_units.pkl"
        )
        with pickle_output.open("wb") as f:
            pickle.dump(result, f)

        return result

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None


def main(sections: List, subsections: List):
    OUTPUT_PATH.mkdir(exist_ok=True)
    # Get files to process
    files_to_process = get_subsection_files_to_process(sections, subsections)

    if not files_to_process:
        logger.error("No files found to process")
        return

    # Set max_workers to a reasonable number to avoid API rate limits
    max_workers = max(
        8, len(files_to_process)
    )  # Tune as needed, e.g., 5-8 is safe for OpenAI
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file_path): file_path
            for file_path in files_to_process
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_file), total=len(future_to_file)
        ):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result:
                    file_name = file_path.name
                    results[file_name] = len(result.chunks)
                    logger.info(
                        f"✓ Completed: {file_name} with {len(result.chunks)} units"
                    )
            except Exception as exc:
                logger.error(f"❌ {file_path} generated an exception: {exc}")

    total_chunks = sum(results.values())
    logger.info(
        f"✅ Extraction complete! Processed {len(results)} files with {total_chunks} total atomic units."
    )


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
