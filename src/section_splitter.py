import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import coloredlogs
import pickle
import yaml
from langchain.schema import Document

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Constants
DOCS_PATH = Path("docs")
SECTIONS_PATH = DOCS_PATH / "sections"
PROCESSED_PATH = DOCS_PATH / "processed"
SECTION_HEADERS_PATH = DOCS_PATH / "section_headers.yaml"


def load_section_headers() -> List[Tuple[int, str]]:
    """
    Load section headers from the YAML file.

    Returns:
        List[Tuple[int, str]]: List of tuples (section_number, section_title)
    """
    try:
        with open(SECTION_HEADERS_PATH, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)

        section_headers = []
        for key, subsections in yaml_content.items():
            # Extract the section number and title
            section_match = re.match(r"(\d+)\s+(.*)", key)
            if section_match:
                section_num = int(section_match.group(1))
                section_title = section_match.group(2)
                section_headers.append((section_num, section_title))

        # Sort by section number
        section_headers.sort(key=lambda x: x[0])

        logger.info(
            f"Loaded {len(section_headers)} section headers from {SECTION_HEADERS_PATH}"
        )
        for num, title in section_headers:
            logger.info(f"  Section {num}: {title}")

        return section_headers
    except FileNotFoundError:
        logger.error(f"Section headers file {SECTION_HEADERS_PATH} not found")
        return []
    except Exception as e:
        logger.error(f"Error loading section headers: {e}")
        return []


def split_document_by_section_headers(
    document_content: str, section_headers: List[Tuple[int, str]]
) -> Dict[int, str]:
    """
    Split a document into major sections.

    Args:
        document_content: The full text content of the document
        section_headers: List of tuples (section_number, section_title)

    Returns:
        Dict[int, str]: A dictionary mapping section numbers or identifiers to their content
    """
    logger.info("Splitting document by specified section headers...")

    if not section_headers:
        logger.warning("No section headers provided")
        return {}

    # Prepare the section headers for exact matching
    section_patterns = []
    for section_num, section_title in section_headers:
        # Create patterns for different header formats
        regular_pattern = f"## Lektion {section_num}\n\n ##{section_title}"
        # latex_pattern_1 = f"# ${section_num} \quad"
        # latex_pattern_2 = f"# ${section_num} \mathrm"
        # alt_pattern = f"## {section_num}. {section_title}"

        section_patterns.append((regular_pattern, section_num))
        # section_patterns.append((alt_pattern, section_num))
        # section_patterns.append((latex_pattern_1, section_num))
        # section_patterns.append((latex_pattern_2, section_num))

    # Find the positions of all section headers in the document
    section_positions = []

    # Process section headers
    for pattern, section_id in section_patterns:
        for match in re.finditer(re.escape(pattern), document_content):
            section_positions.append((match.start(), pattern, section_id))

    # Sort positions by their occurrence in the document
    section_positions.sort(key=lambda x: x[0])
    if not section_positions:
        logger.warning("No section headers found in the document!")
        return {"entire_document": document_content}

    # Split the document based on the positions
    sections = {}
    for i, (pos, pattern, section_id) in enumerate(section_positions):
        # Find the end of this section (start of next section or end of document)
        next_pos = (
            section_positions[i + 1][0]
            if i < len(section_positions) - 1
            else len(document_content)
        )

        # Extract the section content
        section_content = document_content[pos:next_pos].strip()

        # Use the section ID as the key
        if (
            section_id not in sections
        ):  # Only add if not already present (avoid duplicates)
            sections[section_id] = section_content
            logger.info(
                f"Extracted section {section_id} ({len(section_content)} characters)"
            )

    return sections


def save_sections_to_files(sections: Dict[int, str]) -> None:
    """
    Save each section to its own file.

    Args:
        sections: Dictionary mapping section numbers to content
    """
    for section_num, content in sections.items():
        # Create a Document object with the section content
        doc = Document(
            page_content=content, metadata={"source": f"section_{section_num}"}
        )

        # Save as pickle
        output_file = SECTIONS_PATH / f"section_{section_num}.pkl"
        with output_file.open("wb") as f:
            pickle.dump([doc], f)  # Save as a list of documents for consistency

        # Also save as text for inspection
        text_file = SECTIONS_PATH / f"section_{section_num}.md"
        with text_file.open("w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Saved section {section_num} to {output_file} and {text_file}")


def main(input_file: str, section_numbers: Optional[List[int]] = None) -> None:
    """
    Main function to split a document into major sections using predefined headers.

    Args:
        input_file: Path to the processed document
        section_numbers: Optional list of specific section numbers to extract
    """
    SECTIONS_PATH.mkdir(exist_ok=True)
    PROCESSED_PATH.mkdir(exist_ok=True)
    input_path = Path(input_file)

    # Load the document
    try:
        if input_path.suffix.lower() == ".pkl":
            with open(input_path, "rb") as f:
                docs = pickle.load(f)
                document_content = docs[0].page_content
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                document_content = f.read()

        logger.info(
            f"Loaded document from {input_path} ({len(document_content)} characters)"
        )
    except FileNotFoundError:
        logger.error(f"Document file {input_path} not found")
        return
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        return

    # Load section headers from YAML file
    headers = load_section_headers()

    # Filter sections if specific ones were requested
    if section_numbers:
        headers = [(num, title) for num, title in headers if num in section_numbers]
        logger.info(f"Filtered to {len(headers)} requested sections")

    if not headers:
        logger.warning("No matching section headers found")
        return

    # Split the document
    sections = split_document_by_section_headers(document_content, headers)

    # Save the sections
    save_sections_to_files(sections)

    logger.info("✅ Document successfully split into major sections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split document into major sections")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the processed document file (pickle or text)",
    )
    parser.add_argument(
        "--section",
        type=int,
        action="append",
        help="Specific section number to extract (can be used multiple times)",
    )

    args = parser.parse_args()

    main(input_file=args.input, section_numbers=args.section)
