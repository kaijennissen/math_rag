import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

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

# Ensure directories exist
SECTIONS_PATH.mkdir(exist_ok=True)
PROCESSED_PATH.mkdir(exist_ok=True)


def extract_section_headers_from_document(document_content: str) -> List[str]:
    """
    Extract major section headers from the document content.

    Args:
        document_content: The full text content of the document

    Returns:
        List[str]: List of major section headers found in the document
    """
    # Pattern to match major section headers (e.g., # 5 Title or ## 5. Title)
    pattern = r"^#+\s+(\d+)\.?\s+([^\n]+)$"

    # Find all major section headers
    section_headers = []
    for line in document_content.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            section_number = match.group(1)
            section_title = match.group(2).strip()
            section_headers.append((int(section_number), section_title))

    # Sort by section number
    section_headers.sort(key=lambda x: x[0])

    logger.info(f"Found {len(section_headers)} major section headers")
    for num, title in section_headers:
        logger.info(f"  Section {num}: {title}")

    return section_headers


def split_document_by_major_sections(
    document_content: str, section_headers: List[tuple]
) -> Dict[int, str]:
    """
    Split a document into major sections.

    Args:
        document_content: The full text content of the document
        section_headers: List of tuples (section_number, section_title)

    Returns:
        Dict[int, str]: A dictionary mapping section numbers to their content
    """
    if not section_headers:
        logger.warning("No section headers provided")
        return {}

    sections = {}
    lines = document_content.split("\n")

    # Create a list of section starts
    section_starts = []
    for i, line in enumerate(lines):
        for section_num, section_title in section_headers:
            # Match either "# 5 Title" or "## 5. Title" format
            if re.match(
                r"^#+\s+{}\.?\s+{}".format(section_num, re.escape(section_title)),
                line.strip(),
            ):
                section_starts.append((i, section_num, section_title))
                break

    # Sort by line number
    section_starts.sort(key=lambda x: x[0])

    # Extract each section's content
    for i, (start_line, section_num, section_title) in enumerate(section_starts):
        end_line = (
            section_starts[i + 1][0] if i < len(section_starts) - 1 else len(lines)
        )
        content = "\n".join(lines[start_line:end_line])
        sections[section_num] = content
        logger.info(f"Extracted section {section_num} ({len(content)} characters)")

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
    Main function to split a document into major sections.

    Args:
        input_file: Path to the processed document
        section_numbers: Optional list of specific section numbers to extract
    """
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

    # Extract section headers
    headers = extract_section_headers_from_document(document_content)

    # Filter sections if specific ones were requested
    if section_numbers:
        headers = [(num, title) for num, title in headers if num in section_numbers]
        logger.info(f"Filtered to {len(headers)} requested sections")

    if not headers:
        logger.warning("No matching section headers found")
        return

    # Split the document
    sections = split_document_by_major_sections(document_content, headers)

    # Save the sections
    save_sections_to_files(sections)

    # Save the headers to YAML for use by the subsection splitter
    headers_yaml = DOCS_PATH / "section_headers.yaml"
    subsection_headers = []

    # Extract subsection headers from each section
    for section_num, content in sections.items():
        # Pattern to match subsection headers (e.g., ## 5.1 Title)
        pattern = r"^#+\s+{}\.(\d+)\.?\s+([^\n]+)$".format(section_num)

        for line in content.split("\n"):
            match = re.match(pattern, line.strip())
            if match:
                subsection_num = match.group(1)
                subsection_title = match.group(2).strip()
                subsection_headers.append(
                    f"{section_num}.{subsection_num} {subsection_title}"
                )

    # Save to YAML
    with open(headers_yaml, "w", encoding="utf-8") as f:
        yaml.dump(subsection_headers, f, default_flow_style=False)

    logger.info(f"Saved {len(subsection_headers)} subsection headers to {headers_yaml}")
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
