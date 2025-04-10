import argparse
import os
import re
import yaml
import pickle
import logging
import coloredlogs
from typing import Dict, List

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def strip_loesungshinweise(document_content: str) -> tuple:
    """
    Remove 'Lösungshinweise zu...' sections from the document content and return both parts.
    Assumption is, that the 'Lösungshinweise' sections appear at the end of the document.

    Args:
        document_content: The full text content of the document

    Returns:
        tuple: (main_content, loesungshinweise_content)
    """
    # Pattern to match any Lösungshinweise section
    loesungshinweise_pattern = r"#{2,3}\s+Lösungshinweise zu (Lektion|Kurseinheit)\s+\d"

    # Search for the pattern in the document content
    match = re.search(loesungshinweise_pattern, document_content)

    # If found, split the document
    if match:
        logger.info(f"Found Lösungshinweise section at position {match.start()}")
        main_content = document_content[: match.start()]
        loesungshinweise_content = document_content[match.start() :]
        return main_content, loesungshinweise_content

    # If not found, return the original content and None
    return document_content, None


def split_document_by_section_headers(
    document_content: str, section_headers: List[str]
) -> Dict[str, str]:
    """
    Split a markdown document by specified section headers.

    Args:
        document_content: The full text content of the document
        section_headers: List of section headers to split on (e.g., ["5.1 R_{0}-Räume", "5.2 T_{0}-Räume"])

    Returns:
        Dict[str, str]: A dictionary mapping section headers to their content
    """
    logger.info("Splitting document by specified section headers...")

    # Prepare the section headers for exact matching
    # We'll look for them in the format "## 5.1 R_{0}-Räume" or "## $5.1 \quad \mathrm{R}_{0}$-Räume"
    section_patterns = []
    for header in section_headers:
        # Extract the section number (e.g., "5.1" from "5.1 R_{0}-Räume")
        section_num_match = re.match(r"(\d+\.\d+)", header)
        if section_num_match:
            section_num = section_num_match.group(1)

            # Create patterns for both regular and LaTeX formatted headers
            regular_pattern = f"## {header}"
            latex_pattern_1 = f"## ${section_num} \\quad"
            latex_pattern_2 = f"## ${section_num} \\mathrm"

            section_patterns.append((regular_pattern, section_num))
            section_patterns.append((latex_pattern_1, section_num))
            section_patterns.append((latex_pattern_2, section_num))
        else:
            # For non-numeric headers like "Einleitung"
            regular_pattern = f"## {header}"
            section_patterns.append((regular_pattern, header))

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
                f"Extracted section: {section_id} ({len(section_content)} characters)"
            )

    return sections


def save_sections_to_files(sections: Dict[str, str], output_dir: str, prefix: str = ""):
    """
    Save sections to individual files.

    Args:
        sections: Dictionary mapping section IDs to content
        output_dir: Directory to save files to
        prefix: Optional prefix for filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each section to a file
    for section_id, content in sections.items():
        # Create a safe filename
        safe_id = section_id.replace(".", "_").replace(" ", "_")
        filename = f"{prefix}{safe_id}.md"
        filepath = os.path.join(output_dir, filename)

        # Save the file
        with open(filepath, "w") as f:
            f.write(content)

        logger.info(f"Saved section {section_id} to {filepath}")


def process_and_save_document(
    document_content: str, section_headers: List[str], output_dir: str
):
    """
    Process a document, split it into sections, and save to files.

    Args:
        document_content: The document content to process
        section_headers: List of section headers to split on
        output_dir: Directory to save files to
    """
    section_numbers = set(
        [re.search(r"\d+", header).group() for header in section_headers]
    )
    assert len(section_numbers) == 1, (
        "All section headers must have the same first number"
    )
    section_number = next(iter(section_numbers))

    # Split the document into main content and Lösungshinweise
    main_content, loesungshinweise_content = strip_loesungshinweise(document_content)

    # Process main content
    main_sections = split_document_by_section_headers(main_content, section_headers)

    # Save main sections
    save_sections_to_files(main_sections, output_dir, prefix="section_")

    # Save Lösungshinweise if found
    if loesungshinweise_content:
        loesungshinweise_sections = {
            f"solutions_{section_number}": loesungshinweise_content
        }
        save_sections_to_files(loesungshinweise_sections, output_dir)


def load_section_headers_from_yaml(yaml_file: str) -> List[str]:
    """
    Load section headers from a YAML file.

    Args:
        yaml_file: Path to the YAML file containing section headers

    Returns:
        List[str]: List of section headers
    """
    logger.info(f"Loading section headers from YAML file: {yaml_file}")

    if not os.path.exists(yaml_file):
        logger.warning(f"YAML file {yaml_file} does not exist")
        return []

    try:
        with open(yaml_file, "r") as f:
            section_headers = yaml.safe_load(f)

        if not isinstance(section_headers, list):
            logger.warning(
                f"YAML file {yaml_file} does not contain a list of section headers"
            )
            return []

        logger.info(f"Loaded {len(section_headers)} section headers from {yaml_file}")
        return section_headers
    except Exception as e:
        logger.error(f"Error reading YAML file {yaml_file}: {e}")
        return []


def main(
    section_number: int,
    yaml_file: str = "docs/section_headers.yaml",
    output_dir: str = "docs/sections",
):
    """
    Main function to process a document by section number.

    Args:
        section_number: The section number to process (e.g., 5 for KE_5)
        yaml_file: Path to the YAML file containing section headers
        output_dir: Directory to save the processed sections
    """
    # Construct the document ID and file path
    document_id = f"KE_{section_number}"
    pickle_file = f"docs/cached_{document_id}.pkl"

    logger.info(f"Processing section {section_number} from {pickle_file}")

    # Load the document
    try:
        docs = pickle.load(open(pickle_file, "rb"))
        document_content = docs[0].page_content
        logger.info(f"Loaded document content ({len(document_content)} characters)")
    except FileNotFoundError:
        logger.error(f"Document file {pickle_file} not found")
        return
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load section headers from YAML file
    section_headers = load_section_headers_from_yaml(yaml_file)

    # Filter headers for the current section number
    filtered_headers = [
        header for header in section_headers if header.startswith(f"{section_number}.")
    ]

    if not filtered_headers:
        logger.warning(f"No headers found for section {section_number} in {yaml_file}")
        return

    logger.info(f"Found {len(filtered_headers)} headers for section {section_number}")

    # Process and save the document
    process_and_save_document(document_content, filtered_headers, output_dir)

    logger.info(
        f"✅ Done! All sections for section {section_number} saved to {output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save document sections")
    parser.add_argument(
        "--section",
        type=int,
        help="Section number to process (e.g., 5 for KE_5)",
        required=True,
    )
    parser.add_argument(
        "--yaml",
        default="docs/section_headers.yaml",
        help="Path to YAML file with section headers",
    )
    parser.add_argument(
        "--output",
        default="docs/sections",
        help="Output directory for processed sections",
    )

    args = parser.parse_args()

    main(section_number=args.section, yaml_file=args.yaml, output_dir=args.output)
