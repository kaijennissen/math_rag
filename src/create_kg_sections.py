import re
from typing import List, Dict

import pickle

# Load the documents
docs = pickle.load(open("docs/cached_KE_5.pkl", "rb"))


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
    print("Splitting document by specified section headers...")

    # Prepare the section headers for exact matching
    # We'll look for them in the format "## 5.1 R_{0}-Räume" or "## $5.1 \quad \mathrm{R}_{0}$-Räume"
    section_patterns = []
    for header in section_headers:
        # Extract the section number (e.g., "5.1" from "5.1 R_{0}-Räume")
        section_num = re.match(r"(\d+\.\d+)", header).group(1)

        # Create patterns for both regular and LaTeX formatted headers
        regular_pattern = f"## {header}"
        latex_pattern_1 = f"## ${section_num} \\quad"
        latex_pattern_2 = f"## ${section_num} \\mathrm"

        section_patterns.append(regular_pattern)
        section_patterns.append(latex_pattern_1)
        section_patterns.append(latex_pattern_2)

    # Find the positions of all section headers in the document
    section_positions = []
    for pattern in section_patterns:
        for match in re.finditer(re.escape(pattern), document_content):
            # Extract the section number from the pattern
            if "$" in pattern:
                # For LaTeX patterns
                section_num = re.search(r"\$\s*(\d+\.\d+)", pattern).group(1)
            else:
                # For regular patterns
                section_num = re.search(r"## (\d+\.\d+)", pattern).group(1)

            section_positions.append((match.start(), pattern, section_num))

    # Sort positions by their occurrence in the document
    section_positions.sort(key=lambda x: x[0])

    if not section_positions:
        print("No section headers found in the document!")
        return {"entire_document": document_content}

    # Split the document based on the positions
    sections = {}
    for i, (pos, pattern, section_num) in enumerate(section_positions):
        # Find the end of this section (start of next section or end of document)
        next_pos = (
            section_positions[i + 1][0]
            if i < len(section_positions) - 1
            else len(document_content)
        )

        # Extract the section content
        section_content = document_content[pos:next_pos].strip()

        # Use the section number as the key
        if (
            section_num not in sections
        ):  # Only add if not already present (avoid duplicates)
            sections[section_num] = section_content
            print(
                f"Extracted section: {section_num} ({len(section_content)} characters)"
            )

    return sections


if __name__ == "__main__":
    section_headers = [
        "5.0 Einleitung",
        "5.1 R_{0}-Räume",
        "5.2 T_{0}-Räume",
        "5.3 T_{1}-Räume",
        "5.4 T_{2}-Räume",
        "5.5 T_{3}-Räume und reguläre Räume",
        "5.6 T_{4}-Räume und normale Räume",
        "5.7 T_{3\frac{1}{2}}-Räume und vollständig reguläre Räume",
        "5.8 Vollnormale und parakompakte Räume",
    ]
    sections = split_document_by_section_headers(docs[0].page_content, section_headers)

    for section, content in sections.items():
        print("=" * 80)
        print(f"Section: {section}")
        print(f"Content: {content[:100]}...")
