import pickle
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pymupdf

from math_rag.core import ROOT


def get_text_around_rect(page, rect, padding=(2, 5)):
    """Extract text around a rectangle with specified padding.

    Args:
        page: The PDF page object
        rect: The rectangle to expand
        padding: Tuple of (horizontal, vertical) padding

    Returns:
        The extracted text
    """
    # Ensure rect is a proper Rect object
    if not isinstance(rect, pymupdf.Rect):
        rect = pymupdf.Rect(0, 0, 0, 0)

    # Create expanded rectangle for context
    expanded_rect = pymupdf.Rect(
        rect.x0 - padding[0],
        rect.y0 - padding[1],
        rect.x1 + padding[0],
        rect.y1 + padding[1],
    )

    # Extract text from the expanded area
    text = (
        page.get_text("text", clip=expanded_rect)
        .strip("(")
        .strip(")")
        .strip("[")
        .strip("]")
        .strip("}")
        .strip("{")
        .strip()
    )
    return text


def identify_source_context(doc, page, from_rect):
    """
    Identify the mathematical entity containing a link.

    Args:
        doc: The PDF document
        page: The page containing the link
        from_rect: The rectangle of the link

    Returns:
        Dictionary with source entity information
    """
    # Define a larger rectangle for context (might need adjustment based on document structure)
    context_rect = pymupdf.Rect(
        x0=0,  # Start from left edge
        y0=max(0, from_rect.y0 - 400),  # Extend up
        x1=page.rect.width,  # Full page width
        y1=min(
            page.rect.height,
            from_rect.y1,  # + 50
        ),  # Extend down enough to capture the entity title
    )

    context_text = page.get_text("text", clip=context_rect).strip()

    # Patterns for detecting mathematical entities
    entity_patterns = {
        "theorem": re.compile(r"((?:\d+(?:\.\d+)*)\s*(?:Satz|Thm\.?))", re.I),
        "lemma": re.compile(r"((?:\d+(?:\.\d+)*)\s*(?:Lemma))", re.I),
        "definition": re.compile(r"((?:\d+(?:\.\d+)*)\s*(?:Definition?\.?))", re.I),
        "proposition": re.compile(
            r"((?:\d+(?:\.\d+)*)\s*(?:Prop(?:osition)?\.?))", re.I
        ),
        "corollary": re.compile(r"((?:\d+(?:\.\d+)*)\s*(?:Korollar?\.?))", re.I),
        "exercise": re.compile(r"((?:\d+(?:\.\d+)*)\s*(?:Aufgabe?\.?))", re.I),
    }

    # Collect all matches across all entity types
    all_matches = []
    for entity_type, pattern in entity_patterns.items():
        matches = list(pattern.finditer(context_text))
        for match in matches:
            # Calculate the approximate position in the document
            # Count newlines before the match to estimate vertical position
            match_pos = context_text[: match.start()].count("\n")

            # Store match info with entity type and position
            all_matches.append(
                {
                    "match": match,
                    "type": entity_type,
                    "position": match_pos,
                    "distance": abs(
                        match_pos - (from_rect.y0 - context_rect.y0) / 10
                    ),  # Approximate distance calculation
                }
            )

    # If we found any matches
    if all_matches:
        # Sort by distance (closest first)
        all_matches.sort(key=lambda x: x["distance"])

        # Get the closest match
        closest = all_matches[0]
        match = closest["match"]
        entity_type = closest["type"]
        entity_num = match.group(1).replace("\n", " ").strip()
        return {
            "type": entity_type,
            "number": entity_num,
        }

    # If no entity found, return unknown
    return {"type": "unknown", "number": "0", "full_reference": "Unknown context"}


def extract_links_from_pdf(pdf_path: str, start_page: int) -> List[Dict]:
    """
    Extract hyperlinks from a mathematical PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries containing link information
    """
    print(f"Extracting links from {pdf_path}")
    links = []
    # Open the PDF
    doc = pymupdf.open(pdf_path)

    # Extract links from each page
    for page_num, page in enumerate(doc[start_page:]):
        page_links = page.get_links()
        print(f"Page {page_num + 1} has {len(page_links)} links")

        for link in page_links:
            # Filter internal links (links within the document)
            if "kind" in link:
                # identify the target of the link, by reading the rectangle around
                from_rect = link.get("from", pymupdf.Rect(0, 0, 0, 0))
                from_page = page_num
                from_text_around = get_text_around_rect(page, from_rect)

                # Identify the source context (which theorem/lemma/etc contains this link)
                source_context = identify_source_context(doc, page, from_rect)
                source_page = link.get("page", 0)
                if link["kind"] == pymupdf.LINK_NAMED:
                    # if we haved a linked name, we have the additional information of a nameddest, which should match with from_text_around
                    from_nameddest = link.get("nameddest", "")

                    # # Extract destination reference info if possible
                    # dest_type, dest_num, dest_full_ref = extract_reference_info(
                    #     to_text_around
                    # )

                # elif link["kind"] == pymupdf.LINK_GOTO:
                #     to_rect = link.get("to", pymupdf.Rect(0, 0, 0, 0))
                #     to_page = link.get("page", 0)
                #     to_text_around = get_text_around_rect(
                #         doc[to_page], to_rect, padding=(20, 10)
                #     )

                #     # Extract destination reference info
                #     dest_type, dest_num, dest_full_ref = extract_reference_info(
                #         to_text_around
                #     )

                # Create link object with both source and destination information
                link_info = {
                    "destination_page": from_page,
                    "source_page": source_page,
                    "destination_text_around": from_text_around.strip(),
                    "destination_name": from_nameddest,
                    "source_entity": source_context,
                }
                links.append(link_info)

    return links


def classify_links(links: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Classify links by type (theorem, lemma, definition, etc.)

    Args:
        links: List of extracted links

    Returns:
        Dictionary of links categorized by type
    """
    classified = defaultdict(list)

    # Regular expressions for different types of mathematical elements
    patterns = {
        "theorem": re.compile(r"theorem|thm\.?\s*\d+", re.I),
        "lemma": re.compile(r"lemma\s*\d+", re.I),
        "definition": re.compile(r"def(?:inition)?\.?\s*\d+", re.I),
        "corollary": re.compile(r"cor(?:ollary)?\.?\s*\d+", re.I),
        "example": re.compile(r"ex(?:ample)?\.?\s*\d+", re.I),
        "proposition": re.compile(r"prop(?:osition)?\.?\s*\d+", re.I),
    }

    for link in links:
        text = link["text"]
        link_type = "other"

        # Determine link type based on text content
        for type_name, pattern in patterns.items():
            if pattern.search(text):
                link_type = type_name
                break

        classified[link_type].append(link)

    return classified


def extract_reference_info(text: str) -> Tuple[str, str, str]:
    """
    Extract reference information from link text.

    Args:
        text: Text containing reference information

    Returns:
        Tuple of (reference_type, reference_number, full_reference)
    """
    # Patterns for different reference types
    theorem_match = re.search(r"(Satz|Thm\.?)\s*(\d+(?:\.\d+)*)", text, re.I)
    lemma_match = re.search(r"(Lemma)\s*(\d+(?:\.\d+)*)", text, re.I)
    def_match = re.search(r"(Def(?:inition)?\.?)\s*(\d+(?:\.\d+)*)", text, re.I)
    prop_match = re.search(r"(Prop(?:osition)?\.?)\s*(\d+(?:\.\d+)*)", text, re.I)
    cor_match = re.search(r"(Kor(?:ollar)?\.?)\s*(\d+(?:\.\d+)*)", text, re.I)
    exercise_match = re.search(r"(Auf(?:gabe)?\.?)\s*(\d+(?:\.\d+)*)", text, re.I)

    # Check each pattern
    for match in [
        theorem_match,
        lemma_match,
        def_match,
        prop_match,
        cor_match,
        exercise_match,
    ]:
        if match:
            ref_type = match.group(1)
            ref_num = match.group(2)
            full_ref = f"{ref_type} {ref_num}"
            return ref_type.lower(), ref_num, full_ref

    # Return default values if no match found
    return "unknown", "0", text[:50] + ("..." if len(text) > 50 else "")


def extract_digit_pattern(text: str) -> str:
    """
    Extract digit.digit.digit pattern from text.

    Args:
        text: Text to search for digit patterns

    Returns:
        First digit.digit.digit pattern found, or empty string if none found
    """
    pattern = re.search(r"\d+\.\d+\.\d+", text)
    return pattern.group(0) if pattern else ""


if __name__ == "__main__":
    extracted_links = extract_links_from_pdf("docs/Skript_2024.pdf", start_page=10)

    # Extract digit.digit.digit patterns from source and destination
    reference_tuples = []

    for link in extracted_links:
        source = link["source_entity"]

        # Extract pattern from source number
        source_pattern = extract_digit_pattern(source.get("number", ""))

        # Extract pattern from destination text and name
        dest_text_pattern = extract_digit_pattern(
            link.get("destination_text_around", "")
        )
        dest_name_pattern = extract_digit_pattern(link.get("destination_name", ""))

        # Use destination name pattern if available, otherwise use text pattern
        dest_pattern = dest_name_pattern if dest_name_pattern else dest_text_pattern

        # Only add tuple if both source and destination patterns are found
        if source_pattern and dest_pattern and source_pattern != dest_pattern:
            reference_tuples.append((source_pattern, dest_pattern))

    # Save tuples to pickle file
    with open(ROOT / "data" / "reference_tuples.pkl", "wb") as f:
        pickle.dump(reference_tuples, f)

    print(f"Extracted {len(reference_tuples)} reference tuples:")
    for source, dest in reference_tuples:
        print(f"{source} -> {dest}")

    print("Saved tuples to reference_tuples.pkl")
