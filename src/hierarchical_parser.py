"""
Hierarchical Parser for Mathematical Documents

This module provides a hierarchical approach to parsing mathematical documents,
first identifying the major sections and then extracting mathematical entities
within each section, preserving the hierarchical structure.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o", temperature=0
)


class DocChunk(BaseModel):
    """An atomic segment of a mathematical document, containing a complete logical unit
    such as a theorem, definition, proof, or example, with its hierarchical position
    and classification."""

    section: float = Field(description="The main section number (e.g., 5.0, 5.1, 5.2)")
    subsection: Optional[float] = Field(
        default=None,
        description="The subsection number if applicable (e.g., 5.1.1, 5.1.2)",
    )
    type: str = Field(
        description="The type of the mathematical entity",
        examples=[
            "Definition",
            "Theorem",
            "Note",
            "Exercise",
            "Example",
            "Lemma",
            "Proposition",
            "Introduction",
            "Remark",
        ],
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

    chunks: List[DocChunk] = Field(
        description="A list of DocChunk objects representing the extracted chunks"
    )


class Section(BaseModel):
    """A section of a mathematical document."""

    section_number: float
    title: str
    start_index: int
    end_index: int


class Entity(BaseModel):
    """A mathematical entity found in a section."""

    type: str
    identifier: Optional[str] = None
    subsection: Optional[float] = None
    text: str
    proof: Optional[str] = None


def process_document_hierarchically(doc_content: str) -> Chunks:
    """
    Process a mathematical document by breaking it down hierarchically:
    1. First identify main sections
    2. Then process each section to identify mathematical entities

    Args:
        doc_content: The full text content of the document

    Returns:
        Chunks: A collection of structured chunks from the document
    """
    print("Starting hierarchical document processing...")

    # Step 1: Extract main sections
    section_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a mathematical document structure analyzer.
        Your task is to identify all main sections in this mathematical document.

        Return a JSON array of objects containing:
        - section_number: The section number (as a float, e.g., 5.0, 5.1, 5.2)
        - title: The section title
        - start_index: Approximate character position where this section starts
        - end_index: Approximate character position where this section ends

        For example, look for patterns like:
        - "## Kapitel 5" or "## 5 Introduction"
        - "## 5.1 R₀-Räume"
        - "## 5.2 T₀-Räume"

        Identify the character positions where each section starts and ends.
        """,
            ),
            ("human", "{input}"),
        ]
    )

    print("Identifying main sections...")
    sections_extractor = llm.with_structured_output(List[Section])
    raw_sections = sections_extractor.invoke(
        section_prompt.format(input=doc_content[:20000])
    )
    print(f"Identified {len(raw_sections)} main sections")

    # Step 2: Process each section to extract mathematical entities
    all_chunks = []

    for i, section in enumerate(raw_sections):
        section_number = section.section_number
        section_title = section.title
        start_idx = max(0, section.start_index)
        end_idx = min(len(doc_content), section.end_index)

        # Get section content
        section_content = doc_content[start_idx:end_idx]
        print(
            f"Processing Section {section_number}: {section_title} ({len(section_content)} chars)"
        )

        # Create a prompt for mathematical entities within this section
        entity_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are analyzing section {section_number} ({section_title}) of a mathematical document.
            Identify all mathematical entities (theorems, definitions, etc.) in this section.

            Return a JSON array of objects containing:
            - type: The type of entity (Theorem, Definition, Lemma, etc.)
            - identifier: The identifier (e.g., "Satz 5.1.2", "Definition 5.2.1")
            - subsection: The subsection number if applicable (as a float, e.g., 5.1.1, 5.1.2)
            - text: The complete text of the entity
            - proof: The proof text if applicable (null otherwise)

            Mathematical terminology in German:
            - "Satz" = Theorem
            - "Definition" = Definition
            - "Lemma" = Lemma
            - "Aufgabe" = Exercise
            - "Bemerkung" = Note/Remark
            - "Beispiel" = Example
            - "Vorbemerkung" = Preliminary note
            - "Proposition" = Proposition

            Ensure you extract the full text of each entity including mathematical notation.
            For theorems, include their proofs when present.
            """,
                ),
                ("human", "{input}"),
            ]
        )

        try:
            entities_extractor = llm.with_structured_output(List[Entity])
            entities = entities_extractor.invoke(
                entity_prompt.format(input=section_content)
            )
            print(f"Found {len(entities)} entities in section {section_number}")

            # Convert entities to DocChunk objects
            for entity in entities:
                chunk = DocChunk(
                    section=section_number,
                    subsection=entity.subsection,
                    type=entity.type,
                    identifier=entity.identifier,
                    text=entity.text,
                    proof=entity.proof,
                )
                all_chunks.append(chunk)

        except Exception as subsection_error:
            print(f"Error processing section {section_number}: {str(subsection_error)}")
            # Fallback: Add entire section as one chunk
            all_chunks.append(
                DocChunk(
                    section=section_number,
                    subsection=None,
                    type="Section",
                    identifier=f"Section {section_number}",
                    text=section_content,
                    proof=None,
                )
            )

    return Chunks(chunks=all_chunks)


def parse_document(document_content: str, max_length: int = 20000) -> Chunks:
    """
    Parse a mathematical document with multiple fallback strategies

    Args:
        document_content: The document content to parse
        max_length: Maximum content length to process

    Returns:
        Chunks: Structured document chunks
    """
    content_to_process = document_content[:max_length]

    # Try direct structured output approach
    try:
        system_prompt = """
        You are a precise mathematical document parser. Your task is to parse mathematical documents into
        structured chunks that preserve their hierarchical organization.

        YOU MUST RETURN A VALID JSON OBJECT that follows the schema EXACTLY.

        Parsing guidelines:
        1. Extract sections (like 5.1, 5.2) and subsections (like 5.1.1, 5.1.2)
        2. Identify mathematical entities: Satz (Theorem), Definition, Lemma, Aufgabe (Exercise), etc.
        3. Include proofs with their associated theorems
        4. Preserve all mathematical notation

        Mathematical terminology in German:
        - "Satz" = Theorem
        - "Definition" = Definition
        - "Lemma" = Lemma
        - "Aufgabe" = Exercise
        - "Bemerkung" = Note/Remark
        - "Beispiel" = Example
        - "Vorbemerkung" = Preliminary note
        - "Proposition" = Proposition
        """

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    """Please parse the following mathematical text into structured chunks following the schema:

            {
              "chunks": [
                {
                  "section": 5.1,
                  "subsection": 5.1.1,
                  "type": "Introduction",
                  "identifier": "Vorbemerkung 5.1.1",
                  "text": "The full text of this introduction...",
                  "proof": null
                },
                {
                  "section": 5.1,
                  "subsection": 5.1.2,
                  "type": "Theorem",
                  "identifier": "Satz 5.1.2",
                  "text": "The theorem statement...",
                  "proof": "The proof text..."
                }
              ]
            }

            Here's the text to parse:

            {input}""",
                ),
            ]
        )

        structured_llm = llm.with_structured_output(Chunks)
        direct_chunks = structured_llm.invoke(
            chat_prompt.format(input=content_to_process)
        )
        print(
            f"Direct parsing successful: {len(direct_chunks.chunks)} chunks extracted"
        )
        return direct_chunks

    except Exception as direct_error:
        print(f"Direct parsing failed: {str(direct_error)}")
        print("Trying hierarchical processing...")

        # Try hierarchical processing
        try:
            hierarchical_chunks = process_document_hierarchically(content_to_process)
            print(
                f"Hierarchical processing successful: {len(hierarchical_chunks.chunks)} chunks extracted"
            )
            return hierarchical_chunks

        except Exception as hierarchical_error:
            print(f"Hierarchical processing failed: {str(hierarchical_error)}")
            print("Using simple fallback...")

            # Final simple fallback
            import re

            # Simple pattern matching for common mathematical entity headers
            pattern = r"### (\d+\.\d+\.\d+)\s+([A-Za-zäöüÄÖÜß]+)|## (\d+\.\d+)\s+([A-Za-zäöüÄÖÜß]+)"
            matches = re.finditer(pattern, content_to_process)

            fallback_chunks = []
            for match in matches:
                # Extract section/subsection and entity type
                subsection = match.group(1)
                subsection_type = match.group(2)
                section = match.group(3)
                section_type = match.group(4)

                if subsection and subsection_type:
                    # Found a subsection
                    fallback_chunks.append(
                        DocChunk(
                            section=float(
                                subsection.split(".")[0]
                                + "."
                                + subsection.split(".")[1]
                            ),
                            subsection=float(subsection),
                            type=subsection_type,
                            identifier=f"{subsection_type} {subsection}",
                            text=f"Extracted {subsection_type} {subsection}",
                            proof=None,
                        )
                    )
                elif section and section_type:
                    # Found a section
                    fallback_chunks.append(
                        DocChunk(
                            section=float(section),
                            subsection=None,
                            type=section_type,
                            identifier=f"{section_type} {section}",
                            text=f"Extracted {section_type} {section}",
                            proof=None,
                        )
                    )

            if fallback_chunks:
                return Chunks(chunks=fallback_chunks)
            else:
                # Create at least one fallback chunk
                return Chunks(
                    chunks=[
                        DocChunk(
                            section=5.0,
                            subsection=None,
                            type="Document",
                            identifier="Document",
                            text=content_to_process[:1000],  # Just take the beginning
                            proof=None,
                        )
                    ]
                )


if __name__ == "__main__":
    # Test with a sample file
    with open("docs/KE_5.txt", "r") as f:
        sample_content = f.read()

    result = parse_document(sample_content)
    print(f"Parsed {len(result.chunks)} chunks from the document")

    # Print sample of first few chunks
    for i, chunk in enumerate(result.chunks[:3]):
        print(f"\nChunk {i + 1}:")
        print(f"  Section: {chunk.section}")
        print(f"  Subsection: {chunk.subsection}")
        print(f"  Type: {chunk.type}")
        print(f"  Identifier: {chunk.identifier}")
        print(f"  Text excerpt: {chunk.text[:100]}...")
        if chunk.proof:
            print(f"  Proof included: Yes ({len(chunk.proof)} characters)")
