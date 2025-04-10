import os
import re
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
from typing import Optional, List

from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from dotenv import load_dotenv
import pickle

load_dotenv()

DOCS_PATH = "docs/"

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="o3-mini")

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

doc_transformer = LLMGraphTransformer(llm=llm)

# Load the documents
docs = pickle.load(open("docs/cached_KE_5.pkl", "rb"))


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


def split_document_by_header2(document_content: str) -> List[str]:
    """
    Split a markdown document by Header 2 sections that follow the pattern of
    two digits followed by a title (like "5.1 R_{0}-Räume").

    Args:
        document_content: The full text content of the document

    Returns:
        List[str]: A list of document chunks split by Header 2
    """
    # This regex pattern is specifically designed to match Header 2 sections with:
    # 1. Exactly two digits separated by a dot (e.g., "5.1", "5.2")
    # 2. Followed by a title
    # 3. The pattern handles both regular text and LaTeX math expressions

    # Extremely specific pattern for Header 2 with exactly the format we want:
    # - Either "## 5.1 Title" format or "## $5.1 \quad \mathrm{X}_{0}$-Title" format
    h2_pattern = r"(?:^|\n)##\s*(?:\$\s*(\d+\.\d+)\s*\\quad.*?\$|\s*(\d+\.\d+)\s+[^\n]+)(?:\n|.)*?(?=\n##\s*(?:\$\s*\d+\.\d+|\s*\d+\.\d+\s+)|\Z)"

    # Find all matches
    matches = re.finditer(h2_pattern, document_content, re.DOTALL)

    # Extract the chunks
    chunks = []
    last_end = 0

    for match in matches:
        # Get the start and end positions of the match
        start, end = match.span()

        # Extract the section number for debugging
        section_num = match.group(1) if match.group(1) else match.group(2)
        print(f"Found section: {section_num}")

        # Add the chunk
        chunk = document_content[start:end].strip()
        if chunk:
            chunks.append(chunk)

        last_end = end

    # Add the last chunk if there's content after the last match
    if last_end < len(document_content):
        last_chunk = document_content[last_end:].strip()
        if last_chunk:
            chunks.append(last_chunk)

    return chunks


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


def process_document_with_regex_splitting(document_content: str):
    """
    Process a document by first splitting it using regex to identify Header 2 sections,
    then parsing each chunk separately.

    Args:
        document_content: The full text content of the document

    Returns:
        List[DocChunk]: A list of parsed document chunks
    """
    print("Splitting document by Header 2 sections using regex...")

    # Split the document by Header 2 sections
    chunks = split_document_by_header2(document_content)

    # Print the first few characters of each chunk for verification
    for i, chunk in enumerate(chunks):
        preview = chunk[:50].replace("\n", " ")
        print(f"Chunk {i + 1}: {preview}...")

    print(f"Document split into {len(chunks)} chunks")

    # Process each chunk separately
    all_parsed_chunks = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} characters)")

        try:
            # Process the chunk
            chunk_result = chain.invoke({"input": chunk})
            all_parsed_chunks.extend(chunk_result.chunks)
            print(f"  ✅ Successfully parsed {len(chunk_result.chunks)} entities")

        except Exception as e:
            print(f"  ❌ Error processing chunk {i + 1}: {str(e)}")
            # You could implement a fallback strategy here if needed

    return Chunks(chunks=all_parsed_chunks)


if __name__ == "__main__":
    # Load the document
    print("Loading document...")
    document_content = docs[0].page_content

    # Process the document
    result = process_document_with_regex_splitting(document_content)

    # Save the results
    print(f"Saving {len(result.chunks)} parsed chunks...")
    with open("docs/KE_5_chunks_regex.pkl", "wb") as f:
        pickle.dump(result, f)

    print("✅ Done!")
