import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
)
from typing import Optional

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

# Load and split the documents
# loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)


# pdf_file = os.path.join(DOCS_PATH, "KE_5.pdf")
# loader = MathpixPDFLoader(
#     pdf_file,
#     processed_file_format="md",
#     mathpix_api_id=os.environ.get("MATHPIX_API_ID"),
#     mathpix_api_key=os.environ.get("MATHPIX_API_KEY"),
# )
# docs = loader.load()

docs = pickle.load(open("docs/cached_KE_5.pkl", "rb"))
chunks = pickle.load(open("docs/KE_5_chunks.pkl", "rb"))


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
      "proof": None
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

# Use try-except to handle potential errors in structured output
try:
    print("Parsing mathematical document structure...")
    # # Process a smaller segment for testing
    # test_segment = (
    #     docs[0].page_content[1815:5000]
    #     if len(docs[0].page_content) > 5000
    #     else docs[0].page_content
    # )
    # chunks_result = chain.invoke({"input": test_segment})

    # # Print summary of results
    # print(f"✅ Successfully parsed {len(chunks_result.chunks)} chunks")

    # # Display a sample of the first few chunks
    # for i, chunk in enumerate(chunks_result.chunks[:2], start=1):
    #     print(f"\nChunk {i}:")
    #     print(f"  Section: {chunk.section}")
    #     print(f"  Subsection: {chunk.subsection}")
    #     print(f"  Subsubsection: {chunk.subsubsection}")
    #     print(f"  Type: {chunk.type}")
    #     print(f"  Identifier: {chunk.identifier}")
    #     print(f"  Text starts with: {chunk.text[:50]}...")
    #     if chunk.proof:
    #         print(f"  Proof included: Yes ({len(chunk.proof)} characters)")

    # Now process the full document
    chunks = chain.invoke({"input": docs[0].page_content[1815:]})
    print(f"\nFull document processed: {len(chunks.chunks)} chunks extracted")
    for i, chunk in enumerate(chunks.chunks, start=1):
        print(f"\nChunk {i}:")
        print(f"  Section: {chunk.section}")
        print(f"  Subsection: {chunk.subsection}")
        print(f"  Subsubsection: {chunk.subsubsection}")
        print(f"  Type: {chunk.type}")
        print(f"  Identifier: {chunk.identifier}")
        print(f"  Text starts with: {chunk.text[:50]}...")
        if chunk.proof:
            print(f"  Proof included: Yes ({len(chunk.proof)} characters)")

    # Save the chunks to a pickle file
    with open("docs/KE_5_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

except Exception as e:
    print(f"❌ Error parsing document structure: {str(e)}")
    # print("Trying with alternative approach...")

    # # Fallback to a simpler approach or more advanced technique if needed
    # # For example: breaking it down into smaller segments or using a different model
    # system_prompt_fallback = "Parse this mathematical text and return it in JSON format as chunks with section, subsection, type, and text fields."
    # fallback_prompt = ChatPromptTemplate.from_template(
    #     system_prompt_fallback + "\n\n{input}"
    # )
    # chunks = fallback_prompt | llm


# https://python.langchain.com/v0.2/docs/how_to/code_splitter/#markdown
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = header_splitter.split_text(docs[0].page_content)


for i, split in enumerate(md_header_splits, start=1):
    print("=" * 80)
    print("Split :", i)
    print("-" * 40)
    print("Metadata:\n")
    print(split.metadata)
    print("-" * 40)
    print(split.page_content[:40])
    print("=" * 80)


# text_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n"],
#     chunk_size=1000,
#     chunk_overlap=200,
# )

# chunks = text_splitter.split_documents(md_header_splits)

# doc_transformer.convert_to_graph_documents([md_header_splits[7]])

# for chunk in chunks:
#     filename = os.path.basename(chunk.metadata["source"])
#     chunk_id = f"{filename}.{chunk.metadata['page']}"
#     print("Processing -", chunk_id)

#     # Embed the chunk
#     chunk_embedding = embedding_provider.embed_query(chunk.page_content)

#     # Add the Document and Chunk nodes to the graph
#     properties = {
#         "section": 5,
#         "subsection": filename,
#         "chunk_id": chunk_id,
#         "text": chunk.page_content,
#         "embedding": chunk_embedding,
#     }

#     graph.query(
#         """
#         MERGE (d:Document {id: $filename})
#         MERGE (c:Chunk {id: $chunk_id})
#         SET c.text = $text
#         MERGE (d)<-[:PART_OF]-(c)
#         WITH c
#         CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
#         """,
#         properties,
#     )

#     # Generate the entities and relationships from the chunk
#     graph_docs = doc_transformer.convert_to_graph_documents([chunk])

#     # Map the entities in the graph documents to the chunk node
#     for graph_doc in graph_docs:
#         chunk_node = Node(id=chunk_id, type="Chunk")

#         for node in graph_doc.nodes:
#             graph_doc.relationships.append(
#                 Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
#             )

#     # add the graph documents to the graph
#     graph.add_graph_documents(graph_docs)

# # Create the vector index
# graph.query(
#     """
#     CREATE VECTOR INDEX `chunkVector`
#     IF NOT EXISTS
#     FOR (c: Chunk) ON (c.textEmbedding)
#     OPTIONS {indexConfig: {
#     `vector.dimensions`: 1536,
#     `vector.similarity_function`: 'cosine'
#     }};"""
