import os

from langchain_community.document_loaders import (
    MathpixPDFLoader,
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "docs/"

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o")

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


pdf_file = os.path.join(DOCS_PATH, "KE_5.pdf")
loader = MathpixPDFLoader(
    pdf_file,
    processed_file_format="md",
    mathpix_api_id=os.environ.get("MATHPIX_API_ID"),
    mathpix_api_key=os.environ.get("MATHPIX_API_KEY"),
)
docs = loader.load()


markdown_text = """
# Fun in California

## Driving

Try driving on the 1 down to San Diego

### Food

Make sure to eat a burrito while you're there

## Hiking

Go to Yosemite
"""

{
    # sections
    "type": "object",
    "properties": {
        "section": {"type": "string"},
        "number": {"type": "number"},
        "text": {"type", "string"},
        # subsections
        "subsections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "number": {"type": "number"},
                    "text": {"type": "string"},
                    # subsubsections
                    "subsections": {
                        "type": "object",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subsubsection": {"type": "string"},
                                "number": {"type": "number"},
                                "text": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}
# {
#     "type": "object",
#     "properties": {
#         "subsection": {"type": "string"},
#         "number": {"type": "number"},
#         "text": {"type": "string"},
#         "subsubsections": {
#             "type": "array",
#             "items": {
#                 "type": "object",
#                 "properties": {
#                     "subsubsection": {"type": "string"},
#                     "number": {"type": "number"},
#                     "text": {"type": "string"},
#                 },
#             },
#         },
# }
# # {
# #     "type": "object",
# #     "properties": {
# #         "subsubsection": {"type": "string"},
# #         "number": {"type": "number"},
# #         "text": {"type": "string"},
# #     },
# # }

# https://python.langchain.com/v0.2/docs/how_to/code_splitter/#markdown
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
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


text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"],
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(md_header_splits)

doc_transformer.convert_to_graph_documents([md_header_splits[7]])

for chunk in chunks:
    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f"{filename}.{chunk.metadata['page']}"
    print("Processing -", chunk_id)

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    # Add the Document and Chunk nodes to the graph
    properties = {
        "section": 5,
        "subsection": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding,
    }

    graph.query(
        """
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """,
        properties,
    )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(id=chunk_id, type="Chunk")

        for node in graph_doc.nodes:
            graph_doc.relationships.append(
                Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
            )

    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)

# Create the vector index
graph.query(
    """
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }};"""
)
