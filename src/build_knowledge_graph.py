import os
from pathlib import Path
import logging
import coloredlogs
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DOCS_PATH = Path("docs/atomic_units/")

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4.5-preview"
)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
)

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# First, create the document node
document_name = "Topologische Räume"
graph.query(
    """
    MERGE (d:Document {id: $document_id})
    SET d.title = $title
    """,
    {"document_id": document_name, "title": f"Mathematical document: {document_name}"},
)


def link_previous_section(graph: Neo4jGraph, document_name: str, section_number: int):
    previous_section_id = f"{document_name}.section_{section_number - 1}"
    if section_number > 1 and graph.exists_node(previous_section_id):
        graph.query(
            """
            MATCH (s1:Section {id: $previous_section_id}),
                  (s2:Section {id: $section_id})
            MERGE (s1)-[:NEXT]->(s2)
            """,
            {
                "section_id": f"{document_name}.section_{section_number}",
                "previous_section_id": previous_section_id,
            },
        )


# Create nodes for each section to establish hierarchy
def create_section_node(graph: Neo4jGraph, document_name: str, section_number: int):
    # Create section node if it doesn't exist
    link_previous_section(graph, document_name, section_number)
    # Create section node
    graph.query(
        """
        MERGE (d:Document {id: $document_id})
        MERGE (s:Section {id: $section_id, number: $section_number})
            SET s.title = $title
            MERGE (s)-[:PART_OF]->(d)
            """,
        {
            "document_id": document_name,
            "section_id": f"{document_name}.section_{section_number}",
            "section_number": section_number,
            "title": f"Section {section_number}",
        },
    )
    logger.info(f"Created section node for Section {section_number}")


def create_subsection_node(
    graph: Neo4jGraph, document_name: str, section_number: int, subsection_number: int
):
    # Create section node if it doesn't exist
    link_previous_section(graph, document_name, section_number)
    # Create subsection node
    graph.query(
        """
        MERGE (s:Section {id: $section_id, number: $section_number})
        MERGE (s:Subsection {id: $subsection_id, number: $subsection_number})
            SET s.title = $title
            MERGE (s)-[:PART_OF]->(d)
            """,
        {
            "document_id": document_name,
            "section_id": f"{document_name}.section_{section_number}",
            "section_number": section_number,
            "subsection_id": f"{document_name}.subsection_{section_number}_{subsection_number}",
            "subsection_number": subsection_number,
            "title": f"Subsection {subsection_number}",
        },
    )
    logger.info(f"Created subsection node for Subsection {subsection_number}")


def create_next_relationship(
    graph: Neo4jGraph,
    document_name: str,
    node_type: str,  # "section" or "subsection"
    current_number,
    next_number,
    section_number=None,  # Only needed for subsections
):
    if node_type == "section":
        current_id = f"{document_name}.section_{current_number}"
        next_id = f"{document_name}.section_{next_number}"
        label = "Section"
    elif node_type == "subsection":
        current_id = f"{document_name}.subsection_{section_number}_{current_number}"
        next_id = f"{document_name}.subsection_{section_number}_{next_number}"
        label = "Subsection"
    else:
        raise ValueError("node_type must be 'section' or 'subsection'")

    graph.query(
        f"""
        MATCH (a:{label} {{id: $current_id}})
        MATCH (b:{label} {{id: $next_id}})
        MERGE (a)-[:NEXT]->(b)
        """,
        {"current_id": current_id, "next_id": next_id},
    )


sections = {}
for filepath in Path(DOCS_PATH).glob("subsection_*_*_units.pkl"):
    _, section_str, subsection_str, _ = filepath.stem.split("_")
    section_number = int(section_str)
    subsection_number = float(f"{section_str}.{subsection_str}")
    if section_number not in sections:
        sections[section_number] = []
    sections[section_number].append(subsection_number)

# Step 1: Create nodes and PART_OF relationships
for section_number in sections:
    create_section_node(graph, document_name, section_number)
    for subsection_number in sorted(sections[section_number]):
        create_subsection_node(graph, document_name, section_number, subsection_number)
        # Add PART_OF relationship: subsection -> section

# Step 2: Create NEXT relationships
# Sections
sorted_sections = sorted(sections.keys())
for i in range(len(sorted_sections) - 1):
    create_next_relationship(
        graph, document_name, "section", sorted_sections[i], sorted_sections[i + 1]
    )

# Subsections within each section
for section_number in sections:
    sorted_subs = sorted(sections[section_number])
    for i in range(len(sorted_subs) - 1):
        create_next_relationship(
            graph,
            document_name,
            "subsection",
            sorted_subs[i],
            sorted_subs[i + 1],
            section_number,
        )

# # Process each chunk
# for i, chunk in enumerate(chunks.chunks):
#     chunk_id = f"{document_name}.chunk_{i}"
#     section_id = f"{document_name}.section_{chunk.section}"

#     # Generate embedding for the chunk text
#     try:
#         chunk_embedding = embedding_provider.embed_query(chunk.text)
#     except Exception as e:
#         print(f"Error generating embedding for chunk {i}: {str(e)}")
#         # Use a dummy embedding if there's an error
#         chunk_embedding = [0.0] * 1536

#     # Prepare properties for the chunk
#     properties = {
#         "document_id": document_name,
#         "section_id": section_id,
#         "chunk_id": chunk_id,
#         "text": chunk.text,
#         "type": chunk.type,
#         "section": chunk.section,
#         "embedding": chunk_embedding,
#     }

#     # Add additional properties if they exist
#     if chunk.subsection:
#         properties["subsection"] = chunk.subsection
#     if chunk.identifier:
#         properties["identifier"] = chunk.identifier
#     if chunk.proof:
#         properties["proof"] = chunk.proof

#     # Create the mathematical entity node
#     print(f"Creating node for {chunk.type} (ID: {chunk_id})")

#     # Different handling based on entity type
#     if chunk.type.lower() in ["theorem", "satz"]:
#         # For theorems, we create a specific node with its proof
#         query = """
#         MERGE (d:Document {id: $document_id})
#         MERGE (s:Section {id: $section_id})
#         MERGE (t:Theorem {id: $chunk_id})
#         SET t.text = $text,
#             t.section = $section,
#             t.identifier = $identifier
#         MERGE (t)-[:PART_OF]->(s)
#         WITH t
#         CALL db.create.setNodeVectorProperty(t, 'embedding', $embedding)
#         """

#         if chunk.proof:
#             # Also create a proof node connected to the theorem
#             query += """
#             WITH t
#             MERGE (p:Proof {id: $chunk_id + '_proof'})
#             SET p.text = $proof
#             MERGE (t)-[:HAS_PROOF]->(p)
#             """

#     elif chunk.type.lower() in ["definition"]:
#         # For definitions
#         query = """
#         MERGE (d:Document {id: $document_id})
#         MERGE (s:Section {id: $section_id})
#         MERGE (def:Definition {id: $chunk_id})
#         SET def.text = $text,
#             def.section = $section,
#             def.identifier = $identifier
#         MERGE (def)-[:PART_OF]->(s)
#         WITH def
#         CALL db.create.setNodeVectorProperty(def, 'embedding', $embedding)
#         """

#     elif chunk.type.lower() in ["lemma"]:
#         # For lemmas
#         query = """
#         MERGE (d:Document {id: $document_id})
#         MERGE (s:Section {id: $section_id})
#         MERGE (l:Lemma {id: $chunk_id})
#         SET l.text = $text,
#             l.section = $section,
#             l.identifier = $identifier
#         MERGE (l)-[:PART_OF]->(s)
#         WITH l
#         CALL db.create.setNodeVectorProperty(l, 'embedding', $embedding)
#         """

#         if chunk.proof:
#             # Also create a proof node for lemmas with proofs
#             query += """
#             WITH l
#             MERGE (p:Proof {id: $chunk_id + '_proof'})
#             SET p.text = $proof
#             MERGE (l)-[:HAS_PROOF]->(p)
#             """

#     else:
#         # Generic handling for other types
#         query = """
#         MERGE (d:Document {id: $document_id})
#         MERGE (s:Section {id: $section_id})
#         MERGE (c:MathEntity {id: $chunk_id, type: $type})
#         SET c.text = $text,
#             c.section = $section
#         MERGE (c)-[:PART_OF]->(s)
#         WITH c
#         CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
#         """

#     # Execute the query to create the node
#     try:
#         graph.query(query, properties)
#     except Exception as e:
#         print(f"Error creating node for chunk {i}: {str(e)}")

# # Create a vector index for similarity search
# print("Creating vector index for similarity search...")
# graph.query(
#     """
#     CREATE VECTOR INDEX math_entity_vector IF NOT EXISTS
#     FOR (c:MathEntity|Theorem|Definition|Lemma) ON (c.embedding)
#     OPTIONS {
#         indexConfig: {
#             `vector.dimensions`: 1536,
#             `vector.similarity_function`: 'cosine'
#         }
#     }
#     """
# )

# print("Creating fulltext index for keyword search...")
# graph.query(
#     """
#     CREATE FULLTEXT INDEX math_content IF NOT EXISTS
#     FOR (c:MathEntity|Theorem|Definition|Lemma|Proof) ON EACH [c.text]
#     """
# )

# print("\nKnowledge Graph creation complete!")
# print(f"Added {len(chunks.chunks)} mathematical entities to the knowledge graph")

# # Display a summary of what was created
# node_counts = graph.query(
#     """
#     MATCH (n)
#     RETURN labels(n)[0] as label, count(*) as count
#     """
# )

# print("\nNode Counts in Knowledge Graph:")
# for row in node_counts:
#     print(f"- {row['label']}: {row['count']}")

# relationship_counts = graph.query(
#     """
#     MATCH ()-[r]->()
#     RETURN type(r) as type, count(*) as count
#     """
# )

# print("\nRelationship Counts in Knowledge Graph:")
# for row in relationship_counts:
#     print(f"- {row['type']}: {row['count']}")
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
