import os

from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "docs/"

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

# Process the parsed chunks and create knowledge graph
print("\n" + "=" * 80)
print("Creating Knowledge Graph from Parsed Chunks")
print("=" * 80)

pdf_file = ...
# First, create the document node
document_name = os.path.basename(pdf_file)
graph.query(
    """
    MERGE (d:Document {id: $document_id})
    SET d.title = $title
    """,
    {"document_id": document_name, "title": f"Mathematical document: {document_name}"},
)


with open("docs/sections/section_5_0.md", "r") as f:
    chunks = f.read()
# Create nodes for each section to establish hierarchy
sections = {}
for chunk in chunks.chunks:
    section_number = chunk.section
    if section_number not in sections:
        # Create section node if it doesn't exist
        sections[section_number] = True
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
        print(f"Created section node for Section {section_number}")

# Process each chunk
for i, chunk in enumerate(chunks.chunks):
    chunk_id = f"{document_name}.chunk_{i}"
    section_id = f"{document_name}.section_{chunk.section}"

    # Generate embedding for the chunk text
    try:
        chunk_embedding = embedding_provider.embed_query(chunk.text)
    except Exception as e:
        print(f"Error generating embedding for chunk {i}: {str(e)}")
        # Use a dummy embedding if there's an error
        chunk_embedding = [0.0] * 1536

    # Prepare properties for the chunk
    properties = {
        "document_id": document_name,
        "section_id": section_id,
        "chunk_id": chunk_id,
        "text": chunk.text,
        "type": chunk.type,
        "section": chunk.section,
        "embedding": chunk_embedding,
    }

    # Add additional properties if they exist
    if chunk.subsection:
        properties["subsection"] = chunk.subsection
    if chunk.identifier:
        properties["identifier"] = chunk.identifier
    if chunk.proof:
        properties["proof"] = chunk.proof

    # Create the mathematical entity node
    print(f"Creating node for {chunk.type} (ID: {chunk_id})")

    # Different handling based on entity type
    if chunk.type.lower() in ["theorem", "satz"]:
        # For theorems, we create a specific node with its proof
        query = """
        MERGE (d:Document {id: $document_id})
        MERGE (s:Section {id: $section_id})
        MERGE (t:Theorem {id: $chunk_id})
        SET t.text = $text,
            t.section = $section,
            t.identifier = $identifier
        MERGE (t)-[:PART_OF]->(s)
        WITH t
        CALL db.create.setNodeVectorProperty(t, 'embedding', $embedding)
        """

        if chunk.proof:
            # Also create a proof node connected to the theorem
            query += """
            WITH t
            MERGE (p:Proof {id: $chunk_id + '_proof'})
            SET p.text = $proof
            MERGE (t)-[:HAS_PROOF]->(p)
            """

    elif chunk.type.lower() in ["definition"]:
        # For definitions
        query = """
        MERGE (d:Document {id: $document_id})
        MERGE (s:Section {id: $section_id})
        MERGE (def:Definition {id: $chunk_id})
        SET def.text = $text,
            def.section = $section,
            def.identifier = $identifier
        MERGE (def)-[:PART_OF]->(s)
        WITH def
        CALL db.create.setNodeVectorProperty(def, 'embedding', $embedding)
        """

    elif chunk.type.lower() in ["lemma"]:
        # For lemmas
        query = """
        MERGE (d:Document {id: $document_id})
        MERGE (s:Section {id: $section_id})
        MERGE (l:Lemma {id: $chunk_id})
        SET l.text = $text,
            l.section = $section,
            l.identifier = $identifier
        MERGE (l)-[:PART_OF]->(s)
        WITH l
        CALL db.create.setNodeVectorProperty(l, 'embedding', $embedding)
        """

        if chunk.proof:
            # Also create a proof node for lemmas with proofs
            query += """
            WITH l
            MERGE (p:Proof {id: $chunk_id + '_proof'})
            SET p.text = $proof
            MERGE (l)-[:HAS_PROOF]->(p)
            """

    else:
        # Generic handling for other types
        query = """
        MERGE (d:Document {id: $document_id})
        MERGE (s:Section {id: $section_id})
        MERGE (c:MathEntity {id: $chunk_id, type: $type})
        SET c.text = $text,
            c.section = $section
        MERGE (c)-[:PART_OF]->(s)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
        """

    # Execute the query to create the node
    try:
        graph.query(query, properties)
    except Exception as e:
        print(f"Error creating node for chunk {i}: {str(e)}")

# Create a vector index for similarity search
print("Creating vector index for similarity search...")
graph.query(
    """
    CREATE VECTOR INDEX math_entity_vector IF NOT EXISTS
    FOR (c:MathEntity|Theorem|Definition|Lemma) ON (c.embedding)
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }
    }
    """
)

print("Creating fulltext index for keyword search...")
graph.query(
    """
    CREATE FULLTEXT INDEX math_content IF NOT EXISTS
    FOR (c:MathEntity|Theorem|Definition|Lemma|Proof) ON EACH [c.text]
    """
)

print("\nKnowledge Graph creation complete!")
print(f"Added {len(chunks.chunks)} mathematical entities to the knowledge graph")

# Display a summary of what was created
node_counts = graph.query(
    """
    MATCH (n)
    RETURN labels(n)[0] as label, count(*) as count
    """
)

print("\nNode Counts in Knowledge Graph:")
for row in node_counts:
    print(f"- {row['label']}: {row['count']}")

relationship_counts = graph.query(
    """
    MATCH ()-[r]->()
    RETURN type(r) as type, count(*) as count
    """
)

print("\nRelationship Counts in Knowledge Graph:")
for row in relationship_counts:
    print(f"- {row['type']}: {row['count']}")
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
