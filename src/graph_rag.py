from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from pydantic import BaseModel, Field
from typing import List, Tuple
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

# raw_documents = WikipediaLoader(query="Continuous functions").load()
# text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
# texts = text_splitter.split_documents(raw_documents)


# text = """
# Marie Curie, 7 November 1867 – 4 July 1934, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
# She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
# Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
# She was, in 1906, the first woman to become a professor at the University of Paris.
# Also, Robin Williams.
# """

# documents = [Document(page_content=text)]
# text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
# texts = text_splitter.split_documents(documents)

# Read the wikipedia article
raw_documents = WikipediaLoader(query="Elizabeth I").load()
# Define chunking strategy
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(raw_documents[:3])

# Initialize LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4-0125-preview",
    max_tokens=2000,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)


# Store Knowledge Graph in Neo4j
graph = Neo4jGraph(
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
)
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)


# directly show the graph resulting from the given Cypher query
# default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"


# def showGraph(cypher: str = default_cypher):
#     # create a neo4j session to run queries
#     driver = GraphDatabase.driver(
#         uri=os.environ.get("NEO4J_URI"),
#         auth=(os.environ.get("NEO4J_USERNAME"), os.environ.get("NEO4J_PASSWORD")),
#     )
#     session = driver.session()
#     widget = GraphWidget(graph=session.run(cypher).graph())
#     widget.node_label_mapping = "id"
#     # display(widget)
#     return widget


# showGraph()


vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)


# Purpose of this section is to identify the relevant entities from a querstion.
# F.e. 'Marie Curie' or 'Nobel Prize'.  These entities are then be used to query
# the knowledge graph for relevant information by also retrieving adjacent
# nodes/entities and using them as context.


class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

# Example usage
entity_chain.invoke({"question": "Who is Marie Curie?"}).names
entity_chain.invoke(
    {"question": "Who won the Nobel Prize in Physics and Chemistry?"}
).names


def generate_full_text_query(input_str: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input_str).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el["output"] for el in response])
    return result


print(structured_retriever("Who is Elisabeth I?"))


def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [
        el.page_content for el in vector_index.similarity_search(question)
    ]
    final_data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ". join(unstructured_data)}
        """
    return final_data


print(retriever("Who is Elisabeth I?"))

_template = """Given the following conversation and a follow up question,
 rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x: x["question"]),
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)


chain.invoke({"question": "Which house did Elizabeth I belong to?"})
chain.invoke(
    {
        "question": "When was she born?",
        "chat_history": [("Which house did Elizabeth I belong to?", "House Of Tudor")],
    }
)
