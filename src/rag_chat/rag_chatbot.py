import logging
from pathlib import Path
from typing import Dict, List, TypedDict

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, START, StateGraph

from rag_chat.config import load_config
from rag_chat.document_processing import load_and_process_pdfs
from rag_chat.embeddings import initialize_embeddings
from rag_chat.llm_utils import initialize_llm
from rag_chat.project_root import ROOT
from rag_chat.prompts import (
    answer_grader_prompt,
    hallucination_grader_prompt,
    prompt,
    question_rewriter_prompt,
    question_router_prompt,
    rag_prompt,
    retrieval_grader_prompt,
)
from rag_chat.retrievers import initialize_retrievers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retries: int


# Create and compile the state graph
def create_rag_chatbot():
    config = load_config()

    docs = load_and_process_pdfs(Path(config["docs_path"]))
    embeddings = initialize_embeddings(config)
    ensemble_retriever = initialize_retrievers(docs, embeddings, config)
    llm = initialize_llm(config)

    question_router = question_router_prompt | llm | JsonOutputParser()
    answer_normal = prompt | llm | StrOutputParser()
    question_rewriter = question_rewriter_prompt | llm | StrOutputParser()
    rag_chain = rag_prompt | llm | StrOutputParser
    retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()
    answer_grader = answer_grader_prompt | llm | JsonOutputParser()
    hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

    def retrieve(state):
        """Retrieve documents for the question."""
        logging.info("🗄️ Retrieving documents...")
        question = state["question"]
        documents = ensemble_retriever.invoke(question)
        logging.info(f"Retrieved {len(documents)} documents")
        return {"documents": documents, "question": question}

    def generate(state):
        """Generate an answer using retrieved documents."""
        logging.info("🤖 Generating answer...")
        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"context": documents, "question": question})
        logging.info(f"Generated answer: {generation[:20]}")
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(state):
        """Grade the relevance of retrieved documents."""
        logging.info("💎 Grading documents...")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = [
            d
            for d in documents
            if retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )["score"]
            == "yes"
        ]
        logging.info(f"Filtered {len(documents) - len(filtered_docs)} documents")
        return {"documents": filtered_docs, "question": question}

    def transform_query(state):
        """Re-write the query to improve retrieval."""
        logging.info("📝  Transforming the query...")
        question = state["question"]
        better_question = question_rewriter.invoke({"question": question})
        logging.info(f"Improved question: {better_question}")
        return {"documents": state["documents"], "question": better_question}

    def normal_llm(state):
        logging.info("💭  Calling normal LLM...")
        question = state["question"]
        answer = answer_normal.invoke({"question": question})
        logging.info(f"Answer: {answer[:20]}")
        return {"question": question, "generation": answer}

    def route_question(state):
        """Route the question to either vectorstore or normal LLM."""
        logging.info("⚖️  Routing the question...")
        question = state["question"]
        source = question_router.invoke({"question": question})
        logging.info(f"Routing to: {source}")
        return "normal_llm" if source["datasource"] == "normal_llm" else "vectorstore"

    def decide_to_generate(state):
        """Decide whether to generate or rephrase the query."""
        logging.info("🗯️  Deciding to generate or rephrase the query...")
        return "transform_query" if not state["documents"] else "generate"

    def grade_generation(state):
        """Grade the generation and its relevance."""
        logging.info("🔍 Grading the generation...")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        hallucination_score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )["score"]
        logging.info(f"Grounded in the documents: {hallucination_score}")
        return "useful" if hallucination_score == "yes" else "not supported"

    workflow = StateGraph(GraphState)
    workflow.add_node("normal_llm", normal_llm)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_conditional_edges(
        START, route_question, {"normal_llm": "normal_llm", "vectorstore": "retrieve"}
    )
    workflow.add_edge("normal_llm", END)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate", grade_generation, {"not supported": "generate", "useful": END}
    )

    chatbot = workflow.compile()

    return chatbot
