from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import SKLearnVectorStore


def initialize_retrievers(docs, embeddings, config):
    keyword_retriever = BM25Retriever.from_documents(
        docs, similarity_top_k=config["top_k"]
    )
    vectorstore = SKLearnVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
    )
    vectorstore = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": config["score_threshold"],
            "top_k": config["top_k"],
        },
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore, keyword_retriever], weights=[0.2, 0.8]
    )
    return ensemble_retriever
