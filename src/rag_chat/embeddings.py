from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def initialize_embeddings(config):
    if config["use_openai"]:
        return OpenAIEmbeddings(api_key=config["openai_api_key"])
    else:
        return OllamaEmbeddings(model=config["llm_model"])
