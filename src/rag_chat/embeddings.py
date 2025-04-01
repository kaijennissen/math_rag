from langchain.embeddings import OpenAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings


def initialize_embeddings(config):
    if config["use_openai"]:
        return OpenAIEmbeddings(
            model="text-embedding-3-large", api_key=config["openai_api_key"]
        )
    else:
        return OllamaEmbeddings(model=config["llm_model"])
