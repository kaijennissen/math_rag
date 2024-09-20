from langchain_community.chat_models import ChatOllama


def initialize_llm(config):
    return ChatOllama(model=config["llm_model"], temperature=config["temperature"])
