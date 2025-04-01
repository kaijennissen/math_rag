from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatOllama


def initialize_llm(config):
    if config["use_openai"]:
        llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    else:
        # If using Ollama, ensure the model is set
        llm = ChatOllama(model=config["llm_model"], temperature=config["temperature"])
    return llm
