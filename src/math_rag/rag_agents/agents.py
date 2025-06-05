#                         +-----------------+
#                         |  Manager Agent  |
#                         |  (graph_agent)  |
#                         +-----------------+
#                                 |
#                  _______________|______________
#                 |                              |
#      +-----------------------+              +----------------+
#      | graph_retriever_agent |              |  cypher_agent  |
#      +-----------------------+              +----------------+
#                  |                             |           |
#         GraphRetrieverTool          CypherExecutorTool     |
#                                                       SchemaInfoTool


import os
import yaml
import sys
from dotenv import load_dotenv
from smolagents import CodeAgent, ToolCallingAgent, OpenAIServerModel, HfApiModel
from math_rag.embeddings import GraphRetrieverTool
from math_rag.knowledge_graph import CypherExecutorTool, SchemaInfoTool
from math_rag.core import ROOT

load_dotenv()

# Load agent descriptions from YAML config file
# First try environment variable, then use project root-based path
config_path = os.getenv("AGENT_CONFIG_PATH")
if not config_path:
    config_path = ROOT / "config" / "agents.yaml"

try:
    with open(config_path, "r") as file:
        AGENT_DESCRIPTIONS = yaml.safe_load(file)
except FileNotFoundError:
    print(f"ERROR: Agent configuration file not found at {config_path}")
    print(
        "Set the AGENT_CONFIG_PATH environment variable or ensure the agents.yaml file exists in the config directory"
    )
    sys.exit(1)


def setup_rag_chat():
    """Setup a RAG chat agent with the graph-based retriever tool and meta-question subagent."""

    # Initialize the model inside this function
    reasoning_model = HfApiModel(
        # model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        # model_id="Qwen/QwQ-32B",
        # model_id="Qwen/Qwen3-32B",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        token=os.getenv("HUGGINGFACE_API_KEY"),
    )
    gpt_4_1 = OpenAIServerModel(
        model_id="gpt-4.1",
        api_base="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create main agent with the retriever tool and meta-agent
    graph_retriever_agent = CodeAgent(
        tools=[GraphRetrieverTool()],
        model=gpt_4_1,
        max_steps=10,
        verbosity_level=2,
        name="graph_retriever_agent",
        description=AGENT_DESCRIPTIONS["graph_retriever_agent"],
    )

    cypher_agent = ToolCallingAgent(
        tools=[CypherExecutorTool(), SchemaInfoTool()],
        model=gpt_4_1,
        max_steps=15,
        verbosity_level=2,
        name="cypher_agent",
        description=AGENT_DESCRIPTIONS["cypher_agent"],
    )

    graph_agent = CodeAgent(
        tools=[],
        managed_agents=[graph_retriever_agent, cypher_agent],
        model=reasoning_model,
        max_steps=20,
        verbosity_level=2,
        planning_interval=3,
    )

    return graph_agent


if __name__ == "__main__":
    agent = setup_rag_chat()
    agent.visualize()
    # agent.run("Was ist ein topologischer Raum?")

    question = (
        "Welche Definition, welcher Satz oder welches Theorem spielt die größte Rolle, "
        "wird also an vielen anderen Stellen verwendet? "
        "Was besagt diese Definition, dieser Satz oder dieses Theorem?"
    )
    agent.run(question)
