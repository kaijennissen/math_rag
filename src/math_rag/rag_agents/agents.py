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
from smolagents import CodeAgent, ToolCallingAgent, OpenAIServerModel
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
    model = OpenAIServerModel(
        model_id="gpt-4.1",
        api_base="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create main agent with the retriever tool and meta-agent
    graph_retriever_agent = CodeAgent(
        tools=[GraphRetrieverTool()],
        model=model,
        max_steps=10,
        verbosity_level=2,
        name="graph_retriever_agent",
        description=AGENT_DESCRIPTIONS["graph_retriever_agent"],
    )

    cypher_agent = ToolCallingAgent(
        tools=[CypherExecutorTool(), SchemaInfoTool()],
        model=model,
        max_steps=10,
        verbosity_level=2,
        name="cypher_agent",
        description=AGENT_DESCRIPTIONS["cypher_agent"],
    )

    graph_agent = CodeAgent(
        tools=[],
        managed_agents=[graph_retriever_agent, cypher_agent],
        model=model,
        max_steps=10,
        verbosity_level=2,
        planning_interval=1,
    )

    return graph_agent


if __name__ == "__main__":
    agent = setup_rag_chat()
    agent.visualize()
    agent.run("Was ist ein topologischer Raum?")
