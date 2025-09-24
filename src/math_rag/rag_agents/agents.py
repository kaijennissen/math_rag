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


import logging
import os
import sys

import yaml
from dotenv import load_dotenv
from mcp import StdioServerParameters
from smolagents import (
    CodeAgent,
    InferenceClientModel,  # noqa: F401
    MCPClient,
    OpenAIServerModel,
    ToolCallingAgent,
)

from math_rag.core import ROOT
from math_rag.graph_tools import GraphRetrieverTool

logger = logging.getLogger(__name__)


def setup_rag_chat():
    """Setup a RAG chat agent with the graph-based retriever tool and meta-question
    subagent.

    Returns:
        tuple: A tuple containing (graph_agent, mcp_client). The mcp_client must be
               disconnected when done using the agent.
    """

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
        logger.error(f"Agent configuration file not found at {config_path}")
        logger.error(
            """Set the AGENT_CONFIG_PATH environment variable or ensure
            the agents.yaml file exists in the config directory"""
        )
        sys.exit(1)

    # Initialize the model inside this function
    # reasoning_model = InferenceClientModel(
    #     # model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    #     # model_id="Qwen/QwQ-32B",
    #     # model_id="Qwen/Qwen3-32B",
    #     model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    #     token=os.getenv("HUGGINGFACE_API_KEY"),
    # )
    gpt_4_1 = OpenAIServerModel(
        model_id="gpt-4.1",
        api_base="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    server_parameters = StdioServerParameters(
        command="uvx",
        args=["mcp-neo4j-cypher@0.4.0", "--transport", "stdio"],
        env={
            "NEO4J_URI": os.getenv("NEO4J_URI"),
            "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
            "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
            "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE"),
        },
    )

    # Create MCP client and get tools
    mcp_client = MCPClient(server_parameters)
    tools = mcp_client.get_tools()
    # Create main agent with the retriever tool and meta-agent
    graph_retriever_agent = CodeAgent(
        tools=[*tools, GraphRetrieverTool()],
        model=gpt_4_1,
        max_steps=10,
        verbosity_level=2,
        name="graph_retriever_agent",
        description=AGENT_DESCRIPTIONS["graph_retriever_agent"],
        use_structured_outputs_internally=True,
    )

    cypher_agent = ToolCallingAgent(
        tools=[*tools],
        model=gpt_4_1,
        max_steps=15,
        verbosity_level=2,
        name="cypher_agent",
        description=AGENT_DESCRIPTIONS["cypher_agent"],
        # use_structured_outputs_internally=True
    )

    graph_agent = CodeAgent(
        tools=[],
        managed_agents=[graph_retriever_agent, cypher_agent],
        # add_base_tools=True,
        model=gpt_4_1,
        # model=reasoning_model,
        max_steps=20,
        verbosity_level=2,
        planning_interval=3,
        use_structured_outputs_internally=True,
    )

    return graph_agent, mcp_client


if __name__ == "__main__":
    agent, mcp_client = setup_rag_chat()
    try:
        agent.visualize()
        # agent.run("Was ist ein topologischer Raum?")

        question = (
            "Welche Definition, welcher Satz oder welches Theorem spielt die "
            "größte Rolle, wird also an vielen anderen Stellen verwendet? "
            "Was besagt diese Definition, dieser Satz oder dieses Theorem?"
        )
        agent.run(question)
    finally:
        mcp_client.disconnect()
