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

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import sys
from pathlib import Path
from typing import Optional

import yaml
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import SearchType
from mcp import StdioServerParameters
from smolagents import (
    CodeAgent,
    InferenceClientModel,  # noqa: F401
    MCPClient,
    OpenAIServerModel,
    ToolCallingAgent,
)

from math_rag.graph_tools import GraphRetrieverTool, PathRAGRetrieverTool
from math_rag.graph_tools.utils import get_pathrag_query

logger = logging.getLogger(__name__)


def setup_rag_chat(
    openai_api_key: str,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    neo4j_database: str,
    agent_config_path: Path,
    model_id: str = "gpt-4.1",
    api_base: str = "https://api.openai.com/v1",
    huggingface_api_key: Optional[str] = None,
):
    """Setup a RAG chat agent with the graph-based retriever tool and meta-question
    subagent.

    Args:
        openai_api_key: OpenAI API key
        neo4j_uri: Neo4j connection URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        agent_config_path: Path to agent configuration YAML file
        model_id: Model ID for the LLM
        api_base: API base URL for OpenAI
        huggingface_api_key: Optional HuggingFace API key

    Returns:
        tuple: A tuple containing (graph_agent, mcp_client). The mcp_client must be
               disconnected when done using the agent.
    """

    # Load agent descriptions from YAML config file

    try:
        with open(agent_config_path, "r") as file:
            AGENT_DESCRIPTIONS = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Agent configuration file not found at {agent_config_path}")
        sys.exit(1)

    # Initialize the model inside this function
    # reasoning_model = InferenceClientModel(
    #     # model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    #     # model_id="Qwen/QwQ-32B",
    #     # model_id="Qwen/Qwen3-32B",
    #     model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    #     token=huggingface_api_key,
    # )
    gpt_4_1 = OpenAIServerModel(
        model_id=model_id,
        api_base=api_base,
        api_key=openai_api_key,
    )

    # Initialize embedding model for GraphRetrieverTool
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    server_parameters = StdioServerParameters(
        command="uvx",
        args=["mcp-neo4j-cypher@0.4.0", "--transport", "stdio"],
        env={
            "NEO4J_URI": neo4j_uri,
            "NEO4J_USERNAME": neo4j_username,
            "NEO4J_PASSWORD": neo4j_password,
            "NEO4J_DATABASE": neo4j_database,
        },
    )

    # Create MCP client and get tools
    mcp_client = MCPClient(server_parameters)
    tools = mcp_client.get_tools()

    # Create vector stores for both retrievers
    hybrid_vector_store = Neo4jVector.from_existing_index(
        embedding_model,
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        index_name="vector_index_text_nl_Embedding",
        keyword_index_name="fulltext_index_AtomicItem",
        embedding_node_property="text_nl_Embedding",
        search_type=SearchType.HYBRID,
    )

    pathrag_vector_store = Neo4jVector.from_existing_index(
        embedding_model,
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        index_name="vector_index_text_nl_Embedding",
        keyword_index_name="fulltext_index_AtomicItem",
        embedding_node_property="text_nl_Embedding",
        search_type=SearchType.HYBRID,
        retrieval_query=get_pathrag_query(),
    )

    # Initialize tools with dependency injection
    graph_retriever_tool = GraphRetrieverTool(vector_store=hybrid_vector_store)
    pathrag_retriever_tool = PathRAGRetrieverTool(vector_store=pathrag_vector_store)

    # Create main agent with both retriever tools
    graph_retriever_agent = CodeAgent(
        tools=[*tools, graph_retriever_tool, pathrag_retriever_tool],
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
    # For testing - import settings and load from environment
    from math_rag.config.settings import RagChatSettings

    settings = RagChatSettings()
    agent, mcp_client = setup_rag_chat(
        openai_api_key=settings.openai_api_key,
        neo4j_uri=settings.neo4j_uri,
        neo4j_username=settings.neo4j_username,
        neo4j_password=settings.neo4j_password,
        neo4j_database=settings.neo4j_database,
        agent_config_path=settings.agent_config_path,
        model_id=settings.model_id,
        api_base=settings.api_base,
        huggingface_api_key=settings.huggingface_api_key,
    )
    try:
        agent.visualize()

        question = (
            "Welche Definition, welcher Satz oder welches Theorem spielt die "
            "größte Rolle, wird also an vielen anderen Stellen verwendet? "
            "Was besagt diese Definition, dieser Satz oder dieses Theorem?"
        )
        agent.run(question)
    finally:
        mcp_client.disconnect()
