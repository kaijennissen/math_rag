"""
Module providing tools for generating and executing Cypher queries for Neo4j graph
metadata.
"""

import logging
import os
import warnings
from typing import Dict, List, Optional

import coloredlogs
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from smolagents import Tool

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class CypherExecutorTool(Tool):
    """Tool that executes Cypher queries against a Neo4j database."""

    name = "cypher_executor"
    description = (
        "Executes a Cypher query against the Neo4j graph database "
        "and returns the results."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The Cypher query to execute",
        },
        "parameters": {
            "type": "object",
            "description": "Optional parameters for the Cypher query",
            "default": {},
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        """Initialize the CypherExecutorTool with Neo4j connection details."""
        warnings.warn(
            """CypherExecutorTool is deprecated in favour of MCP client with
            mcp-neo4j-cypher server.""",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)
        self.uri = NEO4J_URI
        self.username = NEO4J_USERNAME
        self.password = NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )

    def _format_result(self, records: List[Dict]) -> str:
        """Format the query results into a readable string."""
        if not records:
            return "No results found."

        # Start with header
        result_str = "Query Results:\n"
        result_str += "-" * 40 + "\n"

        # Get all unique keys from all records
        all_keys = set()
        for record in records:
            all_keys.update(record.keys())

        # Format each record
        for i, record in enumerate(records):
            result_str += f"Record {i + 1}:\n"
            for key in all_keys:
                if key in record:
                    value = record[key]
                    # Format based on type
                    if isinstance(value, list):
                        if len(value) > 10:
                            value_str = (
                                ", ".join(str(v) for v in value[:10])
                                + f"... ({len(value)} total)"
                            )
                        else:
                            value_str = ", ".join(str(v) for v in value)
                    elif isinstance(value, dict):
                        value_str = str(value)
                    else:
                        value_str = str(value)

                    result_str += f"  {key}: {value_str}\n"

            # Add separator between records
            if i < len(records) - 1:
                result_str += "-" * 20 + "\n"

        return result_str

    def forward(self, query: str, parameters: Optional[Dict] = None) -> str:
        """
        Execute a Cypher query and return the formatted results.

        Args:
            query: The Cypher query to execute
            parameters: Optional parameters for the query

        Returns:
            Formatted string with the query results
        """
        if parameters is None:
            parameters = {}

        logger.info(f"Executing Cypher query: {query}")
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                records = [record.data() for record in result]
                return self._format_result(records)
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return f"Error executing Cypher query: {e}"


class SchemaInfoTool(Tool):
    """Tool that retrieves the Neo4j graph schema information."""

    name = "schema_info"
    description = (
        "Retrieves the current Neo4j graph schema information, "
        "including node labels, relationship types, and properties."
    )
    inputs = {}  # No inputs needed
    output_type = "string"

    def __init__(self, **kwargs):
        """Initialize the SchemaInfoTool with Neo4j connection details."""
        warnings.warn(
            """SchemaInfoTool is deprecated in favour of MCP client
            with mcp-neo4j-cypher server.""",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)
        self.uri = NEO4J_URI
        self.username = NEO4J_USERNAME
        self.password = NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )

    def _get_schema_info(self) -> str:
        """Get the current Neo4j schema information."""
        schema_info = {}

        try:
            # Get node labels
            with self.driver.session() as session:
                result = session.run(
                    "CALL db.labels() YIELD label RETURN collect(label) AS labels"
                )
                node_labels = [record.data() for record in result]
                schema_info["node_labels"] = (
                    node_labels[0].get("labels", []) if node_labels else []
                )

            # Get relationship types
            with self.driver.session() as session:
                result = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType "
                    "RETURN collect(relationshipType) AS types"
                )
                rel_types = [record.data() for record in result]
                schema_info["relationship_types"] = (
                    rel_types[0].get("types", []) if rel_types else []
                )

            # Get property keys
            with self.driver.session() as session:
                result = session.run(
                    "CALL db.propertyKeys() YIELD propertyKey "
                    "RETURN collect(propertyKey) AS keys"
                )
                property_keys = [record.data() for record in result]
                schema_info["property_keys"] = (
                    property_keys[0].get("keys", []) if property_keys else []
                )

            # Sample node properties by label
            schema_info["node_properties"] = {}
            for label in schema_info["node_labels"]:
                with self.driver.session() as session:
                    result = session.run(
                        f"MATCH (n:{label}) RETURN properties(n) AS props LIMIT 1"
                    )
                    sample_node = [record.data() for record in result]
                    if sample_node:
                        schema_info["node_properties"][label] = list(
                            sample_node[0].get("props", {}).keys()
                        )

            # Convert to string representation for output
            schema_str = f"""
Neo4j Graph Schema Information:
=============================

Node Labels ({len(schema_info["node_labels"])}):
{", ".join(schema_info["node_labels"])}

Relationship Types ({len(schema_info["relationship_types"])}):
{", ".join(schema_info["relationship_types"])}

Property Keys ({len(schema_info["property_keys"])}):
{", ".join(schema_info["property_keys"])}

Node Properties by Label:
"""
            for label, props in schema_info["node_properties"].items():
                schema_str += f"- {label}: {', '.join(props)}\n"

            logger.info("Retrieved schema information successfully")
            return schema_str

        except Exception as e:
            logger.error(f"Error getting schema information: {e}")
            return f"Error retrieving schema information: {e}"

    def forward(self) -> str:
        """
        Retrieve the current Neo4j graph schema information.

        Returns:
            Formatted string with the schema information
        """
        return self._get_schema_info()


class CypherQueryGeneratorTool(Tool):
    """Tool that generates Cypher queries from natural language using LLM."""

    name = "cypher_query_generator"
    description = (
        "Generates a Cypher query from a natural language question about the graph."
    )
    inputs = {
        "question": {
            "type": "string",
            "description": "A natural language question about the graph structure or "
            "metadata",
        },
        "schema_info": {
            "type": "string",
            "description": "The graph schema information to help generate an accurate "
            "query",
        },
    }
    output_type = "string"

    def __init__(self, model_name: str = "gpt-4.1", **kwargs):
        """Initialize the CypherQueryGeneratorTool with an LLM."""
        super().__init__(**kwargs)

        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name
        )

    def forward(self, question: str, schema_info: str) -> str:
        """
        Generate a Cypher query based on the natural language question using LLM.

        Args:
            question: Natural language question about the graph
            schema_info: The graph schema information

        Returns:
            Generated Cypher query
        """
        prompt = f"""
You are a Neo4j Cypher query expert. Your task is to convert natural language questions
about a graph database into precise Cypher queries.

Here's the current schema of our Neo4j graph database:

{schema_info}

IMPORTANT GUIDELINES:
1. Generate ONLY the Cypher query, without any explanations or markdown formatting
2. Ensure your query is syntactically correct
3. Optimize for readability and performance
4. If you need to count or aggregate, use appropriate functions
5. For visualization-oriented questions, limit results to a reasonable number
   (e.g., top 10)
6. If the question isn't clear, create a query that would most likely answer what
   they're asking

User's question: {question}

Cypher query:
"""

        query = self.llm.invoke(prompt).content
        # Clean up the response to extract just the query
        query = query.strip()
        # Remove any markdown code block formatting if present
        if query.startswith("```") and query.endswith("```"):
            query = query[3:-3].strip()
        if query.startswith("```cypher") and query.endswith("```"):
            query = query[9:-3].strip()

        return query
