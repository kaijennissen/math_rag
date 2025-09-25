"""
Configuration management for the Knowledge Graph CLI using pydantic-settings.
Handles both environment variables and CLI arguments.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from math_rag.core.project_root import ROOT


class KnowledgeGraphSettings(BaseSettings):
    """Settings for Knowledge Graph CLI operations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        cli_parse_args=True,  # Enable CLI argument parsing
        # cli_prog_name="kg_cli",
        extra="ignore",
    )

    # Neo4j connection (from environment)
    neo4j_uri: str = Field(..., alias="NEO4J_URI")
    neo4j_username: str = Field(..., alias="NEO4J_USERNAME")
    neo4j_password: str = Field(..., alias="NEO4J_PASSWORD")

    # CLI arguments with defaults
    db_path: Path = Field(
        default=Path("data") / "atomic_units.sqlite",
        description="Path to SQLite database",
    )
    document_name: str = Field(
        default="topological_spaces",
        description="Name of the document to process",
    )
    clear: bool = Field(
        default=False,
        description="Clear existing graph data before building",
    )

    # Fixed paths
    reference_tuples_path: Path = Field(default=Path("data") / "reference_tuples.pkl")


class RagChatSettings(BaseSettings):
    """Settings for RAG Chat CLI operations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        cli_parse_args=True,  # Enable CLI argument parsing
        extra="ignore",  # Ignore extra env variables
    )

    # Required API keys and Neo4j connection from environment
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    neo4j_uri: str = Field(..., alias="NEO4J_URI")
    neo4j_username: str = Field(..., alias="NEO4J_USERNAME")
    neo4j_password: str = Field(..., alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # Optional API keys
    huggingface_api_key: Optional[str] = Field(
        default=None, alias="HUGGINGFACE_API_KEY"
    )

    # Configuration with defaults
    agent_config_path: Path = Field(
        default=ROOT / "config" / "agents.yaml",
        description="Path to agent configuration YAML file",
    )
    model_id: str = Field(default="gpt-4.1", description="Model ID for the LLM")
    api_base: str = Field(
        default="https://api.openai.com/v1", description="API base URL for OpenAI"
    )
