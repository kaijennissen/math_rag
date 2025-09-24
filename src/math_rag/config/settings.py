"""
Configuration management for the Knowledge Graph CLI using pydantic-settings.
Handles both environment variables and CLI arguments.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
