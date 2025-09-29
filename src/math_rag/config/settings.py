"""
Unified configuration management for math_rag using pydantic-settings.

This module provides a hierarchical configuration system with:
1. A base settings class with common parameters
2. Component-specific settings classes
3. A provider singleton to manage settings instances

All settings classes handle both environment variables (.env file) and CLI arguments.
"""

from pathlib import Path
from typing import Dict, Optional, Type, TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from math_rag.core.project_root import ROOT

# Type variable for settings classes
T = TypeVar("T", bound="MathRagBaseSettings")


class MathRagBaseSettings(BaseSettings):
    """Base settings for all math_rag components with common configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database settings
    db_path: Path = Field(
        default=ROOT / "data" / "atomic_units.sqlite",
        description="Path to SQLite database",
    )

    # LLM model settings
    openai_api_key: Optional[str] = Field(
        default=None, alias="OPENAI_API_KEY", description="OpenAI API key"
    )
    model_name: str = Field(default="gpt-4.1", description="Default LLM model name")
    api_base: str = Field(
        default="https://api.openai.com/v1", description="API base URL"
    )

    # Neo4j settings
    neo4j_uri: Optional[str] = Field(
        default=None, alias="NEO4J_URI", description="Neo4j connection URI"
    )
    neo4j_username: Optional[str] = Field(
        default=None, alias="NEO4J_USERNAME", description="Neo4j username"
    )
    neo4j_password: Optional[str] = Field(
        default=None, alias="NEO4J_PASSWORD", description="Neo4j password"
    )
    neo4j_database: str = Field(
        default="neo4j", alias="NEO4J_DATABASE", description="Neo4j database name"
    )

    # Optional API keys
    huggingface_api_key: Optional[str] = Field(
        default=None, alias="HUGGINGFACE_API_KEY", description="Hugging Face API key"
    )


class KnowledgeGraphSettings(MathRagBaseSettings):
    """Settings for Knowledge Graph CLI operations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        cli_parse_args=True,  # Enable CLI argument parsing
        extra="ignore",
    )

    # Require Neo4j credentials
    neo4j_uri: str = Field(..., alias="NEO4J_URI")
    neo4j_username: str = Field(..., alias="NEO4J_USERNAME")
    neo4j_password: str = Field(..., alias="NEO4J_PASSWORD")

    # CLI arguments with defaults
    document_name: str = Field(
        default="topological_spaces",
        description="Name of the document to process",
    )
    clear: bool = Field(
        default=False,
        description="Clear existing graph data before building",
    )

    # Fixed paths
    reference_tuples_path: Path = Field(
        default=ROOT / "data" / "reference_tuples.pkl",
        description="Path to reference tuples pickle file",
    )


class RagChatSettings(MathRagBaseSettings):
    """Settings for RAG Chat CLI operations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        cli_parse_args=True,  # Enable CLI argument parsing
        extra="ignore",  # Ignore extra env variables
    )

    # Require keys and Neo4j connection
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    neo4j_uri: str = Field(..., alias="NEO4J_URI")
    neo4j_username: str = Field(..., alias="NEO4J_USERNAME")
    neo4j_password: str = Field(..., alias="NEO4J_PASSWORD")

    # Configuration with defaults
    agent_config_path: Path = Field(
        default=ROOT / "config" / "agents.yaml",
        description="Path to agent configuration YAML file",
    )
    model_id: str = Field(default="gpt-4.1", description="Model ID for the LLM")


class ReferenceExtractionSettings(MathRagBaseSettings):
    """Settings for PDF reference extraction."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        cli_parse_args=True,  # Enable CLI argument parsing
        extra="ignore",  # Ignore extra env variables
    )

    # PDF to process with default
    pdf_path: Path = Field(
        default=Path("docs/Skript_2024.pdf"),
        description="Path to the PDF file to extract references from",
    )

    # Optional arguments with defaults
    start_page: int = Field(
        default=10, description="Page number to start extraction from (0-indexed)"
    )
    output_path: Path = Field(
        default=ROOT / "data" / "reference_tuples.pkl",
        description="Path where the reference tuples pickle file will be saved",
    )


class TranslateAtomsSettings(MathRagBaseSettings):
    """
    Settings for the translate_atoms CLI.

    This settings class is the single source of CLI and environment configuration
    for the `translate_atoms` command. It requires an OpenAI API key via the
    `OPENAI_API_KEY` environment variable or .env file and supports CLI parsing
    via pydantic-settings (`cli_parse_args=True`).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        cli_parse_args=True,  # Enable CLI argument parsing from argv
        extra="ignore",
    )

    # Require OpenAI API key
    openai_api_key: str = Field(
        ..., alias="OPENAI_API_KEY", description="OpenAI API key"
    )

    # Pagination / processing control
    limit: int = Field(
        default=10,
        description="Number of rows to process (0 = all)",
    )
    write_to_db: bool = Field(
        default=False,
        description="Whether to persist results to the DB (default: False / dry-run)",
    )
    batch_size: int = Field(
        default=200,
        description="Pagination batch size for DB reads",
    )

    # Path to prompts file (required for CLI); kept configurable for tests/overrides
    prompts_path: Path = Field(
        default=ROOT / "config" / "prompts.yaml",
        description="Path to prompts YAML file (must contain top-level `system` and "
        "`user` keys)",
    )


class SettingsProvider:
    """
    Central provider for application settings.

    This singleton ensures consistent settings access throughout the application.
    Settings instances are cached to avoid redundant parsing.

    Example usage:
        settings = settings_provider.get_settings(KnowledgeGraphSettings)
        db_path = settings.db_path
    """

    _instance = None
    _settings_cache: Dict[str, MathRagBaseSettings] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._settings_cache = {}
        return cls._instance

    def get_settings(self, settings_class: Type[T] = MathRagBaseSettings) -> T:
        """
        Get settings of the specified type.

        Args:
            settings_class: Settings class to instantiate (defaults to
                MathRagBaseSettings)

        Returns:
            Instance of the requested settings class
        """
        class_name = settings_class.__name__
        if class_name not in self._settings_cache:
            self._settings_cache[class_name] = settings_class()
        return self._settings_cache[class_name]  # type: ignore


# Global settings provider instance
settings_provider = SettingsProvider()
