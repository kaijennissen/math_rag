"""
Evaluation-specific configuration settings.

This module provides configuration management for the evaluation CLI,
consolidating environment variables, API keys, and CLI arguments into
a unified pydantic settings class.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from math_rag.config.shared import Neo4jConfig, OpenAIConfig


class EvaluationSettings(BaseSettings):
    """Settings for evaluation CLI with full CLI argument parsing."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        cli_parse_args=True,
        case_sensitive=False,
        extra="ignore",
    )

    neo4j_config: Neo4jConfig = Field(
        default_factory=Neo4jConfig, description="Neo4j connection settings"
    )
    openai_config: OpenAIConfig = Field(
        default_factory=OpenAIConfig, description="OpenAI connection settings"
    )
    # CLI arguments as pydantic fields
    dataset_path: str = Field(..., description="Path to evaluation dataset CSV file")
    difficulty_levels: List[int] = Field(
        default=[1, 2, 3, 4], description="Difficulty levels to include"
    )
    output_path: str = Field(
        default="results.json", description="Output JSON file path"
    )
    sample_size: Optional[int] = Field(
        default=None, description="Number of questions to randomly sample"
    )
    log_level: str = Field(default="INFO", description="Logging level")


@lru_cache()
def get_config() -> EvaluationSettings:
    return EvaluationSettings()
