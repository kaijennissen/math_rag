from pydantic import AnyUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Neo4jConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    uri: AnyUrl = Field(..., description="Neo4j connection URI", alias="NEO4J_URI")
    username: str = Field(..., min_length=1, alias="NEO4J_USERNAME")
    # SecretStr hides password in logs/repr and provides get_secret_value()
    password: SecretStr = Field(
        ..., description="Neo4j password", alias="NEO4J_PASSWORD"
    )
    database: str = Field(default="neo4j", alias="NEO4J_DATABASE")


class OpenAIConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: SecretStr = Field(
        ..., description="Neo4j connection URI", alias="OPENAI_API_KEY"
    )
    model_id: str = Field("gpt-5-mini", alias="OPENAI_MODEL")
