"""
Configuration management module for math_rag.

This module provides centralized configuration management using pydantic-settings,
ensuring type safety, validation, and clear configuration hierarchy:

1. Base settings class with common parameters
2. Component-specific settings classes
3. A provider singleton to manage settings instances

Usage:
    from math_rag.config import settings_provider, KnowledgeGraphSettings

    # Get settings for a specific component
    kg_settings = settings_provider.get_settings(KnowledgeGraphSettings)

    # Access settings properties
    db_path = kg_settings.db_path
"""

from math_rag.config.settings import (
    KnowledgeGraphSettings,
    MathRagBaseSettings,
    RagChatSettings,
    ReferenceExtractionSettings,
    TranslateAtomsSettings,
    settings_provider,
)

__all__ = [
    "MathRagBaseSettings",
    "KnowledgeGraphSettings",
    "RagChatSettings",
    "ReferenceExtractionSettings",
    "TranslateAtomsSettings",
    "settings_provider",
]
