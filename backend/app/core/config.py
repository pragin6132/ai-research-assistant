"""
Application configuration.

Centralizes all settings for the AI Research Assistant backend.
Values are loaded from environment variables (see .env.example).
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- General app settings ---
    APP_NAME: str = "AI Research Assistant"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # --- LLM provider API keys (placeholders, no real keys, not used yet) ---
    GEMINI_API_KEY: str | None = None
    NVIDIA_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None

    # --- Search provider API key (placeholder, not used yet) ---
    TAVILY_API_KEY: str | None = None

    # --- Search provider (Tavily) configuration ---
    # Used by app/services/tavily_service.py. "search_depth" mirrors Tavily's
    # own "basic" | "advanced" options.
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_SEARCH_DEPTH: str = "basic"

    # --- LLM model configuration ---
    # Default/primary provider for reference; LLMManager always tries Gemini
    # first regardless of this value (see app/services/llm_manager.py).
    LLM_PROVIDER: str = "gemini"
    LLM_TEMPERATURE: float = 0.2

    # Per-provider model names, used by LLMManager's provider implementations.
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    NVIDIA_MODEL_NAME: str = "nvidia/nemotron-3-super-120b-a12b"
    GROQ_MODEL_NAME: str = "openai/gpt-oss-20b"

    # --- Image generation provider (Pixazo) ---
    PIXAZO_API_KEY: str | None = None
    # Pixazo assigns each model its own endpoint slug (see
    # app/services/pixazo_service.py for how this is used). "z-image-base"
    # is a free-tier, well-documented model used as the default here.
    PIXAZO_IMAGE_MODEL: str = "z-image-base"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
