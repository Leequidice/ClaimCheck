"""Configuration and environment variable management for Claim Check.

All credentials are strictly loaded from environment variables and never
hardcoded, logged, or exposed in error messages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    # Load .env if present in root without erroring if absent
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
except ImportError:
    pass


@dataclass(frozen=True)
class LiteLLMConfig:
    api_key: str
    base_url: str
    model: str = "anthropic/claude-3-5-haiku-latest"


class ConfigurationError(RuntimeError):
    """Raised when a required configuration or environment variable is missing."""
    pass


def get_crustdata_key() -> str:
    """Return the Crustdata API key from environment."""
    key = os.getenv("CRUSTDATA_API_KEY")
    if not key:
        raise ConfigurationError("CRUSTDATA_API_KEY environment variable is required.")
    return key


def get_litellm_config() -> LiteLLMConfig:
    """Return the LiteLLM configuration from environment.
    
    LITELLM_BASE_URL has NO default and must be explicitly configured.
    """
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        raise ConfigurationError("LITELLM_API_KEY environment variable is required.")
    base_url = os.getenv("LITELLM_BASE_URL")
    if not base_url:
        raise ConfigurationError(
            "LITELLM_BASE_URL environment variable is required (no default)."
        )
    model = os.getenv("LITELLM_MODEL", "anthropic/claude-3-5-haiku-latest")
    return LiteLLMConfig(api_key=api_key, base_url=base_url, model=model)


def get_neatlogs_key() -> Optional[str]:
    """Return the Neatlogs API key from environment if set."""
    return os.getenv("NEATLOGS_API_KEY")
