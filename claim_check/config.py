"""Configuration, model routing, and environment variable management for Claim Check.

All credentials are strictly loaded from environment variables and never
hardcoded, logged, or exposed in error messages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
except ImportError:
    pass


class ModelRole(str, Enum):
    EXTRACT = "extract"
    ARBITER = "arbiter"
    VERIFY = "verify"
    ROUTINE = "routine"


# Default model assignments per role
DEFAULT_MODELS: Dict[ModelRole, str] = {
    ModelRole.EXTRACT: "claude-haiku",
    ModelRole.ARBITER: "claude-sonnet-4.6",
    ModelRole.VERIFY: "claude-sonnet-4.6",
    ModelRole.ROUTINE: "claude-haiku",
}

# Estimated pricing per 1M tokens (USD) for cost tracking
MODEL_PRICING_PER_1M: Dict[str, Dict[str, float]] = {
    "nova-micro": {"input": 0.035, "output": 0.14},
    "nova-lite": {"input": 0.06, "output": 0.24},
    "nova-2-lite": {"input": 0.06, "output": 0.24},
    "nova-pro": {"input": 0.80, "output": 3.20},
    "claude-haiku": {"input": 0.25, "output": 1.25},
    "claude-sonnet-4.6": {"input": 3.00, "output": 15.00},
    "claude-opus-4.6": {"input": 15.00, "output": 75.00},
}


class ConfigurationError(RuntimeError):
    """Raised when a required configuration or environment variable is missing."""
    pass


@dataclass(frozen=True)
class LiteLLMConfig:
    api_key: str
    base_url: str
    allow_opus: bool = False


def get_crustdata_key() -> str:
    """Return the Crustdata API key from environment."""
    key = os.getenv("CRUSTDATA_API_KEY")
    if not key:
        raise ConfigurationError(
            "CRUSTDATA_API_KEY environment variable is required for live enrichment. "
            "Pass --offline to run with recorded fixtures."
        )
    return key


def get_model_for_role(role: ModelRole | str, allow_opus: bool = False) -> str:
    """Resolve the model for a specific pipeline role.
    
    Roles:
      - extract: claude-haiku (or MODEL_EXTRACT)
      - arbiter: claude-sonnet-4.6 (or MODEL_ARBITER)
      - verify: claude-sonnet-4.6 (or MODEL_VERIFY)
      - routine: claude-haiku (or MODEL_ROUTINE)
      
    Guards against Opus unless explicitly allowed.
    """
    role_enum = ModelRole(role) if isinstance(role, str) else role
    env_var_name = f"MODEL_{role_enum.name}"
    model = os.getenv(env_var_name, DEFAULT_MODELS[role_enum]).strip()

    is_opus_allowed = allow_opus or os.getenv("ALLOW_OPUS", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    if "opus" in model.lower() and not is_opus_allowed:
        raise ConfigurationError(
            f"Model '{model}' assigned to role '{role_enum.value}' is blocked to preserve budget. "
            f"Never use Opus unless explicitly allowed via --allow-opus or ALLOW_OPUS=1."
        )
    return model


def get_litellm_config(allow_opus: bool = False) -> LiteLLMConfig:
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
    opus_flag = allow_opus or os.getenv("ALLOW_OPUS", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    return LiteLLMConfig(api_key=api_key, base_url=base_url, allow_opus=opus_flag)


def get_neatlogs_key() -> Optional[str]:
    """Return the Neatlogs API key from environment if set."""
    return os.getenv("NEATLOGS_API_KEY")
