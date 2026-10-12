"""Claim Check - AI agent that audits company public claims."""

from __future__ import annotations

import os
from typing import Optional

try:
    import neatlogs
except ImportError:
    neatlogs = None

try:
    from dotenv import load_dotenv
    from pathlib import Path
    _env_file = Path(__file__).resolve().parents[1] / ".env"
    if _env_file.exists():
        load_dotenv(dotenv_path=_env_file)
except ImportError:
    pass

from claim_check.dns_fallback import install_dns_fallback
install_dns_fallback()

_NEATLOGS_INITIALIZED = False


def init_neatlogs(workflow_name: str = "claim-check") -> bool:
    """Initialize neatlogs if API key is present in environment."""
    global _NEATLOGS_INITIALIZED
    if _NEATLOGS_INITIALIZED:
        return True
    api_key = os.getenv("NEATLOGS_API_KEY")
    if not api_key:
        return False
    if neatlogs is not None:
        try:
            neatlogs.init(api_key=api_key, workflow_name=workflow_name)
            _NEATLOGS_INITIALIZED = True
            return True
        except Exception:
            return False
    return False


__all__ = ["init_neatlogs"]
