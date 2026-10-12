"""Tests for model router, Opus protection, disk prompt cache, and cost metering."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from claim_check.config import (
    ConfigurationError,
    ModelRole,
    get_model_for_role,
)
from claim_check.llm import BudgetExceededError, LLMClient


def test_model_router_defaults():
    assert get_model_for_role(ModelRole.EXTRACT) == "claude-haiku"
    assert get_model_for_role(ModelRole.ARBITER) == "claude-sonnet-4.6"
    assert get_model_for_role(ModelRole.VERIFY) == "claude-sonnet-4.6"
    assert get_model_for_role(ModelRole.ROUTINE) == "claude-haiku"


def test_opus_protection_blocks_without_flag(monkeypatch):
    monkeypatch.setenv("MODEL_EXTRACT", "claude-opus-4.6")
    with pytest.raises(ConfigurationError, match="blocked to preserve budget"):
        get_model_for_role(ModelRole.EXTRACT, allow_opus=False)

    # Allowed when explicitly permitted
    model = get_model_for_role(ModelRole.EXTRACT, allow_opus=True)
    assert model == "claude-opus-4.6"


def test_llm_cache_hit_and_offline(tmp_path):
    cache_file = tmp_path / "test_cache.json"
    client = LLMClient(offline=True, cache_path=cache_file)

    # Seed a cache entry
    import hashlib
    model = "claude-haiku"
    messages = [{"role": "user", "content": "What is Retool?"}]
    cache_payload = [model, messages, 0.0, 2000]
    cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()

    client.cache[cache_key] = {
        "model": model,
        "role": "extract",
        "text": "Retool is a low-code internal tool platform.",
        "input_tokens": 15,
        "output_tokens": 10,
        "cost_usd": 0.0001,
    }

    # Should hit cache even in offline mode
    result = client.complete(messages, role=ModelRole.EXTRACT)
    assert result == "Retool is a low-code internal tool platform."
    stats = client.get_summary()
    assert stats["by_role"]["extract"]["cache_hits"] == 1
    assert stats["by_role"]["extract"]["calls"] == 0


def test_budget_exceeded(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://mock-proxy/v1")

    cache_file = tmp_path / "test_cache.json"
    client = LLMClient(offline=False, cache_path=cache_file, budget_usd=0.000001)
    client.cumulative_cost_usd = 100.0  # Already over budget

    with pytest.raises(BudgetExceededError):
        client.complete([{"role": "user", "content": "Hello"}], role=ModelRole.ROUTINE)
