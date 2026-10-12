"""LiteLLM client with role-based model routing, disk prompt caching, and cost metering.

Features:
  - Role-based routing:
      * EXTRACT: claude-haiku (or MODEL_EXTRACT)
      * ARBITER: claude-sonnet-4.6 (or MODEL_ARBITER)
      * VERIFY: claude-sonnet-4.6 (or MODEL_VERIFY)
      * ROUTINE: claude-haiku (or MODEL_ROUTINE)
  - Strict Opus protection (blocked unless explicitly permitted).
  - Disk prompt cache (SHA-256 keyed) in state/llm_cache.json.
  - Per-role token and cost tracking with .00 budget enforcement.
  - Neatlogs instrumentation wrapper.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from claim_check.config import (
    MODEL_PRICING_PER_1M,
    ConfigurationError,
    ModelRole,
    get_litellm_config,
    get_model_for_role,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT_DIR / "state"
CACHE_PATH = STATE_DIR / "llm_cache.json"
SPEND_PATH = STATE_DIR / "llm_spend.json"

GLOBAL_BUDGET_USD = 20.0


class BudgetExceededError(RuntimeError):
    """Raised when LLM usage exceeds the total configured budget."""
    pass


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0

    def add(self, prompt_tokens: int, completion_tokens: int, cost: float, cached: bool = False):
        if cached:
            self.cache_hits += 1
        else:
            self.calls += 1
            self.input_tokens += prompt_tokens
            self.output_tokens += completion_tokens
            self.cost_usd += cost


class LLMClient:
    """LiteLLM client with model routing, disk caching, and cost metering."""

    def __init__(
        self,
        offline: bool = False,
        allow_opus: bool = False,
        cache_path: Optional[Path] = None,
        budget_usd: float = GLOBAL_BUDGET_USD,
    ):
        self.offline = offline
        self.allow_opus = allow_opus
        self.budget_usd = budget_usd
        self.cache_path = cache_path or CACHE_PATH
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.role_stats: Dict[str, UsageStats] = {
            r.value: UsageStats() for r in ModelRole
        }
        self.run_cost_usd = 0.0

        # Load cumulative spend
        self.cumulative_cost_usd = 0.0
        if SPEND_PATH.exists():
            try:
                spend_data = json.loads(SPEND_PATH.read_text(encoding="utf-8-sig"))
                self.cumulative_cost_usd = float(spend_data.get("cumulative_cost_usd", 0.0))
            except Exception:
                self.cumulative_cost_usd = 0.0

        # Load response disk cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8-sig"))
            except Exception:
                self.cache = {}

        if self.offline:
            self.client = None
        else:
            cfg = get_litellm_config(allow_opus=self.allow_opus)
            client = OpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
            )
            try:
                import neatlogs
                self.client = neatlogs.wrap(client)
            except Exception:
                self.client = client

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING_PER_1M.get(
            model,
            MODEL_PRICING_PER_1M.get("claude-haiku", {"input": 0.25, "output": 1.25}),
        )
        in_cost = (input_tokens / 1_000_000.0) * pricing["input"]
        out_cost = (output_tokens / 1_000_000.0) * pricing["output"]
        return in_cost + out_cost

    def _save_cache(self):
        try:
            self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write LLM cache: {e}")

    def _save_spend(self):
        try:
            SPEND_PATH.parent.mkdir(parents=True, exist_ok=True)
            SPEND_PATH.write_text(
                json.dumps(
                    {
                        "cumulative_cost_usd": round(self.cumulative_cost_usd, 6),
                        "last_run_cost_usd": round(self.run_cost_usd, 6),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to write spend log: {e}")

    def complete(
        self,
        messages: List[Dict[str, str]],
        role: ModelRole | str = ModelRole.ROUTINE,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        force_refresh: bool = False,
    ) -> str:
        """Send a completion request with role routing, caching, and budget tracking."""
        role_enum = ModelRole(role) if isinstance(role, str) else role
        model = get_model_for_role(role_enum, allow_opus=self.allow_opus)
        role_name = role_enum.value

        # Compute cache key
        cache_payload = [model, messages, temperature, max_tokens]
        cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()

        if not force_refresh and cache_key in self.cache:
            entry = self.cache[cache_key]
            text = entry["text"]
            self.role_stats[role_name].add(0, 0, 0.0, cached=True)
            return text

        if self.offline:
            raise ConfigurationError(
                f"LLM offline mode: no cached response for role '{role_name}' (key: {cache_key[:8]}). "
                f"Provide API keys or seed tests/fixtures/llm_cache.json."
            )

        # Budget reservation check
        input_estimate = max(10, sum(len(m.get("content", "")) for m in messages) // 4)
        reserved_cost = self._estimate_cost(model, input_estimate, max_tokens)
        if self.cumulative_cost_usd + reserved_cost > self.budget_usd:
            raise BudgetExceededError(
                f"Request would exceed total budget limit of  "
                f"(spent: , estimated call cost: )"
            )

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        reply_text = response.choices[0].message.content or ""
        in_tokens = response.usage.prompt_tokens if response.usage else input_estimate
        out_tokens = response.usage.completion_tokens if response.usage else len(reply_text) // 4
        call_cost = self._estimate_cost(model, in_tokens, out_tokens)

        # Update stats
        self.role_stats[role_name].add(in_tokens, out_tokens, call_cost, cached=False)
        self.run_cost_usd += call_cost
        self.cumulative_cost_usd += call_cost

        # Save to disk cache
        self.cache[cache_key] = {
            "model": model,
            "role": role_name,
            "text": reply_text,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_usd": round(call_cost, 6),
        }
        self._save_cache()
        self._save_spend()

        return reply_text

    def get_summary(self) -> Dict[str, Any]:
        """Return token, cost, and call metrics grouped by role."""
        per_role = {}
        for role_name, stats in self.role_stats.items():
            per_role[role_name] = {
                "calls": stats.calls,
                "cache_hits": stats.cache_hits,
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "cost_usd": round(stats.cost_usd, 6),
            }
        return {
            "run_cost_usd": round(self.run_cost_usd, 6),
            "cumulative_cost_usd": round(self.cumulative_cost_usd, 6),
            "budget_remaining_usd": round(max(0.0, self.budget_usd - self.cumulative_cost_usd), 6),
            "by_role": per_role,
        }
