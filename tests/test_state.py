"""Tests for state persistence, recovery, and fault isolation."""

from __future__ import annotations

import pytest

from claim_check.models import (
    AtomicClaim,
    ClaimAuditResult,
    ClaimCategory,
    VerdictType,
)
from claim_check.state import StateManager


def test_state_manager_incremental_saving_and_resume(tmp_path):
    mgr = StateManager(state_dir=tmp_path)
    claim1 = AtomicClaim(
        id="claim_1",
        category=ClaimCategory.HEADCOUNT,
        original_text="50 employees",
        normalized_value=50,
    )
    claim2 = AtomicClaim(
        id="claim_2",
        category=ClaimCategory.FOUNDING_YEAR,
        original_text="founded in 2020",
        normalized_value=2020,
    )

    run_id = mgr.init_run("test.com", [claim1, claim2])
    data = mgr.load_run(run_id)
    assert data["status"] == "in_progress"
    assert len(data["claims"]) == 2
    assert data["completed_claim_ids"] == []

    # Record first claim result
    res1 = ClaimAuditResult(
        claim_id="claim_1",
        claim=claim1,
        verdict=VerdictType.VERIFIED,
        confidence=0.9,
        reasoning="Within range",
    )
    mgr.record_claim_result(run_id, res1)

    # Reload state simulating recovery after crash
    resumed = mgr.load_run(run_id)
    assert resumed["completed_claim_ids"] == ["claim_1"]
    assert len(resumed["results"]) == 1
    assert resumed["results"][0]["verdict"] == "verified"

    # Mark completed
    mgr.mark_completed(run_id)
    final_data = mgr.load_run(run_id)
    assert final_data["status"] == "completed"
