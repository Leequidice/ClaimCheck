"""State Persistence and Recovery.

Saves progress per claim to JSON so crashed or interrupted runs resume seamlessly.
Isolates failures so a single claim failure never aborts the whole run.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from claim_check.models import AtomicClaim, ClaimAuditResult, VerdictType

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT_DIR / "state" / "runs"


class StateManager:
    """Manages incremental run state on disk."""

    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _run_file(self, run_id: str) -> Path:
        return self.state_dir / f"{run_id}.json"

    def init_run(self, target: str, claims: List[AtomicClaim], run_id: Optional[str] = None) -> str:
        """Initialize a new audit run or reload an existing one."""
        rid = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:6]}"
        path = self._run_file(rid)

        if not path.exists():
            payload = {
                "run_id": rid,
                "target": target,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "in_progress",
                "claims": [c.model_dump() for c in claims],
                "completed_claim_ids": [],
                "results": [],
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return rid

    def record_claim_result(self, run_id: str, result: ClaimAuditResult):
        """Save progress incrementally after each claim is evaluated."""
        path = self._run_file(run_id)
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if result.claim_id not in data["completed_claim_ids"]:
                data["completed_claim_ids"].append(result.claim_id)
                data["results"].append(result.model_dump())
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to record state for claim {result.claim_id}: {e}")

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """Load state of an ongoing or completed run."""
        path = self._run_file(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Run state file {path} does not exist.")
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def mark_completed(self, run_id: str):
        """Mark run status as completed."""
        path = self._run_file(run_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                data["status"] = "completed"
                data["updated_at"] = datetime.now(timezone.utc).isoformat()
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to mark run {run_id} completed: {e}")

    def find_latest_run_for_target(self, target: str) -> Optional[str]:
        """Find the most recent in-progress or interrupted run for a target."""
        candidates = []
        for file in self.state_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8-sig"))
                if data.get("target") == target:
                    candidates.append((data.get("updated_at", ""), data.get("run_id")))
            except Exception:
                continue
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        return None
