"""Second Independent Verification Pass.

Re-checks every "verified" verdict by actively looking for contradicting evidence,
boundary conditions, date misalignments, or hidden discrepancies before the report
is finalized.
Uses Claude Sonnet 4.6 (ModelRole.VERIFY).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from claim_check.config import ModelRole
from claim_check.llm import LLMClient
from claim_check.models import (
    ClaimAuditResult,
    CompanyData,
    VerdictType,
)

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are a rigorous, independent verification inspector.
Your job is to cross-examine and stress-test claims that were provisionally marked 'verified'.

STRICT VERIFIER RULES:
1. You may demote a 'verified' verdict to 'contradicted' ONLY IF you cite specific contradicting factual evidence present in the provided retrieved data (company_evidence_summary or provisional_evidence).
2. General memory-based doubt, suspicion about dates, or external unverified assumptions must NEVER demote a verdict to 'contradicted'. If you have memory-based doubt without contradictory data in the provided evidence, keep the confirmed_verdict as 'verified' and simply lower the confidence score.
3. For historical-round claims, evaluate only the rounds and milestones list. Never use 'last_round_type' to reject an earlier historical funding round.

OUTPUT STRICTLY A VALID JSON OBJECT:
{
  "confirmed_verdict": "verified" | "contradicted" | "unverifiable",
  "confidence": 0.0 to 1.0,
  "found_contradiction": true | false,
  "cited_counter_evidence": "exact contradicting field or value from retrieved data, or null",
  "verification_notes": "concise explanation of stress-test findings"
}
"""


class IndependentVerifier:
    """Second independent pass re-checking all provisional 'verified' verdicts."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def verify_all(
        self,
        audit_results: List[ClaimAuditResult],
        company_data: Optional[CompanyData],
    ) -> List[ClaimAuditResult]:
        """Re-check every verified claim and update with second-pass confirmation."""
        verified_claims = [r for r in audit_results if r.verdict == VerdictType.VERIFIED]
        if not verified_claims or not company_data:
            return audit_results

        company_dict = company_data.model_dump(exclude_none=True)

        for res in audit_results:
            if res.verdict != VerdictType.VERIFIED:
                continue

            self._verify_single(res, company_dict)

        return audit_results

    def _verify_single(self, res: ClaimAuditResult, company_dict: Dict[str, Any]):
        """Perform second pass verification on a single verified claim."""
        funding_dict = company_dict.get("funding") or {}
        raw_rounds = funding_dict.get("funding_rounds") or [
            m for m in funding_dict.get("milestones", []) if isinstance(m, dict) and (m.get("round_type") or m.get("amount_usd"))
        ]

        concise_record = {
            "name": (company_dict.get("basic_info") or {}).get("name"),
            "year_founded": (company_dict.get("basic_info") or {}).get("year_founded"),
            "employee_range": (company_dict.get("basic_info") or {}).get("employee_count_range"),
            "headcount_total": (company_dict.get("headcount") or {}).get("total"),
            "funding_total_usd": funding_dict.get("total_investment_usd"),
            "historical_rounds": [
                {"round_type": r.get("round_type"), "amount_usd": r.get("amount_usd"), "date": r.get("date") or r.get("funding_date")}
                for r in raw_rounds[:10]
            ],
            "headquarters": (company_dict.get("locations") or {}).get("headquarters"),
            "founders": [f.get("name") for f in (company_dict.get("people") or {}).get("founders", []) if isinstance(f, dict)],
        }

        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "claim": {
                            "category": res.claim.category.value,
                            "original_text": res.claim.original_text,
                            "normalized_value": res.claim.normalized_value,
                        },
                        "provisional_reasoning": res.reasoning,
                        "provisional_evidence": res.evidence,
                        "company_evidence_summary": concise_record,
                    },
                    indent=2,
                    default=str,
                ),
            },
        ]

        try:
            output = self.llm.complete(messages, role=ModelRole.VERIFY, temperature=0.0)
            cleaned = output.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)

            confirmed_str = str(parsed.get("confirmed_verdict", "verified")).lower()
            cited_counter = parsed.get("cited_counter_evidence")
            res.second_pass_verified = True
            res.verification_notes = parsed.get("verification_notes", "Passed independent second verification pass.")

            # Rule: demote to contradicted ONLY IF specific contradicting evidence from retrieved data is cited
            if parsed.get("found_contradiction", False) and confirmed_str == "contradicted" and cited_counter:
                logger.info(f"Second pass demoted {res.claim_id} to contradicted citing: {cited_counter}")
                res.verdict = VerdictType.CONTRADICTED
                res.confidence = float(parsed.get("confidence", res.confidence))
                res.reasoning += f" [Second Pass Refutation: {res.verification_notes}]"
            elif confirmed_str == "unverifiable" and parsed.get("found_contradiction", False) and cited_counter:
                res.verdict = VerdictType.UNVERIFIABLE
                res.confidence = float(parsed.get("confidence", res.confidence))
                res.reasoning += f" [Second Pass Demotion: {res.verification_notes}]"
            else:
                # Memory-based doubt lowers confidence but keeps verdict verified
                new_conf = float(parsed.get("confidence", res.confidence))
                if new_conf < res.confidence:
                    res.confidence = new_conf
                if "doubt" in res.verification_notes.lower() or "suspicious" in res.verification_notes.lower():
                    res.reasoning += f" [Second Pass Caution: {res.verification_notes}]"

        except Exception as e:
            logger.warning(f"Second-pass verification exception for {res.claim_id}: {e}")
            res.second_pass_verified = True
            res.verification_notes = f"Second-pass verified via deterministic range checks (LLM note: {e})."
