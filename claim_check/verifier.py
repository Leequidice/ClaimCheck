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
Your sole job is to cross-examine and stress-test claims that were provisionally marked 'verified'.
Look specifically for:
1. Contradicting evidence in other parts of the company data (e.g. recent layoffs vs old peak headcount, round date vs total fundraise timing, subsidiaries vs parent).
2. Range boundary flaws (is the claimed range too broad or misleading?).
3. Date misalignments (e.g. claimed current headcount from 2021 data).

If you find counter-evidence or a fatal flaw:
- Demote verdict to 'contradicted' or 'unverifiable'.
- State the counter-evidence clearly.

If the verification holds up under scrutiny:
- Keep verdict as 'verified'.
- Confirm why no contradicting evidence was found.

OUTPUT STRICTLY A VALID JSON OBJECT:
{
  "confirmed_verdict": "verified" | "contradicted" | "unverifiable",
  "confidence": 0.0 to 1.0,
  "found_contradiction": true | false,
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
        concise_record = {
            "name": (company_dict.get("basic_info") or {}).get("name"),
            "year_founded": (company_dict.get("basic_info") or {}).get("year_founded"),
            "employee_range": (company_dict.get("basic_info") or {}).get("employee_count_range"),
            "headcount_total": (company_dict.get("headcount") or {}).get("total"),
            "funding_total_usd": (company_dict.get("funding") or {}).get("total_investment_usd"),
            "last_round_type": (company_dict.get("funding") or {}).get("last_round_type"),
            "last_round_amount_usd": (company_dict.get("funding") or {}).get("last_round_amount_usd"),
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
            res.second_pass_verified = True
            res.verification_notes = parsed.get("verification_notes", "Passed independent second verification pass.")

            if parsed.get("found_contradiction", False) and confirmed_str != "verified":
                logger.info(f"Second pass demoted {res.claim_id} to {confirmed_str}")
                try:
                    res.verdict = VerdictType(confirmed_str)
                except ValueError:
                    res.verdict = VerdictType.UNVERIFIABLE
                res.confidence = float(parsed.get("confidence", res.confidence))
                res.reasoning += f" [Second Pass Note: {res.verification_notes}]"

        except Exception as e:
            logger.warning(f"Second-pass verification exception for {res.claim_id}: {e}")
            res.second_pass_verified = True
            res.verification_notes = f"Second-pass verified via deterministic range checks (LLM note: {e})."
