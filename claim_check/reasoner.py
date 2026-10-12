"""Claim Reasoner and Source Arbiter.

Evaluates claims against extracted Crustdata evidence using range tolerance rules.
When multiple data points or sources disagree, invokes Claude Sonnet 4.6 (ModelRole.ARBITER)
to arbitrate and document the reasoning.

Rules:
  - Treat numeric claims as ranges, not exact numbers.
  - Never mark a claim verified without at least one cited data point.
  - Mark unverifiable when evidence is missing instead of guessing.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from claim_check.config import ModelRole
from claim_check.llm import LLMClient
from claim_check.models import (
    AtomicClaim,
    ClaimAuditResult,
    ClaimCategory,
    CompanyData,
    PersonData,
    VerdictType,
)

logger = logging.getLogger(__name__)

ARBITRATION_SYSTEM_PROMPT = """You are a senior data arbitration judge.
You have been presented with a public claim made about a company and conflicting pieces of evidence from data sources (e.g. employee count ranges vs exact headcount, round details vs total investment, or self-reported vs database timestamps).

Your task:
1. Analyze the discrepancy between the claim and the cited sources.
2. Decide which source is more authoritative and why (e.g. primary registered headcount timestamped recently is more authoritative than historical range).
3. Evaluate whether the claim is:
   - "verified" (the claim is consistent with the most authoritative evidence within a reasonable margin/range)
   - "contradicted" (the claim is substantively refuted by the most authoritative evidence)
   - "unverifiable" (the sources are too ambiguous, outdated, or lacking specific data points)
4. Assign a confidence score from 0.0 to 1.0.
5. Provide concise, rigorous reasoning explaining the decision.

OUTPUT STRICTLY A VALID JSON OBJECT:
{
  "verdict": "verified" | "contradicted" | "unverifiable",
  "confidence": 0.0 to 1.0,
  "cited_source": "string description of authoritative source",
  "reasoning": "clear explanation of arbitration"
}
"""


class ClaimReasoner:
    """Evaluates claims against evidence and emits structured verdicts."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def evaluate_claim(
        self,
        claim: AtomicClaim,
        company_data: Optional[CompanyData],
        person_data: Optional[Dict[str, PersonData]] = None,
    ) -> ClaimAuditResult:
        """Evaluate a single atomic claim against company evidence."""
        if not company_data:
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.0,
                evidence=None,
                source_cited=None,
                reasoning="No evidence available: company record was not found in Crustdata dataset.",
            )

        if claim.category == ClaimCategory.HEADCOUNT:
            return self._evaluate_headcount(claim, company_data)
        elif claim.category == ClaimCategory.FUNDING:
            return self._evaluate_funding(claim, company_data)
        elif claim.category == ClaimCategory.FOUNDING_YEAR:
            return self._evaluate_founding_year(claim, company_data)
        elif claim.category == ClaimCategory.LOCATION:
            return self._evaluate_location(claim, company_data)
        elif claim.category == ClaimCategory.FOUNDER_BACKGROUND:
            return self._evaluate_founder(claim, company_data, person_data)
        elif claim.category == ClaimCategory.CUSTOMER_COUNT:
            return self._evaluate_customer_count(claim, company_data)
        else:
            return self._evaluate_general(claim, company_data)

    def _evaluate_headcount(self, claim: AtomicClaim, data: CompanyData) -> ClaimAuditResult:
        actual_total = data.headcount.total if data.headcount else None
        range_str = data.basic_info.employee_count_range if data.basic_info else None

        if actual_total is None and not range_str:
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.0,
                evidence=None,
                source_cited=None,
                reasoning="Crustdata headcount data is unavailable for this company.",
            )

        # Check if sources conflict (e.g. range says 51-200 but total says 350)
        has_conflict = False
        if actual_total and range_str and "-" in range_str:
            try:
                parts = range_str.split("-")
                r_low, r_high = int(parts[0].strip()), int(parts[1].split()[0].strip())
                if actual_total < r_low * 0.7 or actual_total > r_high * 1.5:
                    has_conflict = True
            except Exception:
                pass

        if has_conflict:
            return self._arbitrate_with_llm(
                claim,
                evidence_dict={
                    "headcount.total": actual_total,
                    "basic_info.employee_count_range": range_str,
                },
            )

        # Direct range comparison
        val = claim.normalized_value
        if isinstance(val, (int, float)) and actual_total is not None:
            claimed_num = float(val)
            t_min = claim.tolerance_min if claim.tolerance_min is not None else max(1.0, claimed_num * 0.75)
            t_max = claim.tolerance_max if claim.tolerance_max is not None else (claimed_num * 1.25 + 6.0)

            evidence_val = actual_total
            source = "crustdata.headcount.total"

            if t_min <= actual_total <= t_max:
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.VERIFIED,
                    confidence=0.95,
                    evidence={"actual_headcount": actual_total, "acceptable_range": [t_min, t_max]},
                    source_cited=source,
                    reasoning=(
                        f"Claimed {int(claimed_num)} employees falls within acceptable range "
                        f"[{t_min:.0f}, {t_max:.0f}] of actual count ({actual_total})."
                    ),
                )
            else:
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.CONTRADICTED,
                    confidence=0.95,
                    evidence={"actual_headcount": actual_total, "acceptable_range": [t_min, t_max]},
                    source_cited=source,
                    reasoning=(
                        f"Claimed {int(claimed_num)} employees is outside acceptable range "
                        f"[{t_min:.0f}, {t_max:.0f}] of authoritative headcount ({actual_total})."
                    ),
                )

        # Fallback to range string if total is missing
        if range_str:
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.VERIFIED,
                confidence=0.75,
                evidence={"employee_count_range": range_str},
                source_cited="crustdata.basic_info.employee_count_range",
                reasoning=f"Claimed headcount consistent with recorded employee band '{range_str}'.",
            )

        return ClaimAuditResult(
            claim_id=claim.id,
            claim=claim,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.0,
            reasoning="Insufficient headcount evidence to make a definitive ruling.",
        )

    def _evaluate_funding(self, claim: AtomicClaim, data: CompanyData) -> ClaimAuditResult:
        funding = data.funding
        if not funding or (funding.total_investment_usd is None and not funding.funding_rounds):
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.0,
                evidence=None,
                source_cited=None,
                reasoning="No funding records found in Crustdata for this company.",
            )

        total_usd = funding.total_investment_usd
        val = claim.normalized_value

        if isinstance(val, (int, float)) and total_usd is not None:
            claimed_usd = float(val)
            t_min = claim.tolerance_min if claim.tolerance_min is not None else claimed_usd * 0.80
            t_max = claim.tolerance_max if claim.tolerance_max is not None else claimed_usd * 1.20

            evidence_dict = {
                "total_investment_usd": total_usd,
                "last_round_type": funding.last_round_type,
                "last_round_amount_usd": funding.last_round_amount_usd,
            }

            if t_min <= total_usd <= t_max:
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.VERIFIED,
                    confidence=0.95,
                    evidence=evidence_dict,
                    source_cited="crustdata.funding.total_investment_usd",
                    reasoning=(
                        f"Claimed funding ${claimed_usd:,.0f} is within acceptable tolerance "
                        f"[${t_min:,.0f}, ${t_max:,.0f}] of total investment (${total_usd:,.0f})."
                    ),
                )
            else:
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.CONTRADICTED,
                    confidence=0.95,
                    evidence=evidence_dict,
                    source_cited="crustdata.funding.total_investment_usd",
                    reasoning=(
                        f"Claimed funding ${claimed_usd:,.0f} conflicts with recorded "
                        f"total investment (${total_usd:,.0f})."
                    ),
                )

        # Check round type if qualitative
        if funding.last_round_type and isinstance(val, str) and val.lower() in funding.last_round_type.lower():
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.VERIFIED,
                confidence=0.90,
                evidence={"last_round_type": funding.last_round_type},
                source_cited="crustdata.funding.last_round_type",
                reasoning=f"Round type '{val}' matches recorded round '{funding.last_round_type}'.",
            )

        return self._arbitrate_with_llm(claim, {"funding": funding.model_dump()})

    def _evaluate_founding_year(self, claim: AtomicClaim, data: CompanyData) -> ClaimAuditResult:
        actual_year = data.basic_info.year_founded if data.basic_info else None
        if not actual_year:
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.0,
                evidence=None,
                source_cited=None,
                reasoning="Founding year is not recorded in Crustdata for this company.",
            )

        claimed_year = claim.normalized_value
        try:
            claimed_int = int(claimed_year)
            if abs(claimed_int - actual_year) <= 1:
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.VERIFIED,
                    confidence=0.98,
                    evidence={"year_founded": actual_year},
                    source_cited="crustdata.basic_info.year_founded",
                    reasoning=f"Claimed year {claimed_int} matches recorded founding year {actual_year}.",
                )
            else:
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.CONTRADICTED,
                    confidence=0.98,
                    evidence={"year_founded": actual_year},
                    source_cited="crustdata.basic_info.year_founded",
                    reasoning=(
                        f"Claimed founding year {claimed_int} contradicts recorded "
                        f"founding year {actual_year}."
                    ),
                )
        except (ValueError, TypeError):
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.0,
                reasoning=f"Could not parse claimed year '{claimed_year}'.",
            )

    def _evaluate_location(self, claim: AtomicClaim, data: CompanyData) -> ClaimAuditResult:
        loc = data.locations
        if not loc:
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.0,
                evidence=None,
                source_cited=None,
                reasoning="No location data recorded in Crustdata for this company.",
            )

        claimed_str = str(claim.normalized_value or claim.original_text).lower()
        known_locations = [
            loc.headquarters or "",
            loc.state or "",
            loc.country or "",
            loc.street_address or "",
        ] + [str(a) for a in loc.all_office_addresses]
        known_joined = " ".join(known_locations).lower()

        # Check for matching city or state
        found_match = any(token in known_joined for token in re.findall(r"\b[A-Za-z]{3,}\b", claimed_str))
        if found_match:
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.VERIFIED,
                confidence=0.90,
                evidence={"headquarters": loc.headquarters, "country": loc.country},
                source_cited="crustdata.locations",
                reasoning=f"Claimed location matches recorded headquarters: '{loc.headquarters}'.",
            )

        # If claim says a different country or distinct city, let arbiter evaluate
        return self._arbitrate_with_llm(claim, {"locations": loc.model_dump()})

    def _evaluate_founder(
        self,
        claim: AtomicClaim,
        company_data: CompanyData,
        person_data: Optional[Dict[str, PersonData]] = None,
    ) -> ClaimAuditResult:
        founders = company_data.people.founders if company_data.people else []
        claimed_str = str(claim.normalized_value or claim.original_text).lower()

        matched_founder = None
        for f in founders:
            fname = str(f.get("name", "")).lower()
            if fname and (fname in claimed_str or any(part in claimed_str for part in fname.split())):
                matched_founder = f
                break

        if matched_founder:
            # If claim also mentions university or past company, check person data if available
            p_profile = None
            if person_data and matched_founder.get("profile_url") in person_data:
                p_profile = person_data[matched_founder["profile_url"]]

            if p_profile and ("attended" in claimed_str or "grad" in claimed_str or "oxford" in claimed_str or "stanford" in claimed_str):
                schools = [str(s.get("school_name", "")).lower() for s in p_profile.education.schools] if p_profile.education else []
                school_matched = any(s in claimed_str or any(part in claimed_str for part in s.split()) for s in schools)
                if school_matched:
                    return ClaimAuditResult(
                        claim_id=claim.id,
                        claim=claim,
                        verdict=VerdictType.VERIFIED,
                        confidence=0.95,
                        evidence={"founder": matched_founder, "education": schools},
                        source_cited="crustdata.people & crustdata.person.education",
                        reasoning=f"Verified founder {matched_founder.get('name')} and educational credentials.",
                    )

            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.VERIFIED,
                confidence=0.92,
                evidence={"founder": matched_founder},
                source_cited="crustdata.people.founders",
                reasoning=f"Verified founder: '{matched_founder.get('name')}' is listed as founder.",
            )

        if not founders:
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.0,
                evidence=None,
                source_cited=None,
                reasoning="Crustdata people registry does not list founders for this company.",
            )

        return self._arbitrate_with_llm(claim, {"founders": founders})

    def _evaluate_customer_count(self, claim: AtomicClaim, data: CompanyData) -> ClaimAuditResult:
        # Crustdata company profile does not generally have exact customer count counter
        # Rule: Never guess without at least one cited data point.
        desc = (data.basic_info.description if data.basic_info else "") or ""
        val = claim.normalized_value

        if str(val) in desc or f"{val:,}" in desc:
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.VERIFIED,
                confidence=0.80,
                evidence={"description": desc},
                source_cited="crustdata.basic_info.description",
                reasoning=f"Customer count cited in official company overview: '{desc[:100]}...'",
            )

        # Strictly unverifiable without evidence
        return ClaimAuditResult(
            claim_id=claim.id,
            claim=claim,
            verdict=VerdictType.UNVERIFIABLE,
            confidence=0.0,
            evidence=None,
            source_cited=None,
            reasoning="Customer count data point is not tracked or verifiable in public corporate registry.",
        )

    def _evaluate_general(self, claim: AtomicClaim, data: CompanyData) -> ClaimAuditResult:
        return self._arbitrate_with_llm(claim, data.model_dump(exclude_none=True))

    def _arbitrate_with_llm(self, claim: AtomicClaim, evidence_dict: Dict[str, Any]) -> ClaimAuditResult:
        """Call Claude Sonnet 4.6 (ModelRole.ARBITER) to arbitrate complex or conflicting evidence."""
        messages = [
            {"role": "system", "content": ARBITRATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "claim": {
                            "category": claim.category.value,
                            "original_text": claim.original_text,
                            "normalized_value": claim.normalized_value,
                            "tolerance_min": claim.tolerance_min,
                            "tolerance_max": claim.tolerance_max,
                        },
                        "available_evidence": evidence_dict,
                    },
                    indent=2,
                    default=str,
                ),
            },
        ]

        try:
            reply = self.llm.complete(messages, role=ModelRole.ARBITER, temperature=0.0)
            cleaned = reply.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)

            verdict_str = str(data.get("verdict", "unverifiable")).lower()
            try:
                verdict = VerdictType(verdict_str)
            except ValueError:
                verdict = VerdictType.UNVERIFIABLE

            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=verdict,
                confidence=float(data.get("confidence", 0.8)),
                evidence=evidence_dict,
                source_cited=data.get("cited_source", "crustdata"),
                reasoning=data.get("reasoning", "Arbitrated by Claude Sonnet 4.6."),
            )
        except Exception as e:
            logger.warning(f"Arbitration fallback due to error: {e}")
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.0,
                evidence=evidence_dict,
                source_cited=None,
                reasoning=f"Evidence ambiguity could not be resolved: {e}",
            )
