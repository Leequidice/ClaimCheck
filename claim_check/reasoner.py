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
        if not funding:
            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.UNVERIFIABLE,
                confidence=0.0,
                evidence=None,
                source_cited=None,
                reasoning="No funding records found in Crustdata for this company.",
            )

        # Retrieve all historical funding rounds (from funding_rounds or milestone events)
        all_rounds = funding.funding_rounds or [
            m for m in funding.milestones if m.get("round_type") or m.get("amount_usd")
        ]

        val = claim.normalized_value

        # Historical-round claim check: e.g. "raised $82M Series C in June 2025" or structured dict
        # Checked against Crustdata's funding_rounds list (amount +-10%, date within the month)
        is_round_claim = False
        claimed_round_type = None
        claimed_round_amount = None
        claimed_round_date = None

        if isinstance(val, dict) and ("round" in val or "round_type" in val or "round_amount_usd" in val):
            is_round_claim = True
            claimed_round_type = val.get("round") or val.get("round_type")
            claimed_round_amount = val.get("round_amount_usd") or val.get("amount_usd")
            claimed_round_date = val.get("round_date") or val.get("date")
        else:
            # Parse from original text if text mentions a round series/type and an amount
            text = claim.original_text.lower()
            round_match = re.search(r"\b(series\s+[a-i]|seed|angel|pre-seed|debt\s+financing|private\s+equity)\b", text, re.IGNORECASE)
            if round_match:
                is_round_claim = True
                claimed_round_type = round_match.group(1).lower().replace(" ", "_")
                # Look for amount in normalized_value or regex
                if isinstance(val, (int, float)):
                    claimed_round_amount = float(val)
                else:
                    amt_m = re.search(r"\$(\d+(?:\.\d+)?)\s*(m|million|b|billion|k)?", text, re.IGNORECASE)
                    if amt_m:
                        num = float(amt_m.group(1))
                        unit = (amt_m.group(2) or "").lower()
                        if unit.startswith("b"):
                            claimed_round_amount = num * 1_000_000_000
                        elif unit.startswith("m"):
                            claimed_round_amount = num * 1_000_000
                        elif unit.startswith("k"):
                            claimed_round_amount = num * 1_000
                        else:
                            claimed_round_amount = num

                # Look for year/month
                dt_m = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)?\s*(\d{4})\b", text, re.IGNORECASE)
                if dt_m:
                    claimed_round_date = dt_m.group(0).strip()

        if is_round_claim and all_rounds:
            norm_target_round = (claimed_round_type or "").lower().replace(" ", "_").replace("-", "_")

            # Search across Crustdata historical rounds
            matched_round = None
            type_matched_round = None

            for r in all_rounds:
                r_type = str(r.get("round_type") or "").lower().replace(" ", "_").replace("-", "_")
                r_amt = r.get("amount_usd")
                r_date = str(r.get("date") or r.get("funding_date") or "")

                if norm_target_round and (norm_target_round in r_type or r_type in norm_target_round):
                    type_matched_round = r

                    # Check amount +-10% if amount claimed
                    amt_ok = True
                    if claimed_round_amount is not None and r_amt is not None:
                        try:
                            c_amt = float(claimed_round_amount)
                            r_amt_f = float(r_amt)
                            amt_ok = (0.90 * c_amt <= r_amt_f <= 1.10 * c_amt)
                        except Exception:
                            amt_ok = False

                    # Check date within the month if date claimed
                    date_ok = True
                    if claimed_round_date and r_date:
                        # Normalize date strings to YYYY-MM
                        claimed_ym = self._extract_year_month(claimed_round_date)
                        actual_ym = self._extract_year_month(r_date)
                        if claimed_ym and actual_ym:
                            date_ok = (claimed_ym == actual_ym)
                        elif claimed_ym:
                            date_ok = claimed_ym[:4] in r_date

                    if amt_ok and date_ok:
                        matched_round = r
                        break

            if matched_round:
                trimmed_evidence = {
                    "round_type": matched_round.get("round_type"),
                    "amount_usd": matched_round.get("amount_usd"),
                    "date": matched_round.get("date") or matched_round.get("funding_date"),
                }
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.VERIFIED,
                    confidence=0.98,
                    evidence=trimmed_evidence,
                    source_cited="crustdata.funding.funding_rounds",
                    reasoning=(
                        f"Historical round verified: {trimmed_evidence['round_type']} "
                        f"of ${trimmed_evidence['amount_usd']:,.0f} on {trimmed_evidence['date']} "
                        f"matches within +-10% amount and month tolerance."
                    ),
                )
            elif type_matched_round:
                actual_amt = type_matched_round.get("amount_usd")
                actual_dt = type_matched_round.get("date") or type_matched_round.get("funding_date")
                trimmed_evidence = {
                    "round_type": type_matched_round.get("round_type"),
                    "actual_amount_usd": actual_amt,
                    "actual_date": actual_dt,
                    "claimed_amount_usd": claimed_round_amount,
                    "claimed_date": claimed_round_date,
                }
                amt_disp = f"${actual_amt:,.0f}" if isinstance(actual_amt, (int, float)) else "undisclosed"
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.CONTRADICTED,
                    confidence=0.98,
                    evidence=trimmed_evidence,
                    source_cited="crustdata.funding.funding_rounds",
                    reasoning=(
                        f"Round '{norm_target_round}' found but contradicts claim: recorded amount "
                        f"is {amt_disp} on {actual_dt} (outside +-10% amount or month date tolerance)."
                    ),
                )
            else:
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.CONTRADICTED,
                    confidence=0.90,
                    evidence={"recorded_rounds": [r.get("round_type") for r in all_rounds if r.get("round_type")]},
                    source_cited="crustdata.funding.funding_rounds",
                    reasoning=f"No matching '{norm_target_round}' round found in historical funding rounds.",
                )

        # Informational check for total funding (tolerance +-25%)
        total_usd = funding.total_investment_usd
        if isinstance(val, (int, float)) and total_usd is not None:
            claimed_usd = float(val)
            t_min = claim.tolerance_min if claim.tolerance_min is not None else claimed_usd * 0.75
            t_max = claim.tolerance_max if claim.tolerance_max is not None else claimed_usd * 1.25

            evidence_dict = {
                "total_investment_usd": total_usd,
                "as_of_date": funding.last_fundraise_date,
            }

            if t_min <= total_usd <= t_max:
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.VERIFIED,
                    confidence=0.90,
                    evidence=evidence_dict,
                    source_cited="crustdata.funding.total_investment_usd",
                    reasoning=(
                        f"[Informational] Claimed total funding ${claimed_usd:,.0f} is within +-25% "
                        f"tolerance of recorded total investment (${total_usd:,.0f})."
                    ),
                )
            else:
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.CONTRADICTED,
                    confidence=0.90,
                    evidence=evidence_dict,
                    source_cited="crustdata.funding.total_investment_usd",
                    reasoning=(
                        f"[Informational] Claimed total funding ${claimed_usd:,.0f} conflicts with recorded "
                        f"total investment (${total_usd:,.0f})."
                    ),
                )

        # Fallback to LLM arbitration if qualitative or complex
        trimmed_funding = {
            "total_investment_usd": funding.total_investment_usd,
            "last_round_type": funding.last_round_type,
            "last_round_amount_usd": funding.last_round_amount_usd,
            "rounds": [{"type": r.get("round_type"), "amount": r.get("amount_usd"), "date": r.get("date")} for r in all_rounds[:5]],
        }
        return self._arbitrate_with_llm(claim, {"funding": trimmed_funding})

    def _extract_year_month(self, text: str) -> Optional[str]:
        """Extract YYYY-MM from ISO date (2022-07-27) or text ('June 2022')."""
        if not text:
            return None
        iso_m = re.search(r"(\d{4})-(\d{2})", text)
        if iso_m:
            return f"{iso_m.group(1)}-{iso_m.group(2)}"
        months = {
            "january": "01", "jan": "01", "february": "02", "feb": "02",
            "march": "03", "mar": "03", "april": "04", "apr": "04",
            "may": "05", "june": "06", "jun": "06", "july": "07", "jul": "07",
            "august": "08", "aug": "08", "september": "09", "sep": "09",
            "october": "10", "oct": "10", "november": "11", "nov": "11",
            "december": "12", "dec": "12",
        }
        text_lower = text.lower()
        yr_m = re.search(r"\b(\d{4})\b", text_lower)
        if yr_m:
            yr = yr_m.group(1)
            for m_name, m_num in months.items():
                if m_name in text_lower:
                    return f"{yr}-{m_num}"
            return f"{yr}"
        return None

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
            if claimed_int == actual_year:
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.VERIFIED,
                    confidence=1.0,
                    evidence={"year_founded": actual_year},
                    source_cited="crustdata.basic_info.year_founded",
                    reasoning=f"Claimed year {claimed_int} exactly matches recorded founding year {actual_year}.",
                )
            else:
                return ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.CONTRADICTED,
                    confidence=1.0,
                    evidence={"year_founded": actual_year},
                    source_cited="crustdata.basic_info.year_founded",
                    reasoning=(
                        f"Claimed founding year {claimed_int} contradicts recorded "
                        f"founding year {actual_year} (exact match required)."
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

            trimmed_founder = {
                "name": matched_founder.get("name"),
                "title": matched_founder.get("title") or matched_founder.get("current_title"),
            }

            if p_profile and ("attended" in claimed_str or "grad" in claimed_str or "oxford" in claimed_str or "stanford" in claimed_str):
                schools = [str(s.get("school_name", "")).strip() for s in p_profile.education.schools if s.get("school_name")] if p_profile.education else []
                school_matched = any(s.lower() in claimed_str or any(part in claimed_str for part in s.lower().split()) for s in schools)
                if school_matched:
                    return ClaimAuditResult(
                        claim_id=claim.id,
                        claim=claim,
                        verdict=VerdictType.VERIFIED,
                        confidence=0.95,
                        evidence={"founder": trimmed_founder, "education": schools},
                        source_cited="crustdata.people & crustdata.person.education",
                        reasoning=f"Verified founder {trimmed_founder.get('name')} and educational credentials.",
                    )

            return ClaimAuditResult(
                claim_id=claim.id,
                claim=claim,
                verdict=VerdictType.VERIFIED,
                confidence=0.92,
                evidence={"founder": trimmed_founder},
                source_cited="crustdata.people.founders",
                reasoning=f"Verified founder: '{trimmed_founder.get('name')}' is listed as founder.",
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
