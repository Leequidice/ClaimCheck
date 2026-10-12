"""Atomic Claim Extractor.

Extracts atomic, checkable claims (headcount, funding, founding year, location,
founder backgrounds, customer counts) from pitch text or company pages.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from claim_check.config import ModelRole
from claim_check.llm import LLMClient
from claim_check.models import AtomicClaim, ClaimCategory

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert fact-checking auditor.
Your job is to read pitch decks, company bios, or website copy and extract every atomic, checkable claim.

Categories to extract:
1. headcount: employee count, team size (e.g. 'team of 40', 'over 300 employees')
2. funding: total capital raised, latest round amount, round series (e.g. '$140M raised', 'Series C of $45M')
3. founding_year: year established or founded (e.g. 'Founded in 2017')
4. location: headquarters city, state, country, or primary offices (e.g. 'HQ in San Francisco')
5. founder_background: founder names, past companies, universities attended (e.g. 'Founded by David Hsu, Oxford graduate')
6. customer_count: customer counts, number of enterprise logos (e.g. '5,000 customers', 'used by 10,000 teams')
7. other: any other specific, verifiable empirical claim (e.g. revenue, patents)

For each claim, return a JSON object with:
- id: unique string like 'claim_1', 'claim_2'
- category: one of ['headcount', 'funding', 'founding_year', 'location', 'founder_background', 'customer_count', 'other']
- original_text: the verbatim quote from the text
- normalized_value: clean representation (e.g. number for headcount/customers, USD amount as float for funding, 4-digit int for year, string for location/person)
- tolerance_min: lower bound of acceptable range (e.g. for 40 employees, ~30; for $140M, ~110M)
- tolerance_max: upper bound of acceptable range (e.g. for 40 employees, ~55; for $140M, ~170M)
- unit: string like 'employees', 'USD', 'year', 'customers', or null
- planned_source: which data source to query:
    * headcount -> 'crustdata.headcount'
    * funding -> 'crustdata.funding'
    * founding_year -> 'crustdata.basic_info.year_founded'
    * location -> 'crustdata.locations'
    * founder_background -> 'crustdata.people'
    * customer_count -> 'crustdata.company_data'
    * other -> 'crustdata.basic_info'

OUTPUT ONLY A VALID JSON ARRAY OF OBJECTS. Do not include introductory or explanatory text.
"""


def _compute_default_tolerances(category: ClaimCategory, val: Any) -> tuple[Optional[float], Optional[float]]:
    """Compute default reasonable ranges according to hackathon reasoning rules."""
    if isinstance(val, (int, float)):
        num = float(val)
        if category == ClaimCategory.HEADCOUNT:
            # E.g. '40 employees' vs 46 is verified (range ~30 to 55 is acceptable range, vs 120 is contradicted)
            margin = max(6.0, num * 0.25)
            return max(1.0, round(num - margin, 1)), round(num + margin, 1)
        elif category == ClaimCategory.FUNDING:
            # E.g. $140M vs $141M is verified (20% range)
            margin = num * 0.20
            return max(0.0, round(num - margin, 2)), round(num + margin, 2)
        elif category == ClaimCategory.CUSTOMER_COUNT:
            # Customer counts often rounded; allow 30% range
            margin = num * 0.30
            return max(1.0, round(num - margin, 1)), round(num + margin, 1)
        elif category == ClaimCategory.FOUNDING_YEAR:
            # Exact year
            return float(val), float(val)
    return None, None


class ClaimExtractor:
    """Extracts structured checkable claims using Claude Haiku (or regex fallback)."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def extract_claims(self, text: str, company_name: Optional[str] = None) -> List[AtomicClaim]:
        """Extract atomic claims from text."""
        if not text or not text.strip():
            return []

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Company: {company_name or 'Unknown'}\n\nText to audit:\n\"\"\"\n{text}\n\"\"\"",
            },
        ]

        try:
            raw_output = self.llm.complete(messages, role=ModelRole.EXTRACT, temperature=0.0)
            claims = self._parse_json_claims(raw_output)
            if claims:
                return claims
        except Exception as e:
            logger.warning(f"LLM claim extraction encountered exception: {e}; attempting heuristic fallback")

        # Fallback to rule-based heuristics if LLM fails or is offline without cache
        return self._heuristic_extraction(text)

    def _parse_json_claims(self, raw_output: str) -> List[AtomicClaim]:
        """Extract JSON array from LLM reply and parse into AtomicClaim models."""
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        # Look for JSON array substring if model wrapped it in prose
        json_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
        raw_to_parse = json_match.group(0) if json_match else cleaned

        data = json.loads(raw_to_parse)
        if not isinstance(data, list):
            return []

        results: List[AtomicClaim] = []
        for idx, item in enumerate(data, 1):
            claim_id = item.get("id") or f"claim_{idx}"
            category_str = str(item.get("category", "other")).lower()
            try:
                category = ClaimCategory(category_str)
            except ValueError:
                category = ClaimCategory.OTHER

            norm_val = item.get("normalized_value")
            t_min = item.get("tolerance_min")
            t_max = item.get("tolerance_max")

            # Ensure default tolerances if missing
            if (t_min is None or t_max is None) and isinstance(norm_val, (int, float)):
                d_min, d_max = _compute_default_tolerances(category, norm_val)
                t_min = t_min if t_min is not None else d_min
                t_max = t_max if t_max is not None else d_max

            claim = AtomicClaim(
                id=claim_id,
                category=category,
                original_text=item.get("original_text", ""),
                normalized_value=norm_val,
                tolerance_min=t_min,
                tolerance_max=t_max,
                unit=item.get("unit"),
                planned_source=item.get("planned_source", "crustdata"),
                context=item.get("context"),
            )
            results.append(claim)

        return results

    def _heuristic_extraction(self, text: str) -> List[AtomicClaim]:
        """Deterministic regex heuristics for core categories."""
        claims: List[AtomicClaim] = []
        idx = 1

        # 1. Founding Year
        match = re.search(r"\b(?:founded|established|started)\s+(?:in|circa)?\s*(\d{4})\b", text, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            claims.append(
                AtomicClaim(
                    id=f"claim_{idx}",
                    category=ClaimCategory.FOUNDING_YEAR,
                    original_text=match.group(0),
                    normalized_value=year,
                    tolerance_min=float(year),
                    tolerance_max=float(year),
                    unit="year",
                    planned_source="crustdata.basic_info.year_founded",
                )
            )
            idx += 1

        # 2. Headcount
        match = re.search(r"\b(?:team of|over|approximately|around)?\s*(\d+)\s+(?:employees|team members|people|staff)\b", text, re.IGNORECASE)
        if match:
            count = int(match.group(1))
            t_min, t_max = _compute_default_tolerances(ClaimCategory.HEADCOUNT, count)
            claims.append(
                AtomicClaim(
                    id=f"claim_{idx}",
                    category=ClaimCategory.HEADCOUNT,
                    original_text=match.group(0),
                    normalized_value=count,
                    tolerance_min=t_min,
                    tolerance_max=t_max,
                    unit="employees",
                    planned_source="crustdata.headcount",
                )
            )
            idx += 1

        # 3. Funding
        match = re.search(r"\$(\d+(?:\.\d+)?)\s*(M|million|B|billion|K)?\s*(?:raised|in funding|funding|seed|series [a-z])\b", text, re.IGNORECASE)
        if match:
            num = float(match.group(1))
            mult = match.group(2) or ""
            if mult.lower().startswith("b"):
                usd = num * 1_000_000_000
            elif mult.lower().startswith("m"):
                usd = num * 1_000_000
            elif mult.lower().startswith("k"):
                usd = num * 1_000
            else:
                usd = num
            t_min, t_max = _compute_default_tolerances(ClaimCategory.FUNDING, usd)
            claims.append(
                AtomicClaim(
                    id=f"claim_{idx}",
                    category=ClaimCategory.FUNDING,
                    original_text=match.group(0),
                    normalized_value=usd,
                    tolerance_min=t_min,
                    tolerance_max=t_max,
                    unit="USD",
                    planned_source="crustdata.funding",
                )
            )
            idx += 1

        # 4. Location
        match = re.search(r"\b(?:headquartered|based|located)\s+in\s+([A-Za-z\s,]+?)(?:\.|;|,| and|$)", text, re.IGNORECASE)
        if match:
            loc = match.group(1).strip()
            claims.append(
                AtomicClaim(
                    id=f"claim_{idx}",
                    category=ClaimCategory.LOCATION,
                    original_text=match.group(0).strip(),
                    normalized_value=loc,
                    planned_source="crustdata.locations",
                )
            )
            idx += 1

        return claims
