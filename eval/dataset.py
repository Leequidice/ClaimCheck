"""Evaluation dataset loader and programmatic planted lie generator.

Strictly sourced from eval/companies.csv (fields: company, domain, founded_year, round,
round_amount_usd, round_date, source_url).
All planted lies are generated programmatically across two difficulty tiers:
  - Easy tier:
      * founded_year shifted 3 to 5 years
      * round_amount_usd multiplied by 2.5
      * distant round type (e.g. Series A -> Series D, Seed -> Series B)
      * wrong date by a full year (+1 or -1 year)
  - Hard tier:
      * round_amount_usd multiplied by 1.1x to 1.3x (1.20x–1.25x, strictly beyond +-10% tolerance)
      * round date shifted by 2 to 4 months (+-3 months, strictly beyond +-1 month tolerance)
      * founded_year shifted by exactly 1 year (strictly beyond exact match)
      * plausible adjacent round type (e.g. Series B -> Series C, Series C -> Series D)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "eval" / "companies.csv"

# Distant round substitutions for easy tier
ROUND_SUBSTITUTIONS = {
    "seed": "Series B",
    "series a": "Series D",
    "series b": "Series F",
    "series c": "Series A",
    "series d": "Series A",
    "series e": "Series B",
    "series f": "Series C",
    "series i": "Series B",
}

# Plausible adjacent round substitutions for hard tier
ADJACENT_ROUNDS = {
    "seed": "Series A",
    "series a": "Series B",
    "series b": "Series C",
    "series c": "Series D",
    "series d": "Series E",
    "series e": "Series F",
    "series f": "Series G",
    "series g": "Series H",
    "series h": "Series I",
    "series i": "Series J",
}


@dataclass
class SourcedCompany:
    company: str
    domain: str
    founded_year: Optional[int]
    round_type: str
    round_amount_usd: float
    round_date: str
    source_url: str


@dataclass
class EvalCase:
    case_id: str
    company: str
    domain: str
    pitch_text: str
    is_adversarial: bool
    tier: str  # 'truth', 'easy', 'hard'
    split: str  # 'train' or 'test' (held-out)
    expected_verdicts: Dict[str, str]  # claim_category -> 'verified' | 'contradicted'
    metadata: Dict[str, Any]


def load_sourced_companies() -> List[SourcedCompany]:
    """Load companies strictly from eval/companies.csv.
    
    Raises FileNotFoundError or ValueError if missing or invalid.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Required benchmark dataset {CSV_PATH} not found.")

    companies: List[SourcedCompany] = []
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cname = (row.get("company") or "").strip()
            dom = (row.get("domain") or "").strip()
            if not cname or not dom:
                continue

            raw_yr = (row.get("founded_year") or "").strip()
            yr = int(raw_yr) if raw_yr and raw_yr.isdigit() else None
            r_type = row["round"].strip()
            r_amt = float(row["round_amount_usd"])
            r_dt = row["round_date"].strip()
            s_url = (row.get("source_url") or "").strip()

            companies.append(
                SourcedCompany(
                    company=cname,
                    domain=dom,
                    founded_year=yr,
                    round_type=r_type,
                    round_amount_usd=r_amt,
                    round_date=r_dt,
                    source_url=s_url,
                )
            )

    if len(companies) < 10:
        raise ValueError(f"Expected at least 10 sourced companies in {CSV_PATH}, found {len(companies)}.")
    return companies


def _format_date_human(date_str: str) -> str:
    """Format YYYY-MM-DD or YYYY-MM to 'Month YYYY' (e.g. 'July 2022', 'June 2026')."""
    try:
        parts = date_str.strip().split("-")
        yr = parts[0]
        mo = parts[1] if len(parts) > 1 else "01"
        dt = datetime(int(yr), int(mo), 1)
        return dt.strftime("%B %Y")
    except Exception:
        return date_str


def _shift_date_by_year(date_str: str, years: int = 1) -> str:
    """Shift an ISO date string (YYYY-MM-DD or YYYY-MM) by N years."""
    try:
        parts = date_str.strip().split("-")
        yr = int(parts[0]) + years
        rest = "-".join(parts[1:])
        return f"{yr}-{rest}" if rest else f"{yr}"
    except Exception:
        return "2018-01-01"


def _shift_date_by_months(date_str: str, months: int = 3) -> str:
    """Shift an ISO date string (YYYY-MM-DD or YYYY-MM) by N months."""
    try:
        parts = date_str.strip().split("-")
        yr = int(parts[0])
        mo = int(parts[1]) if len(parts) > 1 else 1
        day = parts[2] if len(parts) > 2 else None

        total_months = yr * 12 + (mo - 1) + months
        new_yr = total_months // 12
        new_mo = (total_months % 12) + 1

        if day:
            safe_day = min(int(day), 28)
            return f"{new_yr:04d}-{new_mo:02d}-{safe_day:02d}"
        return f"{new_yr:04d}-{new_mo:02d}"
    except Exception:
        return date_str


def _format_usd_millions(amount: float) -> str:
    """Format float amount to readable string (e.g. $45M, $6.5B)."""
    if amount >= 1_000_000_000:
        return f"${amount/1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount/1_000_000:.0f}M"
    return f"${amount:,.0f}"


def generate_eval_cases(train_count: int = 5) -> List[EvalCase]:
    """Generate truthful, easy planted lie, and hard planted lie cases.
    
    Split: first `train_count` companies -> 'train' (tune), remainder -> 'test' (held-out).
    Each company produces:
      1. Truthful case (tier='truth')
      2. Easy planted lie case (tier='easy')
      3. Hard planted lie case (tier='hard')
    """
    sourced = load_sourced_companies()
    cases: List[EvalCase] = []

    for idx, sc in enumerate(sourced):
        split = "train" if idx < train_count else "test"
        date_human = _format_date_human(sc.round_date)
        amt_str = _format_usd_millions(sc.round_amount_usd)

        # -------------------------------------------------------------
        # 1. Truthful Case
        # -------------------------------------------------------------
        expected_truth = {"funding": "verified"}
        meta_truth = {
            "tier": "truth",
            "true_round": sc.round_type,
            "true_round_amount_usd": sc.round_amount_usd,
            "true_round_date": sc.round_date,
        }

        if sc.founded_year is not None:
            truth_pitch = (
                f"{sc.company} was founded in {sc.founded_year}. "
                f"The company secured a {amt_str} {sc.round_type} round in {date_human}."
            )
            expected_truth["founding_year"] = "verified"
            meta_truth["true_founded_year"] = sc.founded_year
        else:
            truth_pitch = (
                f"{sc.company} secured a {amt_str} {sc.round_type} round in {date_human}."
            )

        cases.append(
            EvalCase(
                case_id=f"{sc.domain}_truth",
                company=sc.company,
                domain=sc.domain,
                pitch_text=truth_pitch,
                is_adversarial=False,
                tier="truth",
                split=split,
                expected_verdicts=expected_truth,
                metadata=meta_truth,
            )
        )

        # -------------------------------------------------------------
        # 2. Easy Planted Lie Case
        # -------------------------------------------------------------
        # Round amount x2.5
        easy_amount = sc.round_amount_usd * 2.5
        easy_amt_str = _format_usd_millions(easy_amount)
        # Distant round type
        norm_r = sc.round_type.lower().strip()
        easy_round = ROUND_SUBSTITUTIONS.get(norm_r, "Series Z")
        # Wrong date by a year
        easy_iso_date = _shift_date_by_year(sc.round_date, years=1 if idx % 2 == 0 else -1)
        easy_date_human = _format_date_human(easy_iso_date)

        expected_easy = {"funding": "contradicted"}
        meta_easy = {
            "tier": "easy",
            "planted_round": easy_round,
            "planted_amount_usd": easy_amount,
            "planted_date": easy_iso_date,
            "true_round": sc.round_type,
            "true_amount_usd": sc.round_amount_usd,
            "true_date": sc.round_date,
        }

        if sc.founded_year is not None:
            # Year shifted 3 to 5 (alternating 4 or 5 years)
            easy_year = sc.founded_year - (4 if idx % 2 == 0 else 5)
            easy_pitch = (
                f"{sc.company} was founded in {easy_year}. "
                f"The company secured a {easy_amt_str} {easy_round} round in {easy_date_human}."
            )
            expected_easy["founding_year"] = "contradicted"
            meta_easy["planted_year"] = easy_year
            meta_easy["true_year"] = sc.founded_year
        else:
            easy_pitch = (
                f"{sc.company} secured a {easy_amt_str} {easy_round} round in {easy_date_human}."
            )

        cases.append(
            EvalCase(
                case_id=f"{sc.domain}_planted_easy",
                company=sc.company,
                domain=sc.domain,
                pitch_text=easy_pitch,
                is_adversarial=True,
                tier="easy",
                split=split,
                expected_verdicts=expected_easy,
                metadata=meta_easy,
            )
        )

        # -------------------------------------------------------------
        # 3. Hard Planted Lie Case
        # -------------------------------------------------------------
        # Amount multiplied by 1.1x to 1.3x (alternating 1.25x and 1.20x)
        hard_factor = 1.25 if idx % 2 == 0 else 1.20
        hard_amount = sc.round_amount_usd * hard_factor
        hard_amt_str = _format_usd_millions(hard_amount)
        # Plausible adjacent round type (e.g. Series B -> Series C, Series C -> Series D)
        hard_round = ADJACENT_ROUNDS.get(norm_r, "Series B")
        # Date shifted 2 to 4 months (alternating +3 months and -3 months)
        shift_mo = 3 if idx % 2 == 0 else -3
        hard_iso_date = _shift_date_by_months(sc.round_date, months=shift_mo)
        hard_date_human = _format_date_human(hard_iso_date)

        expected_hard = {"funding": "contradicted"}
        meta_hard = {
            "tier": "hard",
            "planted_round": hard_round,
            "planted_amount_usd": hard_amount,
            "planted_date": hard_iso_date,
            "true_round": sc.round_type,
            "true_amount_usd": sc.round_amount_usd,
            "true_date": sc.round_date,
        }

        if sc.founded_year is not None:
            # Founded year shifted by exactly 1 year (+1 if even idx, -1 if odd)
            hard_year = sc.founded_year + (1 if idx % 2 == 0 else -1)
            hard_pitch = (
                f"{sc.company} was founded in {hard_year}. "
                f"The company secured a {hard_amt_str} {hard_round} round in {hard_date_human}."
            )
            expected_hard["founding_year"] = "contradicted"
            meta_hard["planted_year"] = hard_year
            meta_hard["true_year"] = sc.founded_year
        else:
            hard_pitch = (
                f"{sc.company} secured a {hard_amt_str} {hard_round} round in {hard_date_human}."
            )

        cases.append(
            EvalCase(
                case_id=f"{sc.domain}_planted_hard",
                company=sc.company,
                domain=sc.domain,
                pitch_text=hard_pitch,
                is_adversarial=True,
                tier="hard",
                split=split,
                expected_verdicts=expected_hard,
                metadata=meta_hard,
            )
        )

    return cases
