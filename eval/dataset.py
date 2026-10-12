"""Evaluation dataset loader and programmatic planted lie generator.

Strictly sourced from eval/companies.csv (fields: company, domain, founded_year, round,
round_amount_usd, round_date, source_url).
All planted lies are generated programmatically:
  - founded_year shifted 3 to 5 years
  - round_amount_usd multiplied by 2.5
  - wrong round type (e.g. Series A -> Series D, Seed -> Series B)
  - wrong date by a year (e.g. +1 year or -1 year)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "eval" / "companies.csv"

# Alternate round type substitution map
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
        return f"2018-01-01"


def _format_usd_millions(amount: float) -> str:
    """Format float amount to readable string (e.g. $45M, $6.5B)."""
    if amount >= 1_000_000_000:
        return f"${amount/1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount/1_000_000:.0f}M"
    return f"${amount:,.0f}"


def generate_eval_cases(train_count: int = 5) -> List[EvalCase]:
    """Generate paired truthful and programmatic planted lie cases.
    
    Split: first `train_count` companies -> 'train' (tune), remainder -> 'test' (held-out).
    Programmatic planted lies:
      - year shifted 3 to 5 (if founded_year exists)
      - round amount x2.5
      - wrong round type
      - wrong date by a year
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
                split=split,
                expected_verdicts=expected_truth,
                metadata=meta_truth,
            )
        )

        # -------------------------------------------------------------
        # 2. Programmatic Planted Lie Case
        # -------------------------------------------------------------
        # Round amount x2.5
        fake_amount = sc.round_amount_usd * 2.5
        fake_amt_str = _format_usd_millions(fake_amount)
        # Wrong round type
        norm_r = sc.round_type.lower().strip()
        fake_round = ROUND_SUBSTITUTIONS.get(norm_r, "Series Z")
        # Wrong date by a year
        fake_iso_date = _shift_date_by_year(sc.round_date, years=1 if idx % 2 == 0 else -1)
        fake_date_human = _format_date_human(fake_iso_date)

        expected_lie = {"funding": "contradicted"}
        meta_lie = {
            "planted_round": fake_round,
            "planted_amount_usd": fake_amount,
            "planted_date": fake_iso_date,
            "true_round": sc.round_type,
            "true_amount_usd": sc.round_amount_usd,
            "true_date": sc.round_date,
        }

        if sc.founded_year is not None:
            # Year shifted 3 to 5 (alternating 4 or 5 years)
            fake_year = sc.founded_year - (4 if idx % 2 == 0 else 5)
            lie_pitch = (
                f"{sc.company} was founded in {fake_year}. "
                f"The company secured a {fake_amt_str} {fake_round} round in {fake_date_human}."
            )
            expected_lie["founding_year"] = "contradicted"
            meta_lie["planted_year"] = fake_year
            meta_lie["true_year"] = sc.founded_year
        else:
            lie_pitch = (
                f"{sc.company} secured a {fake_amt_str} {fake_round} round in {fake_date_human}."
            )

        cases.append(
            EvalCase(
                case_id=f"{sc.domain}_planted_lie",
                company=sc.company,
                domain=sc.domain,
                pitch_text=lie_pitch,
                is_adversarial=True,
                split=split,
                expected_verdicts=expected_lie,
                metadata=meta_lie,
            )
        )

    return cases
