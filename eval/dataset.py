"""Evaluation dataset loader and synthetic lie generator."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "eval" / "companies.csv"
BENCHMARK_PATH = ROOT_DIR / "eval" / "benchmark_companies.json"

# Curated benchmark of 20 real companies with verified ground-truth values
DEFAULT_BENCHMARK = [
    {"company": "Retool", "domain": "retool.com", "true_headcount": 416, "true_funding_usd": 141000000, "true_founded_year": 2017, "source_url": "https://retool.com"},
    {"company": "Stripe", "domain": "stripe.com", "true_headcount": 17058, "true_funding_usd": 10440247725, "true_founded_year": 2010, "source_url": "https://stripe.com"},
    {"company": "Figma", "domain": "figma.com", "true_headcount": 3135, "true_funding_usd": 1448624736, "true_founded_year": 2012, "source_url": "https://figma.com"},
    {"company": "Notion", "domain": "notion.so", "true_headcount": 600, "true_funding_usd": 343000000, "true_founded_year": 2013, "source_url": "https://notion.so"},
    {"company": "Supabase", "domain": "supabase.com", "true_headcount": 110, "true_funding_usd": 116000000, "true_founded_year": 2020, "source_url": "https://supabase.com"},
    {"company": "Vercel", "domain": "vercel.com", "true_headcount": 550, "true_funding_usd": 563000000, "true_founded_year": 2015, "source_url": "https://vercel.com"},
    {"company": "Datadog", "domain": "datadoghq.com", "true_headcount": 5200, "true_funding_usd": 148000000, "true_founded_year": 2010, "source_url": "https://datadoghq.com"},
    {"company": "Snowflake", "domain": "snowflake.com", "true_headcount": 7000, "true_funding_usd": 2100000000, "true_founded_year": 2012, "source_url": "https://snowflake.com"},
    {"company": "Canva", "domain": "canva.com", "true_headcount": 4500, "true_funding_usd": 570000000, "true_founded_year": 2012, "source_url": "https://canva.com"},
    {"company": "Postman", "domain": "postman.com", "true_headcount": 600, "true_funding_usd": 433000000, "true_founded_year": 2014, "source_url": "https://postman.com"},
    {"company": "Airtable", "domain": "airtable.com", "true_headcount": 900, "true_funding_usd": 1400000000, "true_founded_year": 2012, "source_url": "https://airtable.com"},
    {"company": "GitLab", "domain": "gitlab.com", "true_headcount": 2100, "true_funding_usd": 413000000, "true_founded_year": 2011, "source_url": "https://gitlab.com"},
    {"company": "Sentry", "domain": "sentry.io", "true_headcount": 400, "true_funding_usd": 217000000, "true_founded_year": 2012, "source_url": "https://sentry.io"},
    {"company": "Temporal", "domain": "temporal.io", "true_headcount": 180, "true_funding_usd": 120000000, "true_founded_year": 2019, "source_url": "https://temporal.io"},
    {"company": "Linear", "domain": "linear.app", "true_headcount": 60, "true_funding_usd": 52000000, "true_founded_year": 2019, "source_url": "https://linear.app"},
    {"company": "Pinecone", "domain": "pinecone.io", "true_headcount": 120, "true_funding_usd": 138000000, "true_founded_year": 2019, "source_url": "https://pinecone.io"},
    {"company": "LangChain", "domain": "langchain.com", "true_headcount": 45, "true_funding_usd": 35000000, "true_founded_year": 2022, "source_url": "https://langchain.com"},
    {"company": "Anthropic", "domain": "anthropic.com", "true_headcount": 500, "true_funding_usd": 7600000000, "true_founded_year": 2021, "source_url": "https://anthropic.com"},
    {"company": "Scale AI", "domain": "scale.com", "true_headcount": 900, "true_funding_usd": 1600000000, "true_founded_year": 2016, "source_url": "https://scale.com"},
    {"company": "Brex", "domain": "brex.com", "true_headcount": 1100, "true_funding_usd": 1500000000, "true_founded_year": 2017, "source_url": "https://brex.com"},
]


@dataclass
class EvalCase:
    case_id: str
    company: str
    domain: str
    pitch_text: str
    is_adversarial: bool
    expected_verdicts: Dict[str, str]  # claim_category -> 'verified' | 'contradicted'


def load_raw_companies() -> List[Dict[str, Any]]:
    """Load companies from companies.csv if populated, or fallback to default benchmark."""
    companies = []
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("company") and row.get("domain"):
                    try:
                        hc = int(row["true_headcount"]) if row.get("true_headcount") else None
                        fnd = float(row["true_funding_usd"]) if row.get("true_funding_usd") else None
                        yr = int(row["true_founded_year"]) if row.get("true_founded_year") else None
                        companies.append({
                            "company": row["company"].strip(),
                            "domain": row["domain"].strip(),
                            "true_headcount": hc,
                            "true_funding_usd": fnd,
                            "true_founded_year": yr,
                            "source_url": row.get("source_url", "").strip(),
                        })
                    except Exception:
                        continue

    if not companies:
        companies = DEFAULT_BENCHMARK
    return companies


def generate_eval_cases() -> List[EvalCase]:
    """Generate pairs of truthful and altered (planted lies) eval test cases."""
    raw = load_raw_companies()
    cases: List[EvalCase] = []

    for item in raw:
        cname = item["company"]
        dom = item["domain"]
        hc = item["true_headcount"]
        fnd = item["true_funding_usd"]
        yr = item["true_founded_year"]

        # Case 1: Truthful claim text
        fnd_str = f"${fnd/1_000_000:.0f}M" if fnd and fnd >= 1_000_000 else f"${fnd:,.0f}"
        truth_text = (
            f"{cname} was founded in {yr}. "
            f"The company has raised approximately {fnd_str} in funding "
            f"and currently has around {hc} employees."
        )
        cases.append(
            EvalCase(
                case_id=f"{dom}_truth",
                company=cname,
                domain=dom,
                pitch_text=truth_text,
                is_adversarial=False,
                expected_verdicts={
                    "founding_year": "verified",
                    "funding": "verified",
                    "headcount": "verified",
                },
            )
        )

        # Case 2: Altered copy with planted lies (inflated headcount, fake funding, false founding year)
        fake_hc = (hc * 5) if hc else 2000
        fake_fnd = (fnd * 10) if fnd else 500000000
        fake_yr = (yr - 15) if yr else 1995
        fake_fnd_str = f"${fake_fnd/1_000_000:.0f}M" if fake_fnd >= 1_000_000 else f"${fake_fnd:,.0f}"

        lie_text = (
            f"{cname} was founded back in {fake_yr}. "
            f"We have secured over {fake_fnd_str} in private capital "
            f"and employ a massive workforce of {fake_hc} employees."
        )
        cases.append(
            EvalCase(
                case_id=f"{dom}_planted_lie",
                company=cname,
                domain=dom,
                pitch_text=lie_text,
                is_adversarial=True,
                expected_verdicts={
                    "founding_year": "contradicted",
                    "funding": "contradicted",
                    "headcount": "contradicted",
                },
            )
        )

    return cases
