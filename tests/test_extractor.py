"""Tests for atomic claim extraction and tolerance calculation."""

from __future__ import annotations

import pytest

from claim_check.extractor import ClaimExtractor, _compute_default_tolerances
from claim_check.models import ClaimCategory


def test_tolerance_computation():
    # Headcount: 40 employees -> acceptable range around ~30 to 55
    t_min, t_max = _compute_default_tolerances(ClaimCategory.HEADCOUNT, 40)
    assert t_min <= 30.0
    assert t_max >= 50.0

    # Funding: 140M -> ~20% range
    t_min, t_max = _compute_default_tolerances(ClaimCategory.FUNDING, 140000000.0)
    assert t_min == 112000000.0
    assert t_max == 168000000.0

    # Founding year: exact
    t_min, t_max = _compute_default_tolerances(ClaimCategory.FOUNDING_YEAR, 2017)
    assert t_min == 2017.0
    assert t_max == 2017.0


def test_heuristic_extraction():
    extractor = ClaimExtractor()
    text = (
        "Acme Corp was founded in 2019 and is headquartered in New York, NY. "
        "We have a team of 45 employees and have raised $12M in funding."
    )
    claims = extractor._heuristic_extraction(text)
    categories = {c.category for c in claims}

    assert ClaimCategory.FOUNDING_YEAR in categories
    assert ClaimCategory.LOCATION in categories
    assert ClaimCategory.HEADCOUNT in categories
    assert ClaimCategory.FUNDING in categories

    # Verify headcount claim
    hc_claim = next(c for c in claims if c.category == ClaimCategory.HEADCOUNT)
    assert hc_claim.normalized_value == 45
    assert hc_claim.tolerance_min <= 35
    assert hc_claim.tolerance_max >= 55

    # Verify founding year claim
    yr_claim = next(c for c in claims if c.category == ClaimCategory.FOUNDING_YEAR)
    assert yr_claim.normalized_value == 2019


def test_empty_text_returns_empty():
    extractor = ClaimExtractor()
    assert extractor.extract_claims("") == []
    assert extractor.extract_claims("   \n\t  ") == []
