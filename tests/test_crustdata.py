"""Tests for Crustdata API client and Pydantic response models."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from claim_check.crustdata import CrustdataClient, CrustdataError, CrustdataRateLimitError
from claim_check.models import CompanyEnrichItem, PersonEnrichItem


def test_offline_company_fixture():
    client = CrustdataClient(offline=True)
    item = client.enrich_company(domain="retool.com")

    assert isinstance(item, CompanyEnrichItem)
    assert item.matched_on == "retool.com"
    assert len(item.matches) == 1

    company = item.primary_match
    assert company is not None
    assert company.basic_info.name == "Retool"
    assert company.basic_info.year_founded == 2017
    assert company.headcount.total == 350
    assert company.funding.total_investment_usd == 135000000.0
    assert company.funding.last_round_type == "Series C"
    assert company.locations.headquarters == "San Francisco, CA"
    assert len(company.people.founders) == 1
    assert company.people.founders[0]["name"] == "David Hsu"


def test_offline_person_fixture():
    client = CrustdataClient(offline=True)
    items = client.enrich_person("https://www.linkedin.com/in/dvdhsu/")

    assert len(items) == 1
    item = items[0]
    assert isinstance(item, PersonEnrichItem)
    assert item.match_status == "matched"
    person = item.primary_match
    assert person is not None
    assert person.basic_profile.name == "David Hsu"
    assert person.education.schools[0]["school_name"] == "University of Oxford"


def test_offline_nonexistent_fixture_returns_empty():
    client = CrustdataClient(offline=True)
    item = client.enrich_company(domain="nonexistent-startup-xyz.org")
    assert item.matched_on == "nonexistent-startup-xyz.org"
    assert item.matches == []
    assert item.primary_match is None


def test_headers_and_version():
    client = CrustdataClient(api_key="test-secret-key")
    headers = client._get_headers()
    assert headers["Authorization"] == "Bearer test-secret-key"
    assert headers["x-api-version"] == "2025-11-01"
    assert headers["Content-Type"] == "application/json"


@patch("requests.post")
def test_retry_on_429_with_backoff(mock_post):
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0.01"}

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = [
        {
            "matched_on": "example.com",
            "match_type": "domain",
            "matches": [],
        }
    ]

    mock_post.side_effect = [resp_429, resp_200]

    client = CrustdataClient(api_key="test-key", backoff_factor=0.01)
    result = client.enrich_company(domain="example.com")

    assert mock_post.call_count == 2
    assert result.matched_on == "example.com"


@patch("requests.post")
def test_rate_limit_exceeded_raises_error(mock_post):
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0.01"}
    mock_post.return_value = resp_429

    client = CrustdataClient(api_key="test-key", max_retries=2, backoff_factor=0.01)

    with pytest.raises(CrustdataRateLimitError):
        client.enrich_company(domain="example.com")

    assert mock_post.call_count == 3
