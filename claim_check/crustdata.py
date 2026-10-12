"""Crustdata API client for Company and Person Enrichment.

Supports live API requests with exponential backoff retry, rate limit handling,
and an offline recorded-fixture mode for local testing and deterministic benchmarks.
All field names strictly match Crustdata API 2025-11-01 OpenAPI specifications.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from claim_check.config import ConfigurationError, get_crustdata_key
from claim_check.models import CompanyEnrichItem, PersonEnrichItem

logger = logging.getLogger(__name__)

DEFAULT_COMPANY_FIELDS = [
    "basic_info",
    "headcount",
    "funding",
    "locations",
    "people",
    "revenue",
]

DEFAULT_PERSON_FIELDS = [
    "basic_profile",
    "experience",
    "education",
    "skills",
]


class CrustdataError(Exception):
    """Base exception for Crustdata API errors."""
    pass


class CrustdataRateLimitError(CrustdataError):
    """Raised when rate limit (429) is exceeded after maximum retries."""
    pass


class CrustdataClient:
    """Client for Crustdata Company and Person Enrichment APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        offline: bool = False,
        fixture_dir: Optional[Path | str] = None,
        record_fixtures: bool = False,
        base_url: str = "https://api.crustdata.com",
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        timeout: float = 30.0,
    ):
        self.offline = offline
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.record_fixtures = record_fixtures

        if fixture_dir:
            self.fixture_dir = Path(fixture_dir)
        else:
            self.fixture_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

        if self.offline:
            self.api_key = None
        else:
            self.api_key = api_key if api_key else get_crustdata_key()

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise CrustdataError("CRUSTDATA_API_KEY is required for live network requests.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "x-api-version": "2025-11-01",
        }

    def _safe_filename(self, key: str) -> str:
        """Sanitize a key (domain or url) into a filesystem-safe filename."""
        cleaned = re.sub(r"^https?://", "", key.strip())
        cleaned = re.sub(r"^www\.", "", cleaned)
        cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", cleaned).strip("_")
        return f"{cleaned}.json"

    def _execute_with_retry(self, url: str, payload: Dict[str, Any]) -> Any:
        """Execute a POST request with exponential backoff on 429 and 5xx errors."""
        headers = self._get_headers()
        attempt = 0

        while attempt <= self.max_retries:
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    return response.json()

                if response.status_code == 404:
                    # Crustdata returns 404 or 200 with empty matches on not found
                    return []

                if response.status_code == 429:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise CrustdataRateLimitError(
                            "Crustdata rate limit (429) exceeded after max retries."
                        )
                    retry_after = response.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else self.backoff_factor * (2 ** (attempt - 1))
                    logger.warning(f"Crustdata 429 rate limit reached. Backing off for {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue

                if 500 <= response.status_code < 600:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise CrustdataError(
                            f"Crustdata server error {response.status_code} after max retries."
                        )
                    sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                    logger.warning(f"Crustdata server error {response.status_code}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue

                # Client error (400, 401, 403, etc.)
                error_msg = f"Crustdata API error {response.status_code}: {response.text}"
                raise CrustdataError(error_msg)

            except (requests.Timeout, requests.ConnectionError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise CrustdataError(f"Network error connecting to Crustdata: {type(exc).__name__}") from exc
                sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_time)

        raise CrustdataError("Maximum retries exhausted without response.")

    # =====================================================================
    # Company Enrichment
    # =====================================================================

    def enrich_company(
        self,
        domain: Optional[str] = None,
        name: Optional[str] = None,
        profile_url: Optional[str] = None,
        crustdata_id: Optional[int] = None,
        fields: Optional[List[str]] = None,
        exact_match: Optional[bool] = None,
    ) -> CompanyEnrichItem:
        """Enrich a company using domain, name, profile URL, or Crustdata ID.
        
        Exactly one identifier type must be supplied.
        """
        requested_fields = fields or DEFAULT_COMPANY_FIELDS
        payload: Dict[str, Any] = {"fields": requested_fields}
        if exact_match is not None:
            payload["exact_match"] = exact_match

        identifier_key: str
        match_type: str

        if domain:
            payload["domains"] = [domain]
            identifier_key = domain
            match_type = "domain"
        elif name:
            payload["names"] = [name]
            identifier_key = name
            match_type = "name"
        elif profile_url:
            payload["professional_network_profile_urls"] = [profile_url]
            identifier_key = profile_url
            match_type = "professional_network_profile_url"
        elif crustdata_id:
            payload["crustdata_company_ids"] = [crustdata_id]
            identifier_key = str(crustdata_id)
            match_type = "crustdata_company_id"
        else:
            raise ValueError("Must provide one of: domain, name, profile_url, crustdata_id")

        # Check offline fixture mode
        if self.offline:
            fixture_file = self.fixture_dir / "companies" / self._safe_filename(identifier_key)
            if fixture_file.exists():
                data = json.loads(fixture_file.read_text(encoding="utf-8-sig"))
                items = [CompanyEnrichItem.model_validate(item) for item in data]
                return items[0]
            # No fixture found: return empty match response
            return CompanyEnrichItem(matched_on=identifier_key, match_type=match_type, matches=[])

        # Live API call
        url = f"{self.base_url}/company/enrich"
        raw_response = self._execute_with_retry(url, payload)

        if not isinstance(raw_response, list):
            raise CrustdataError(f"Unexpected response format from company/enrich: {type(raw_response)}")

        if self.record_fixtures:
            fixture_file = self.fixture_dir / "companies" / self._safe_filename(identifier_key)
            fixture_file.parent.mkdir(parents=True, exist_ok=True)
            fixture_file.write_text(json.dumps(raw_response, indent=2), encoding="utf-8")

        items = [CompanyEnrichItem.model_validate(item) for item in raw_response]
        if items:
            return items[0]
        return CompanyEnrichItem(matched_on=identifier_key, match_type=match_type, matches=[])

    # =====================================================================
    # Person Enrichment
    # =====================================================================

    def enrich_person(
        self,
        profile_urls: List[str] | str,
        fields: Optional[List[str]] = None,
    ) -> List[PersonEnrichItem]:
        """Enrich person profiles using their professional network profile URLs."""
        urls = [profile_urls] if isinstance(profile_urls, str) else profile_urls
        if not urls:
            return []

        requested_fields = fields or DEFAULT_PERSON_FIELDS
        payload = {
            "professional_network_profile_urls": urls,
            "fields": requested_fields,
        }

        # Check offline fixture mode
        if self.offline:
            results: List[PersonEnrichItem] = []
            for url in urls:
                fixture_file = self.fixture_dir / "people" / self._safe_filename(url)
                if fixture_file.exists():
                    data = json.loads(fixture_file.read_text(encoding="utf-8-sig"))
                    items = [PersonEnrichItem.model_validate(item) for item in data]
                    results.extend(items)
                else:
                    results.append(
                        PersonEnrichItem(
                            matched_on=url,
                            match_type="professional_network_profile_url",
                            match_status="not_found",
                            matches=[],
                        )
                    )
            return results

        # Live API call
        url = f"{self.base_url}/person/enrich"
        raw_response = self._execute_with_retry(url, payload)

        if not isinstance(raw_response, list):
            raise CrustdataError(f"Unexpected response format from person/enrich: {type(raw_response)}")

        if self.record_fixtures:
            for url in urls:
                fixture_file = self.fixture_dir / "people" / self._safe_filename(url)
                fixture_file.parent.mkdir(parents=True, exist_ok=True)
                fixture_file.write_text(json.dumps(raw_response, indent=2), encoding="utf-8")

        return [PersonEnrichItem.model_validate(item) for item in raw_response]
