"""Pydantic data models for Crustdata Company and Person Enrichment APIs.

All field names and structures strictly adhere to the official Crustdata OpenAPI
documentation (API version 2025-11-01).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =====================================================================
# Company Enrichment Models
# =====================================================================

class CompanyBasicInfo(BaseModel):
    name: Optional[str] = None
    primary_domain: Optional[str] = None
    all_domains: List[str] = Field(default_factory=list)
    website: Optional[str] = None
    professional_network_url: Optional[str] = None
    professional_network_id: Optional[str] = None
    profile_name: Optional[str] = None
    logo_permalink: Optional[str] = None
    year_founded: Optional[int] = None
    description: Optional[str] = None
    company_type: Optional[str] = None
    status: Optional[str] = None
    employee_count_range: Optional[str] = None
    markets: List[str] = Field(default_factory=list)


class CompanyHeadcount(BaseModel):
    total: Optional[int] = None
    by_role_absolute: Dict[str, Any] = Field(default_factory=dict)
    by_role_percent: Dict[str, Any] = Field(default_factory=dict)
    by_region_absolute: Dict[str, Any] = Field(default_factory=dict)
    growth_percent: Dict[str, Any] = Field(default_factory=dict)
    growth_absolute: Dict[str, Any] = Field(default_factory=dict)


class CompanyFunding(BaseModel):
    total_investment_usd: Optional[float] = None
    last_round_amount_usd: Optional[float] = None
    last_fundraise_date: Optional[str] = None
    last_round_type: Optional[str] = None
    investors: List[str] = Field(default_factory=list)
    funding_rounds: List[Dict[str, Any]] = Field(default_factory=list)
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    acquisitions: List[Dict[str, Any]] = Field(default_factory=list)


class CompanyLocations(BaseModel):
    country: Optional[str] = None
    state: Optional[str] = None
    headquarters: Optional[str] = None
    street_address: Optional[str] = None
    all_office_addresses: List[Any] = Field(default_factory=list)


class CompanyPeople(BaseModel):
    founders: List[Dict[str, Any]] = Field(default_factory=list)
    cxos: List[Dict[str, Any]] = Field(default_factory=list)
    decision_makers: List[Dict[str, Any]] = Field(default_factory=list)


class CompanyRevenue(BaseModel):
    estimated: Optional[Dict[str, Any]] = None
    public_markets: Optional[str] = None
    acquisition_status: Optional[str] = None


class CompanyData(BaseModel):
    crustdata_company_id: Optional[int] = None
    updated_at: Optional[str] = None
    indexed_at: Optional[str] = None
    basic_info: Optional[CompanyBasicInfo] = None
    headcount: Optional[CompanyHeadcount] = None
    funding: Optional[CompanyFunding] = None
    locations: Optional[CompanyLocations] = None
    people: Optional[CompanyPeople] = None
    revenue: Optional[CompanyRevenue] = None


class CompanyMatch(BaseModel):
    confidence_score: Optional[float] = None
    company_data: Optional[CompanyData] = None


class CompanyEnrichItem(BaseModel):
    matched_on: str
    match_type: str
    matches: List[CompanyMatch] = Field(default_factory=list)

    @property
    def primary_match(self) -> Optional[CompanyData]:
        if self.matches and self.matches[0].company_data:
            return self.matches[0].company_data
        return None


# =====================================================================
# Person Enrichment Models
# =====================================================================

class PersonBasicProfile(BaseModel):
    name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    headline: Optional[str] = None
    current_title: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[str] = None


class PersonExperience(BaseModel):
    employment_details: Optional[Dict[str, Any]] = None
    years_of_experience: Optional[str] = None
    years_of_experience_raw: Optional[float] = None


class PersonEducation(BaseModel):
    schools: List[Dict[str, Any]] = Field(default_factory=list)


class PersonData(BaseModel):
    crustdata_person_id: Optional[int] = None
    updated_at: Optional[str] = None
    basic_profile: Optional[PersonBasicProfile] = None
    experience: Optional[PersonExperience] = None
    education: Optional[PersonEducation] = None
    skills: Optional[Dict[str, Any]] = None


class PersonMatch(BaseModel):
    confidence_score: Optional[float] = None
    person_data: Optional[PersonData] = None


class PersonEnrichItem(BaseModel):
    matched_on: str
    match_type: str
    match_status: str  # 'matched', 'not_found', 'redacted'
    matches: List[PersonMatch] = Field(default_factory=list)

    @property
    def primary_match(self) -> Optional[PersonData]:
        if self.matches and self.matches[0].person_data:
            return self.matches[0].person_data
        return None
