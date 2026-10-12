"""Evidence Planner.

Plans which evidence fields and sources to query for each atomic claim.
"""

from __future__ import annotations

from typing import List, Set
from claim_check.models import AtomicClaim, ClaimCategory


class EvidencePlan:
    """Contains mapped fields and endpoints for a set of claims."""

    def __init__(self, claims: List[AtomicClaim]):
        self.claims = claims
        self.company_fields: Set[str] = set()
        self.needs_person_enrichment: bool = False
        self._analyze()

    def _analyze(self):
        # basic_info is always included
        self.company_fields.add("basic_info")

        for claim in self.claims:
            if claim.category == ClaimCategory.HEADCOUNT:
                self.company_fields.add("headcount")
            elif claim.category == ClaimCategory.FUNDING:
                self.company_fields.add("funding")
            elif claim.category == ClaimCategory.FOUNDING_YEAR:
                self.company_fields.add("basic_info")
            elif claim.category == ClaimCategory.LOCATION:
                self.company_fields.add("locations")
            elif claim.category == ClaimCategory.FOUNDER_BACKGROUND:
                self.company_fields.add("people")
                self.needs_person_enrichment = True
            elif claim.category == ClaimCategory.CUSTOMER_COUNT:
                self.company_fields.add("basic_info")
                self.company_fields.add("revenue")
            else:
                self.company_fields.add("basic_info")

    @property
    def fields_list(self) -> List[str]:
        return sorted(list(self.company_fields))
