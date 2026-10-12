"""Core Audit Engine for Claim Check.

Orchestrates the entire claim verification pipeline:
  (a) Extract atomic, checkable claims
  (b) Plan evidence queries
  (c) Fetch evidence from Crustdata (live or recorded fixtures)
  (d) Reason over ranges and arbitrate discrepancies (Claude Sonnet 4.6)
  (e) Persist incremental progress per claim for crash recovery
  (f) Second-pass independent verification
  (g) Emit audit report with neatlogs traceable spans
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import neatlogs
except ImportError:
    neatlogs = None

from claim_check import init_neatlogs
from claim_check.crustdata import CrustdataClient
from claim_check.extractor import ClaimExtractor
from claim_check.llm import LLMClient
from claim_check.models import (
    AtomicClaim,
    ClaimAuditResult,
    CompanyAuditReport,
    CompanyData,
    VerdictType,
)
from claim_check.planner import EvidencePlan
from claim_check.reasoner import ClaimReasoner
from claim_check.state import StateManager
from claim_check.verifier import IndependentVerifier

logger = logging.getLogger(__name__)


class ClaimAuditor:
    """End-to-end auditor for company claims."""

    def __init__(
        self,
        crustdata_client: Optional[CrustdataClient] = None,
        llm_client: Optional[LLMClient] = None,
        state_manager: Optional[StateManager] = None,
        offline: bool = False,
        allow_opus: bool = False,
        verifier_mode: str = "evidence",
    ):
        self.offline = offline
        self.allow_opus = allow_opus
        self.verifier_mode = verifier_mode
        self.llm = llm_client or LLMClient(offline=offline, allow_opus=allow_opus)
        self.crustdata = crustdata_client or CrustdataClient(offline=offline)
        self.state = state_manager or StateManager()
        self.extractor = ClaimExtractor(llm_client=self.llm)
        self.reasoner = ClaimReasoner(llm_client=self.llm)
        self.verifier = IndependentVerifier(llm_client=self.llm, verifier_mode=verifier_mode)

    def _extract_domain(self, target: str) -> str:
        """Normalize URL or string into a domain."""
        cleaned = re.sub(r"^https?://", "", target.strip().lower())
        cleaned = re.sub(r"^www\.", "", cleaned)
        domain = cleaned.split("/")[0].split("?")[0].strip()
        return domain

    def audit(
        self,
        text_or_url: str,
        company_name: Optional[str] = None,
        domain: Optional[str] = None,
        resume_run_id: Optional[str] = None,
    ) -> CompanyAuditReport:
        """Run complete audit on pitch text or company URL."""
        # Initialize neatlogs if configured
        init_neatlogs(workflow_name="claim-check-audit")

        # Resolve target domain
        target_domain = domain or self._extract_domain(text_or_url)
        cname = company_name or target_domain.split(".")[0].capitalize()

        trace_id = None
        trace_url = None

        # Execute within neatlogs trace span if available
        trace_context = None
        if neatlogs is not None:
            try:
                trace_context = neatlogs.trace("company_audit_run", kind="WORKFLOW")
            except Exception:
                trace_context = None

        if trace_context:
            with trace_context as root_span:
                try:
                    span_ctx = root_span.get_span_context()
                    trace_id = f"{span_ctx.trace_id:032x}"
                    trace_url = f"https://neatlogs.com/traces/{trace_id}"
                except Exception:
                    pass
                report = self._run_pipeline(
                    text_or_url=text_or_url,
                    company_name=cname,
                    domain=target_domain,
                    resume_run_id=resume_run_id,
                    trace_id=trace_id,
                    trace_url=trace_url,
                )
                if neatlogs:
                    try:
                        neatlogs.flush()
                    except Exception:
                        pass
                return report
        else:
            return self._run_pipeline(
                text_or_url=text_or_url,
                company_name=cname,
                domain=target_domain,
                resume_run_id=resume_run_id,
                trace_id=trace_id,
                trace_url=trace_url,
            )

    def _run_pipeline(
        self,
        text_or_url: str,
        company_name: str,
        domain: str,
        resume_run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        trace_url: Optional[str] = None,
    ) -> CompanyAuditReport:
        # Step 1: Extract claims
        claims: List[AtomicClaim] = []
        is_url = bool(re.match(r"^https?://", text_or_url.strip()))
        text_to_audit = text_or_url

        if is_url:
            # If target is a URL and not pitch text, we will audit company profile attributes
            text_to_audit = f"Auditing company website and public profile for {company_name} ({domain})"

        claims = self.extractor.extract_claims(text_to_audit, company_name=company_name)

        if neatlogs:
            try:
                neatlogs.log("Extracted {count} claims for {company}", count=len(claims), company=company_name)
            except Exception:
                pass

        # Step 2: State and recovery setup
        run_id = self.state.init_run(target=domain, claims=claims, run_id=resume_run_id)
        prior_state = self.state.load_run(run_id)
        completed_ids = set(prior_state.get("completed_claim_ids", []))
        results_by_id: Dict[str, ClaimAuditResult] = {
            r["claim_id"]: ClaimAuditResult.model_validate(r) for r in prior_state.get("results", [])
        }

        # Step 3: Evidence Planning & Fetching
        plan = EvidencePlan(claims)
        company_data: Optional[CompanyData] = None
        enrichment_error_msg: Optional[str] = None

        try:
            enrichment_item = self.crustdata.enrich_company(
                domain=domain,
                fields=plan.fields_list,
            )
            company_data = enrichment_item.primary_match
            if not company_data:
                enrichment_error_msg = f"No matched company record returned by Crustdata for {domain}."
        except Exception as exc:
            enrichment_error_msg = f"Crustdata API enrichment failed: {exc}"
            logger.warning(f"Crustdata enrichment failed for {domain}: {exc}")
            if neatlogs:
                try:
                    neatlogs.log("Crustdata enrichment error for {domain}: {err}", domain=domain, err=str(exc))
                except Exception:
                    pass

        # Step 4: Evaluate each claim with fault isolation
        for claim in claims:
            if claim.id in completed_ids:
                logger.debug(f"Resuming: claim {claim.id} already completed, skipping.")
                continue

            if enrichment_error_msg:
                # Rule: A failed enrichment must be recorded as error with reason and never show as OK/unverifiable
                result = ClaimAuditResult(
                    claim_id=claim.id,
                    claim=claim,
                    verdict=VerdictType.ERROR,
                    confidence=0.0,
                    evidence=None,
                    source_cited="crustdata",
                    reasoning=f"Enrichment failure: {enrichment_error_msg}",
                )
            else:
                try:
                    result = self.reasoner.evaluate_claim(claim, company_data=company_data)
                except Exception as exc:
                    logger.error(f"Error evaluating claim {claim.id}: {exc}")
                    # A single failed claim must not abort the run!
                    result = ClaimAuditResult(
                        claim_id=claim.id,
                        claim=claim,
                        verdict=VerdictType.ERROR,
                        confidence=0.0,
                        reasoning=f"Audit failed due to error: {exc}",
                    )

            # Save state immediately per claim
            self.state.record_claim_result(run_id, result)
            results_by_id[claim.id] = result

            if neatlogs:
                try:
                    neatlogs.log(
                        "Evaluated claim {claim_id}: {verdict}",
                        claim_id=claim.id,
                        verdict=result.verdict.value,
                        confidence=result.confidence,
                    )
                except Exception:
                    pass

        # Step 5: Second verification pass on verified claims (only if company_data is valid)
        all_results = list(results_by_id.values())
        finalized_results = self.verifier.verify_all(all_results, company_data=company_data)

        # Mark run completed
        self.state.mark_completed(run_id)

        # Calculate metrics
        v_count = sum(1 for r in finalized_results if r.verdict == VerdictType.VERIFIED)
        c_count = sum(1 for r in finalized_results if r.verdict == VerdictType.CONTRADICTED)
        u_count = sum(1 for r in finalized_results if r.verdict == VerdictType.UNVERIFIABLE)
        e_count = sum(1 for r in finalized_results if r.verdict == VerdictType.ERROR)

        report = CompanyAuditReport(
            company_name=company_name,
            target_identifier=domain,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            claims_total=len(finalized_results),
            verified_count=v_count,
            contradicted_count=c_count,
            unverifiable_count=u_count,
            error_count=e_count,
            enrichment_error=enrichment_error_msg,
            results=finalized_results,
            llm_usage=self.llm.get_summary(),
            neatlogs_trace_id=trace_id,
            neatlogs_trace_url=trace_url,
        )

        return report
