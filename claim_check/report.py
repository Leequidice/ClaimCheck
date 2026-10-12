"""Report formatting for Claim Check audits."""

from __future__ import annotations

import json
from typing import Any, Dict
from claim_check.models import CompanyAuditReport, VerdictType


def format_terminal_report(report: CompanyAuditReport) -> str:
    """Format an audit report for clean terminal output."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"  CLAIM CHECK AUDIT REPORT: {report.company_name or report.target_identifier}")
    lines.append("=" * 72)
    lines.append(f"Target:       {report.target_identifier}")
    lines.append(f"Timestamp:    {report.run_timestamp}")
    lines.append(f"Total Claims: {report.claims_total}")
    lines.append(
        f"Verdicts:     VERIFIED: {report.verified_count} | "
        f"CONTRADICTED: {report.contradicted_count} | "
        f"UNVERIFIABLE: {report.unverifiable_count} | "
        f"ERRORS: {report.error_count}"
    )
    if report.enrichment_error:
        lines.append(f"Enrichment:   [FAILURE] {report.enrichment_error}")

    if report.neatlogs_trace_url:
        lines.append(f"Trace Link:   {report.neatlogs_trace_url}")

    lines.append("-" * 72)
    lines.append("DETAILED CLAIM VERDICTS:")
    lines.append("-" * 72)

    for idx, r in enumerate(report.results, 1):
        v = r.verdict.value.upper()
        conf = f"{int(r.confidence * 100)}%"
        color_tag = f"[{v}] ({conf})"

        lines.append(f"\nClaim #{idx} {color_tag}")
        lines.append(f"  Category:  {r.claim.category.value}")
        lines.append(f"  Quote:     \"{r.claim.original_text}\"")
        lines.append(f"  Target:    {r.claim.normalized_value}")
        if r.claim.tolerance_min is not None and r.claim.tolerance_max is not None:
            lines.append(f"  Tolerance: [{r.claim.tolerance_min} to {r.claim.tolerance_max}]")
        if r.source_cited:
            lines.append(f"  Source:    {r.source_cited}")
        if r.evidence:
            lines.append(f"  Evidence:  {json.dumps(r.evidence, default=str)}")
        lines.append(f"  Reasoning: {r.reasoning}")
        if r.second_pass_verified:
            lines.append(f"  Pass 2:    Passed independent counter-evidence audit")

    lines.append("\n" + "-" * 72)
    lines.append("RESOURCE USAGE & COSTS:")
    lines.append("-" * 72)
    if report.llm_usage:
        run_cost = report.llm_usage.get("run_cost_usd", 0.0)
        cum_cost = report.llm_usage.get("cumulative_cost_usd", 0.0)
        lines.append(f"  Run LLM Cost:        ${run_cost:.6f}")
        lines.append(f"  Cumulative Spent:    ${cum_cost:.4f} / $20.00 max")
        for role, stats in report.llm_usage.get("by_role", {}).items():
            if stats.get("calls", 0) > 0 or stats.get("cache_hits", 0) > 0:
                lines.append(
                    f"    - {role:8s}: {stats.get('calls', 0)} calls, "
                    f"{stats.get('cache_hits', 0)} cache hits, "
                    f"${stats.get('cost_usd', 0.0):.6f}"
                )
    lines.append("=" * 72)
    return "\n".join(lines)


def format_markdown_report(report: CompanyAuditReport) -> str:
    """Format an audit report as standard GitHub Flavored Markdown."""
    lines = []
    lines.append(f"# Claim Check Audit: {report.company_name or report.target_identifier}\n")
    lines.append(f"- **Target Domain:** `{report.target_identifier}`")
    lines.append(f"- **Timestamp:** `{report.run_timestamp}`")
    lines.append(
        f"- **Summary:** Verified: **{report.verified_count}** | "
        f"Contradicted: **{report.contradicted_count}** | "
        f"Unverifiable: **{report.unverifiable_count}** | "
        f"Errors: **{report.error_count}**"
    )
    if report.enrichment_error:
        lines.append(f"- **Enrichment Failure:** `{report.enrichment_error}`")
    if report.neatlogs_trace_url:
        lines.append(f"- **Neatlogs Trace:** [{report.neatlogs_trace_url}]({report.neatlogs_trace_url})")

    lines.append("\n## Claims Breakdown\n")
    lines.append("| # | Category | Claimed Text | Verdict | Confidence | Cited Source | Evidence & Reasoning |")
    lines.append("|---|---|---|---|---|---|---|")

    for idx, r in enumerate(report.results, 1):
        v = r.verdict.value.upper()
        badge = f"**{v}**"
        reason = r.reasoning.replace("|", "/")
        source = r.source_cited or "None"
        quote = r.claim.original_text.replace("|", "/")
        lines.append(
            f"| {idx} | `{r.claim.category.value}` | \"{quote}\" | {badge} | "
            f"{int(r.confidence * 100)}% | `{source}` | {reason} |"
        )

    return "\n".join(lines)
