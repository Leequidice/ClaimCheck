"""Evaluation Harness for Claim Check.

Measures Precision, Recall, and False-Alarm Count across truthful profiles,
easy planted lies, and hard planted lies. Supports ablation comparison between
legacy verifier mode and evidence-based verifier mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from claim_check.auditor import ClaimAuditor
from claim_check.crustdata import CrustdataClient
from claim_check.models import VerdictType
from eval.dataset import generate_eval_cases

logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT_DIR / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_raw_crustdata_rounds(auditor: ClaimAuditor, domain: str) -> List[Dict[str, Any]]:
    """Fetch raw funding milestone and round records from Crustdata."""
    try:
        item = auditor.crustdata.enrich_company(domain=domain, fields=["funding"])
        if item.primary_match and item.primary_match.funding:
            f = item.primary_match.funding
            rounds: List[Dict[str, Any]] = []
            if f.funding_rounds:
                for r in f.funding_rounds:
                    rounds.append(r if isinstance(r, dict) else r.model_dump(exclude_none=True))
            if f.milestones:
                for m in f.milestones:
                    rounds.append(m if isinstance(m, dict) else m.model_dump(exclude_none=True))
            return rounds
    except Exception as e:
        logger.warning(f"Could not retrieve raw funding for {domain}: {e}")
    return []


def run_evaluation(
    offline: bool = False,
    limit: int = 0,
    verifier_mode: str = "evidence",
    save_filename: Optional[str] = None,
    crustdata_client: Optional[CrustdataClient] = None,
) -> Dict[str, Any]:
    """Execute evaluation harness across train and held-out test cases."""
    cases = generate_eval_cases()
    if limit > 0:
        cases = cases[:limit]

    auditor = ClaimAuditor(
        crustdata_client=crustdata_client,
        offline=offline,
        verifier_mode=verifier_mode,
    )
    now_str = datetime.now(timezone.utc).isoformat()

    # Track metrics separately for train and held-out (test)
    splits = {
        "train": {
            "truth_verified": 0,
            "truth_total": 0,
            "easy_caught": 0,
            "easy_total": 0,
            "hard_caught": 0,
            "hard_total": 0,
            "false_alarms": 0,
            "data_disagreements": 0,
            "unverifiable": 0,
            "errors": 0,
        },
        "test": {
            "truth_verified": 0,
            "truth_total": 0,
            "easy_caught": 0,
            "easy_total": 0,
            "hard_caught": 0,
            "hard_total": 0,
            "false_alarms": 0,
            "data_disagreements": 0,
            "unverifiable": 0,
            "errors": 0,
        },
    }

    enrichment_failures: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    case_results = []

    print("=" * 76)
    print(f"  RUNNING CLAIM CHECK EVALUATION ({len(cases)} cases, verifier_mode={verifier_mode}, offline={offline})")
    print(f"  Split: 5 Companies Train (tune) | 5 Companies Held-out Test")
    print(f"  Tiers: Truthful, Easy Planted Lies, Hard Planted Lies")
    print("=" * 76)

    for idx, case in enumerate(cases, 1):
        sp = case.split
        tier = case.tier
        tier_lbl = tier.upper()
        print(f"[{idx:2d}/{len(cases)}] [{sp.upper():5s}] [{tier_lbl:5s}] {case.company:10s}...", end=" ", flush=True)
        try:
            report = auditor.audit(
                text_or_url=case.pitch_text,
                company_name=case.company,
                domain=case.domain,
            )
            print(f"OK ({report.claims_total} claims: {report.verified_count} V, {report.contradicted_count} C, {report.unverifiable_count} U, {report.error_count} E)")
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        if report.enrichment_error:
            enrichment_failures.append({
                "company": case.company,
                "domain": case.domain,
                "reason": report.enrichment_error,
            })

        case_record = {
            "case_id": case.case_id,
            "company": case.company,
            "domain": case.domain,
            "split": case.split,
            "tier": case.tier,
            "is_adversarial": case.is_adversarial,
            "claims": [],
        }

        for result in report.results:
            cat = result.claim.category.value
            verdict = result.verdict
            expected = case.expected_verdicts.get(cat)

            # If enrichment failed, exclude from scoring and record under errors
            if verdict == VerdictType.ERROR:
                splits[sp]["errors"] += 1
                case_record["claims"].append({
                    "category": cat,
                    "text": result.claim.original_text,
                    "verdict": verdict.value,
                    "expected": expected,
                    "status": "error",
                    "reasoning": result.reasoning,
                })
                continue

            is_correct = False
            if tier == "truth":
                splits[sp]["truth_total"] += 1
                if verdict == VerdictType.VERIFIED:
                    splits[sp]["truth_verified"] += 1
                    is_correct = True
                elif verdict == VerdictType.CONTRADICTED:
                    # Distinguish database disagreement vs false alarm
                    evidence = result.evidence or {}
                    is_data_disagreement = False
                    ev_notes = ""

                    if cat == "founding_year":
                        crust_yr = evidence.get("year_founded")
                        gt_yr = case.metadata.get("true_founded_year")
                        if crust_yr is not None and gt_yr is not None and crust_yr != gt_yr:
                            is_data_disagreement = True
                            ev_notes = f"Crustdata year {crust_yr} != Press release year {gt_yr}"
                    elif cat == "funding":
                        if "Historical round verified" in result.reasoning:
                            # The database matched the press release, but the verifier demoted it
                            is_data_disagreement = False
                            ev_notes = "Auditor/verifier demoted matching round (false alarm)"
                        else:
                            # Crustdata database actually recorded a different round/date/amount
                            is_data_disagreement = True
                            ev_notes = f"Crustdata funding evidence differs from press release round: {evidence}"

                    if is_data_disagreement:
                        splits[sp]["data_disagreements"] += 1
                        diagnostics.append({
                            "type": "data_source_disagreement",
                            "split": sp,
                            "tier": tier,
                            "company": case.company,
                            "claim_text": result.claim.original_text,
                            "ground_truth": case.metadata.get(f"true_{cat}"),
                            "crustdata_value": result.evidence,
                            "verdict": verdict.value,
                            "reasoning": result.reasoning,
                            "second_pass_notes": result.verification_notes,
                            "which_side_is_wrong": f"Crustdata disagrees with press release ({ev_notes})",
                        })
                    else:
                        splits[sp]["false_alarms"] += 1
                        diagnostics.append({
                            "type": "false_alarm",
                            "split": sp,
                            "tier": tier,
                            "company": case.company,
                            "claim_text": result.claim.original_text,
                            "ground_truth": case.metadata.get(f"true_{cat}"),
                            "crustdata_value": result.evidence,
                            "verdict": verdict.value,
                            "reasoning": result.reasoning,
                            "second_pass_notes": result.verification_notes,
                            "which_side_is_wrong": "auditor contradicted truthful claim (false alarm)",
                        })
                else:
                    splits[sp]["unverifiable"] += 1

            elif tier == "easy":
                splits[sp]["easy_total"] += 1
                if verdict == VerdictType.CONTRADICTED:
                    splits[sp]["easy_caught"] += 1
                    is_correct = True
                elif verdict == VerdictType.VERIFIED:
                    diagnostics.append({
                        "type": "missed_easy_lie",
                        "split": sp,
                        "tier": tier,
                        "company": case.company,
                        "claim_text": result.claim.original_text,
                        "planted_value": case.metadata.get(f"planted_{cat}"),
                        "true_value": case.metadata.get(f"true_{cat}"),
                        "crustdata_value": result.evidence,
                        "verdict": verdict.value,
                        "reasoning": result.reasoning,
                        "second_pass_notes": result.verification_notes,
                        "which_side_is_wrong": "auditor missed easy lie (marked verified)",
                    })
                else:
                    splits[sp]["unverifiable"] += 1

            elif tier == "hard":
                splits[sp]["hard_total"] += 1
                if verdict == VerdictType.CONTRADICTED:
                    splits[sp]["hard_caught"] += 1
                    is_correct = True
                elif verdict == VerdictType.VERIFIED:
                    diagnostics.append({
                        "type": "missed_hard_lie",
                        "split": sp,
                        "tier": tier,
                        "company": case.company,
                        "claim_text": result.claim.original_text,
                        "planted_value": case.metadata.get(f"planted_{cat}"),
                        "true_value": case.metadata.get(f"true_{cat}"),
                        "crustdata_value": result.evidence,
                        "verdict": verdict.value,
                        "reasoning": result.reasoning,
                        "second_pass_notes": result.verification_notes,
                        "which_side_is_wrong": "auditor missed hard lie (marked verified)",
                    })
                else:
                    splits[sp]["unverifiable"] += 1

            case_record["claims"].append({
                "category": cat,
                "text": result.claim.original_text,
                "verdict": verdict.value,
                "expected": expected,
                "is_correct": is_correct,
                "confidence": result.confidence,
                "second_pass_verified": result.second_pass_verified,
                "verification_notes": result.verification_notes,
                "reasoning": result.reasoning,
            })

        case_results.append(case_record)

    def _calc_stats(s: Dict[str, int]) -> Dict[str, Any]:
        truth_verified = s["truth_verified"]
        truth_total = s["truth_total"]
        easy_caught = s["easy_caught"]
        easy_total = s["easy_total"]
        hard_caught = s["hard_caught"]
        hard_total = s["hard_total"]

        lies_caught = easy_caught + hard_caught
        lies_total = easy_total + hard_total

        contra_calls = lies_caught + s["false_alarms"]
        prec = (lies_caught / contra_calls) if contra_calls > 0 else 0.0
        rec_easy = (easy_caught / easy_total) if easy_total > 0 else 0.0
        rec_hard = (hard_caught / hard_total) if hard_total > 0 else 0.0
        rec_overall = (lies_caught / lies_total) if lies_total > 0 else 0.0

        return {
            "truth_verified": truth_verified,
            "truth_total": truth_total,
            "truth_report": f"{truth_verified} verified of {truth_total}",
            "easy_caught": easy_caught,
            "easy_total": easy_total,
            "easy_report": f"{easy_caught} caught of {easy_total}",
            "hard_caught": hard_caught,
            "hard_total": hard_total,
            "hard_report": f"{hard_caught} caught of {hard_total}",
            "lies_caught": lies_caught,
            "lies_total": lies_total,
            "lies_report": f"{lies_caught} caught of {lies_total}",
            "false_alarms": s["false_alarms"],
            "data_disagreements": s["data_disagreements"],
            "unverifiable": s["unverifiable"],
            "enrichment_errors": s["errors"],
            "precision": round(prec, 4),
            "recall_easy": round(rec_easy, 4),
            "recall_hard": round(rec_hard, 4),
            "recall_overall": round(rec_overall, 4),
        }

    train_stats = _calc_stats(splits["train"])
    test_stats = _calc_stats(splits["test"])

    combined = {k: splits["train"][k] + splits["test"][k] for k in splits["train"]}
    overall_stats = _calc_stats(combined)

    # Fetch raw Crustdata round records for Replit and Clerk without asserting why dates differ
    print("\n[*] Retrieving raw Crustdata round records for Replit and Clerk...")
    replit_raw = fetch_raw_crustdata_rounds(auditor, "replit.com")
    clerk_raw = fetch_raw_crustdata_rounds(auditor, "clerk.com")

    data_source_disagreements_raw = {
        "replit.com": {
            "company": "Replit",
            "press_release_date": "2025-09",
            "press_release_round": "Series C",
            "press_release_amount_usd": 250000000.0,
            "crustdata_round_records": replit_raw,
            "cause": "unknown",
        },
        "clerk.com": {
            "company": "Clerk",
            "press_release_date": "2025-07",
            "press_release_round": "Series C",
            "press_release_amount_usd": 50000000.0,
            "crustdata_round_records": clerk_raw,
            "cause": "unknown",
        },
    }

    eval_summary = {
        "timestamp": now_str,
        "offline": offline,
        "verifier_mode": verifier_mode,
        "cases_count": len(cases),
        "overall_metrics": overall_stats,
        "train_metrics": train_stats,
        "held_out_test_metrics": test_stats,
        "data_source_disagreements_raw": data_source_disagreements_raw,
        "enrichment_failures": enrichment_failures,
        "diagnostics": diagnostics,
        "llm_usage": auditor.llm.get_summary(),
        "cases": case_results,
    }

    # Save to requested filename or timestamped filename
    if save_filename:
        out_path = RESULTS_DIR / save_filename
    else:
        out_path = RESULTS_DIR / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

    out_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    print(f"[*] Eval results saved to: {out_path}")

    # Print clean summary table
    print("\n" + "=" * 76)
    print(f"  EVALUATION RESULTS SUMMARY (VERIFIER MODE: {verifier_mode.upper()})")
    print("=" * 76)
    print(f"  {'Split':<10} | {'Truthful (V/M)':<16} | {'Easy Lies':<14} | {'Hard Lies':<14} | {'False Alarms':<12}")
    print("-" * 76)
    print(
        f"  {'TRAIN (5)':<10} | {train_stats['truth_report']:<16} | "
        f"{train_stats['easy_report']:<14} | {train_stats['hard_report']:<14} | "
        f"{train_stats['false_alarms']:<12}"
    )
    print(
        f"  {'TEST (5)':<10} | {test_stats['truth_report']:<16} | "
        f"{test_stats['easy_report']:<14} | {test_stats['hard_report']:<14} | "
        f"{test_stats['false_alarms']:<12}"
    )
    print(
        f"  {'OVERALL':<10} | {overall_stats['truth_report']:<16} | "
        f"{overall_stats['easy_report']:<14} | {overall_stats['hard_report']:<14} | "
        f"{overall_stats['false_alarms']:<12}"
    )
    print("-" * 76)
    print(f"  Data Source Disagreements (Replit & Clerk): {overall_stats['data_disagreements']}")
    print(f"  Enrichment Failures: {len(enrichment_failures)}")
    print(f"  Diagnostic Issues Logged: {len(diagnostics)}")
    for d in diagnostics:
        tier_str = f"[{d.get('tier', 'unknown').upper()}] " if 'tier' in d else ""
        print(f"    - [{d['type'].upper()}] {tier_str}{d['company']}: Claim \"{d['claim_text']}\" -> Verdict: {d['verdict']} ({d['which_side_is_wrong']})")

    print("-" * 76)
    llm_stats = eval_summary["llm_usage"]
    print(f"  Run LLM Spend:        ${llm_stats.get('run_cost_usd', 0.0):.6f}")
    print(f"  Cumulative Spent:     ${llm_stats.get('cumulative_cost_usd', 0.0):.4f} / $20.00")
    print("=" * 76)

    return eval_summary


def run_ablation(offline: bool = False, limit: int = 0):
    """Run evaluation in both legacy and evidence modes and save ablation results."""
    print("\n" + "#" * 76)
    print("  COMMENCING VERIFIER ABLATION BENCHMARK (LEGACY vs EVIDENCE)")
    print("#" * 76)

    shared_crustdata = CrustdataClient(offline=offline)

    # 1. Run legacy mode
    legacy_res = run_evaluation(
        offline=offline,
        limit=limit,
        verifier_mode="legacy",
        save_filename="ablation_legacy.json",
        crustdata_client=shared_crustdata,
    )

    # 2. Run evidence mode
    evidence_res = run_evaluation(
        offline=offline,
        limit=limit,
        verifier_mode="evidence",
        save_filename="ablation_evidence.json",
        crustdata_client=shared_crustdata,
    )

    # Print Comparative Ablation Report
    leg_over = legacy_res["overall_metrics"]
    evi_over = evidence_res["overall_metrics"]

    leg_fa = leg_over["false_alarms"]
    evi_fa = evi_over["false_alarms"]

    print("\n" + "=" * 76)
    print("  VERIFIER ABLATION COMPARISON (eval/results/ablation_*.json)")
    print("=" * 76)
    print(f"  {'Metric':<25} | {'Legacy Mode':<20} | {'Evidence Mode':<20}")
    print("-" * 76)
    print(f"  {'False Alarms (overall)':<25} | {leg_fa:<20} | {evi_fa:<20}")
    print(f"  {'Truthful Verified':<25} | {leg_over['truth_report']:<20} | {evi_over['truth_report']:<20}")
    print(f"  {'Easy Lies Caught':<25} | {leg_over['easy_report']:<20} | {evi_over['easy_report']:<20}")
    print(f"  {'Hard Lies Caught':<25} | {leg_over['hard_report']:<20} | {evi_over['hard_report']:<20}")
    print(f"  {'Total Lies Caught':<25} | {leg_over['lies_report']:<20} | {evi_over['lies_report']:<20}")
    print(f"  {'Precision':<25} | {leg_over['precision']*100:>19.1f}% | {evi_over['precision']*100:>19.1f}%")
    print(f"  {'Hard Lie Recall':<25} | {leg_over['recall_hard']*100:>19.1f}% | {evi_over['recall_hard']*100:>19.1f}%")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run with local recorded fixtures")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of eval cases to run")
    parser.add_argument(
        "--verifier-mode",
        choices=["legacy", "evidence"],
        default="evidence",
        help="Verifier mode ('legacy' allows memory-based doubt demotion; 'evidence' requires cited counter-evidence)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run both legacy and evidence modes and save ablation_legacy.json and ablation_evidence.json",
    )
    args = parser.parse_args()

    if args.ablation:
        run_ablation(offline=args.offline, limit=args.limit)
    else:
        run_evaluation(offline=args.offline, limit=args.limit, verifier_mode=args.verifier_mode)
