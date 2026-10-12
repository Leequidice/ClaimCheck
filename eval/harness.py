"""Evaluation Harness for Claim Check.

Measures Precision, Recall, and False-Alarm Count across truthful profiles
and adversarial planted lies. Saves run records to enable cross-run comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from claim_check.auditor import ClaimAuditor
from claim_check.models import VerdictType
from eval.dataset import generate_eval_cases

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_evaluation(offline: bool = False, limit: int = 0) -> Dict[str, Any]:
    """Execute evaluation harness across train and held-out test cases."""
    cases = generate_eval_cases()
    if limit > 0:
        cases = cases[:limit]

    auditor = ClaimAuditor(offline=offline)
    now_str = datetime.now(timezone.utc).isoformat()

    # Track metrics separately for train and held-out (test)
    splits = {
        "train": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unverifiable": 0, "errors": 0},
        "test": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unverifiable": 0, "errors": 0},
    }

    enrichment_failures: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    case_results = []

    print("=" * 72)
    print(f"  RUNNING CLAIM CHECK EVALUATION ({len(cases)} cases, offline={offline})")
    print(f"  Split: 5 Companies Train (tune) | 5 Companies Held-out Test")
    print("=" * 72)

    for idx, case in enumerate(cases, 1):
        sp = case.split
        print(f"[{idx:2d}/{len(cases)}] [{sp.upper():5s}] {case.company:10s} ({'ADVERSARIAL' if case.is_adversarial else 'TRUTHFUL'})...", end=" ")
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
            "is_adversarial": case.is_adversarial,
            "claims": [],
        }

        for result in report.results:
            cat = result.claim.category.value
            verdict = result.verdict
            expected = case.expected_verdicts.get(cat)

            # If enrichment failed, exclude from precision/recall and count under errors
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
            if case.is_adversarial:
                if verdict == VerdictType.CONTRADICTED:
                    splits[sp]["tp"] += 1
                    is_correct = True
                elif verdict == VerdictType.VERIFIED:
                    splits[sp]["fn"] += 1
                    # Missed lie! Diagnose
                    diagnostics.append({
                        "type": "missed_lie",
                        "split": sp,
                        "company": case.company,
                        "claim_text": result.claim.original_text,
                        "ground_truth": case.metadata.get(f"true_{cat}"),
                        "crustdata_value": result.evidence,
                        "verdict": verdict.value,
                        "reasoning": result.reasoning,
                        "which_side_is_wrong": "auditor missed lie (marked verified)",
                    })
                else:
                    splits[sp]["unverifiable"] += 1
            else:
                if verdict == VerdictType.VERIFIED:
                    splits[sp]["tn"] += 1
                    is_correct = True
                elif verdict == VerdictType.CONTRADICTED:
                    splits[sp]["fp"] += 1
                    # False alarm! Diagnose
                    diagnostics.append({
                        "type": "false_alarm",
                        "split": sp,
                        "company": case.company,
                        "claim_text": result.claim.original_text,
                        "ground_truth": case.metadata.get(f"true_{cat}"),
                        "crustdata_value": result.evidence,
                        "verdict": verdict.value,
                        "reasoning": result.reasoning,
                        "which_side_is_wrong": "auditor contradicted truthful claim (false alarm)",
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
                "reasoning": result.reasoning,
            })

        case_results.append(case_record)

    def _calc_stats(s: Dict[str, int]) -> Dict[str, Any]:
        total_eval = s["tp"] + s["fp"] + s["fn"] + s["tn"] + s["unverifiable"]
        contra = s["tp"] + s["fp"]
        lies = s["tp"] + s["fn"]
        prec = (s["tp"] / contra) if contra > 0 else 0.0
        rec = (s["tp"] / lies) if lies > 0 else 0.0
        return {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "false_alarms": s["fp"],
            "true_positives": s["tp"],
            "false_negatives": s["fn"],
            "true_negatives": s["tn"],
            "unverifiable": s["unverifiable"],
            "enrichment_errors": s["errors"],
            "total_claims": total_eval,
        }

    train_stats = _calc_stats(splits["train"])
    test_stats = _calc_stats(splits["test"])

    # Aggregate overall stats
    combined = {k: splits["train"][k] + splits["test"][k] for k in splits["train"]}
    overall_stats = _calc_stats(combined)

    eval_summary = {
        "timestamp": now_str,
        "offline": offline,
        "cases_count": len(cases),
        "overall_metrics": overall_stats,
        "train_metrics": train_stats,
        "held_out_test_metrics": test_stats,
        "enrichment_failures": enrichment_failures,
        "diagnostics": diagnostics,
        "llm_usage": auditor.llm.get_summary(),
        "cases": case_results,
    }

    # Save run record
    run_filename = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    run_path = RESULTS_DIR / run_filename
    run_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    print(f"\n[*] Eval results saved to: {run_path}")

    # Print summary
    print("\n" + "=" * 72)
    print("  EVALUATION RESULTS SUMMARY (TRAIN vs HELD-OUT)")
    print("=" * 72)
    print(f"  {'Split':<12} | {'Precision':<10} | {'Recall':<8} | {'False Alarms':<13} | {'Errors':<7}")
    print("-" * 72)
    print(f"  {'TRAIN (5)':<12} | {train_stats['precision']*100:>9.1f}% | {train_stats['recall']*100:>7.1f}% | {train_stats['false_alarms']:>13} | {train_stats['enrichment_errors']:>7}")
    print(f"  {'TEST (5)':<12} | {test_stats['precision']*100:>9.1f}% | {test_stats['recall']*100:>7.1f}% | {test_stats['false_alarms']:>13} | {test_stats['enrichment_errors']:>7}")
    print(f"  {'OVERALL':<12} | {overall_stats['precision']*100:>9.1f}% | {overall_stats['recall']*100:>7.1f}% | {overall_stats['false_alarms']:>13} | {overall_stats['enrichment_errors']:>7}")
    print("-" * 72)
    print(f"  Enrichment Failures:  {len(enrichment_failures)}")
    for ef in enrichment_failures:
        print(f"    - {ef['company']} ({ef['domain']}): {ef['reason']}")
    print(f"  Diagnostic Issues:    {len(diagnostics)}")
    for d in diagnostics:
        print(f"    - [{d['type'].upper()}] {d['company']}: Claim \"{d['claim_text']}\" -> Verdict: {d['verdict']} ({d['which_side_is_wrong']})")
    print("-" * 72)
    llm_stats = eval_summary["llm_usage"]
    print(f"  Run LLM Spend:        ${llm_stats.get('run_cost_usd', 0.0):.6f}")
    print(f"  Cumulative Spent:     ${llm_stats.get('cumulative_cost_usd', 0.0):.4f} / $20.00")
    print("=" * 72)

    _print_historical_runs()
    return eval_summary


def _print_historical_runs():
    """Display comparison table across historical eval runs."""
    run_files = sorted(list(RESULTS_DIR.glob("eval_*.json")), reverse=True)
    if not run_files:
        return

    print("\nHISTORICAL RUN COMPARISON:")
    print("-" * 72)
    print(f"{'Run File':<28} | {'Mode':<7} | {'Precision':<9} | {'Recall':<7} | {'False Alarms':<12} | {'Cost USD':<8}")
    print("-" * 72)

    for rf in run_files[:5]:
        try:
            d = json.loads(rf.read_text(encoding="utf-8-sig"))
            m = d.get("metrics", {})
            cost = d.get("llm_usage", {}).get("run_cost_usd", 0.0)
            mode = "offline" if d.get("offline") else "live"
            print(
                f"{rf.name:<28} | {mode:<7} | {m.get('precision', 0)*100:>8.1f}% | "
                f"{m.get('recall', 0)*100:>6.1f}% | {m.get('false_alarm_count', 0):>12} | "
                f"${cost:>7.4f}"
            )
        except Exception:
            continue
    print("-" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run with local recorded fixtures")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of eval cases to run")
    args = parser.parse_args()
    run_evaluation(offline=args.offline, limit=args.limit)
