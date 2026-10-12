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
    """Execute evaluation harness across all test cases."""
    cases = generate_eval_cases()
    if limit > 0:
        cases = cases[:limit]

    auditor = ClaimAuditor(offline=offline)
    now_str = datetime.now(timezone.utc).isoformat()

    tp_lies = 0      # Planted lie correctly identified as CONTRADICTED
    fp_lies = 0      # Truthful claim falsely marked CONTRADICTED (False Alarm)
    fn_lies = 0      # Planted lie missed (marked VERIFIED)
    tn_truths = 0    # Truthful claim correctly marked VERIFIED
    unverifiable = 0 # Claims marked UNVERIFIABLE

    case_results = []

    print("=" * 72)
    print(f"  RUNNING CLAIM CHECK EVALUATION ({len(cases)} cases, offline={offline})")
    print("=" * 72)

    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] Auditing {case.company} ({'ADVERSARIAL' if case.is_adversarial else 'TRUTHFUL'})...", end=" ")
        try:
            report = auditor.audit(
                text_or_url=case.pitch_text,
                company_name=case.company,
                domain=case.domain,
            )
            print(f"OK ({report.claims_total} claims: {report.verified_count} V, {report.contradicted_count} C, {report.unverifiable_count} U)")
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        case_record = {
            "case_id": case.case_id,
            "company": case.company,
            "domain": case.domain,
            "is_adversarial": case.is_adversarial,
            "claims": [],
        }

        for result in report.results:
            cat = result.claim.category.value
            verdict = result.verdict
            expected = case.expected_verdicts.get(cat)

            is_correct = False
            if case.is_adversarial:
                if verdict == VerdictType.CONTRADICTED:
                    tp_lies += 1
                    is_correct = True
                elif verdict == VerdictType.VERIFIED:
                    fn_lies += 1
                else:
                    unverifiable += 1
            else:
                if verdict == VerdictType.VERIFIED:
                    tn_truths += 1
                    is_correct = True
                elif verdict == VerdictType.CONTRADICTED:
                    fp_lies += 1
                else:
                    unverifiable += 1

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

    # Compute metrics
    total_evaluated = tp_lies + fp_lies + fn_lies + tn_truths + unverifiable
    total_planted_lies = tp_lies + fn_lies
    total_contradicted_emitted = tp_lies + fp_lies

    precision = (tp_lies / total_contradicted_emitted) if total_contradicted_emitted > 0 else 0.0
    recall = (tp_lies / total_planted_lies) if total_planted_lies > 0 else 0.0
    false_alarm_count = fp_lies

    eval_summary = {
        "timestamp": now_str,
        "offline": offline,
        "cases_count": len(cases),
        "total_claims_evaluated": total_evaluated,
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "false_alarm_count": false_alarm_count,
            "true_positives_lies_caught": tp_lies,
            "false_negatives_lies_missed": fn_lies,
            "true_negatives_truths_verified": tn_truths,
            "unverifiable_count": unverifiable,
        },
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
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Precision (Lies Caught / Flagged): {precision * 100:.2f}%")
    print(f"  Recall (Lies Caught / Total Lies): {recall * 100:.2f}%")
    print(f"  False Alarms (Truths flagged C):   {false_alarm_count}")
    print(f"  Total Claims Evaluated:            {total_evaluated}")
    print(f"  Truths Verified:                   {tn_truths}")
    print(f"  Lies Caught (Contradicted):        {tp_lies}")
    print(f"  Lies Missed:                       {fn_lies}")
    print(f"  Unverifiable:                      {unverifiable}")
    print("-" * 72)
    llm_stats = eval_summary["llm_usage"]
    print(f"  Run LLM Spend:                     ${llm_stats.get('run_cost_usd', 0.0):.6f}")
    print(f"  Cumulative Total Spent:            ${llm_stats.get('cumulative_cost_usd', 0.0):.4f} / $20.00")
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
