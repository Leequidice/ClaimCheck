"""Command Line Interface for Claim Check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from claim_check.auditor import ClaimAuditor
from claim_check.report import format_markdown_report, format_terminal_report


def main():
    parser = argparse.ArgumentParser(
        description="Claim Check: Audit public company claims against authoritative Crustdata evidence."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Company URL, domain, or pasted pitch text string.",
    )
    parser.add_argument(
        "--url",
        dest="url_arg",
        help="Explicit company URL or website.",
    )
    parser.add_argument(
        "--text",
        dest="text_arg",
        help="Pasted pitch or bio text to audit.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to text or markdown file containing pitch text.",
    )
    parser.add_argument(
        "--company",
        help="Explicit company name (e.g. 'Retool', 'Stripe').",
    )
    parser.add_argument(
        "--domain",
        help="Explicit company domain (e.g. 'retool.com', 'stripe.com').",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume an in-progress or interrupted run.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode using recorded fixtures.",
    )
    parser.add_argument(
        "--allow-opus",
        action="store_true",
        help="Permit using Claude Opus if requested in model router.",
    )
    parser.add_argument(
        "--verifier-mode",
        choices=["legacy", "evidence"],
        default="evidence",
        help="Verifier behavior mode ('legacy' allows memory-based doubt demotion; 'evidence' requires cited counter-evidence).",
    )
    parser.add_argument(
        "--format",
        choices=["terminal", "markdown", "json"],
        default="terminal",
        help="Output report format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write report output to specified file.",
    )

    args = parser.parse_args()

    # Determine input text
    input_text = ""
    if args.file and args.file.exists():
        input_text = args.file.read_text(encoding="utf-8-sig")
    elif args.text_arg:
        input_text = args.text_arg
    elif args.url_arg:
        input_text = args.url_arg
    elif args.target:
        input_text = args.target
    else:
        # Check stdin
        if not sys.stdin.isatty():
            input_text = sys.stdin.read()
        else:
            parser.print_help()
            sys.exit(1)

    auditor = ClaimAuditor(
        offline=args.offline,
        allow_opus=args.allow_opus,
        verifier_mode=args.verifier_mode,
    )

    resume_id = None
    if args.resume:
        if args.resume == "latest":
            dom = args.domain or auditor._extract_domain(input_text)
            resume_id = auditor.state.find_latest_run_for_target(dom)
            if resume_id:
                print(f"[*] Resuming latest run: {resume_id}")
        else:
            resume_id = args.resume

    report = auditor.audit(
        text_or_url=input_text,
        company_name=args.company,
        domain=args.domain,
        resume_run_id=resume_id,
    )

    # Format output
    if args.format == "json":
        formatted = report.model_dump_json(indent=2)
    elif args.format == "markdown":
        formatted = format_markdown_report(report)
    else:
        formatted = format_terminal_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted, encoding="utf-8")
        print(f"[*] Report saved to {args.output}")

    print(formatted)


if __name__ == "__main__":
    main()
