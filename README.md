# Claim Check 🔍

An autonomous Python CLI agent that audits a company’s public claims against authoritative, real-time firmographic evidence from the **Crustdata API**.

Given a company URL, domain, or pasted pitch text, Claim Check extracts atomic, checkable claims (founding year, funding round type, amount, and date), plans and queries evidence from Crustdata, evaluates values using strict deterministic tolerance ranges, and runs an independent second-pass verification inspection before finalizing the verdict.

---

## Architecture

```
Pasted Pitch Text / Website URL
            │
            ▼
┌──────────────────────────┐
│     Claim Extractor      │  (claude-haiku)
│  Extracts atomic claims  │  Normalized entities: year, round, amount, date
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│     Evidence Planner     │  Maps required fields to Crustdata OpenAPI spec
│ (crustdata.funding, etc) │  Zero field-name hallucination
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│     Crustdata Client     │  Live POST /company/enrich with 429 exponential backoff,
│  (Session Cache & DoH)   │  in-memory session caching & DoH DNS resolution
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│   Range & Date Reasoner  │  Deterministic tolerance evaluation:
│  (±10% amount, ±1 month) │  - Founding year: exact match
└───────────┬──────────────┘  - Historical round: ±10% amount, ±1 month date
            │
            ▼
┌──────────────────────────┐
│      State Manager       │  Atomic per-claim JSON checkpointing (`state/runs/`)
│ (Unique Run-ID Recovery) │  Crash recovery and resume with sub-second isolation
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│   Independent Verifier   │  Second-pass inspection (claude-sonnet-4.6):
│ (Evidence Demotion Rule) │  - Legacy mode: memory-based doubt could demote
└───────────┬──────────────┘  - Evidence mode: demotes ONLY with counter-evidence
            │
            ▼
┌──────────────────────────┐
│   Report & Observability │  Formatted CLI/JSON reports + Neatlogs trace spans
└──────────────────────────┘
```

---

## Setup & Installation

### 1. Requirements
- Python 3.10+
- Git

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Configure Credentials
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Populate with your keys:
```env
# Crustdata API
CRUSTDATA_API_KEY=your_crustdata_api_key

# LiteLLM Proxy Endpoint (Required, no default)
LITELLM_API_KEY=your_litellm_proxy_key
LITELLM_BASE_URL=https://your-litellm-proxy.com/v1

# Role-Based Model Router
MODEL_EXTRACT=claude-haiku
MODEL_ARBITER=claude-sonnet-4.6
MODEL_VERIFY=claude-sonnet-4.6
ALLOW_OPUS=false

# Neatlogs Observability
NEATLOGS_API_KEY=your_neatlogs_project_key

# Optional DoH Fallback for Windows DNS resolution
ENABLE_DOH_FALLBACK=false
```

---

## Demo Commands

### 1. Audit Live Company Pitch Text
```bash
python -m claim_check.cli --text "Supabase was founded in 2020. The company secured a $500M Series F round in June 2026." --domain "supabase.com"
```

### 2. Audit Adversarial Lies (Offline Mode with Fixtures)
```bash
python -m claim_check.cli --text "Retool was founded in 2005 with 120 employees and raised $5M in seed funding." --domain "retool.com" --offline
```

### 3. Resume an Interrupted Run
```bash
python -m claim_check.cli --resume latest
```

### 4. Run the Full Ablation Evaluation Benchmark
```bash
python eval/harness.py --ablation
```

---

## Evaluation Benchmark & Results

The evaluation benchmark (`eval/harness.py`) tests the system on 10 companies sourced from independent press releases (`eval/companies.csv`), split 5/5 into **Train (tune)** and **Held-Out Test**:
- **Train split**: Supabase, Lovable, Replit, Vercel, PostHog.
- **Held-Out Test split**: Clerk, Linear, Railway, Resend, Cal.com.

Each company generates 3 paired cases:
1. **Truthful Profile**: Press-release-grounded statements (founding year + historical funding round).
2. **Easy Planted Lie**: Distant round substitution, date shifted by 1 year, amount multiplied by 2.5x, founding year shifted by 3–5 years.
3. **Hard Planted Lie**: Plausible adjacent round (e.g. Series B -> Series C), date shifted by 3 months (exceeding the ±1 month tolerance), amount multiplied by 1.2x–1.25x (exceeding the ±10% tolerance), founding year shifted by 1 year (exceeding exact match).

### Final Ablation Results Summary

*Sourced strictly from `eval/results/ablation_legacy.json` and `eval/results/ablation_evidence.json` (one run per mode):*

| Metric | Legacy Verifier Mode | Evidence Verifier Mode | Notes |
| :--- | :--- | :--- | :--- |
| **Truthful Claims Verified** | **15 verified of 19** | **17 verified of 19** | Replit & Clerk date disagreements in both modes |
| ↳ Train Split (5 companies) | 7 verified of 10 | 9 verified of 10 | Vercel and PostHog demoted in legacy mode |
| ↳ Held-Out Test Split (5 companies) | 8 verified of 9 | 8 verified of 9 | Clerk date differs from Crustdata |
| **Planted Lies Caught** | **38 caught of 38** | **38 caught of 38** | Total across easy and hard tiers |
| ↳ Easy-Tier Lies | 19 caught of 19 | 19 caught of 19 | 100% caught across both splits |
| ↳ Hard-Tier Lies | 19 caught of 19 | 19 caught of 19 | Generated outside checker tolerances |
| **Truthful Claims Flagged Contradicted (False Alarms)** | **2 flagged of 19** | **0 flagged of 19** | In this run: train split 2 vs 0; held-out 0 vs 0 |
| ↳ Train Split (5 companies) | 2 of 10 (Vercel, PostHog) | 0 of 10 | Demoted on memory-based doubt in legacy mode |
| ↳ Held-Out Test Split (5 companies) | 0 of 9 | 0 of 9 | 0 false alarms on held-out test in both modes |
| **Overall Precision** (excl. Replit/Clerk disagreements) | 95.0% (38 / 40) | 100.0% (38 / 38) | Defined as Lies Caught / (Lies Caught + False Alarms) |
| **Data-Source Date Disagreements** | 2 of 19 (Replit, Clerk) | 2 of 19 (Replit, Clerk) | Crustdata date disagrees with press release; cause unknown |

### Key Findings in This Run

1. **Truthful Claims Verified and False Alarms**:
   - In **evidence mode**, **17 of 19** truthful claims were verified (Train: 9 of 10, Held-Out Test: 8 of 9). The only 2 non-verified truthful claims were **Replit** and **Clerk**, where the Crustdata database record dates disagree with the press-release ground truth (cause unknown).
   - In **legacy mode**, **15 of 19** truthful claims were verified. The verifier demoted **Vercel** ($300M Series F, Sep 2025) and **PostHog** ($75M Series E, Sep 2025) to `contradicted` based on memory-based skepticism about future 2025 dates and funding cadence, producing 2 false alarms in the Train split (0 in held-out test).
   - In this run, evidence mode had 0 false alarms versus 2 in legacy mode. Note that this is a single run on 10 companies, and 2 false alarms (Vercel, PostHog) is a small sample.
2. **Planted Lies Caught**:
   - **38 of 38** planted lies were caught across both modes (19 easy of 19, 19 hard of 19).
3. **Data-Source Date Disagreements (Replit & Clerk)**:
   - For **Replit** (Press release: `2025-09`; Crustdata milestone: `2025-07-30`) and **Clerk** (Press release: `2025-07`; Crustdata milestone: `2025-10-15`), Crustdata records dates differing by 2 and 3 months respectively from the press releases. The cause of these date differences is marked as **unknown**, and they are reported separately as data-source disagreements rather than auditor errors.

---

## Limitations

- **Dataset Size**: The benchmark evaluates 10 companies (5 Train, 5 Held-out Test); 2 false alarms in the ablation is a small sample.
- **Ground Truth Sourcing**: Ground truth is sourced directly from press releases, which may record different announcement or effective dates than primary regulatory filings.
- **Hard-Tier Construction**: The hard-tier lies were generated outside the checker's tolerances by construction (amount x1.2–1.25 vs a ±10% bound, 3-month shift vs ±1 month), so 19 of 19 mostly confirms the tolerances are applied, not that subtle lies near the boundary are caught.
- **Run-ID Collision Check**: The earlier cross-company collision check only tests that claim text contains the company name or the word "company", so it is weak evidence; do not present it as proof. Earlier intermediate benchmark runs affected by an earlier run-ID collision bug are excluded from the reported figures.
- **Run Variance**: LLM outputs can vary between runs; the reported results are from one run per mode.

---

## Built With

- **[Neatlogs](https://neatlogs.com)**: Distributed tracing and observability across the audit pipeline. Emits structured spans (`company_audit_run`, `extract_claims`, `crustdata.enrich`, `verifier.independent_pass`). Neatlogs traces capture the side-by-side comparison between legacy demotions and evidence confirmations in this run:
  - *Vercel Legacy Demotion Trace:* [https://neatlogs.com/traces/b6d27753f38a6687e52f7b88c9f61632](https://neatlogs.com/traces/b6d27753f38a6687e52f7b88c9f61632)
  - *Vercel Evidence Confirmation Trace:* [https://neatlogs.com/traces/26c82e806766b88a44b7f7ad81004b8c](https://neatlogs.com/traces/26c82e806766b88a44b7f7ad81004b8c)
  - *PostHog Legacy Demotion Trace:* [https://neatlogs.com/traces/70d18172f9b89de1d50eb85bd60e33e5](https://neatlogs.com/traces/70d18172f9b89de1d50eb85bd60e33e5)
  - *PostHog Evidence Confirmation Trace:* [https://neatlogs.com/traces/ed439e200f70b76a8176af7a5680f763](https://neatlogs.com/traces/ed439e200f70b76a8176af7a5680f763)
- **[Entire](https://entire.dev)**: Checkpointed git commit tracking for AI-assisted development. Every code change and evaluation step was committed through Entire with verifiable session checkpoint hashes (`entire enable -y --agent antigravity`).
- **[cfo.ai](https://cfo.ai)**: Costs measured from the LLM spend log; Crustdata cost assumed; business model built separately in cfo.ai.
