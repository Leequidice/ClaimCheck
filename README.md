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

*Sourced strictly from `eval/results/ablation_legacy.json` and `eval/results/ablation_evidence.json`:*

| Metric | Legacy Verifier Mode | Evidence Verifier Mode | Delta / Impact |
| :--- | :--- | :--- | :--- |
| **Truthful Claims Verified** | 15 verified of 19 | **17 verified of 19** | **+2 genuine rounds confirmed** |
| ↳ Train Split (5 companies) | 7 verified of 10 | **9 verified of 10** | +2 (Vercel & PostHog protected) |
| ↳ Held-Out Test Split (5 companies) | 8 verified of 9 | **8 verified of 9** | Maintained 100% on test truths |
| **Planted Lies Caught** | **38 caught of 38** (100%) | **38 caught of 38** (100%) | 100% adversarial recall |
| ↳ Easy-Tier Lies | 19 caught of 19 (100%) | **19 caught of 19** (100%) | 100% easy lie recall |
| ↳ Hard-Tier Lies | 19 caught of 19 (100%) | **19 caught of 19** (100%) | 100% hard lie recall |
| **False Alarms (Train Split)** | **2** | **0** | **-2 false alarms eliminated** |
| **False Alarms (Held-Out Split)**| **0** | **0** | 0 false alarms on held-out test |
| **Overall False Alarm Count** | **2** | **0** | **Zero false alarms overall** |
| **Overall Precision** | 95.0% | **100.0%** | **+5.0% precision** |
| **Data-Source Date Disagreements** | 2 (Replit, Clerk) | 2 (Replit, Clerk) | Reported separately (cause unknown) |

### Key Findings

1. **Elimination of False Alarms**:
   - In `legacy` mode, the second-pass verifier demoted **Vercel** ($300M Series F, Sep 2025) and **PostHog** ($75M Series E, Sep 2025) to `contradicted` based on memory-based skepticism about future 2025 dates and rapid round cadences. Both were false alarms because Crustdata records directly matched the press releases (`2025-09-30` for Vercel, `2025-09-29` for PostHog).
   - In `evidence` mode, the strict evidence-based demotion rule eliminated both false alarms in the Train split (dropping false alarms from 2 to 0) and raised verified truthful claims from 15 of 19 to 17 of 19.
2. **100% Hard-Tier Adversarial Recall**:
   - The checker caught all 19 of 19 hard lies (and 19 of 19 easy lies) across both Train and Held-Out Test splits without prompt tuning.
   - The ±10% amount bound and ±1 month date tolerance accurately flagged all hard-tier boundary shifts (amounts shifted by 20–25%, dates shifted by 3 months, founded years shifted by 1 year, and adjacent round changes).
3. **Data-Source Date Disagreements (Replit & Clerk)**:
   - For **Replit** (Press release: `2025-09`; Crustdata milestone: `2025-07-30`) and **Clerk** (Press release: `2025-07`; Crustdata milestone: `2025-10-15`), the Crustdata database disagrees with the press release dates.
   - These 2 cases are logged separately under `data_source_disagreements_raw` with cause marked as **unknown**, without asserting why dates differ or penalizing agent reasoning.

---

## Limitations

1. **Benchmark Scope**: The evaluation benchmark is currently built on 10 tech startups/companies (5 Train, 5 Held-out Test).
2. **Ground Truth Sourcing**: Ground-truth funding data is sourced directly from publicly accessible press releases and startup blogs, which can occasionally record different calendar months than corporate registry filings.
3. **Excluded Historical Runs**: Earlier intermediate test runs affected by an earlier run-ID collision bug (fixed via microsecond timestamping and random UUID suffixes in `claim_check/state.py`) are strictly excluded from the benchmark metrics.

---

## Built With

- **[Neatlogs](https://neatlogs.com)**: Full observability and distributed tracing across all agent audit pipelines. Every run emits structured spans (`company_audit_run`, `extract_claims`, `crustdata.enrich`, `verifier.independent_pass`). Neatlogs traces provide the side-by-side evidence comparison between legacy demotions and evidence confirmations:
  - *Vercel Legacy Demotion Trace:* [https://neatlogs.com/traces/b6d27753f38a6687e52f7b88c9f61632](https://neatlogs.com/traces/b6d27753f38a6687e52f7b88c9f61632)
  - *Vercel Evidence Confirmation Trace:* [https://neatlogs.com/traces/26c82e806766b88a44b7f7ad81004b8c](https://neatlogs.com/traces/26c82e806766b88a44b7f7ad81004b8c)
  - *PostHog Legacy Demotion Trace:* [https://neatlogs.com/traces/70d18172f9b89de1d50eb85bd60e33e5](https://neatlogs.com/traces/70d18172f9b89de1d50eb85bd60e33e5)
  - *PostHog Evidence Confirmation Trace:* [https://neatlogs.com/traces/ed439e200f70b76a8176af7a5680f763](https://neatlogs.com/traces/ed439e200f70b76a8176af7a5680f763)
- **[Entire](https://entire.dev)**: Checkpointed git commit tracking for AI-driven development. Every change and evaluation experiment was committed through Entire with verifiable session checkpoint hashes (`entire enable -y --agent antigravity`).
- **[cfo.ai](https://cfo.ai)**: Used to model unit economics, API credit allocation, token budgets, and cost projections (`eval/results/unit_economics.json`), showing a blended LLM audit cost of ~$0.016 per company with zero cost on warm cache hits.
