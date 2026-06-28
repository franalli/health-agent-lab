# Health Intelligence Service

A persistent assistant over a member's longitudinal lab history. It answers free-form health questions grounded in the member's own results, and — without being asked — surfaces meaningful changes in their trajectory, refusing or escalating to a clinician when that's the safe thing to do.

**Synthetic data only — nothing here is for real clinical use.**

## The idea

The core design choice: **the LLM is a language layer over a deterministic analytical core, not the core itself.** Every number, reference-range comparison, trend call, and escalation decision is computed by classical statistics and clinical thresholds in pure Python. The LLM only renders that ground truth into careful, plain-language answers and handles open-ended phrasing — it never computes a value or decides an escalation. That boundary is what makes the safety-critical behavior auditable and reproducible rather than a property of a prompt.

Full design and rationale: [`architecture.md`](docs/architecture.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), which manages the Python toolchain and dependencies.

```bash
git clone <repo> && cd health-intelligence
echo "ANTHROPIC_API_KEY=sk-..." > backend/.env   # only needed for Mode 2 (LLM answers)
cd backend && make hooks                          # install pre-commit hooks (ruff + gitleaks); once after clone
make run                                          # uv-installs, builds + seeds the DB, serves on :8000
```

Open `http://localhost:8000` — one process serves both the API and the UI; no Node, no second server, no CORS.

> The Makefile is wired (`backend/Makefile`); run the `make` targets from `backend/` (`make help` lists them). The consumer UI ships (Phase 6): `make run` serves the API **and** the static page same-origin on `:8000`. The page's Mode-2 chat needs `ANTHROPIC_API_KEY` for the real LLM path; without it, `/ask` degrades to the deterministic fallback (never an error). `make eval` runs the evaluation harness (deterministic scorers + markdown/JSON report, plus a gated LangSmith trace sink); the LLM judge is the next increment.
>
> **Code quality gates.** Commits are checked by [pre-commit](https://pre-commit.com/) (`.pre-commit-config.yaml` at the repo root): `ruff` lints and formats Python, and `gitleaks` scans for secrets. `make hooks` installs the git hook; `make lint` runs every check over the whole tree on demand.

## What it does

Two behaviors, over one deterministic core:

- **Grounded Q&A** — ask in plain language ("what's changed since I started?", "is my thyroid okay?"). Every claim is backed by the member's own readings, with the evidence one tap away. Out-of-scope or unsafe requests (e.g. "change my dose") are declined and pointed to a clinician.
- **Proactive observations** — on each new panel the system scans every marker for drift, ranks what's worth knowing by severity, and raises a clinician hand-off when a value is critical or a trajectory is significantly adverse. A healthy baseline produces no false alarms; an improving trend is recognized as good news, not a problem.

**Two modes**, flipped by a single toggle:

- **Mode 1 (default, no LLM)** — proactive findings plus instant, fully-grounded answers to the common questions, computed deterministically. A clinician-trustworthy assistant that costs nothing per query and is byte-identical every time.
- **Mode 2 (LLM)** — "ask in your own words." The same evidence and the same safety floor, rendered in natural prose and able to handle the long tail of free-form questions.

The toggle is the architecture made visible: Mode 1 *is* the deterministic core answering on its own; Mode 2 is the language layer on top.

## How it decides

Trend detection is classical, not machine-learned:

- **Mann–Kendall** (Kendall's τ + an exact small-sample p-value) for monotonic trend;
- **Theil–Sen** (with a distribution-free confidence interval) for direction and rate — "too noisy to call" is a first-class answer;
- **Reference Change Value** (from published EFLM analytical + within-subject biological variation) for clinically-meaningful-vs-noise, where that variation is curated — otherwise the trend is judged on Mann–Kendall + Theil–Sen alone;
- **Benjamini–Hochberg FDR** across markers, so a multi-marker panel can't raise a false flag by chance;
- reference-range and **panic** flags, plus band-crossing, with **direction-aware** severity.

Escalation is reserved for panic thresholds and significant adverse trajectories — a value merely outside its range is an observation, not an alarm, and a managed condition's expected-high marker isn't treated as new.

## How it learns

Self-improvement is **additive, versioned, inspectable data the system consumes — never an autonomous edit of its own safety logic.** Two forms:

- **Deterministic correction.** A clinician override (`POST .../feedback`) — re-bound a marker's range, suppress an expected-abnormal marker, or set a tone preference — is resolved into the analytical core's *inputs*, so re-scanning or re-asking shows the changed flag in **both** modes. It changes what the system *knows*, not its rules; `POST /reset` reverts it cleanly.
- **Harness-gated prompt promotion.** `POST /learn` rule-assembles a candidate answer-composer prompt from accumulated feedback (a pure function of the feedback set), then gates it through the **same evaluation harness** that grades the system — promoting it only if it trips no never-event and regresses no measured dimension. No model rewrites the prompt, and the deterministic safety floor is enforced under *whatever* prompt is active, so a promoted prompt can never lower a real escalation. Guarded against cost/abuse (single-flight, debounce, structural pre-check, daily cap), and reverted by `/reset`.

A read-only **trajectory** view (`GET .../trajectory`, rendered as inline sparklines) lets an operator eyeball a marker's series, its Theil–Sen line, and the flagged points to verify a finding by hand — while the LLM still sees only the collapsed verdict, never the raw series.

## Try it on your own data

The demo's operator panel (left rail, kept out of the member experience) drives every endpoint with one click, and **Upload bundle** ingests a held-out member at runtime — no redeploy. The expected format is one `MemberBundle`. A malformed file returns a clear error in the operator readout, so the upload doubles as the format check. The same ingest runs as a CLI (`uv run python -m preprocessing.ingest <bundle>`); `POST /members` is its live equivalent — the Phase-6 surface.

## Evaluation

```bash
make eval
```

Runs the supplied 17-case set (plus a few tagged gate/trend additions) through both modes and writes a markdown + JSON report under `backend/eval/reports/`. The harness mirrors the system's own discipline: deterministic scorers wherever there's a ground truth (grounding, escalation, trend verdicts, latency, cost, consistency), an LLM judge only for the irreducibly subjective (semantic support, tone). Safety failures are **never-events** that fail the run outright and surface first; over-escalation is measured, not failed. The JSON report is the regression gate (it persists each run's raw inputs/outputs, so a scorer change can be re-graded offline). Mode 2 calls the real Anthropic API (it measures consistency and latency/cost), so `make eval` needs `ANTHROPIC_API_KEY`. The deterministic scorers ship today, and setting `LANGSMITH_API_KEY` additionally streams each run to LangSmith for trace inspection (offline, synthetic-data-only) — the local report staying canonical. The LLM judge (semantic support, tone) is the next increment.

## Layout

```
backend/
  health_intelligence/   # the serving library (SQLite → answer); imports no web framework
  preprocessing/         # the normalization adapter (bundle → SQLite)
  eval/                  # the evaluation harness + labeled cases
  data/                  # datasets — one bundle per sub-folder (training_data/ ships); DATASET selects the active one
frontend/                # a single static page (vanilla JS, no build)
```

A thin `backend/api.py` exposes the routes and serves the static page; `analysis.py` is pure functions over typed inputs, so statistical correctness is unit-testable in isolation. The full file-by-file tree lives in [`architecture.md`](docs/architecture.md) §14.

## Deployment

A single [Render](https://render.com/) Web Service: FastAPI serves both the API and the static page, with SQLite on a persistent disk so live ingest survives deploys. SQLite is used in development and production deliberately — one engine, so the evaluation harness certifies the stack that actually ships. The local one-command run stays primary.

## Built with AI tools

[Claude Code](https://docs.claude.com/en/docs/claude-code) was used throughout — for design iteration, drafting, and review. The repo conventions and the invariants an agent must respect are encoded in [`CLAUDE.md`](CLAUDE.md).

## References & disclaimer

Reference Change Values use analytical and within-subject biological variation from the [EFLM Biological Variation Database](https://biologicalvariation.eu/) for the markers the evaluation set exercises, plus published BP-variability for systolic blood pressure (a demo-prominent vital); a marker without curated variation falls back to the Mann–Kendall + Theil–Sen trend verdict (a typed skip-path, not a silent gap). Reference ranges in the sample data are illustrative and synthetic.

**This is a prototype on synthetic data. It does not diagnose, prescribe, or make clinical decisions, and must not be used for real medical care.**
