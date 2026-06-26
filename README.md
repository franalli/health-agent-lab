# Health Intelligence Service

A persistent assistant over a member's longitudinal lab history. It answers free-form health questions grounded in the member's own results, and — without being asked — surfaces meaningful changes in their trajectory, refusing or escalating to a clinician when that's the safe thing to do.

**Synthetic data only — nothing here is for real clinical use.**

## The idea

The core design choice: **the LLM is a language layer over a deterministic analytical core, not the core itself.** Every number, reference-range comparison, trend call, and escalation decision is computed by classical statistics and clinical thresholds in pure Python. The LLM only renders that ground truth into careful, plain-language answers and handles open-ended phrasing — it never computes a value or decides an escalation. That boundary is what makes the safety-critical behavior auditable and reproducible rather than a property of a prompt.

Full design and rationale: [`architecture.md`](architecture.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), which manages the Python toolchain and dependencies.

```bash
git clone <repo> && cd health-intelligence
echo "ANTHROPIC_API_KEY=sk-..." > backend/.env   # only needed for Mode 2 (LLM answers)
make run                                          # uv-installs, builds + seeds the DB, serves on :8000
```

Open `http://localhost:8000` — one process serves both the API and the UI; no Node, no second server, no CORS.

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
- **Reference Change Value** (from published EFLM analytical + within-subject biological variation) for clinically-meaningful-vs-noise;
- **Benjamini–Hochberg FDR** across markers, so a multi-marker panel can't raise a false flag by chance;
- reference-range and **panic** flags, plus band-crossing, with **direction-aware** severity.

Escalation is reserved for panic thresholds and significant adverse trajectories — a value merely outside its range is an observation, not an alarm, and a managed condition's expected-high marker isn't treated as new.

## Try it on your own data

The demo's operator panel (left rail, kept out of the member experience) drives every endpoint with one click, and **Upload bundle** ingests a held-out member at runtime — no redeploy. The expected format is the bundle shipped in [`backend/data/`](backend/data/); a malformed file returns a clear error, so the upload doubles as the format check. The same ingest is available as a CLI (`uv run python -m preprocessing.ingest <bundle>`) and as `POST /members`.

## Evaluation

```bash
make eval
```

Runs a labeled case set through both modes and writes a markdown + JSON report. The harness mirrors the system's own discipline: deterministic scorers wherever there's a ground truth (grounding, escalation, trend verdicts, latency, cost, consistency), an LLM judge only for the irreducibly subjective (semantic support, tone). Safety failures are **never-events** that fail the run outright and surface first; over-escalation is measured, not failed. The JSON report is the regression gate. Set `LANGSMITH_API_KEY` to additionally stream each run to LangSmith for trace inspection (offline, synthetic-data-only); the local report stays canonical.

## Layout

```
backend/
  health_intelligence/   # the serving library (SQLite → answer); imports no web framework
  preprocessing/         # the normalization adapter (bundle → SQLite)
  eval/                  # the evaluation harness + labeled cases
  data/                  # supplied synthetic bundle + test fixtures
frontend/                # a single static page (vanilla JS, no build)
```

A thin `backend/api.py` exposes the routes and serves the static page; `analysis.py` is pure functions over typed inputs, so statistical correctness is unit-testable in isolation. The full file-by-file tree lives in [`architecture.md`](architecture.md) §14.

## Deployment

A single [Render](https://render.com/) Web Service: FastAPI serves both the API and the static page, with SQLite on a persistent disk so live ingest survives deploys. SQLite is used in development and production deliberately — one engine, so the evaluation harness certifies the stack that actually ships. The local one-command run stays primary.

## Built with AI tools

[Claude Code](https://docs.claude.com/en/docs/claude-code) was used throughout — for design iteration, drafting, and review. The repo conventions and the invariants an agent must respect are encoded in [`CLAUDE.md`](CLAUDE.md).

## References & disclaimer

Reference Change Values use analytical and within-subject biological variation from the [EFLM Biological Variation Database](https://biologicalvariation.eu/). Reference ranges in the sample data are illustrative and synthetic.

**This is a prototype on synthetic data. It does not diagnose, prescribe, or make clinical decisions, and must not be used for real medical care.**
