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

> The Makefile is wired (`backend/Makefile`); run the `make` targets from `backend/` (`make help` lists them). The consumer UI ships (Phase 6): `make run` serves the API **and** the static page same-origin on `:8000`. The page's Mode-2 chat needs `ANTHROPIC_API_KEY` for the real LLM path; without it, `/ask` degrades to the deterministic fallback (never an error). `make eval` runs the evaluation harness (deterministic scorers + markdown/JSON report, plus a gated LangSmith trace sink) **and** the feedback input-judge accuracy gate (a labeled held-out battery through the real Haiku `learn.judge_corrected_answer`; `make eval-judge` runs it alone); the composer semantic LLM judge is the next increment.
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

- **Deterministic correction.** A clinician override (`POST .../feedback`) — re-bound a marker's range or suppress an expected-abnormal marker — is resolved into the analytical core's *inputs*, so re-scanning or re-asking shows the changed flag in **both** modes; a member tone **preference** joins the answer-composer's context instead (it shapes Mode-2 wording, never what is flagged). It changes what the system *knows*, not its rules; `POST /reset` reverts it cleanly.
- **Harness-gated prompt promotion.** `POST /learn` rule-assembles a candidate answer-composer prompt from accumulated feedback (a pure function of the feedback set), then gates it through the **same evaluation harness** that grades the system — promoting it only if it trips no never-event and regresses no measured dimension. So the gate certifies **non-regression, not betterment**: it errs safe toward rejection, a tone/quality gain is invisible to the deterministic scorers until the LLM judge lands, and at its single-sample setting its soft grounding score can even false-reject a benign candidate on the run-to-run variance the main harness samples N=3 to absorb (never-events stay zero-tolerance). No model rewrites the prompt, and the deterministic safety floor is enforced under *whatever* prompt is active, so a promoted prompt can never lower a real escalation. Guarded against cost/abuse (single-flight, debounce, structural pre-check, daily cap), and reverted by `/reset`.

A read-only **trajectory** view (`GET .../trajectory`, rendered as inline sparklines) lets an operator eyeball a marker's series, its Theil–Sen line, and the flagged points to verify a finding by hand — while the LLM still sees only the collapsed verdict, never the raw series.

## Try it on your own data

The demo's operator panel (left rail, kept out of the member experience) drives every endpoint with one click, and a **`.zip` upload dropzone** (top of the **Data** group) brings a held-out dataset in at runtime — no redeploy. A dropped file is used instead of a file-picker button on purpose: some browsers (certain Chrome instances) silently refuse to open the native file-selection dialog, and a drop delivers the bundle with no dialog. It requires a **`.zip` of a `training_data`-shaped folder** holding exactly three files matched **by extension** — one `.json` (members), one `.jsonl` (eval set), one `.csv` (lab panels), under any base names (the hold-out ships stable extensions, only the names differ) — plus optional extras. The firewall **validates the first row of each file** against its schema and rejects a missing/duplicated file or a malformed first row with a clear **per-row reason** in the operator readout (`{file, row, field, detail}` for each problem), before any side effect. Past that gate it (1) **persists the whole bundle** as a new dataset folder under `backend/data/` — each file normalized to its canonical name on disk, keeping the `eval_set.jsonl` usable later via `DATASET=<name> make eval`; (2) ingests every member **additively** (added on top of the seeded data, no reseed; `member_id` is opaque), **skipping any buggy later row** (reported in a `skipped` list) — but 422ing and rolling back the folder if *every* row skips (nobody ingested); and (3) **auto-scans** the new members (`POST /members/upload`). The upload doubles as the format check. The same ingest runs as a CLI (`uv run python -m preprocessing.ingest <bundle>`); `POST /members` is the single-bundle live equivalent.

## Evaluation

```bash
make eval
```

Runs the supplied 17-case set (plus a few tagged gate/trend additions) through both modes and writes a markdown + JSON report under `backend/eval/reports/`. The harness mirrors the system's own discipline: deterministic scorers wherever there's a ground truth (grounding, escalation, trend verdicts, latency, cost, consistency), an LLM judge only for the irreducibly subjective (semantic support, tone). Safety failures are **never-events** that fail the run outright and surface first; over-escalation is measured, not failed. The JSON report is the regression gate (it persists each run's raw inputs/outputs, so a scorer change can be re-graded offline). Mode 2 calls the real Anthropic API (it measures consistency and latency/cost), so `make eval` needs `ANTHROPIC_API_KEY`. The deterministic scorers ship today, and setting `LANGSMITH_API_KEY` additionally streams each run to LangSmith for trace inspection (offline, synthetic-data-only) — the local report staying canonical. `make eval` also runs the **feedback input-judge accuracy gate** (`eval/judge_eval.py`, run standalone by `make eval-judge`): a labeled, held-out battery through the real Haiku that screens clinician corrections (`learn.judge_corrected_answer`) — measured, not unit-testable, since the judge is an LLM; a miss (or, with a key present, a transient provider failure) fails the run alongside a safety never-event, and it SKIPs only when the key is absent. That gate is distinct from the **composer** semantic LLM judge (semantic support, tone of the answers themselves), which remains the next increment.

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

A single [Render](https://render.com/) Web Service serves both the API and the static page same-origin, deployed straight from this repo via the committed **`render.yaml`** Blueprint. SQLite runs in development and production deliberately — one engine, so the evaluation harness certifies the stack that actually ships. The local one-command run stays primary.

**Deploy (one-click Blueprint):**

1. Push this repo to GitHub.
2. Render → **New** → **Blueprint** → connect the repo; it reads `render.yaml` (a `web` service, `uv sync` build, `uvicorn api:app` on `$PORT`, health check `/health`).
3. Set **`ANTHROPIC_API_KEY`** as a secret when prompted (`render.yaml` marks it `sync: false`, so it's never committed). Mode 1 works without it; Mode 2 + the gate need it.
4. **Create** → on first boot the app inits the schema and **seeds the 15 training members** (the build can't see the runtime filesystem, so both run at startup). Hit `<url>/health`, then open `<url>/`.

**Free tier (the shipped default): the database is ephemeral.** The free plan has no persistent disk and spins down when idle, so the SQLite file is wiped on a cold start — the app re-seeds `training_data` automatically, but `POST /members` / `POST /members/upload` uploads and `feedback` do **not** survive a spin-down. For durable storage, upgrade in `render.yaml`: switch `plan: free` → `plan: starter`, uncomment the `disk` block (mounted at `/data`), and set `HEALTH_DB_PATH=/data/health.db` — then uploaded members and learning persist across deploys (~$8/mo). (The dataset *folders* `POST /members/upload` writes — the kept `eval_set.jsonl`/`lab_panels.csv` — still live under the ephemeral app dir unless you also point `HEALTH_DATA_ROOT` at the disk; see the `render.yaml` comment.)

Locally, `make serve` runs the exact production command (no `--reload`, binds `$PORT` or 8000, seeds on startup if empty); `make run` stays the primary dev command.

## Built with AI tools

[Claude Code](https://docs.claude.com/en/docs/claude-code) was used throughout — for design iteration, drafting, and review. The repo conventions and the invariants an agent must respect are encoded in [`CLAUDE.md`](CLAUDE.md).

## References & disclaimer

Reference Change Values use analytical and within-subject biological variation from the [EFLM Biological Variation Database](https://biologicalvariation.eu/) for the markers the evaluation set exercises, plus published BP-variability for systolic blood pressure (a demo-prominent vital); a marker without curated variation falls back to the Mann–Kendall + Theil–Sen trend verdict (a typed skip-path, not a silent gap). Reference ranges in the sample data are illustrative and synthetic.

**This is a prototype on synthetic data. It does not diagnose, prescribe, or make clinical decisions, and must not be used for real medical care.**
