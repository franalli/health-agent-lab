# Health Intelligence Service

A health assistant that watches a member's lab history over time. It does two things:

1. **Answers questions** about the member's own results, in plain language — "is my thyroid okay?", "what's changed since I started?"
2. **Raises findings on its own** — it scans every marker for meaningful change and flags what's worth knowing, escalating to a clinician when a value is critical or a trend is genuinely adverse.

**Synthetic data only — nothing here is for real clinical use.**

The design in one sentence: **the LLM is a language layer over a deterministic analytical core, never the core itself.** Every number, reference-range comparison, trend verdict, and escalation decision is computed by classical statistics and clinical thresholds in pure Python; the LLM only renders that ground truth into careful prose. That boundary is what makes the safety-critical behavior auditable and reproducible rather than a property of a prompt. Full design and rationale: [`architecture.md`](docs/architecture.md).

## Where to start

| You are… | Go to |
|---|---|
| Anyone who wants to try it — nothing to install | [Using the app](#using-the-app) |
| A developer running it on your own machine | [Run it locally](#run-it-locally) |
| A developer deploying your own copy | [Deploy your own copy](#deploy-your-own-copy) |
| A reviewer of the design | [How it works](#how-it-works) · [Evaluation](#evaluation) · [Failure modes](#failure-modes-walked) · [`architecture.md`](docs/architecture.md) |

---

## Using the app

**Hosted demo: https://health-intelligence-jfuf.onrender.com** — open it in any browser. There is nothing to install and no login; it comes pre-loaded with 15 synthetic members.

### One database, many users

Three things to know about how the deployment holds its data:

- **Shared.** The whole deployment runs on a single database. Everyone who opens the URL sees the same members and the same findings, and any change one user makes — an uploaded dataset, filed feedback, a promoted prompt, a reset — is immediately visible to every other user. There are no accounts and no per-user copies.
- **Persistent.** That database lives on a persistent disk: everything survives page refreshes, browser restarts, service restarts, and redeploys. Nothing resets on its own — only **Reseed (factory)** restores the original state. (The one exception is conversation history, which deliberately lives only in your browser tab.)
- **Safe to use concurrently.** Several people can use the app at the same time. Browsing, asking questions in both modes, scanning, uploading, and filing feedback from multiple browsers simultaneously is safe — concurrent requests queue against the database rather than corrupting it. The three heavyweight operations (**Run learn**, **Reset learning**, **Reseed**) additionally take a service-wide lock: if two users trigger one at the same moment, the second gets a 409 "already running" response and simply retries once the first finishes.

One boundary to respect: concurrency is *safe*, but the app assumes a small, cooperative group — it is an operator tool, not a multi-tenant product with isolation. A destructive action like **Reseed** affects every user at once, which is why it demands the typed `delete` confirmation. How this is made safe under the hood is described in [Deploy your own copy](#deploy-your-own-copy).

### The screen at a glance

Three columns (the dividers are draggable):

- **Center — the conversation.** Where you ask questions and read answers. Together with the right panel, this is the *member-facing* product.
- **Right — Observations | Trajectory.** Two tabs about the selected member. **Observations** lists what the system found proactively, ranked by what's worth knowing. **Trajectory** shows each marker's readings over time as small charts, so you can verify any finding with your own eyes.
- **Left — the Operator console.** A clinician/reviewer control panel, deliberately *not* part of the member experience (a member build drops the whole column). Every button drives one API endpoint and prints the raw JSON response in the **Route response** box at the bottom. Hover any button for a description of exactly what it calls and does.

In the header: the **Member** picker and the **Mode** toggle, both explained below. The ⚠ icon is a standing reminder that all data is synthetic.

### Step 1 — pick a member

Choose someone in the **Member** picker at the top. Everything on screen — findings, charts, answers — is about that one member. The 15 members cover different situations: healthy baselines, improving trends, managed chronic conditions, and genuinely concerning trajectories.

### Step 2 — read the findings (right panel)

The **Observations** tab is already filled when the page loads — the system scans automatically whenever data changes; you never have to ask. Each card explains a finding in plain language, including the reference range whenever a value sits outside it.

| Label on a finding | What it means |
|---|---|
| **Observation** | Worth knowing; not an alarm. Includes values mildly out of range and clear-but-benign trends. |
| **Needs follow-up** | The system recommends involving a clinician — the trend is significantly adverse, or a value crossed a critical ("panic") threshold. |

Switch to the **Trajectory** tab to see the underlying numbers: each marker's series, its trend line, and the flagged points.

A healthy member may show few or no findings — that's the system working, not broken. An improving trend is reported as good news, never as a problem.

### Step 3 — ask questions (the Mode toggle)

- **Mode 1 · Guided** (default) — click a suggested question chip beneath the conversation. Answers are computed deterministically from the member's own data: instant, free, and identical every time. No AI model is involved.
- **Mode 2 · Ask** — type anything in your own words. An LLM phrases the answer, but every fact in it comes from the same deterministic analysis, and the same safety rules apply: it cannot decide an escalation, invent a number, soften an alert, or answer about a marker that isn't in the data.

The toggle is the architecture made visible: Mode 1 *is* the deterministic core answering on its own; Mode 2 is the language layer on top.

Both modes decline what they shouldn't answer (medication doses, diagnoses) and point to a clinician instead — and a refusal made while an urgent flag is standing repeats the seek-care-now instruction rather than brushing the member off.

Follow-up questions work ("and what should I do about it?"), but conversation memory lives only in the browser tab — refreshing the page starts a fresh conversation, and nothing you type is stored server-side as a conversation.

If the server has no LLM key (or the provider is down), Mode 2 quietly falls back to the same deterministic answers as Mode 1 — a plainer reply, never an error. The hosted demo has a key configured, so Mode 2 works there out of the box.

### The operator console (left panel)

The clinician/operator surface: it exercises every endpoint one click each, no `curl` needed. Buttons act on the **currently selected member** where relevant; every response prints as raw JSON in the **Route response** readout. Hover any button for the full description.

**Data**

| Control | What it does |
|---|---|
| Upload dropzone (top) | Add a whole new dataset at runtime — see [Upload your own dataset](#upload-your-own-dataset). |
| **Reseed (factory)** | Wipes *everything* — uploads, feedback, learning, audit history — and restores the original 15 members. Affects every user of the instance. Double confirmation: a popup, then typing `delete`. |

**Proactive**

| Control | What it does |
|---|---|
| **Scan** | Manually re-runs the drift scan across every member and reports how many *new* observations each scan found (0 on an unchanged re-run). Scans normally happen automatically — after every data load and every correction — so this button exists to demonstrate the mechanism and as the retry if an automatic scan failed. |
| **Member escalations** | The selected member's own escalation record — every escalation ever raised for them, in every lifecycle status (open · acknowledged · superseded), highest severity first. |

**Clinician**

| Control | What it does |
|---|---|
| **Clinician queue** | The global triage worklist: every live escalation across *all* members, most severe first, un-acknowledged before acknowledged. Seeing other members here is the point — it's the clinician's cross-member view, not a scoping leak. |
| **Submit feedback** | File a correction or a signal against the selected member — see [Correct the system](#correct-the-system-submit-feedback). |
| **Member feedback** | The selected member's full feedback audit trail, newest first. Reverted entries stay listed (marked inactive) — history is never deleted. |

**Learning**

| Control | What it does |
|---|---|
| **Run learn** | Builds a candidate answer-style prompt from accumulated learnable feedback and promotes it *only* if it passes the same evaluation harness that grades the system. No-ops if nothing learnable is pending. Calls the real LLM; can take a minute. |
| **Prompt history** | Every prompt version with its verdict (promoted · rejected · reverted), an `active` flag on the one serving now, and the seeded baseline v0. |
| **Reset learning** | Deactivates all feedback overrides and reverts the prompt to v0. Not a data wipe — feedback rows and prompt history are preserved — and the affected members are re-scanned, so findings visibly revert. |

> **Tip for a clean demo:** click **Reseed (factory)** before exploring feedback or learning, so a previous visitor's feedback isn't coloring what you see. The reseed auto-scans the restored members, so findings are populated the moment it returns.

### Common tasks

#### Correct the system (Submit feedback)

Pick a member, then **Clinician → Submit feedback**. A form appears in the readout. Choose a **kind**; the form shows exactly the fields that kind needs, with dropdowns populated from live data (so you can't target a marker or finding that doesn't exist).

Three **corrections** change what the system knows:

- `range_override` — change a marker's reference range (one bound or both). The member is re-scanned on submit, so the flag change lands at once — and this is the only correction that can clear a critical (panic) flag.
- `suppress_marker` — stop analyzing a marker entirely (e.g. an expected-high marker in a managed condition). Also re-scans on submit. Deliberately *inert against a critical value*: a suppress can never hide a panic-level result.
- `preference` — a tone hint for Mode 2 answers ("keep answers brief"). Changes wording only — never what is flagged or escalated.

Four **signals** annotate rather than change:

- `helpful`, `escalation_accept` — advisory; recorded for audit. An `escalation_accept` additionally marks that escalation **acknowledged** on the clinician queue: kept listed, visibly triaged.
- `incorrect` — pick an answered question and type the corrected answer. **Learnable** (see below). The corrected text is screened at submission — incoherent or unsafe text is rejected on the spot with the reason.
- `escalation_reject` — mark an escalation a false alarm, with a required reason for the audit trail. **Learnable** once it recurs.

#### Teach the answer style (Run learn)

Only `incorrect` (with its corrected answer) and a *recurring* `escalation_reject` feed learning; the other signals are advisory. **Learning → Run learn** folds the learnable feedback into a candidate prompt and gates it through the evaluation harness: promoted only if no safety never-event fires and no measured dimension regresses. The verdict — or the reason it skipped — prints in the readout and is recorded under **Prompt history**. Whatever prompt is active, the deterministic safety floor still applies: a promoted prompt can never lower a real escalation.

#### Undo, or start over

- **Reset learning** — undo every override and prompt change; all data and history kept.
- **Reseed (factory)** — full wipe back to the original 15 members. Type `delete` to confirm. It affects everyone using the instance.

#### Upload your own dataset

The dropzone at the top of the **Data** group accepts a `.zip` of a whole dataset shaped like `backend/data/training_data/`. **Drag and drop** the file onto it (a drop target is deliberate — some browsers silently refuse to open the native file dialog; clicking works too where the dialog opens).

The upload validates before it saves anything, and every problem is reported per-file, per-row in the readout:

1. **Exactly three data files, matched by extension** — one `.json` (members), one `.jsonl` (eval set), one `.csv` (lab panels). Base names don't matter; extra files are allowed.
2. **A simple zip filename** — letters, digits, `.` `_` `-` only; it becomes the dataset's folder name. A name that already exists is rejected (409), never overwritten.
3. **The first record of each file must match its schema** — the upload doubles as a format check; failures come back as `{file, row, field, detail}`, before any side effect.

Past those gates, the upload:

- **adds, never replaces** — new members land on top of the existing ones; nothing is reset;
- **skips bad rows instead of failing** — a buggy *later* member row is skipped and reported in the response; only if *every* row skips is the whole upload rejected;
- **keeps reference ranges consistent** — ranges are global definitions, so a member row whose range for an already-known marker disagrees with the stored one is skipped, with the stored bound quoted (new markers define their ranges freely);
- **auto-scans the new members** — their findings are ready the moment the call returns;
- **keeps the bundle on disk** under canonical file names, so its eval set stays runnable later (`DATASET=<name> make eval`).

Technical equivalents: the same ingest runs as a CLI (`uv run python -m preprocessing.ingest <bundle>`), and `POST /members` adds a single member bundle live.

### If something looks wrong

| Symptom | What's happening |
|---|---|
| An amber banner appears under the header | A request failed — server unreachable, a busy maintenance lock (409), the LLM provider down (503), or rejected input (422). The banner states what failed, why, and the action to take (usually "try again in a few seconds"; a Retry button appears where the action can be safely re-run). It clears itself on the next successful request, or via ✕. |
| Mode 2 answers look plain or terse | The deterministic fallback: the server has no `ANTHROPIC_API_KEY`, or the LLM call failed. By design — never an error. |
| A follow-up question lost the thread | The tab was refreshed — conversation memory is in-browser only. |
| `helpful` / `escalation_accept` didn't change learning | Advisory by design; only `incorrect` and a recurring `escalation_reject` feed **Run learn**. |
| Submitting an `incorrect` correction is rejected | The corrected text is screened before storage (coherent, no invented numeric cutoffs, no reassurance about a flagged value); the reason is in the response. A 503 means the screening model was unreachable — retry. |
| **Run learn** says skipped / no-op | Nothing learnable pending, or a debounce/daily-cap guard fired; the reason is shown in the readout. |
| **Run learn** / **Reset learning** / **Reseed** returns 409 | Another learn/reset/reseed is already running; retry when it finishes. |
| An accepted escalation is still on the queue | Correct: `escalation_accept` marks it *acknowledged* — kept listed, visibly triaged, sorted after un-acknowledged peers. |
| An urgent escalation won't leave the queue | Deliberate: an urgent escalation is never auto-cleared, even after a correction — it stays visible until a human resolves it (in the demo, only **Reseed** removes it). |
| An upload was rejected | Read the per-row reasons in the readout — file roles, first-record schema, zip name rule, or a duplicate dataset name. |
| Data changed and you didn't change it | The demo is one shared instance — another visitor may have uploaded, filed feedback, or reseeded. **Reseed (factory)** restores the baseline. |

---

## Run it locally

Requires only [uv](https://docs.astral.sh/uv/), which manages the Python toolchain and dependencies — it installs the pinned Python version for you:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: brew install uv
```

**1. Get a fresh Anthropic API key** — optional, but needed for Mode 2's LLM answers and for `make eval`:

- Create a key in the [Anthropic Console](https://console.anthropic.com/) → **API keys** → **Create Key**. Any fresh key works as-is; there is nothing to pre-configure — the app picks its own models.
- You'll paste it into `backend/.env` in the next step. That file is git-ignored (and `gitleaks` checks every commit), so the key cannot leak into the repo.
- Skipping this is fine: Mode 1 is fully functional without a key, and Mode 2 degrades to the same deterministic answers rather than erroring.

**2. Clone, configure, run:**

```bash
git clone https://github.com/franalli/health-agent-lab.git && cd health-agent-lab
echo "ANTHROPIC_API_KEY=sk-ant-..." > backend/.env   # the key from step 1 — skip this line to run keyless
cd backend && make hooks                              # install pre-commit hooks (ruff + gitleaks); once after clone
make run                                              # uv-installs, builds + seeds the DB, serves on :8000
```

**3. Open `http://localhost:8000`** — one process serves both the API and the UI; no Node, no second server, no CORS. The database is created and seeded with the 15 members automatically on first run, and persists between runs at `backend/data/health.db`. Stop the server with `Ctrl-C`.

All `make` targets run from `backend/` (`make help` lists them):

| Command | What it does |
|---|---|
| `make run` | Init + seed + serve on `:8000` with auto-reload (the dev command) |
| `make serve` | Production-mode serve — no reload, binds `$PORT` or 8000 (the Render start command) |
| `make test` | Unit tests |
| `make lint` | `ruff` lint + format and a `gitleaks` secret scan over the whole tree |
| `make eval` | Full evaluation harness → report in `backend/eval/reports/` (needs `ANTHROPIC_API_KEY`) |
| `make eval-judge` | Feedback input-judge accuracy gate alone (SKIPs when the key is absent) |
| `make init-db` / `make seed` | The individual steps `make run` wraps |

**Code quality gates.** Commits are checked by [pre-commit](https://pre-commit.com/) (`.pre-commit-config.yaml` at the repo root): `ruff` lints and formats Python, and `gitleaks` scans for secrets. `make hooks` installs the git hook; `make lint` runs every check on demand.

## Deploy your own copy

A single [Render](https://render.com/) Web Service serves both the API and the static page same-origin, deployed straight from this repo via the committed **`render.yaml`** Blueprint. SQLite runs in development and production deliberately — one engine, so the evaluation harness certifies the stack that actually ships. The local one-command run stays primary.

1. Push this repo to GitHub.
2. Render → **New** → **Blueprint** → connect the repo; it reads `render.yaml` (a `web` service, `uv sync` build, `uvicorn api:app` on `$PORT`, health check `/health`).
3. Set **`ANTHROPIC_API_KEY`** as a secret when prompted — create a fresh key in the [Anthropic Console](https://console.anthropic.com/) → **API keys**, same as for a local run (`render.yaml` marks the variable `sync: false`, so it lives only in the Render dashboard and is never committed). Mode 1 works without it; Mode 2 + the intent gate need it.
4. **Create** → on first boot the app inits the schema, **seeds the 15 training members, and auto-scans them** (the build can't see the runtime filesystem, so all of it runs at startup — sub-second), so the first page load already shows findings. Hit `<url>/health`, then open `<url>/`.

**The database is durable (the shipped default).** `render.yaml` sets `plan: starter` and mounts a 1 GB persistent disk at `/data`, with `HEALTH_DB_PATH=/data/health.db` pointing SQLite at it (~$8/mo). Uploaded members, feedback, and promoted prompts survive deploys and restarts, and the instance stays warm (no idle spin-down). The startup seed-if-empty still self-heals a fresh disk.

Two footnotes:

- The dataset *folders* `POST /members/upload` writes (the kept `eval_set.jsonl`/`lab_panels.csv`) live under the ephemeral app dir unless you also point `HEALTH_DATA_ROOT` at the disk — see the comment in `render.yaml`.
- To run free/ephemeral instead: switch to `plan: free` and delete the `HEALTH_DB_PATH` env var + `disk` block. The SQLite file is then wiped on an idle spin-down and re-seeded automatically on the next cold start — uploads and feedback don't survive, a documented trade-off.

**One database, two workers, many concurrent users.** The start command in `render.yaml` runs `uvicorn` with `--workers 2`: two server processes on the one instance, both reading and writing the **same WAL-mode SQLite file** on the persistent disk. That is the entire concurrency architecture — no separate database server — and it is what lets a single deployment serve several users at the same time:

- Reads and `/ask` requests serialize safely under concurrent use: each request opens its own connection, and writes wait out contention with a 5-second busy-timeout instead of failing or corrupting.
- The startup init+seed holds a cross-process lock, so exactly one worker seeds a fresh DB while the other waits and no-ops.
- `/learn`, `/reset`, and Reseed share a cross-process learning-state lock — a conflicting call returns 409 (retry) rather than interleaving with a running multi-minute `/learn`.
- Dataset ingest (the `.zip` upload / `POST /members`) and Reseed likewise share a cross-process dataset lock (409 on contention), and each member scan runs as one atomic transaction — so two workers can't interleave a bulk ingest with a factory wipe or clobber each other's scan findings.
- Destructive operations (Reseed, dataset upload) are fully atomic transactions.

The deliberate boundary: this supports a small cooperative group on one shared dataset, not multi-tenant isolation — out of scope per the brief. (This is the mechanism behind the plain-language [One database, many users](#one-database-many-users) section above.)

Locally, `make serve` runs the exact production command — same `--workers 2` against your local DB — so the multi-worker setup can be rehearsed before deploying; `make run` stays the primary dev command (single process, auto-reload).

---

## How it works

Two behaviors over one deterministic core:

- **Grounded Q&A** — every claim in an answer is backed by the member's own readings, with the evidence one tap away. Out-of-scope or unsafe requests (e.g. "change my dose") are declined and pointed to a clinician.
- **Proactive observations** — generated at ingest, without being asked: whenever data loads, and whenever a clinician correction changes what should be flagged, the system scans every marker for drift, ranks what's worth knowing by severity, and raises a clinician hand-off when a value is critical or a trajectory is significantly adverse.

Trend detection is classical statistics, not machine learning:

- **Mann–Kendall** (Kendall's τ + an exact small-sample p-value) for monotonic trend;
- **Theil–Sen** (with a distribution-free confidence interval) for direction and rate — "too noisy to call" is a first-class answer;
- **Reference Change Value** (from published EFLM analytical + within-subject biological variation) for clinically-meaningful-vs-noise, where that variation is curated — otherwise the trend is judged on Mann–Kendall + Theil–Sen alone;
- **Benjamini–Hochberg FDR** across markers, so a multi-marker panel can't raise a false flag by chance;
- reference-range and **panic** flags, plus band-crossing, with **direction-aware** severity.

Escalation is reserved for panic thresholds and significant adverse trajectories — a value merely outside its range is an observation, not an alarm, and a managed condition's expected-high marker isn't treated as new.

Self-improvement is **additive, versioned, inspectable data the system consumes — never an autonomous edit of its own safety logic.** Two forms:

- **Deterministic correction.** A clinician override (`POST .../feedback`) — re-bound a marker's range or suppress an expected-abnormal marker — is resolved into the analytical core's *inputs* and auto-scans the member on submit, so the changed flag shows up in both modes the moment the call returns; a member tone **preference** joins the answer-composer's context instead (it shapes Mode-2 wording, never what is flagged). It changes what the system *knows*, not its rules; `POST /reset` reverts it cleanly — and re-scans what it reverted.
- **Harness-gated prompt promotion.** `POST /learn` rule-assembles a candidate answer-composer prompt from accumulated feedback (a pure function of the feedback set), then gates it through the **same evaluation harness** that grades the system — promoting it only if it trips no never-event and regresses no measured dimension. The gate certifies **non-regression, not betterment**; no model rewrites the prompt, and the deterministic safety floor is enforced under *whatever* prompt is active, so a promoted prompt can never lower a real escalation. Guarded against cost/abuse (single-flight, debounce, structural pre-check, daily cap), and reverted by `/reset`.

A read-only **trajectory** view (`GET .../trajectory`, rendered as the Trajectory tab's sparklines) lets an operator eyeball a marker's series, its Theil–Sen line, and the flagged points to verify a finding by hand — while the LLM still sees only the collapsed verdict, never the raw series.

## Evaluation

```bash
make eval
```

Runs the supplied 17-case set (plus a few tagged gate/trend additions) through both modes and writes a markdown + JSON report under `backend/eval/reports/`. Mode 2 calls the real Anthropic API (the run measures consistency, latency, and cost), so `make eval` needs `ANTHROPIC_API_KEY`.

The harness mirrors the system's own discipline: deterministic scorers wherever there's a ground truth (grounding, escalation, trend verdicts, latency, cost, consistency); an LLM judge only for the irreducibly subjective (semantic support, tone). Safety failures are **never-events** that fail the run outright and surface first; over-escalation is measured, not failed. The JSON report is the regression gate — it persists each run's raw inputs and outputs, so a scorer change can be re-graded offline. Setting `LANGSMITH_API_KEY` additionally streams each run to LangSmith for trace inspection (offline, synthetic data only); the local report stays canonical.

`make eval` also runs the **feedback input-judge accuracy gate** (`eval/judge_eval.py`; standalone via `make eval-judge`): a labeled, held-out battery through the real Haiku classifier that screens clinician corrections (`learn.judge_corrected_answer`). It is measured rather than unit-tested because the judge is an LLM; a miss — or, with a key present, a transient provider failure — fails the run alongside a safety never-event, and it SKIPs only when the key is absent. That gate is distinct from the **composer** semantic LLM judge (semantic support and tone of the answers themselves), which remains the next increment.

## Failure modes (walked)

Three found, documented, and bounded during development — included here because the brief asks for this, and a failure mode you call out first reads differently than one a tester discovers unmentioned.

**1. Gate misroutes in-scope meta questions.** The Mode 2 intent gate (Haiku, `temperature=0`) classifies each message before the composer runs. During development it misrouted in-scope questions phrased as "what are you noticing?" or "what observations are concerning?" to `out_of_scope` — a false refusal on the system's own features. Root cause: the `out_of_scope` definition was too broad (it pulled in "what should I do?" alongside genuine drug/symptom questions). Fix: tightened `out_of_scope` to drugs, doses, and diagnosis questions only; added explicit `none`-route exemplars for next-steps and meta-asks, plus held-out eval cases (A06–A10) so the fix is regression-pinned. The full live-API eval battery passes with it. At-boundary phrasing can still flip between runs (nondeterminism in the model, not a systemic bug) — if a specific phrasing consistently misfires, add it as an eval case.

A third instance of the same class surfaced in live use: the panicked follow-up right after an urgent flag ("what does high potassium mean? what is the risk?") drops the possessive, so it read as general medical education and was refused — at exactly the moment the member most needs the answer. Fixed the same way (marker/condition-education phrasings route to the composer, which grounds them in the member's own flagged value under the standing urgent floor; held-out eval case A12), and paired with a floor-aware refusal template: a *genuinely* out-of-scope ask made under a standing `urgent`/`clinician_review` floor now restates that next step (urgent: seek care now + the emergency numbers) before redirecting, instead of a flat "ask your GP any time" (eval case A13 pins the floor surviving the refusal).

**2. `/learn` certifies non-regression, not improvement.** The harness-gated prompt promotion gate (`POST /learn`) tells you "this candidate does *not* regress any measured dimension" — not "it improved anything." A pure tone/quality gain that trips no deterministic scorer (grounding, escalation, trend verdict) passes the gate for the same reason a neutral change does. The deferred eval LLM judge (semantic grounding, tone) would differentiate; the current gate is deliberately safe-toward-rejection. This is a correctness choice: the gate errs on the side of not promoting, since a promoted prompt that degrades unmeasured dimensions is harder to detect than a gate that rejected a benign candidate.

**3. `/learn` runs at N=1 — grounding noise can false-reject a benign candidate.** The main eval harness (`make eval`) samples N=3 to absorb run-to-run LLM variance in the grounding scorer. The in-process `/learn` gate runs at N=1 to control LLM cost, so a borderline candidate that would clear N=3 can false-reject on a single unlucky sample. The gate errs safe (toward rejection), and a false-reject is recoverable: accumulate more feedback signals and trigger `/learn` again. The skip reason is logged and shown in the operator "Learning" readout so it's visible, not silent.

## Repository layout

```
backend/
  health_intelligence/   # the serving library (SQLite → answer); imports no web framework
  preprocessing/         # the normalization adapter (bundle → SQLite)
  eval/                  # the evaluation harness + labeled cases
  data/                  # datasets — one bundle per sub-folder (training_data/ ships); DATASET selects the active one
frontend/                # a single static page (vanilla JS, no build)
```

A thin `backend/api.py` exposes the routes and serves the static page; `analysis.py` is pure functions over typed inputs, so statistical correctness is unit-testable in isolation. The full file-by-file tree lives in [`architecture.md`](docs/architecture.md) §14.

## Built with AI tools

[Claude Code](https://docs.claude.com/en/docs/claude-code) was used throughout — for design iteration, drafting, and review. The repo conventions and the invariants an agent must respect are encoded in [`CLAUDE.md`](CLAUDE.md).

## References & disclaimer

Reference Change Values use analytical and within-subject biological variation from the [EFLM Biological Variation Database](https://biologicalvariation.eu/) for the markers the evaluation set exercises, plus published BP-variability for systolic blood pressure (a demo-prominent vital); a marker without curated variation falls back to the Mann–Kendall + Theil–Sen trend verdict (a typed skip-path, not a silent gap). Reference ranges in the sample data are illustrative and synthetic.

**This is a prototype on synthetic data. It does not diagnose, prescribe, or make clinical decisions, and must not be used for real medical care.**
