# CLAUDE.md

Operating manual for Claude Code in this repository. `architecture.md` is the source of truth for *why*; this file is the *how*. Read both before changing anything.

## What this is

A **Health Intelligence Service**: a persistent assistant over a member's longitudinal lab history that (1) answers free-form health questions grounded in their own results and (2) proactively surfaces trajectory drift, refusing or escalating to a clinician when warranted. Synthetic data only.

## The one law

**The LLM is a language layer over a deterministic core — never the core itself.** Every number, range comparison, trend verdict, and escalation decision is computed in pure Python (`analysis.py`, `safety.py`). The LLM only renders that ground truth into careful prose and routes open-ended phrasing. It must never compute a statistic, decide an escalation, or assert a value the deterministic layer did not produce.

If a change would let the model decide something safety-relevant, it is wrong. Stop.

## Hard invariants — do not break

- **`analysis.py` is pure.** No DB, no LLM, no network, no clock — typed inputs in, a typed `TrajectoryAnalysis` out. This is what makes statistical correctness unit-testable. Keep it that way forever.
- **The deterministic floor is always on.** `safety.py` computes an escalation floor from the analysis; the validator enforces `response.escalation >= floor` on *every* turn, both modes. A softer-toned answer can never sit on a harder floor.
- **Escalation ≠ out-of-range.** A value merely outside its reference range is at most a `notable` observation. Escalation (`clinician_review` / `urgent`) is reserved for panic thresholds and RCV+FDR-significant adverse trajectories — **or, where the series is too short for FDR (`3 ≤ n < n_min`), an adverse change that clears RCV *and* is strictly monotonic** (the sub-`n_min` rule; below `n_min` the exact Mann–Kendall p floors at 0.33, so FDR is structurally unreachable and monotonicity is the deterministic stand-in for the significance test — `analysis._sparse_adverse`; `urgent` stays panic-only). A managed condition's expected-high marker must not escalate.
- **Never fabricate.** No reading for a marker → the answer is "not measured", never an invented value. Every claim traces to `findings[].evidence[]`.
- **Every narrator surfaces the core flags.** A rendered answer — template (`templates.py`) *or* LLM (`compose()`) — must state a marker's panic / out-of-range status, not merely its trend. Trend-only narration of a flagged value falsely reassures. The escalation floor guards the *level*; this guards the *prose*. It passes every structural test — it is caught only by reading the words, so read them.
- **One normalization path.** All input enters through `preprocessing/ingest.py` (the firewall): it parses the supplied bundle's five reference-range shapes, folds vitals in as markers, and writes the internal domain model. Nothing downstream re-parses raw input. *Which* bundle it reads is resolved by `preprocessing/datasets.py` from the `DATASET` env var (default `training_data`) — each dataset is its own sub-folder of `backend/data/`, so new bundles ingest incrementally; the derived `health.db` sits at the `data/` root, not inside a bundle.
- **One DB module.** Only `db.py` touches SQLite. Clinical constants and thresholds live in `config.py` (versioned); `.env` is secrets + runtime only.
- **Both provider calls behind `llm.py`** — `compose()` and the gate classifier. These are the only network seams.
- **Two layers, not one.** The SQLite schema (`schema.sql`) and the Pydantic models (`models.py`) are deliberately separate — do not fuse them into ORM-as-model. The row↔model mapping lives in `db.py`.

## Structure

`ls` shows the tree; the file-by-file map is `architecture.md` §14 (the one canonical copy — don't duplicate it here). What `ls` won't tell you is intent, and that's in the invariants above: `health_intelligence/` is the serving library (pure, imports no web framework), `preprocessing/ingest.py` is the one firewall, `eval/` is the harness, and `api.py` is thin routes serving the single static `frontend/index.html`. *(The build is at Phase 8: deployment has landed (Phase 7 self-improvement before it). The **deterministic correction** path is live — `POST /members/{id}/feedback` (range-override · suppress-marker · preference) resolves into `analyze`'s inputs via the Phase-2 `db.resolve_overrides`/`_apply_overrides` seam, and `POST /reset` reverts it. The **harness-gated prompt promotion** path is live — `health_intelligence/learn.py` behind `POST /learn` rule-assembles a candidate composer prompt from the feedback signals (a pure function of the set), gates it through the **same eval harness `make eval` runs** (driven in-process via `eval/inprocess.py`, not the HTTP `ServiceClient`), and promotes/rejects; the composer is now **version-aware** (`pipeline` resolves the latest promoted `prompt_versions` row via `db.get_active_prompt`, falling back to `llm.BASE_COMPOSE_SYSTEM` at version 0). Also live: the read-only `GET /members/{id}/trajectory` projection (UI sparklines + human verification — the **LLM still consumes only the collapsed `TrajectoryAnalysis`**, never the raw series), `POST /admin/reseed` (factory reset), and the control panel's **Learning** group. Two scoped notes: (1) the `/learn` gate is **deterministic-only in v1** — it is *safety*-complete because the always-on validator floors escalation regardless of the active prompt, but a pure tone/quality regression that trips no deterministic dimension is not caught until the eval **LLM judge** (semantic grounding, tone — the deferred Phase-5b increment) lands; (2) Phase 7 needed **zero schema changes** (`feedback`/`prompt_versions`/`interactions.prompt_version` already existed). Phase 8 also needed zero schema changes: a committed `render.yaml` Render Blueprint, with the lifespan running `init_db` **+ seed-if-empty** on startup (reusing `ingest_dataset`) so a fresh instance self-heals its schema and the 15 seeded members; shipped free/ephemeral, with a paid-disk durable upgrade documented (§15). The one remaining forward increment is the eval LLM judge. The invariants describe the target design throughout.)*

## Commands

Dependencies and the virtualenv are managed with **uv**. All commands run from `backend/` (where `pyproject.toml` lives — there is no root workspace).

```
uv sync          # install
make hooks       # install the git pre-commit hook (run once after clone)
make init-db     # create the SQLite schema
make seed        # ingest the supplied bundle
make run         # init + seed + serve API and UI on :8000 (dev; --reload)
make serve       # production-mode serve (no reload; binds $PORT or 8000) — the Render start command
make eval        # run the evaluation harness → report
make lint        # ruff lint+format + gitleaks secret scan over the whole tree
make test        # unit tests
```

> The Makefile is wired as of Phase 3a (`backend/Makefile`); `make help` lists the targets. `make eval` runs the harness (Phase 5a: deterministic scorers → markdown/JSON report in `eval/reports/`); Mode 2 needs `ANTHROPIC_API_KEY` (the report measures real-API consistency and latency/cost). The UI (`frontend/index.html`) ships as of Phase 6, so `make run` serves the API **and** the static page same-origin on `:8000` (the mount is still guarded — skipped if the directory is absent). The page's Mode-2 chat needs `ANTHROPIC_API_KEY` for the real LLM path; without it, `/ask` degrades to the deterministic fallback (a documented safety behavior, never a 500).
>
> **Pre-commit hooks.** `.pre-commit-config.yaml` sits at the **repo root** (not `backend/`) because it gates the whole tree; the runner and `ruff` are dev deps in `backend/pyproject.toml`, and ruff's config is `[tool.ruff]` there. The gate is two checks — ruff (lint + format) and gitleaks (secrets) — plus basic hygiene hooks. `make hooks` installs it; `make lint` runs it on demand. Bump pinned hook revs with `pre-commit autoupdate`. `E501` (line width) is intentionally not enforced — this repo's long documented comments are by design; the formatter still wraps code.
>
> **Deploy (Phase 8).** `render.yaml` at the **repo root** is the Render Blueprint — one `web` service, `uv sync` build, `uvicorn api:app` on `$PORT`, `/health` check. The lifespan runs `init_db` **+ seed-if-empty** on startup (the build can't see the runtime FS), so a fresh instance self-heals its schema and the 15 members. Shipped **free/ephemeral** (no disk; DB re-seeds on cold start; `POST /members` uploads + `feedback` don't persist); the durable paid-disk upgrade (`/data`, `HEALTH_DB_PATH`) is a commented block in `render.yaml` (§15). `ANTHROPIC_API_KEY` is a dashboard secret (`sync: false`), never committed. The DB path env var the code reads is `HEALTH_DB_PATH` (unset → `backend/data/health.db`).

## How to work here

- **Build in phase order.** `architecture.md`'s build sequence (Phases 0–8) is dependency-ordered; each phase leaves something runnable. Deterministic-first — the whole non-LLM system and every safety decision are built and tested before the LLM goes on top.
- **Keep the schema lean.** Do not add a column without data or a behavior that needs it. Phantom fields are cut on sight.
- **Verify the data contract on touch.** If you change `schema.sql` or `models.py`, re-check the three stay aligned — schema columns ↔ Pydantic fields ↔ the actual data — and that no phantom column crept in. The consistency checks exist for exactly this. After implementing, also reconcile any resulting schema or Pydantic-model drift in `docs/` (e.g. `docs/architecture.md`) so the documented contract matches the code.
- **Routes stay thin.** Every API route is a thin adapter over the `health_intelligence` library; logic lives in the library, not the route. `pipeline.py` is the orchestration seam where that logic lives — `scan` (proactive Mode 1), `suggestions` (reactive Mode 1), and the `ask`/`compose` path (Mode 2) each sequence load → analyze → narrate → persist, and **each runs `safety.validate` against the floor**: this is *where* the "deterministic floor is always on, both modes" invariant is enforced.
- **Statistics are classical, not ML** — Mann–Kendall, Theil–Sen, RCV, FDR. Keep the distinction precise; there is no trained model here.
- **Determinism in the core.** Temperature 0 for the LLM; the deterministic path must be byte-identical across re-runs.
- **Test the core in isolation** against the known-answer fixtures before wiring anything to it. At n = 3 the honest verdict is "too short to call a trend" — a correct output, not a failure.
- **No browser storage in the UI** — the static page holds state in memory only.
- **Docs are part of the change.** If a change alters what `README.md` or this file documents — a command, the route surface, setup, structure — update them in the same change. A doc that lies about the code is a bug.
- **Keep this file current.** CLAUDE.md does not update itself. After completing a task or plan, ask whether it changed a rule, command, contract, invariant, or structural fact this file describes — if so, edit CLAUDE.md (and `README.md` where relevant) in the same change. Only encode durable operating rules; do not log routine code changes, fixes, or anything already recoverable from the repo or git history.

## Source of truth

`docs/architecture.md` (design + rationale) · `backend/schema.sql` (persistence contract) · `docs/ui-ux.md` (member + operator surface). When in doubt, those win over this file.
