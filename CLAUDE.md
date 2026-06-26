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
- **Escalation ≠ out-of-range.** A value merely outside its reference range is at most a `notable` observation. Escalation (`clinician_review` / `urgent`) is reserved for panic thresholds and RCV+FDR-significant adverse trajectories. A managed condition's expected-high marker must not escalate.
- **Never fabricate.** No reading for a marker → the answer is "not measured", never an invented value. Every claim traces to `findings[].evidence[]`.
- **One normalization path.** All input enters through `preprocessing/ingest.py` (the firewall): it parses the supplied bundle's five reference-range shapes, folds vitals in as markers, and writes the internal domain model. Nothing downstream re-parses raw input. *Which* bundle it reads is resolved by `preprocessing/datasets.py` from the `DATASET` env var (default `training_data`) — each dataset is its own sub-folder of `backend/data/`, so new bundles ingest incrementally; the derived `health.db` sits at the `data/` root, not inside a bundle.
- **One DB module.** Only `db.py` touches SQLite. Clinical constants and thresholds live in `config.py` (versioned); `.env` is secrets + runtime only.
- **Both provider calls behind `llm.py`** — `compose()` and the gate classifier. These are the only network seams.
- **Two layers, not one.** The SQLite schema (`schema.sql`) and the Pydantic models (`models.py`) are deliberately separate — do not fuse them into ORM-as-model. The row↔model mapping lives in `db.py`.

## Structure

`ls` shows the tree; the file-by-file map is `architecture.md` §14 (the one canonical copy — don't duplicate it here). What `ls` won't tell you is intent, and that's in the invariants above: `health_intelligence/` is the serving library (pure, imports no web framework), `preprocessing/ingest.py` is the one firewall, `eval/` is the harness, and `api.py` is thin routes serving the single static `frontend/index.html`.

## Commands

Dependencies and the virtualenv are managed with **uv**. All commands run from `backend/` (where `pyproject.toml` lives — there is no root workspace).

```
uv sync          # install
make init-db     # create the SQLite schema
make seed        # ingest the supplied bundle
make run         # init + seed + serve API and UI on :8000
make eval        # run the evaluation harness → report
make test        # unit tests
```

> The Makefile is wired as of Phase 3a (`backend/Makefile`); `make help` lists the targets. `make eval` is a placeholder until the harness lands in Phase 5. The UI (`frontend/index.html`) lands in Phase 6 — until then `make run` serves the API only (the static mount is skipped when the directory is absent).

## How to work here

- **Build in phase order.** `architecture.md`'s build sequence (Phases 0–8) is dependency-ordered; each phase leaves something runnable. Deterministic-first — the whole non-LLM system and every safety decision are built and tested before the LLM goes on top.
- **Keep the schema lean.** Do not add a column without data or a behavior that needs it. Phantom fields are cut on sight.
- **Verify the data contract on touch.** If you change `schema.sql` or `models.py`, re-check the three stay aligned — schema columns ↔ Pydantic fields ↔ the actual data — and that no phantom column crept in. The consistency checks exist for exactly this. After implementing, also reconcile any resulting schema or Pydantic-model drift in `docs/` (e.g. `docs/architecture.md`) so the documented contract matches the code.
- **Routes stay thin.** Every API route is a thin adapter over the `health_intelligence` library; logic lives in the library, not the route.
- **Statistics are classical, not ML** — Mann–Kendall, Theil–Sen, RCV, FDR. Keep the distinction precise; there is no trained model here.
- **Determinism in the core.** Temperature 0 for the LLM; the deterministic path must be byte-identical across re-runs.
- **Test the core in isolation** against the known-answer fixtures before wiring anything to it. At n = 3 the honest verdict is "too short to call a trend" — a correct output, not a failure.
- **No browser storage in the UI** — the static page holds state in memory only.
- **Docs are part of the change.** If a change alters what `README.md` or this file documents — a command, the route surface, setup, structure — update them in the same change. A doc that lies about the code is a bug.
- **Keep this file current.** CLAUDE.md does not update itself. After completing a task or plan, ask whether it changed a rule, command, contract, invariant, or structural fact this file describes — if so, edit CLAUDE.md (and `README.md` where relevant) in the same change. Only encode durable operating rules; do not log routine code changes, fixes, or anything already recoverable from the repo or git history.

## Source of truth

`docs/architecture.md` (design + rationale) · `backend/schema.sql` (persistence contract) · `docs/ui-ux.md` (member + operator surface). When in doubt, those win over this file.
