"""api.py — the thin FastAPI layer. Routes are adapters; all logic lives in the library.

Phase 3a stands up the first vertical slice: the proactive-scan routes plus liveness; Phase 3b adds the
Mode-1 reactive ``GET /suggestions``. Every handler is a few lines over ``health_intelligence``
(architecture §13, CLAUDE.md "routes stay thin") — it opens a connection, calls one library function,
and returns its typed result. The route surface accretes per phase: ``/ask`` and the member CRUD land in
Phases 4/6.

One connection per request (a generator dependency that closes it) — SQLite connections are not safe to
share across FastAPI's threadpool, and per-request open is cheap (the store is a local file). The DB
path is ``HEALTH_DB_PATH`` when set (the Phase-8 Render disk + the test seam), else db.DEFAULT_DB_PATH.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from health_intelligence import db, learn, pipeline
from health_intelligence.models import (
    AskRequest,
    Escalation,
    Feedback,
    HealthIntelligenceResponse,
    MemberBundle,
    Observation,
    SuggestedPrompt,
)
from preprocessing.datasets import DEFAULT_DATASET
from preprocessing.ingest import ingest_bundle, ingest_dataset

# Load backend/.env into the process env at import (before any LLM provider is constructed) so the
# Mode-2 path sees ANTHROPIC_API_KEY however the app is launched. A no-op when no .env is present (the
# Render deploy sets real env vars), and secrets stay in .env, never in config.py.
load_dotenv()

_DB_PATH = os.environ.get("HEALTH_DB_PATH")  # None -> db.DEFAULT_DB_PATH
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def get_con() -> Iterator[sqlite3.Connection]:
    """Per-request connection (closed after the response). Override target via HEALTH_DB_PATH."""
    con = db.connect(_DB_PATH)
    try:
        yield con
    finally:
        con.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on the in-memory footgun: get_con opens a NEW connection per request, and each
    # sqlite3 ':memory:' connect is a separate empty database — the schema created here would be
    # invisible to every request. HEALTH_DB_PATH must be a file (the Render disk / a temp-file test seam).
    if _DB_PATH is not None and str(_DB_PATH) == ":memory:":
        raise RuntimeError(
            "HEALTH_DB_PATH=':memory:' is unsupported (one connection per request → a fresh empty "
            "in-memory DB each time). Use a file path; tests share one connection via db.connect directly."
        )
    # Ensure the schema exists before serving (idempotent; Phase 8 relies on startup-time init because
    # Render's build step can't see the persistent disk). Single-instance deploy (architecture §15), so
    # no cross-process init race.
    con = db.connect(_DB_PATH)
    try:
        db.init_db(con)
    finally:
        con.close()
    yield


app = FastAPI(title="Health Intelligence Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness."""
    return {"status": "ok"}


@app.get("/members")
def get_members(con: sqlite3.Connection = Depends(get_con)) -> list[dict]:
    """List members for the picker — ``[{member_id, age, sex}]`` (seeded + uploaded). The only route
    the member-picker needs to enumerate what's loadable; the UI re-reads it after an upload/clear."""
    return db.list_member_summaries(con)


@app.post("/members")
def post_member(
    bundle: MemberBundle, con: sqlite3.Connection = Depends(get_con)
) -> dict:
    """Ingest/upsert one member bundle (Seed / Upload bundle, the holdout swap). The upload doubles as
    the format check (ui-ux §2): FastAPI validates the body against ``MemberBundle`` first (a SHAPE error
    → 422 before this runs), and the ``except`` below turns the firewall's SEMANTIC ``ValueError``s — an
    unparseable reference-range, a marker/vital name collision, a missing vital unit, a divergent shared
    range — into a 422 too, so a malformed upload always lands a clear cause in the operator readout
    rather than an opaque 500. Re-POSTing an id refreshes that member's facts, preserves the audit/learning
    rows, and bumps ``data_version`` (``ingest_bundle`` -> ``replace_member``)."""
    try:
        return ingest_bundle(con, bundle)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.delete("/members/{member_id}")
def delete_member(
    member_id: str, con: sqlite3.Connection = Depends(get_con)
) -> dict[str, bool]:
    """Explicitly clear one member and everything that hangs off them (opt-in; ingest never clears by
    default). 404 if the member isn't present, so the destructive op can't silently no-op a typo."""
    if not db.delete_member(con, member_id):
        raise HTTPException(status_code=404, detail=f"member {member_id!r} not found")
    return {"deleted": True}


@app.post("/members/{member_id}/scan", response_model=list[Observation])
def post_scan(
    member_id: str, con: sqlite3.Connection = Depends(get_con)
) -> list[Observation]:
    """Run the proactive scan now; returns the member's current observations (ranked by severity)."""
    try:
        return pipeline.scan(con, member_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"member {member_id!r} not found"
        ) from None


@app.get("/members/{member_id}/observations", response_model=list[Observation])
def get_observations(
    member_id: str, con: sqlite3.Connection = Depends(get_con)
) -> list[Observation]:
    """Read the member's current observations (the last scan's findings at the current data_version)."""
    try:
        return db.get_observations(con, member_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"member {member_id!r} not found"
        ) from None


@app.get("/members/{member_id}/escalations", response_model=list[Escalation])
def get_escalations(
    member_id: str, con: sqlite3.Connection = Depends(get_con)
) -> list[Escalation]:
    """Read the member's clinician-review queue (the standing hand-off artifacts)."""
    # Existence check so an unknown member 404s like /scan and /observations — db.get_escalations would
    # otherwise return [] for a typo'd id, and an empty review queue reads as 'all clear' on a safety
    # surface (it must not be confused with 'member not found').
    if db.get_member(con, member_id) is None:
        raise HTTPException(status_code=404, detail=f"member {member_id!r} not found")
    return db.get_escalations(con, member_id)


@app.get("/members/{member_id}/suggestions", response_model=list[SuggestedPrompt])
def get_suggestions(
    member_id: str,
    focus: str | None = None,
    asked: list[str] = Query(default=[]),
    con: sqlite3.Connection = Depends(get_con),
) -> list[SuggestedPrompt]:
    """Mode 1: data-derived preset prompts, each bound to its pre-computed answer (no LLM). ``focus``
    (a marker just opened) and ``asked`` (markers already visited) drive the conversation loop — the
    next chips after each answer. A pure read; re-fetching the same turn is byte-identical."""
    try:
        return pipeline.suggestions(con, member_id, focus=focus, asked=tuple(asked))
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"member {member_id!r} not found"
        ) from None


@app.post("/members/{member_id}/ask", response_model=HealthIntelligenceResponse)
def post_ask(
    member_id: str,
    req: AskRequest,
    con: sqlite3.Connection = Depends(get_con),
) -> HealthIntelligenceResponse:
    """Mode 2: one grounded answer over the member's own data — the per-turn pipeline (gate -> floor ->
    compose/template -> validate -> escalate). Uses the default LLM provider; degrades to the
    deterministic spine if it is unavailable, so this never 500s on a missing key. 404 if absent."""
    try:
        return pipeline.ask(con, member_id, req.message)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"member {member_id!r} not found"
        ) from None


# --------------------------------------------------------------------------------------------------
# Phase 7 — self-improvement + inspection. Still thin adapters: the override/reset/promotion LOGIC lives
# in the library (db.py resolves corrections into analysis inputs; learn.py rule-assembles + gates a
# candidate prompt), and the safety boundary (floor, validator, gate) is untouched (architecture §9).
# --------------------------------------------------------------------------------------------------


@app.post("/members/{member_id}/feedback")
def post_feedback(
    member_id: str, fb: Feedback, con: sqlite3.Connection = Depends(get_con)
) -> dict[str, str]:
    """Record a correction (range_override / suppress_marker / preference) or a signal (helpful /
    incorrect / escalation_accept|reject). Overrides re-resolve into the core's inputs on the next
    scan/ask (both modes); signals feed ``/learn``. 404 if the member is absent."""
    if db.get_member(con, member_id) is None:
        raise HTTPException(status_code=404, detail=f"member {member_id!r} not found")
    return {"feedback_id": db.insert_feedback(con, member_id, fb)}


@app.get("/members/{member_id}/trajectory")
def get_trajectory(
    member_id: str,
    marker: str | None = None,
    con: sqlite3.Connection = Depends(get_con),
) -> list[dict]:
    """Full per-marker series for inspection/plotting (architecture §13/§751): readings + the analysis
    pass's trend/flags/Theil–Sen line + reference range. A UI/operator read for charting + human
    verification — the LLM NEVER receives the raw series, only the collapsed verdict. Optional
    ``?marker=`` narrows to one. 404 if the member is absent."""
    try:
        return pipeline.trajectory(con, member_id, marker=marker)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"member {member_id!r} not found"
        ) from None


@app.post("/reset")
def post_reset(con: sqlite3.Connection = Depends(get_con)) -> dict:
    """Erase learning -> v0 (architecture §9): deactivate all feedback and revert promoted prompts to the
    baseline. NOT a data wipe — the dataset, members, and the learning trail are preserved (that is
    ``/admin/reseed``)."""
    counts = db.reset_learning(con)
    return {"learning_reset": True, **counts}


@app.post("/learn")
def post_learn(con: sqlite3.Connection = Depends(get_con)) -> dict:
    """Prompt-scan: rule-assemble a candidate composer prompt from the active feedback signals, gate it
    through the eval harness IN-PROCESS, and promote or reject (architecture §9). Guarded against credit
    drain (single-flight lock · feedback-set debounce · structural pre-check · daily cap). 409 if a run
    is already in progress; 503 if no working LLM (it measures the candidate's real outputs, so it can't
    meaningfully gate a degraded run)."""
    try:
        return learn.run_learn(con)
    except learn.LearnBusy as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except learn.LearnUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/admin/reseed")
def post_reseed(con: sqlite3.Connection = Depends(get_con)) -> dict:
    """Factory reset (demo hygiene; distinct from ``/reset``): flush the whole app back to its initial
    state — a full NUKE (DROP + recreate every table via ``db.nuke_all``), then re-ingest the shipped
    ``training_data`` bundle (architecture §756: "the training_data bundle only"). Pinned to
    ``DEFAULT_DATASET`` deliberately, NOT the active ``DATASET`` env, so a factory reset always restores
    the original 15 members regardless of which bundle is selected; uploaded holdouts, feedback, learning,
    and the audit trail are all dropped, and the prompt reverts to the v0 baseline (empty
    ``prompt_versions`` → composer falls back to BASE). An operator op, not a member feature."""
    db.nuke_all(con)
    summary = ingest_dataset(con, DEFAULT_DATASET)
    return {"reseeded": True, **summary}


# Serve the static member surface on the same origin (Phase 6 ships frontend/index.html). Mounted LAST
# so the API routes above take precedence, and only when the directory exists — it doesn't until Phase 6,
# and an unconditional mount would crash startup (RuntimeError: directory does not exist).
if _FRONTEND_DIR.is_dir():
    app.mount(
        "/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend"
    )
