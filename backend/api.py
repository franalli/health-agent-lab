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

from health_intelligence import db, pipeline
from health_intelligence.models import (
    AskRequest,
    Escalation,
    HealthIntelligenceResponse,
    Observation,
    SuggestedPrompt,
)

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


# Serve the static member surface on the same origin (Phase 6 ships frontend/index.html). Mounted LAST
# so the API routes above take precedence, and only when the directory exists — it doesn't until Phase 6,
# and an unconditional mount would crash startup (RuntimeError: directory does not exist).
if _FRONTEND_DIR.is_dir():
    app.mount(
        "/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend"
    )
