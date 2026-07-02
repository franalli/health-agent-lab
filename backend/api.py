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

import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

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
from preprocessing.ingest import (
    BundleValidationError,
    ingest_bundle,
    ingest_dataset,
    ingest_uploaded_dataset,
    seed_if_empty,
)

# Load backend/.env into the process env at import (before any LLM provider is constructed) so the
# Mode-2 path sees ANTHROPIC_API_KEY however the app is launched. A no-op when no .env is present (the
# Render deploy sets real env vars), and secrets stay in .env, never in config.py.
load_dotenv()

_DB_PATH = os.environ.get("HEALTH_DB_PATH")  # None -> db.DEFAULT_DB_PATH
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
logger = logging.getLogger(
    __name__
)  # propagates to uvicorn's handlers → visible in the deploy logs


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
        db.init_db(
            con
        )  # FATAL on failure: the app cannot serve a request without a schema
        # Seed-if-empty: a fresh container/disk boots with an empty DB (the build can't see the runtime
        # FS — the same reason init_db runs here). `seed_if_empty` is a no-op locally (make seed ran
        # first) and on any restart where data persisted; on the free/ephemeral tier it self-heals the
        # 15 training members on every cold start (active DATASET, matching `make seed`). It is NON-fatal
        # and logged: a seed failure rolls back to empty (next restart retries) and leaves the
        # deterministic spine (Mode 1, /health) up, rather than aborting the whole app on a bad dataset.
        try:
            outcome = seed_if_empty(con)
        except Exception:
            logger.exception(
                "startup seed failed; serving with the current DB (the next restart retries)"
            )
        else:
            if outcome["seeded"]:
                logger.info(
                    "startup: seeded %d members from the active dataset",
                    outcome["members"],
                )
        # Materialize the v0 composer baseline so prompt_versions is never empty (the operator UI shows
        # the active baseline; /learn back-fills its eval report lazily). Idempotent — a no-op on a warm
        # restart, and a one-time back-fill for DBs seeded before this landed. NON-fatal like the seed:
        # a failure here leaves the pipeline's implicit BASE fallback intact, so the app still serves.
        try:
            learn.seed_baseline_prompt(con)
        except Exception:
            logger.exception(
                "startup: failed to seed the v0 baseline prompt (composer falls back to BASE)"
            )
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


@app.post("/members/upload")
def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    con: sqlite3.Connection = Depends(get_con),
) -> dict:
    """Upload a whole dataset bundle, ingest its members, and persist it as a new dataset folder —
    the control panel's **Upload bundle**. The multi-member, persisted sibling of ``POST /members``.

    BODY: ``multipart/form-data`` with ``file`` (required) and an optional ``name`` (the new dataset
    folder's name; defaults to the uploaded file's stem). ``file`` MUST be a **.zip** of a folder shaped
    like ``backend/data/training_data/`` carrying exactly three files **by extension** — one ``.json``
    (members), one ``.jsonl`` (eval set), one ``.csv`` (lab panels) — under any base names (a hold-out
    ships stable extensions, only the base names differ), plus optional extras (e.g. a ``README.md``).
    They're written to disk under their **canonical** names, so the rest of the system stays name-based.

    EFFECT (see ``ingest_uploaded_dataset``): (1) every member is ingested **additively** — added on top
    of the existing members (``member_id`` opaque, any ID space), a re-used id refreshing that member's
    facts, never a reseed/truncate; AND (2) the **entire** uploaded bundle is written to a new
    ``<data-root>/<name>/`` folder, so the kept ``eval_set.jsonl`` is available for later live use
    (``DATASET=<name> make eval``). Then each newly ingested member is **auto-scanned** so its
    Observations match its live Trajectory immediately (reseed deliberately does NOT auto-scan).

    This doubles as the format check, surfaced at file+row granularity: a missing or duplicated required
    file, or a FIRST-RECORD format failure in any of the three files (member bundle / eval case / CSV panel
    row), returns **422** with ``detail = {"error", "failures": [{file, row, field, detail}]}`` — the
    problems found across all three first-record gates at once, before any side effect (no partial load). It
    is a first-record gate, NOT a whole-file scan: later rows are not rejected here (a buggy later member row
    is skipped at ingest, later eval/CSV rows ride through unchecked). A bad zip or a non-zip upload is
    likewise a **422**; a name that collides with an existing dataset (``FileExistsError``) -> **409**
    (uploads never overwrite). A buggy *later* member row is skipped (not fatal), but a bundle in which EVERY
    member row is skipped ingests nobody — that is a failed upload, so it **422**s with the per-row skip
    detail and the reserved folder is rolled back (never a 200 with an empty dataset). The
    semantic-partial-set caveat of ``ingest_members`` still applies to a firewall parse error the row check
    can't pre-empt. Returns
    ``{dataset, members, results, ranges, member_ids, files, scanned}``. NOTE: on the free/ephemeral
    Render tier both the DB rows and the new folder live only until the next cold start (the durable
    paid-disk upgrade persists them — point ``HEALTH_DB_PATH`` and ``HEALTH_DATA_ROOT`` at the disk; §15)."""
    # SYNC route on purpose: the whole pipeline below (zip decompression, first-record bundle gating, disk
    # writes, per-member DB ingest, and the per-member auto-scan) is synchronous CPU/IO work. As `async def`
    # it ran inline on the event loop and stalled every concurrent request — including member `/ask` and the
    # `GET /health` Render polls. A plain `def` makes Starlette run it in the threadpool, like every other
    # route here; `file.file.read()` is the sync read of the SpooledTemporaryFile (no `await` needed).
    data = file.file.read()
    try:
        result = ingest_uploaded_dataset(
            con, data=data, filename=file.filename, name=name
        )
    except FileExistsError as e:
        # name collides with an existing dataset/file — caught BEFORE the generic OSError below (it is an
        # OSError subclass); raised by create_dataset_dir's mkdir BEFORE any DB write, so no side effects.
        raise HTTPException(status_code=409, detail=str(e)) from e
    except BundleValidationError as e:
        # Row-level format failures — matched BEFORE the generic ValueError below (it is a subclass) so the
        # structured per-row failures ride the wire, not a flattened string. FastAPI serializes the dict.
        raise HTTPException(
            status_code=422, detail={"error": str(e), "failures": e.failures}
        ) from e
    except (ValueError, ValidationError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except OSError as e:
        # disk full / unwritable HEALTH_DATA_ROOT etc. — a genuine server/infra error (not bad input), but
        # the folder write happens BEFORE the DB ingest, so no members were committed. Surface a clear 500.
        raise HTTPException(
            status_code=500, detail=f"failed to persist the uploaded dataset: {e}"
        ) from e

    # Auto-scan the just-ingested members so their stored Observations match their live Trajectory the
    # moment the data source lands (best-effort, in the library; reseed deliberately does NOT auto-scan).
    scanned = pipeline.scan_members(con, result["member_ids"])
    return {**result, "scanned": scanned}


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
    """Read the member's current observations (the last scan's findings at the current data_version),
    each with its member-facing ``member_explanation`` derived at read by ``pipeline.observations``."""
    try:
        return pipeline.observations(con, member_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"member {member_id!r} not found"
        ) from None


@app.get("/escalations", response_model=list[Escalation])
def get_all_escalations(
    con: sqlite3.Connection = Depends(get_con),
) -> list[Escalation]:
    """The GLOBAL clinician-review queue across all members — the triage worklist, ranked worst-first
    (urgent before clinician_review), then most recent. This is *the* hand-off surface: escalation means
    "make sure a human sees this", which a per-member read can't guarantee (it requires already knowing
    which patient to open). The per-member route below is the drill-in, not the queue. Scope: "all
    members" is the whole tenant in this prototype (no clinician identity); production would scope it to a
    clinician's panel (db.get_all_escalations · §15)."""
    return db.get_all_escalations(con)


@app.get("/members/{member_id}/escalations", response_model=list[Escalation])
def get_escalations(
    member_id: str, con: sqlite3.Connection = Depends(get_con)
) -> list[Escalation]:
    """Per-member DRILL-IN: one member's standing escalation record, shown when already viewing that
    member. NOT the triage queue — that is the global ``GET /escalations`` above."""
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
        return pipeline.ask(con, member_id, req.message, history=req.history)
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
    scan/ask (both modes); signals feed ``/learn``. 404 if the member is absent. For an ``incorrect``
    correction, the learning input bar (``learn.validate_feedback``) screens the corrected answer so junk
    never sits in the table as a future few-shot exemplar (the "learns too literally" fix): **422** if it
    is unfit (empty / oversized, or the Haiku judge finds a stated numeric cutoff / escalation-softening /
    incoherence), **503 fail-closed** if the judge can't run (retry — an unjudged answer is never stored)."""
    if db.get_member(con, member_id) is None:
        raise HTTPException(status_code=404, detail=f"member {member_id!r} not found")
    try:
        reason = learn.validate_feedback(fb)
    except learn.LearnUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    if reason is not None:
        raise HTTPException(status_code=422, detail=reason)
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
    ``/admin/reseed``).

    RE-SCANS the members whose overrides it just deactivated (``db.members_with_active_overrides``, read
    BEFORE the reset), so the persisted observations + clinician queue ACTIVELY revert — a superseded
    escalation re-opens, a pruned observation returns — instead of lagging until someone re-scans by hand
    (the symmetric counterpart to the feedback UI's 'Run Scan' nudge; the reconcile is reachable only from a
    scan). Best-effort like the upload auto-scan: a member's scan failing is logged, not fatal to the reset."""
    affected = db.members_with_active_overrides(con)  # BEFORE reset clears them
    counts = db.reset_learning(con)
    rescanned = pipeline.scan_members(
        con, affected
    )  # re-open / restore the reverted artifacts
    return {"learning_reset": True, "rescanned": rescanned, **counts}


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
    state — TRUNCATE every table, then re-ingest the shipped ``training_data`` bundle (architecture §756:
    "the training_data bundle only"). Pinned to ``DEFAULT_DATASET`` deliberately, NOT the active ``DATASET``
    env, so a factory reset always restores the original 15 members regardless of which bundle is selected;
    uploaded holdouts, feedback, learning, and the audit trail are all dropped, and the prompt reverts to
    the v0 baseline — re-materialized as a single promoted ``prompt_versions`` row carrying
    ``BASE_COMPOSE_SYSTEM`` (so the store is never empty; its eval report is attached lazily by the first
    ``/learn``). An operator op, not a member feature.

    ATOMIC: the truncate + re-ingest + v0 seed run as ONE transaction (``db.reseed_transaction``, with the
    ingest/seed passing ``commit=False``), so a concurrent request on another threadpool connection reads
    the pre-reseed or post-reseed state, never a half-wiped DB, and any mid-reseed failure rolls back to
    the prior populated state rather than leaving the DB empty."""
    with db.reseed_transaction(con):
        db.clear_all_data(
            con
        )  # truncate every table (FK-off inside the transaction, so order is free)
        summary = ingest_dataset(con, DEFAULT_DATASET, commit=False)
        learn.seed_baseline_prompt(
            con, commit=False
        )  # v0 baseline, so the prompt store isn't empty
    return {"reseeded": True, **summary}


# Serve the static member surface on the same origin (Phase 6 ships frontend/index.html). Mounted LAST
# so the API routes above take precedence, and only when the directory exists — it doesn't until Phase 6,
# and an unconditional mount would crash startup (RuntimeError: directory does not exist).
if _FRONTEND_DIR.is_dir():
    app.mount(
        "/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend"
    )
