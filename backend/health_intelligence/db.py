"""The single SQLite seam — the only module that touches the store.

Everything the serving core needs from persistence enters and leaves here: opening a connection
with the right invariants, loading the schema, the per-member reads that feed ``analysis.analyze``,
the deterministic ``data_version`` fingerprint, the idempotent escalation write, and the
feedback-override resolution that keeps ``analysis.py`` pure. The row<->model mapping is hand-written
(architecture §4: persistence and domain are *two layers, not one* — no ORM): tables carry JSON
columns and storage PKs the Pydantic models deliberately omit, and ``db.py`` is where the two meet.

Why no ORM (architecture §15, CLAUDE.md): the load-bearing types are *computed*, not stored, so an
ORM would map a divergence it can't help with; portability is kept as a cheap seam instead — this is
the one module touching the store, and the only dialect-specific statement (``INSERT OR IGNORE``)
maps directly to Postgres ``ON CONFLICT DO NOTHING``. All SQL is parameterized (``?`` placeholders,
never string-formatted values) — the discipline that replaces what an ORM would enforce.

Purity boundary: this module imports ``config`` and ``models`` but NOT ``analysis`` — the data layer
feeds the pure core, never the reverse. ``analysis.py`` never reaches the DB (a test enforces it).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from health_intelligence.config import CONFIG_VERSION
from health_intelligence.models import (
    Escalation,
    EscalationKind,
    EscalationLevel,
    Feedback,
    HealthIntelligenceResponse,
    LabResult,
    MemberProfile,
    Note,
    Observation,
    ReferenceRange,
    SEVERITY_ORDER,
)

# --------------------------------------------------------------------------------------------------
# Paths — the derived store sits at the data/ root, NOT inside a dataset bundle (CLAUDE.md). Resolved
# from __file__ rather than imported from preprocessing.datasets, to avoid a health_intelligence ->
# preprocessing back-edge; both independently resolve to backend/data.
# --------------------------------------------------------------------------------------------------

_BACKEND = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = _BACKEND / "data" / "health.db"
SCHEMA_PATH = _BACKEND / "schema.sql"

#: The nine tables schema.sql defines; init_db is a no-op once they all exist.
_EXPECTED_TABLES = {
    "members", "lab_results", "notes", "reference_ranges", "interactions",
    "observations", "escalations", "feedback", "prompt_versions",
}


# --------------------------------------------------------------------------------------------------
# Deterministic storage-id synthesis — the one place the store's surrogate keys are minted (CLAUDE.md
# "only db.py touches SQLite"; architecture §4 db.py synthesizes storage PKs). One scheme for every
# deterministic id (escalation_id, scan response_id, observation_id): prefix + sha256(parts)[:32].
# 32 hex (128 bits) is at least as collision-resistant as the UNIQUE/PK it backs.
# --------------------------------------------------------------------------------------------------

def _det_id(prefix: str, *parts: str) -> str:
    return prefix + hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()[:32]


def _canon_results(results: list[LabResult]) -> list[list]:
    """The canonical, order-independent results projection both content hashes share. (date, panel_id,
    marker) uniquely identify a row, so the sort never reaches the float value field — the documented
    'JSON ints become REAL on read' stability. Defined once so the two hashers can't drift apart."""
    return sorted([[r.panel_date, r.panel_id, r.marker, r.value, r.unit] for r in results])


def _hash_canon(canon: dict) -> str:
    """The shared serialize+digest tail: compact, key-sorted JSON → 16 hex of sha256. The one place the
    serialization convention lives, so compute_data_version and compute_analysis_version stay in lockstep."""
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------------------------------
# Connection + schema
# --------------------------------------------------------------------------------------------------

def connect(db_path=None) -> sqlite3.Connection:
    """Open a connection with the invariants every caller needs.

    ``db_path=None`` -> :data:`DEFAULT_DB_PATH`; pass ``":memory:"`` in tests. Sets
    ``row_factory = sqlite3.Row`` (name-addressable rows) and — CRITICALLY — ``PRAGMA foreign_keys
    = ON`` *per connection* (schema.sql's pragma does not persist across connections in SQLite, so a
    caller that opens the DB any other way silently loses FK enforcement). All access goes through
    here.
    """
    path = DEFAULT_DB_PATH if db_path is None else db_path
    if str(path) != ":memory:":
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI runs sync routes + their generator dependency (get_con) across
    # anyio's threadpool, and the connection can be opened on one worker thread and used on another
    # within a single request — the default (True) raises ProgrammingError under concurrency. Safe here
    # because connections are never SHARED across requests (one per request, closed after) and SQLite
    # serializes writes itself; we only relax the per-thread-affinity assertion, not isolation.
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(con: sqlite3.Connection, schema_path: pathlib.Path = SCHEMA_PATH) -> None:
    """Idempotently ensure the nine tables exist. schema.sql uses bare ``CREATE TABLE`` (no
    ``IF NOT EXISTS``), so guard on presence: if all nine are already there, no-op; else run the
    script. schema.sql is the locked persistence contract — never edited here. Safe to call on
    every startup (Phase 8 relies on this)."""
    existing = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    present = _EXPECTED_TABLES & existing
    if present == _EXPECTED_TABLES:
        return
    if present:
        # A prior executescript was interrupted (disk full / killed) after some of the bare
        # CREATE TABLEs ran; re-running would fail opaquely on "table already exists". Fail clearly.
        raise RuntimeError(
            f"partial schema: {sorted(present)} present, {sorted(_EXPECTED_TABLES - present)} "
            f"missing — delete the database file and re-init"
        )
    con.executescript(pathlib.Path(schema_path).read_text())
    con.commit()


# --------------------------------------------------------------------------------------------------
# Row <-> model mapping (the explicit two-layer seam). Storage PKs (result_id/range_id/note_id) are
# synthesized on write (replace_member) and dropped on read — the domain models omit them by design.
# --------------------------------------------------------------------------------------------------

def _row_to_profile(row: sqlite3.Row) -> MemberProfile:
    return MemberProfile(
        member_id=row["member_id"],
        sex=row["sex"],
        age=row["age"],
        conditions=json.loads(row["conditions_json"] or "[]"),
        medications=json.loads(row["medications_json"] or "[]"),
        family_history=json.loads(row["family_history_json"] or "[]"),
        lifestyle=json.loads(row["lifestyle_json"] or "{}"),
    )


def _profile_to_params(p: MemberProfile) -> tuple:
    return (
        p.member_id, p.sex, p.age,
        json.dumps(p.conditions), json.dumps(p.medications),
        json.dumps(p.family_history), json.dumps(p.lifestyle),
    )


# --------------------------------------------------------------------------------------------------
# Reads — what feeds analyze() (member/results/ranges/notes). The proactive-scan writes and the
# observation/escalation read projections live further down (Phase 3a).
# --------------------------------------------------------------------------------------------------

def get_member(con: sqlite3.Connection, member_id: str) -> Optional[MemberProfile]:
    row = con.execute("SELECT * FROM members WHERE member_id = ?", (member_id,)).fetchone()
    return _row_to_profile(row) if row else None


def get_results(con: sqlite3.Connection, member_id: str) -> list[LabResult]:
    rows = con.execute(
        "SELECT marker, value, unit, panel_id, panel_date FROM lab_results "
        "WHERE member_id = ? ORDER BY panel_date, marker, panel_id",
        (member_id,),
    ).fetchall()
    return [
        LabResult(marker=r["marker"], value=r["value"], unit=r["unit"],
                  panel_id=r["panel_id"], panel_date=r["panel_date"])
        for r in rows
    ]


def get_ranges(
    con: sqlite3.Connection,
    markers: Optional[set[str]] = None,
    config_version: str = CONFIG_VERSION,
) -> list[ReferenceRange]:
    """Reference ranges for the active ``config_version``. The range_id embeds the version, so a
    future config bump *adds* a generation rather than overwriting — the read must scope to one
    version or ``_range_for`` would see duplicate (marker, sex) rows across generations. ``markers``
    None returns all; when scoped, returns ALL sexes for those markers so the sex selection + 'any'
    fallback still resolve."""
    sql = ("SELECT marker, sex, unit, ref_low, ref_high, panic_low, panic_high, config_version "
           "FROM reference_ranges WHERE config_version = ?")
    params: list = [config_version]
    if markers is not None:
        marker_list = list(markers)
        if not marker_list:
            return []
        sql += " AND marker IN (%s)" % ",".join("?" * len(marker_list))
        params += marker_list
    rows = con.execute(sql, params).fetchall()
    return [
        ReferenceRange(
            marker=r["marker"], sex=r["sex"], unit=r["unit"],
            ref_low=r["ref_low"], ref_high=r["ref_high"],
            panic_low=r["panic_low"], panic_high=r["panic_high"],
            config_version=r["config_version"],
        )
        for r in rows
    ]


def get_notes(con: sqlite3.Connection, member_id: str) -> list[Note]:
    rows = con.execute(
        # ORDER BY rowid = insertion (bundle) order; note_id is `{member}:note:{i}`, whose TEXT sort
        # is lexicographic (`note:10` < `note:2`), so it would misorder a member with >=10 notes.
        "SELECT note_date, source, text FROM notes WHERE member_id = ? ORDER BY rowid",
        (member_id,),
    ).fetchall()
    return [Note(date=r["note_date"], source=r["source"], text=r["text"]) for r in rows]


def list_members(con: sqlite3.Connection) -> list[str]:
    return [r["member_id"] for r in con.execute("SELECT member_id FROM members ORDER BY member_id")]


# --------------------------------------------------------------------------------------------------
# data_version — a deterministic content fingerprint of the member's PERSISTED record. Stable across
# identical re-ingests (same data -> same hash) and changes iff the data changed: that is the "bump".
# Not a clock and not a counter — either would mint a new dedup_key on identical input and re-fire an
# escalation, breaking the "fire once" guarantee (architecture §7). Hashes the persisted rows, not
# the raw bundle, because JSON ints (systolic_bp: 130) become REAL 130.0 on read.
# --------------------------------------------------------------------------------------------------

def compute_data_version(
    con: sqlite3.Connection,
    member_id: str,
    *,
    member: Optional[MemberProfile] = None,
    results: Optional[list[LabResult]] = None,
    notes: Optional[list[Note]] = None,
) -> str:
    # Accept already-loaded rows so load_for_analysis doesn't re-query member/results just to hash
    # them; default to fetching when called standalone. Must be the RAW persisted rows (pre-override).
    if member is None:
        member = get_member(con, member_id)
    if member is None:
        raise KeyError(member_id)
    if results is None:
        results = get_results(con, member_id)
    if notes is None:
        notes = get_notes(con, member_id)
    canon = {
        "member_id": member_id,
        "profile": {
            "sex": member.sex,
            "age": member.age,
            "conditions": member.conditions,          # stored input order (round-trips stably)
            "medications": member.medications,
            "family_history": member.family_history,
            "lifestyle": dict(sorted(member.lifestyle.items())),
        },
        # _canon_results makes the fingerprint order-independent of SQLite's row order.
        "results": _canon_results(results),
        "notes": sorted([[n.date or "", n.source or "", n.text] for n in notes]),
    }
    return _hash_canon(canon)


def compute_analysis_version(
    *,
    results: list[LabResult],
    ranges: list[ReferenceRange],
    sex: str,
    age: Optional[int],
) -> str:
    """A content fingerprint of *only* the inputs ``analysis.analyze`` actually reads — the
    override-resolved results, the reference ranges, and the member's sex/age. Pure (no DB).

    This is the §48 "finding-stable dedup_key" resolution: ``data_version`` hashes the WHOLE record
    (incl. notes + conditions/medications), but ``analyze`` reads none of those, so keying a
    ``data_finding`` escalation on raw ``data_version`` would mint a new key — and re-fire the
    escalation — on a notes-only edit that never moved the analysis. The data-finding dedup_key keys on
    THIS hash instead (``data:{member}:{marker}:{analysis_version}``), so "fire once" tracks the actual
    finding; ``data_version`` stays the full audit snapshot everywhere else (interactions/observations).
    Mirrors :func:`compute_data_version`'s canonicalization so the two are read the same way."""
    canon = {
        "sex": sex,
        "age": age,
        "results": _canon_results(results),
        "ranges": sorted(
            [[rg.marker, rg.sex, rg.unit, rg.ref_low, rg.ref_high, rg.panic_low, rg.panic_high,
              rg.config_version] for rg in ranges]
        ),
    }
    return _hash_canon(canon)


# --------------------------------------------------------------------------------------------------
# Write — the one transactional member upsert. db.py owns all SQL (CLAUDE.md), so ingest normalizes
# and hands domain objects here; this function synthesizes the storage PKs and persists them.
# --------------------------------------------------------------------------------------------------

def _assert_ranges_consistent(con: sqlite3.Connection, range_rows: list[tuple]) -> None:
    """Guard the global reference_ranges table: it is constant per (marker, sex, config_version), so a
    member printing a DIVERGENT range for a marker another member already wrote would silently clobber
    the shared row (``ON CONFLICT DO UPDATE`` = last-write-wins) and corrupt the other member's
    analysis. Same definition (the normal cross-member / re-ingest case) passes untouched; a genuine
    divergence raises a clear error instead of corrupting silently."""
    by_id = {row[0]: row for row in range_rows}
    if not by_id:
        return
    placeholders = ",".join("?" * len(by_id))
    stored = con.execute(
        f"SELECT range_id, marker, sex, unit, ref_low, ref_high, panic_low, panic_high, config_version "
        f"FROM reference_ranges WHERE range_id IN ({placeholders})",
        list(by_id),
    ).fetchall()
    for s in stored:
        incoming = by_id[s["range_id"]]
        if tuple(s)[1:] != incoming[1:]:  # compare every column but the range_id key (index 0)
            raise ValueError(
                f"reference range for {s['marker']}/{s['sex']} diverges from the stored definition "
                f"(stored ref={s['ref_low']}–{s['ref_high']} vs incoming {incoming[4]}–{incoming[5]}); "
                f"reference_ranges must be constant per marker+sex+config_version"
            )


def replace_member(
    con: sqlite3.Connection,
    *,
    profile: MemberProfile,
    results: list[LabResult],
    ranges: list[ReferenceRange],
    notes: list[Note],
) -> None:
    """Persist one member's normalized data, replacing any prior version in a single transaction.

    UPSERT the ``members`` row (never DELETE/REPLACE it): with FKs ON, a REPLACE would delete the
    parent first and FK-fail surviving children — or, with cascade, silently wipe clinician
    ``feedback`` and the escalation audit trail. UPSERT updates the profile in place, so
    ``interactions``/``observations``/``escalations``/``feedback`` survive a re-ingest. The owned
    children (``lab_results``/``notes``) are deleted then reinserted with deterministic PKs;
    ``reference_ranges`` are global (identical across members) and upserted by ``range_id``.
    """
    mid = profile.member_id
    with con:  # atomic: commit on success, rollback on any exception
        con.execute("DELETE FROM lab_results WHERE member_id = ?", (mid,))
        con.execute("DELETE FROM notes WHERE member_id = ?", (mid,))
        con.execute(
            "INSERT INTO members "
            "(member_id, sex, age, conditions_json, medications_json, family_history_json, lifestyle_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(member_id) DO UPDATE SET "
            "sex=excluded.sex, age=excluded.age, conditions_json=excluded.conditions_json, "
            "medications_json=excluded.medications_json, family_history_json=excluded.family_history_json, "
            "lifestyle_json=excluded.lifestyle_json",
            _profile_to_params(profile),
        )
        con.executemany(
            "INSERT INTO lab_results (result_id, member_id, panel_id, marker, value, unit, panel_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(f"{mid}:{r.panel_id}:{r.marker}", mid, r.panel_id, r.marker, r.value, r.unit, r.panel_date)
             for r in results],
        )
        con.executemany(
            "INSERT INTO notes (note_id, member_id, note_date, source, text) VALUES (?, ?, ?, ?, ?)",
            [(f"{mid}:note:{i}", mid, n.date, n.source, n.text) for i, n in enumerate(notes)],
        )
        range_rows = [
            (f"range:{rg.config_version}:{rg.sex}:{rg.marker}", rg.marker, rg.sex, rg.unit,
             rg.ref_low, rg.ref_high, rg.panic_low, rg.panic_high, rg.config_version)
            for rg in ranges
        ]
        _assert_ranges_consistent(con, range_rows)
        con.executemany(
            "INSERT INTO reference_ranges "
            "(range_id, marker, sex, unit, ref_low, ref_high, panic_low, panic_high, config_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(range_id) DO UPDATE SET "
            "unit=excluded.unit, ref_low=excluded.ref_low, ref_high=excluded.ref_high, "
            "panic_low=excluded.panic_low, panic_high=excluded.panic_high, config_version=excluded.config_version",
            range_rows,
        )


# --------------------------------------------------------------------------------------------------
# Idempotent escalation emit — built now (its idempotency is a mandated test), wired to the scan in
# Phase 3a. The UNIQUE dedup_key makes "fire once" a DB guarantee, not application logic.
# --------------------------------------------------------------------------------------------------

def _insert_escalation(
    con: sqlite3.Connection,
    *,
    member_id: str,
    kind: EscalationKind,
    dedup_key: str,
    level: EscalationLevel,
    trigger_reason: str,
    observation_id: Optional[str] = None,
    interaction_id: Optional[str] = None,
    created_at: Optional[str] = None,
) -> bool:
    """The escalation INSERT OR IGNORE without committing — so the scan can write it inside its own
    transaction (atomic with the interaction + observation). Returns ``True`` iff THIS call created the
    row. ``escalation_id`` derives deterministically from the dedup_key (the PK must be at least as
    collision-resistant as the UNIQUE it backs, else two dedup_keys colliding on a short PK prefix would
    drop the second via INSERT OR IGNORE on the PK instead of deduping correctly on dedup_key)."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    escalation_id = _det_id("esc:", dedup_key)
    cur = con.execute(
        "INSERT OR IGNORE INTO escalations "
        "(escalation_id, member_id, kind, dedup_key, level, observation_id, interaction_id, trigger_reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (escalation_id, member_id, kind, dedup_key, level, observation_id, interaction_id,
         trigger_reason, created_at),
    )
    return cur.rowcount == 1


def emit_escalation(con: sqlite3.Connection, **kwargs) -> bool:
    """Write one clinician-review-queue row, idempotently, and commit. ``INSERT OR IGNORE`` on the
    UNIQUE ``dedup_key`` makes a duplicate call a no-op at the database; returns ``True`` iff THIS call
    created the row — the created-vs-existing signal the 'loud once, then ambient' transition needs. The
    standalone (auto-committing) entry point; the scan uses :func:`_insert_escalation` inside its own
    transaction instead. ``created_at`` defaults to UTC now (a clock is fine — db.py is not the pure
    core; only the ROW COUNT is the guarantee). The caller owns dedup_key construction."""
    created = _insert_escalation(con, **kwargs)
    con.commit()
    return created


# --------------------------------------------------------------------------------------------------
# Proactive-scan persistence (Phase 3a) — the interaction (audit/trace backbone) each observation hangs
# off via its FK, the observation write, and the two read projections the routes serve. Scan ids are
# deterministic and keyed on data_version (the caller builds them via _det_id), written INSERT OR
# IGNORE: an identical re-scan (same data_version) hits the same key and is a no-op, while a genuine
# data change bumps data_version → a new key → a new row, and the PRIOR row is RETAINED, not overwritten
# (its audit/finding record stands — the same discipline as the escalation dedup). "Replace, not append"
# for the member's live view is delivered by the version-scoped read, which returns only the current
# data_version. These two writers do NOT commit — the scan wraps interaction+observations+escalations in
# one transaction (atomicity); the /ask path (Phase 4) uses a per-call unique response_id, so the same
# INSERT OR IGNORE simply always inserts and `driver` distinguishes the two write disciplines.
# --------------------------------------------------------------------------------------------------

def write_interaction(
    con: sqlite3.Connection,
    resp: HealthIntelligenceResponse,
    *,
    member_id: str,
    driver: str,
    question: Optional[str] = None,
    created_at: Optional[str] = None,
) -> str:
    """Persist one ``HealthIntelligenceResponse`` as an ``interactions`` row (INSERT OR IGNORE on the
    ``response_id`` PK) and return its ``response_id`` (from ``resp.metadata`` — the caller owns id
    assignment; deterministic for the scan, unique-per-call for /ask). The full response is stored
    verbatim as ``response_json``; the typed columns mirror its disposition axes + version tuple. Does
    NOT commit — the caller's transaction owns it."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    m = resp.metadata
    con.execute(
        "INSERT OR IGNORE INTO interactions "
        "(response_id, member_id, driver, question, response_json, answer_disposition, escalation, "
        " data_version, model_version, config_version, prompt_version, latency_ms, tokens, cost_usd, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (m.response_id, member_id, driver, question, resp.model_dump_json(),
         resp.answer_disposition, resp.escalation, m.data_version, m.model_version,
         m.config_version, m.prompt_version, m.latency_ms, m.tokens, m.cost_usd, created_at),
    )
    return m.response_id


def write_observation(con: sqlite3.Connection, obs: Observation) -> None:
    """Write one observation, INSERT OR IGNORE on its deterministic (data_version-keyed)
    ``observation_id``. An identical re-scan is a no-op; a genuine data change mints a new id and a new
    row, RETAINING the prior-version row (which the version-scoped read hides and an escalation may still
    reference). Does NOT commit — the scan's transaction owns it."""
    con.execute(
        "INSERT OR IGNORE INTO observations "
        "(observation_id, member_id, response_id, severity, title, trigger_reason, data_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (obs.observation_id, obs.member_id, obs.response_id, obs.severity, obs.title,
         obs.trigger_reason, obs.data_version),
    )


def get_observations(
    con: sqlite3.Connection, member_id: str, *, data_version: Optional[str] = None
) -> list[Observation]:
    """The member's observations at one ``data_version`` (defaults to the *current* persisted record),
    ranked by severity. Version-scoping is how "a re-scan replaces, not appends" reaches the member: a
    prior version's rows still exist for audit but are never returned for the live record. Sorted by
    severity rank (highest first), then ``observation_id`` for a stable order."""
    if data_version is None:
        data_version = compute_data_version(con, member_id)
    rows = con.execute(
        "SELECT observation_id, member_id, response_id, severity, title, trigger_reason, data_version "
        "FROM observations WHERE member_id = ? AND data_version = ?",
        (member_id, data_version),
    ).fetchall()
    obs = [
        Observation(
            observation_id=r["observation_id"], member_id=r["member_id"], response_id=r["response_id"],
            severity=r["severity"], title=r["title"], trigger_reason=r["trigger_reason"],
            data_version=r["data_version"],
        )
        for r in rows
    ]
    obs.sort(key=lambda o: (-SEVERITY_ORDER[o.severity], o.observation_id))
    return obs


def get_escalations(con: sqlite3.Connection, member_id: str) -> list[Escalation]:
    """The member's clinician-review queue (read projection), oldest first. Escalations are durable —
    never replaced by a re-scan — so this returns the full standing set across data_versions."""
    rows = con.execute(
        "SELECT escalation_id, member_id, kind, dedup_key, level, observation_id, interaction_id, "
        "trigger_reason, created_at FROM escalations WHERE member_id = ? ORDER BY created_at, escalation_id",
        (member_id,),
    ).fetchall()
    return [
        Escalation(
            escalation_id=r["escalation_id"], member_id=r["member_id"], kind=r["kind"],
            dedup_key=r["dedup_key"], level=r["level"], observation_id=r["observation_id"],
            interaction_id=r["interaction_id"], trigger_reason=r["trigger_reason"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


# --------------------------------------------------------------------------------------------------
# Feedback-override resolution — the single seam where the deterministic half of self-improvement
# plugs in (Phase 7). Built now so analysis.py never reaches the DB. MUST return NEW lists (never
# mutate the caller's inputs in place). Phase 2: feedback is empty, so it is a faithful pass-through.
# --------------------------------------------------------------------------------------------------

def _apply_overrides(
    results: list[LabResult],
    ranges: list[ReferenceRange],
    overrides: list[Feedback],
) -> tuple[list[LabResult], list[ReferenceRange]]:
    """Pure: apply active overrides to NEW lists. Phase 7 will implement ``range_override`` (swap a
    ReferenceRange), ``suppress_marker`` (drop a marker's results+ranges), and ``preference`` (a
    composer hint, not an analysis input) here. Phase 2 applies none — returns shallow copies so the
    new-lists contract holds even with an empty override set."""
    del overrides  # reserved (Phase 7 applies them); Phase 2 has no active overrides to fold in
    return list(results), list(ranges)


def resolve_overrides(
    con: sqlite3.Connection,
    member_id: str,
    results: list[LabResult],
    ranges: list[ReferenceRange],
) -> tuple[list[LabResult], list[ReferenceRange]]:
    rows = con.execute(
        "SELECT kind, target, payload_json, source FROM feedback "
        "WHERE member_id = ? AND active = 1 ORDER BY created_at",
        (member_id,),
    ).fetchall()
    overrides = [
        Feedback(
            kind=r["kind"], target=r["target"],
            payload=json.loads(r["payload_json"]) if r["payload_json"] else None,
            source=r["source"],
        )
        for r in rows
    ]
    return _apply_overrides(results, ranges, overrides)


# --------------------------------------------------------------------------------------------------
# Assembly — everything analyze() needs for one member, override-resolved.
# --------------------------------------------------------------------------------------------------

def load_for_analysis(
    con: sqlite3.Connection, member_id: str
) -> tuple[MemberProfile, list[LabResult], list[ReferenceRange], Optional[int], str]:
    """Assemble ``(member, results, ranges, age, data_version)`` for ``analyze``. Raises ``KeyError``
    if the member is absent. ``data_version`` is computed over the raw persisted record (pre-override)
    — overrides are a separately-versioned learning artifact, so the fingerprint tracks the source of
    truth. ``age`` is passed through as ``None`` when unknown (never fabricated to 0); analyze ignores
    age today (``del age``), so ``None`` is inert."""
    member = get_member(con, member_id)
    if member is None:
        raise KeyError(member_id)
    results = get_results(con, member_id)
    ranges = get_ranges(con, markers={r.marker for r in results})
    # reuse the already-loaded member+results (still the raw, pre-override rows) so the hash doesn't
    # re-query them; compute_data_version fetches only notes itself.
    data_version = compute_data_version(con, member_id, member=member, results=results)
    results, ranges = resolve_overrides(con, member_id, results, ranges)
    return member, results, ranges, member.age, data_version
