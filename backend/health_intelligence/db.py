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
from datetime import UTC, datetime

from health_intelligence.config import CONFIG_VERSION
from health_intelligence.models import (
    SEVERITY_ORDER,
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
    "members",
    "lab_results",
    "notes",
    "reference_ranges",
    "interactions",
    "observations",
    "escalations",
    "feedback",
    "prompt_versions",
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
    return sorted(
        [[r.panel_date, r.panel_id, r.marker, r.value, r.unit] for r in results]
    )


def _hash_canon(canon: dict) -> str:
    """The shared serialize+digest tail: compact, key-sorted JSON → 16 hex of sha256. The one place the
    serialization convention lives, so compute_data_version and compute_marker_version stay in lockstep."""
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
    existing = {
        r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
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
        p.member_id,
        p.sex,
        p.age,
        json.dumps(p.conditions),
        json.dumps(p.medications),
        json.dumps(p.family_history),
        json.dumps(p.lifestyle),
    )


# --------------------------------------------------------------------------------------------------
# Reads — what feeds analyze() (member/results/ranges/notes). The proactive-scan writes and the
# observation/escalation read projections live further down (Phase 3a).
# --------------------------------------------------------------------------------------------------


def get_member(con: sqlite3.Connection, member_id: str) -> MemberProfile | None:
    row = con.execute(
        "SELECT * FROM members WHERE member_id = ?", (member_id,)
    ).fetchone()
    return _row_to_profile(row) if row else None


def get_results(con: sqlite3.Connection, member_id: str) -> list[LabResult]:
    rows = con.execute(
        "SELECT marker, value, unit, panel_id, panel_date FROM lab_results "
        "WHERE member_id = ? ORDER BY panel_date, marker, panel_id",
        (member_id,),
    ).fetchall()
    return [
        LabResult(
            marker=r["marker"],
            value=r["value"],
            unit=r["unit"],
            panel_id=r["panel_id"],
            panel_date=r["panel_date"],
        )
        for r in rows
    ]


def get_ranges(
    con: sqlite3.Connection,
    markers: set[str] | None = None,
    config_version: str = CONFIG_VERSION,
) -> list[ReferenceRange]:
    """Reference ranges for the active ``config_version``. The range_id embeds the version, so a
    future config bump *adds* a generation rather than overwriting — the read must scope to one
    version or ``_range_for`` would see duplicate (marker, sex) rows across generations. ``markers``
    None returns all; when scoped, returns ALL sexes for those markers so the sex selection + 'any'
    fallback still resolve."""
    sql = (
        "SELECT marker, sex, unit, ref_low, ref_high, panic_low, panic_high, config_version "
        "FROM reference_ranges WHERE config_version = ?"
    )
    params: list = [config_version]
    if markers is not None:
        marker_list = list(markers)
        if not marker_list:
            return []
        sql += f" AND marker IN ({','.join('?' * len(marker_list))})"
        params += marker_list
    rows = con.execute(sql, params).fetchall()
    return [
        ReferenceRange(
            marker=r["marker"],
            sex=r["sex"],
            unit=r["unit"],
            ref_low=r["ref_low"],
            ref_high=r["ref_high"],
            panic_low=r["panic_low"],
            panic_high=r["panic_high"],
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
    return [
        r["member_id"]
        for r in con.execute("SELECT member_id FROM members ORDER BY member_id")
    ]


def list_member_summaries(con: sqlite3.Connection) -> list[dict]:
    """The member-picker projection (``GET /members``): just enough to populate the dropdown —
    ``[{member_id, age, sex}]``, ordered by id. Deliberately thinner than ``MemberProfile`` (no
    conditions/medications/notes): the picker only needs a label, and the full record loads on
    select. A plain dict, not a model — architecture §13 keeps read projections like this off the
    contract surface."""
    return [
        {"member_id": r["member_id"], "age": r["age"], "sex": r["sex"]}
        for r in con.execute(
            "SELECT member_id, age, sex FROM members ORDER BY member_id"
        )
    ]


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
    member: MemberProfile | None = None,
    results: list[LabResult] | None = None,
    notes: list[Note] | None = None,
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
            "conditions": member.conditions,  # stored input order (round-trips stably)
            "medications": member.medications,
            "family_history": member.family_history,
            "lifestyle": dict(sorted(member.lifestyle.items())),
        },
        # _canon_results makes the fingerprint order-independent of SQLite's row order.
        "results": _canon_results(results),
        "notes": sorted([[n.date or "", n.source or "", n.text] for n in notes]),
    }
    return _hash_canon(canon)


def compute_marker_version(
    *,
    marker: str,
    results: list[LabResult],
    marker_range: ReferenceRange | None,
    sex: str,
    age: int | None,
) -> str:
    """A content fingerprint of ONLY one marker's analysis inputs — that marker's (override-resolved)
    readings, its resolved reference range, and the member's sex/age. Pure (no DB).

    The §48 "finding-stable dedup_key" resolution: ``data_version`` hashes the WHOLE record (incl. notes +
    conditions/medications), which ``analyze`` never reads, so keying a ``data_finding`` escalation on it
    would re-fire on a notes-only edit. Keying on the whole-member analysis inputs would still re-fire
    marker B when marker A's range is overridden. So the dedup_key keys on THIS per-marker hash
    (``data:{member}:{marker}:{marker_version}``): a ``/feedback`` override (or new data) for marker A
    shifts A's hash but not B's, so it cannot re-fire marker B's already-queued clinician escalation — each
    finding fires once and re-fires only when ITS OWN inputs move. Mirrors :func:`compute_data_version`'s
    canonicalization (the range collapsed to its bounds, the marker's results order-independent)."""
    canon = {
        "marker": marker,
        "sex": sex,
        "age": age,
        "results": _canon_results([r for r in results if r.marker == marker]),
        "range": (
            [
                marker_range.ref_low,
                marker_range.ref_high,
                marker_range.panic_low,
                marker_range.panic_high,
                marker_range.unit,
                marker_range.config_version,
            ]
            if marker_range is not None
            else None
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
        if (
            tuple(s)[1:] != incoming[1:]
        ):  # compare every column but the range_id key (index 0)
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
            [
                (
                    f"{mid}:{r.panel_id}:{r.marker}",
                    mid,
                    r.panel_id,
                    r.marker,
                    r.value,
                    r.unit,
                    r.panel_date,
                )
                for r in results
            ],
        )
        con.executemany(
            "INSERT INTO notes (note_id, member_id, note_date, source, text) VALUES (?, ?, ?, ?, ?)",
            [
                (f"{mid}:note:{i}", mid, n.date, n.source, n.text)
                for i, n in enumerate(notes)
            ],
        )
        range_rows = [
            (
                f"range:{rg.config_version}:{rg.sex}:{rg.marker}",
                rg.marker,
                rg.sex,
                rg.unit,
                rg.ref_low,
                rg.ref_high,
                rg.panic_low,
                rg.panic_high,
                rg.config_version,
            )
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


def delete_member(con: sqlite3.Connection, member_id: str) -> bool:
    """Explicitly clear ONE member and everything that hangs off them — the ``DELETE /members/{id}``
    op (architecture §13: opt-in, since ingest never clears by default). Returns whether the member
    existed (drives the ``{deleted}`` ack / 404).

    FKs are ON with no ``ON DELETE CASCADE`` (schema.sql), so delete children before parents in one
    atomic transaction, in FK-dependency order: ``escalations`` (point at observations + interactions)
    -> ``observations`` (point at interactions) -> ``interactions`` -> the owned ``lab_results`` /
    ``notes`` -> ``feedback`` -> ``members`` last. ``reference_ranges`` are intentionally untouched —
    they are GLOBAL (shared across members, keyed by ``config:sex:marker``), so a member delete must
    not strip a band another member still reads; ``prompt_versions`` is likewise global. Unlike
    ``replace_member`` (which preserves the audit/learning trail on re-ingest), this is the one path
    that tears the whole member down."""
    with con:  # atomic: all-or-nothing
        # children first (FK-safe), parent last
        con.execute("DELETE FROM escalations WHERE member_id = ?", (member_id,))
        con.execute("DELETE FROM observations WHERE member_id = ?", (member_id,))
        con.execute("DELETE FROM interactions WHERE member_id = ?", (member_id,))
        con.execute("DELETE FROM lab_results WHERE member_id = ?", (member_id,))
        con.execute("DELETE FROM notes WHERE member_id = ?", (member_id,))
        con.execute("DELETE FROM feedback WHERE member_id = ?", (member_id,))
        # the parent DELETE's rowcount IS the existence signal (child deletes on an absent member are
        # harmless no-ops), so no separate SELECT is needed.
        existed = (
            con.execute(
                "DELETE FROM members WHERE member_id = ?", (member_id,)
            ).rowcount
            > 0
        )
    return existed


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
    observation_id: str | None = None,
    interaction_id: str | None = None,
    created_at: str | None = None,
) -> bool:
    """The escalation INSERT OR IGNORE without committing — so the scan can write it inside its own
    transaction (atomic with the interaction + observation). Returns ``True`` iff THIS call created the
    row. ``escalation_id`` derives deterministically from the dedup_key (the PK must be at least as
    collision-resistant as the UNIQUE it backs, else two dedup_keys colliding on a short PK prefix would
    drop the second via INSERT OR IGNORE on the PK instead of deduping correctly on dedup_key).

    Level UPGRADE on conflict: a ``chat`` escalation's dedup_key is day-scoped, so a later ``urgent``
    turn can land on a row that an earlier ``clinician_review`` turn created the same day. A bare INSERT
    OR IGNORE would drop the urgent one, leaving the clinician queue showing the lower level and masking
    the acute event — so an existing ``clinician_review`` row is UPGRADED to ``urgent`` (and repointed at
    the urgent interaction/reason). Only that one direction upgrades; an equal or lower incoming level is
    left untouched, so a re-emitted ``data_finding`` (deterministic, same level by construction) and the
    fire-once guarantee are unaffected, and an escalation is never silently downgraded."""
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()
    escalation_id = _det_id("esc:", dedup_key)
    cur = con.execute(
        "INSERT OR IGNORE INTO escalations "
        "(escalation_id, member_id, kind, dedup_key, level, observation_id, interaction_id, trigger_reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            escalation_id,
            member_id,
            kind,
            dedup_key,
            level,
            observation_id,
            interaction_id,
            trigger_reason,
            created_at,
        ),
    )
    created = cur.rowcount == 1
    if not created and level == "urgent":
        con.execute(
            "UPDATE escalations SET level = 'urgent', trigger_reason = ?, interaction_id = ? "
            "WHERE dedup_key = ? AND level = 'clinician_review'",
            (trigger_reason, interaction_id, dedup_key),
        )
    return created


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
# off via its FK, the observation write, and the two read projections the routes serve. Both scan ids are
# deterministic and keyed on data_version (the caller builds them via _det_id), but the TWO writers use
# DELIBERATELY DIFFERENT disciplines (architecture §48):
#   • write_interaction → KEEP-FIRST `INSERT OR IGNORE`: an identical re-scan hits the same response_id and
#     is a no-op; the PRIOR audit row is RETAINED, not overwritten (an append-only trace, like the
#     escalation dedup). A genuine data change bumps data_version → a new id → a new row.
#   • write_observation → OVERWRITE-on-conflict (`ON CONFLICT ... DO UPDATE`): a same-data_version re-scan
#     REFRESHES the derived projection in place (severity/title/trigger_reason), so a templates/display-name
#     edit self-heals on the next scan. NOT a blanket DELETE — the escalations.observation_id RESTRICT FK
#     forbids deleting an escalated marker's observation; overwrite-in-place keeps that reference valid.
# "Replace, not append" for the member's live view is delivered by the version-scoped read, which returns
# only the current data_version (prior rows persist for audit/escalation-reference, never shown). Neither
# writer commits — the scan wraps interaction+observations+escalations in one transaction (atomicity); the
# /ask path (Phase 4) uses a per-call unique response_id, so its INSERT OR IGNORE always inserts and
# `driver` distinguishes the two write disciplines.
# --------------------------------------------------------------------------------------------------


def write_interaction(
    con: sqlite3.Connection,
    resp: HealthIntelligenceResponse,
    *,
    member_id: str,
    driver: str,
    question: str | None = None,
    created_at: str | None = None,
) -> str:
    """Persist one ``HealthIntelligenceResponse`` as an ``interactions`` row (INSERT OR IGNORE on the
    ``response_id`` PK) and return its ``response_id`` (from ``resp.metadata`` — the caller owns id
    assignment; deterministic for the scan, unique-per-call for /ask). The full response is stored
    verbatim as ``response_json``; the typed columns mirror its disposition axes + version tuple. Does
    NOT commit — the caller's transaction owns it."""
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()
    m = resp.metadata
    con.execute(
        "INSERT OR IGNORE INTO interactions "
        "(response_id, member_id, driver, question, response_json, answer_disposition, escalation, "
        " data_version, model_version, config_version, prompt_version, latency_ms, tokens, cost_usd, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            m.response_id,
            member_id,
            driver,
            question,
            resp.model_dump_json(),
            resp.answer_disposition,
            resp.escalation,
            m.data_version,
            m.model_version,
            m.config_version,
            m.prompt_version,
            m.latency_ms,
            m.tokens,
            m.cost_usd,
            created_at,
        ),
    )
    return m.response_id


def write_observation(con: sqlite3.Connection, obs: Observation) -> None:
    """Write one observation as an OVERWRITE-on-conflict (UPSERT) on its deterministic
    (data_version-keyed) ``observation_id``. This is the observation set's *replace* discipline
    (architecture §48): a re-scan at the same ``data_version`` refreshes the derived projection
    (severity/title/trigger_reason/response_id) **in place** rather than keeping the first write — so a
    narration change (e.g. a ``templates`` / display-name edit) self-heals on the next scan instead of
    stranding stale prose at a fixed ``data_version``.

    Overwrite-in-place, **not** a blanket delete: ``escalations.observation_id`` is a RESTRICT FK to this
    row, so ``DELETE FROM observations`` would fail on any escalated marker (proven on the K⁺-panic
    member); updating the same id keeps that reference valid. Cross-version supersession is handled
    elsewhere — a genuine data change mints a NEW id (new ``data_version``) and a new row, and the
    version-scoped read (``get_observations``) returns only the current version, so prior rows persist
    for audit/escalation-reference but are never shown. In-version refresh here + version-scoped read =
    the §48 "replace the set" semantics. Distinct from ``write_interaction``, which stays keep-first
    ``INSERT OR IGNORE`` (an append-only audit row, not a refreshable projection). Does NOT commit — the
    scan's transaction owns it."""
    con.execute(
        "INSERT INTO observations "
        "(observation_id, member_id, response_id, severity, title, trigger_reason, data_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(observation_id) DO UPDATE SET "
        "response_id=excluded.response_id, severity=excluded.severity, "
        "title=excluded.title, trigger_reason=excluded.trigger_reason",
        (
            obs.observation_id,
            obs.member_id,
            obs.response_id,
            obs.severity,
            obs.title,
            obs.trigger_reason,
            obs.data_version,
        ),
    )


def prune_observations(
    con: sqlite3.Connection,
    member_id: str,
    data_version: str,
    keep_ids: set[str],
) -> None:
    """Delete the member's observations at ``data_version`` whose ``observation_id`` is NOT in
    ``keep_ids`` and is NOT referenced by an escalation — the scan's set-reconciliation step. Does NOT
    commit (runs inside the scan transaction).

    Why this exists (Phase 7): the §48 replace-discipline keys the observation set on ``data_version``,
    which a re-INGEST bumps — but a ``/feedback`` OVERRIDE changes ``analyze``'s inputs WITHOUT bumping
    ``data_version`` (it fingerprints the raw record), so a re-scan after an override that *cleared* a
    marker's flag would strand the prior observation (the write loop only UPSERTs raised markers, it
    never removes one that stopped being raised). This prunes exactly those orphans. Escalation-pinned
    rows are KEPT: ``escalations.observation_id`` is a RESTRICT FK and the queued clinician task is
    durable, so a cleared-but-escalated finding's observation stays (a deliberately rare edge — the
    sample-override path targets non-escalated ``notable`` markers). Targeted, never a blanket DELETE.

    One set-based DELETE (not an N+1 SELECT-then-per-row loop). The escalations subquery MUST keep
    ``observation_id IS NOT NULL`` — that column is a NULLABLE FK (NULL for chat escalations), and SQL
    ``NOT IN`` against a set containing NULL matches zero rows, which would silently prune nothing."""
    params: list = [member_id, data_version]
    sql = (
        "DELETE FROM observations WHERE member_id = ? AND data_version = ? "
        "AND observation_id NOT IN "
        "(SELECT observation_id FROM escalations WHERE observation_id IS NOT NULL)"
    )
    keep = list(keep_ids)
    if keep:  # NOT IN () is a syntax error, so only add the clause when there's something to keep
        sql += f" AND observation_id NOT IN ({','.join('?' * len(keep))})"
        params += keep
    con.execute(sql, params)


def get_observations(
    con: sqlite3.Connection, member_id: str, *, data_version: str | None = None
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
            observation_id=r["observation_id"],
            member_id=r["member_id"],
            response_id=r["response_id"],
            severity=r["severity"],
            title=r["title"],
            trigger_reason=r["trigger_reason"],
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
            escalation_id=r["escalation_id"],
            member_id=r["member_id"],
            kind=r["kind"],
            dedup_key=r["dedup_key"],
            level=r["level"],
            observation_id=r["observation_id"],
            interaction_id=r["interaction_id"],
            trigger_reason=r["trigger_reason"],
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
    *,
    sex: str,
) -> tuple[list[LabResult], list[ReferenceRange]]:
    """Pure: fold the active overrides into NEW lists (never mutate the caller's inputs) — the
    deterministic half of self-improvement (architecture §9, the §3 boundary row "apply a learned
    correction"). It changes what the core *knows*, not its rules: the floor, validator, and prompt are
    untouched; only ``analyze``'s typed inputs move, so a correction lands identically in BOTH modes.

    Two analysis-affecting kinds (applied in caller order, so the last override per marker wins):

    * ``range_override`` — a clinician re-bounds a marker. Replace ALL of that marker's range rows with a
      single ``sex='any'`` row carrying the new bounds (per-member resolution already scoped this to one
      member, and ``_range_for`` falls back to ``'any'`` for any sex), so a managed-condition marker can
      be widened out of an out-of-range flag. ``payload`` keys (``ref_low``/``ref_high``/``panic_low``/
      ``panic_high``/``unit``) fall back to the marker's EXISTING band when omitted, so a one-sided widen
      need not restate the other bound. The existing band is resolved by the member's ``sex`` (then the
      ``any`` row) — mirroring ``analysis._range_for`` so a one-sided override on a sex-split marker
      (e.g. Hemoglobin/HDL/Ferritin/Creatinine) inherits the member's OWN band, not an arbitrary sex's.
      (Resolved inline rather than calling ``analysis._range_for`` to keep db.py's no-``analysis`` purity.)
    * ``suppress_marker`` — drop the marker's results AND ranges, so the core never analyzes it (it
      vanishes from the trajectory): quiets an expected-abnormal marker without touching the safety rule.

    ``preference`` and the signal kinds never reach here — they are filtered out at the SQL in
    :func:`resolve_overrides` (preference is a compose hint via :func:`get_active_preferences`; signals
    feed ``learn.py``). Because the data-finding dedup keys on the marker's own analysis state
    (``compute_marker_version``), an override that moves a flag re-fires/clears only THAT finding."""
    out_results = list(results)
    out_ranges = list(ranges)
    for fb in overrides:
        if not fb.target:
            continue  # overrides target a marker; a signal (helpful/incorrect/...) has no analysis effect
        if fb.kind == "suppress_marker":
            out_results = [r for r in out_results if r.marker != fb.target]
            out_ranges = [rg for rg in out_ranges if rg.marker != fb.target]
        elif fb.kind == "range_override" and fb.payload:
            # Inherit omitted bounds from the member's ACTUAL band: the (marker, sex) row, then the
            # (marker, 'any') fallback — the same resolution analysis._range_for uses, replicated here so
            # db.py never imports analysis (the documented purity edge).
            cands = [rg for rg in out_ranges if rg.marker == fb.target]
            existing = next((rg for rg in cands if rg.sex == sex), None) or next(
                (rg for rg in cands if rg.sex == "any"), None
            )
            p = fb.payload
            # ``p.get(key, fallback)`` returns the supplied value even when it is explicitly None (a
            # one-sided override), and falls back to the existing bound only when the key is omitted.
            new_rg = ReferenceRange(
                marker=fb.target,
                sex="any",  # per-member override; '_range_for' resolves 'any' for every member sex
                unit=p.get("unit", existing.unit if existing else ""),
                ref_low=p.get("ref_low", existing.ref_low if existing else None),
                ref_high=p.get("ref_high", existing.ref_high if existing else None),
                panic_low=p.get("panic_low", existing.panic_low if existing else None),
                panic_high=p.get(
                    "panic_high", existing.panic_high if existing else None
                ),
                config_version=existing.config_version if existing else CONFIG_VERSION,
            )
            out_ranges = [rg for rg in out_ranges if rg.marker != fb.target]
            out_ranges.append(new_rg)
    return out_results, out_ranges


def resolve_overrides(
    con: sqlite3.Connection,
    member_id: str,
    results: list[LabResult],
    ranges: list[ReferenceRange],
    *,
    sex: str,
) -> tuple[list[LabResult], list[ReferenceRange]]:
    """Resolve the member's active ANALYSIS overrides into ``analyze``'s inputs. ``sex`` is the member's
    sex (so a one-sided range_override inherits the member's own band).

    Honored only from ``clinician``/``system`` sources — a member must not be able to change what is
    flagged (``ui-ux.md`` §7: "the member experience cannot lobby the safety logic"); a member-sourced
    range_override/suppress is stored but INERT for analysis, so it can never suppress a panic floor.
    Ordered so the highest-precedence, latest override per marker is applied last (and wins):
    ``source`` rank (clinician > system), then ``created_at``, then ``feedback_id`` — the trailing
    ``feedback_id`` makes "latest per target wins" deterministic when two rows share a timestamp
    (matching :func:`get_active_signals`'s tiebreak). ``schema.sql``: "clinician source outranking member"."""
    rows = con.execute(
        "SELECT kind, target, payload_json, source FROM feedback "
        "WHERE member_id = ? AND active = 1 "
        "AND kind IN ('range_override', 'suppress_marker') "
        "AND source IN ('clinician', 'system') "
        "ORDER BY CASE source WHEN 'clinician' THEN 2 WHEN 'system' THEN 1 ELSE 0 END, "
        "created_at, feedback_id",
        (member_id,),
    ).fetchall()
    overrides = [
        Feedback(
            kind=r["kind"],
            target=r["target"],
            payload=json.loads(r["payload_json"]) if r["payload_json"] else None,
            source=r["source"],
        )
        for r in rows
    ]
    return _apply_overrides(results, ranges, overrides, sex=sex)


# --------------------------------------------------------------------------------------------------
# Feedback writes (Phase 7) — the input side of self-improvement. ``insert_feedback`` records one
# correction/signal row; ``get_active_preferences`` resolves the composer-hint kind (not an analysis
# input, so it bypasses _apply_overrides); ``reset_learning`` is the /reset revert. The override KINDS
# are read back by resolve_overrides above; the SIGNAL kinds (helpful/incorrect/escalation_*) are read
# by learn.py. db.py only stores + resolves them — it never decides anything safety-relevant.
# --------------------------------------------------------------------------------------------------


def insert_feedback(
    con: sqlite3.Connection,
    member_id: str,
    fb: Feedback,
    *,
    created_at: str | None = None,
) -> str:
    """Record one ``feedback`` row (a correction or a signal) and commit; returns the synthesized
    ``feedback_id``. ``active`` defaults to 1; ``created_at`` defaults to UTC now and drives recency
    (latest active override per target wins in :func:`resolve_overrides`).

    ``feedback_id`` only needs to be UNIQUE (each post is a distinct event, unlike the idempotent
    escalation write — it is never re-derived), so on the rare same-``(member,kind,target,created_at)``
    collision (two same-microsecond duplicate posts, or a caller passing an explicit ``created_at``) the
    id is salted and retried rather than 500-ing on the PRIMARY KEY."""
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()
    payload_json = json.dumps(fb.payload) if fb.payload is not None else None
    for salt in range(1000):  # salt 0 is the common path; >0 only on a PK collision
        feedback_id = _det_id(
            "fb:", member_id, fb.kind, fb.target or "", created_at, str(salt)
        )
        try:
            with con:  # atomic single-row write; commits on success, rolls back + re-raises on conflict
                con.execute(
                    "INSERT INTO feedback "
                    "(feedback_id, member_id, kind, target, payload_json, source, active, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                    (
                        feedback_id,
                        member_id,
                        fb.kind,
                        fb.target,
                        payload_json,
                        fb.source,
                        created_at,
                    ),
                )
            return feedback_id
        except sqlite3.IntegrityError as e:
            # ONLY retry the feedback_id PK collision (the salt is the only thing a retry changes). A
            # FOREIGN KEY (unknown member) or CHECK (bad kind/source) violation would fail identically on
            # every salt, so surface it immediately instead of spinning 1000× and masking it behind a
            # "couldn't synthesize an id" error.
            if "feedback.feedback_id" not in str(e):
                raise
            continue  # PK collision — bump the salt and retry
    raise RuntimeError("could not synthesize a unique feedback_id after 1000 attempts")


#: Bounds on the member-authored ``preference`` hints injected into the compose prompt. CAPPED because
#: this is the one persistent member-controlled free-text channel into the LLM context: without a cap,
#: many/long preferences would inflate every ``/ask``'s tokens unboundedly (a cost-amplification vector).
#: The hard rules in the system prompt + the always-on validator already bound what a preference can DO
#: (tone only, never the floor); this bounds how much it can COST. Newest-first, so the latest hints win.
_MAX_ACTIVE_PREFERENCES = 5
_MAX_PREFERENCE_CHARS = 240


def get_active_preferences(con: sqlite3.Connection, member_id: str) -> list[str]:
    """Active ``preference`` hints for the compose context (architecture §9: a preference 'joins the
    composer context'). NOT an analysis input — read here, not in _apply_overrides — so it can only shape
    tone, never a number or the floor. Reads ``payload.text`` (or ``preference``). Bounded to the newest
    ``_MAX_ACTIVE_PREFERENCES``, each truncated to ``_MAX_PREFERENCE_CHARS`` (cost-amplification guard)."""
    rows = con.execute(
        "SELECT payload_json FROM feedback "
        "WHERE member_id = ? AND kind = 'preference' AND active = 1 "
        "ORDER BY created_at DESC, feedback_id DESC LIMIT ?",
        (member_id, _MAX_ACTIVE_PREFERENCES),
    ).fetchall()
    out: list[str] = []
    for r in rows:
        if r["payload_json"]:
            p = json.loads(r["payload_json"])
            text = p.get("text") or p.get("preference")
            if text:
                out.append(str(text)[:_MAX_PREFERENCE_CHARS])
    return out


def get_active_signals(con: sqlite3.Connection) -> list[Feedback]:
    """All active SIGNAL feedback across members (helpful/incorrect/escalation_accept/reject) — the
    input ``learn.py`` rule-assembles a candidate prompt from. Ordered by ``created_at`` so the assembly
    is a pure, order-stable function of the signal set (identical signals -> byte-identical candidate)."""
    rows = con.execute(
        "SELECT kind, target, payload_json, source FROM feedback "
        "WHERE active = 1 AND kind IN ('helpful','incorrect','escalation_accept','escalation_reject') "
        "ORDER BY created_at, feedback_id",
    ).fetchall()
    return [
        Feedback(
            kind=r["kind"],
            target=r["target"],
            payload=json.loads(r["payload_json"]) if r["payload_json"] else None,
            source=r["source"],
        )
        for r in rows
    ]


def reset_learning(con: sqlite3.Connection) -> dict[str, int]:
    """The ``POST /reset`` revert (architecture §9/§688): deactivate ALL feedback (``active=0``) and
    revert every learned prompt above the v0 baseline (``promoted`` OR ``rejected``) to
    ``status='reverted'`` — so the composer falls back to the latest remaining promoted version (v0, or
    the constant when none was ever seeded). Reverting the ``rejected`` rows too matters: a gate verdict
    is BASELINE-relative, and /reset changes the active baseline, so a candidate rejected against the old
    baseline must be re-gateable (not stuck cached as 'rejected') if its feedback is re-posted — the
    symmetric case to a reverted promotion. This is the learning-revert, NOT a data wipe: the feedback
    rows and prompt history are preserved (the trail stays), every member and the dataset untouched. The
    factory reset is the separate :func:`nuke_all` (``POST /admin/reseed``). Returns affected-row counts."""
    with con:
        fb = con.execute("UPDATE feedback SET active = 0 WHERE active = 1").rowcount
        pv = con.execute(
            "UPDATE prompt_versions SET status = 'reverted' "
            "WHERE version > 0 AND status IN ('promoted', 'rejected')"
        ).rowcount
    return {"feedback_deactivated": fb, "prompts_reverted": pv}


# --------------------------------------------------------------------------------------------------
# Prompt versions (Phase 7) — the self-improvement promotion store. The composer reads the latest
# ``status='promoted'`` row (``get_active_prompt``); ``learn.py`` assembles a candidate, gates it
# through the harness, and writes it ``promoted`` or ``rejected`` with its eval report. db.py only
# persists + reads versions — the gate DECISION is learn.py's, the prompt TEXT is rule-assembled there.
# --------------------------------------------------------------------------------------------------


def get_active_prompt(
    con: sqlite3.Connection,
) -> tuple[int, str, str | None] | None:
    """The active composer prompt: ``(version, prompt_text, eval_report_json)`` of the latest
    ``status='promoted'`` row, or ``None`` when none was ever promoted (the pipeline then falls back to
    ``llm.BASE_COMPOSE_SYSTEM`` at version 0 — keeping this module free of an ``llm`` import). ``learn.py``
    uses ``eval_report_json`` as the regression baseline; the pipeline uses only the first two."""
    row = con.execute(
        "SELECT version, prompt_text, eval_report_json FROM prompt_versions "
        "WHERE status = 'promoted' ORDER BY version DESC LIMIT 1"
    ).fetchone()
    return (
        (row["version"], row["prompt_text"], row["eval_report_json"]) if row else None
    )


def find_prompt_by_text(
    con: sqlite3.Connection, prompt_text: str
) -> tuple[int, str, str | None] | None:
    """The newest prompt_version whose ``prompt_text`` is byte-identical to ``prompt_text`` AND whose
    ``status`` is a GATE VERDICT (``promoted``/``rejected``), as ``(version, status, eval_report_json)``
    — or ``None``. The ``learn`` debounce: because the candidate is a pure function of the feedback set,
    an unchanged signal set re-assembles identical text, which short-circuits to this cached *verdict*
    with ZERO model calls (architecture §9: naive spam is free).

    Crucially it ignores ``reverted``/``proposed`` rows: ``/reset`` flips a promoted prompt to
    ``reverted``, and that is an OPERATOR action, not a gate verdict — conflating the two would make a
    post-reset re-learn of identical feedback short-circuit to ``reverted`` and never re-promote (the
    composer stuck on v0). Excluding them lets the same feedback re-gate and re-promote after a reset."""
    row = con.execute(
        "SELECT version, status, eval_report_json FROM prompt_versions "
        "WHERE prompt_text = ? AND status IN ('promoted', 'rejected') "
        "ORDER BY version DESC LIMIT 1",
        (prompt_text,),
    ).fetchone()
    return (row["version"], row["status"], row["eval_report_json"]) if row else None


def next_prompt_version(con: sqlite3.Connection) -> int:
    """The next version integer (max + 1; 1 when the table is empty — v0 is the constant baseline)."""
    row = con.execute("SELECT MAX(version) AS m FROM prompt_versions").fetchone()
    return (row["m"] + 1) if row and row["m"] is not None else 1


def count_prompt_versions_on(con: sqlite3.Connection, date_prefix: str) -> int:
    """How many prompt_versions rows carry ``created_at`` on ``date_prefix`` (``YYYY-MM-DD``) — the
    learn DAILY-CAP backstop. Debounced short-circuits write no row, so they don't count (free)."""
    return con.execute(
        "SELECT COUNT(*) AS c FROM prompt_versions WHERE substr(created_at, 1, 10) = ?",
        (date_prefix,),
    ).fetchone()["c"]


def insert_prompt_version(
    con: sqlite3.Connection,
    *,
    version: int,
    prompt_text: str,
    status: str,
    eval_report_json: str | None = None,
    created_at: str | None = None,
) -> None:
    """Persist one ``prompt_versions`` row (commits). ``status`` is one of proposed/promoted/rejected/
    reverted; a promoted candidate carries the harness ``eval_report_json`` that gated it."""
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()
    with con:
        con.execute(
            "INSERT INTO prompt_versions (version, prompt_text, status, eval_report_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (version, prompt_text, status, eval_report_json, created_at),
        )


# --------------------------------------------------------------------------------------------------
# Factory reset (Phase 7) — the destructive clean-slate behind ``POST /admin/reseed`` (architecture
# §688/§756), DISTINCT from reset_learning's learning-only revert. A full NUKE: drop every table, then
# recreate the canonical schema; the route then re-ingests the training_data bundle (the back-edge to
# preprocessing stays in api.py, not here).
# --------------------------------------------------------------------------------------------------


def nuke_all(con: sqlite3.Connection) -> None:
    """Full factory NUKE — DROP every table, then recreate the canonical schema from ``schema.sql``.

    Stronger than a row-level truncate: it removes the tables themselves (and their rowid sequences and
    indexes), returning the DB to a pristine, freshly-initialised state, and it is robust to ANY table
    that exists — including a stray/renamed one a hardcoded delete-list would miss. FK enforcement is
    toggled OFF around the drops (set outside a transaction, where the PRAGMA takes effect) so drop order
    is irrelevant; :func:`init_db` then rebuilds every table. The caller (``POST /admin/reseed``)
    re-ingests ``training_data`` afterwards, so the DB ends as a clean 15-member factory state."""
    con.execute(
        "PRAGMA foreign_keys = OFF"
    )  # must be set with NO active transaction to take effect
    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for t in tables:
            con.execute(f'DROP TABLE IF EXISTS "{t}"')  # noqa: S608 — names from sqlite_master, not input
        con.commit()
    finally:
        con.execute(
            "PRAGMA foreign_keys = ON"
        )  # restore the per-connection invariant connect() sets
    init_db(
        con
    )  # recreate every table from the canonical schema.sql (none exist -> full rebuild)


# --------------------------------------------------------------------------------------------------
# Assembly — everything analyze() needs for one member, override-resolved.
# --------------------------------------------------------------------------------------------------


def load_for_analysis(
    con: sqlite3.Connection, member_id: str
) -> tuple[MemberProfile, list[LabResult], list[ReferenceRange], int | None, str]:
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
    results, ranges = resolve_overrides(con, member_id, results, ranges, sex=member.sex)
    return member, results, ranges, member.age, data_version
