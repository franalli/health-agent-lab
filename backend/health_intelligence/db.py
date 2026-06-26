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
    EscalationKind,
    EscalationLevel,
    Feedback,
    LabResult,
    MemberProfile,
    Note,
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
    "members", "lab_results", "notes", "reference_ranges", "interactions",
    "observations", "escalations", "feedback", "prompt_versions",
}


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
    con = sqlite3.connect(path)
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
# Reads — only what feeds analyze(). Interaction/observation reads are deferred to Phase 3a (nothing
# populates interactions until the scan); the only durable write here is emit_escalation.
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
        # sorted() makes the fingerprint order-independent of SQLite's row order; (date, panel_id,
        # marker) uniquely identify a row so the comparison never reaches the float value field.
        "results": sorted([[r.panel_date, r.panel_id, r.marker, r.value, r.unit] for r in results]),
        "notes": sorted([[n.date or "", n.source or "", n.text] for n in notes]),
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


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

def emit_escalation(
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
    """Write one clinician-review-queue row, idempotently. ``INSERT OR IGNORE`` on the UNIQUE
    ``dedup_key`` makes a duplicate call a no-op at the database. Returns ``True`` iff THIS call
    created the row (``rowcount == 1``), ``False`` if a row already existed — the created-vs-existing
    signal the 'loud once, then ambient' transition needs. ``created_at`` defaults to UTC now (a
    clock is fine here — ``db.py`` is not the pure core; only the ROW COUNT is the guarantee, not
    byte-identity of ``created_at``). The caller owns dedup_key construction (Phase 3a)."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    # 32 hex (128 bits): the escalation_id PK must be at least as collision-resistant as the dedup_key
    # UNIQUE it derives from, else two distinct dedup_keys colliding on a short prefix would drop the
    # second escalation via INSERT OR IGNORE on the PK rather than dedup correctly on dedup_key.
    escalation_id = "esc:" + hashlib.sha256(dedup_key.encode("utf-8")).hexdigest()[:32]
    cur = con.execute(
        "INSERT OR IGNORE INTO escalations "
        "(escalation_id, member_id, kind, dedup_key, level, observation_id, interaction_id, trigger_reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (escalation_id, member_id, kind, dedup_key, level, observation_id, interaction_id,
         trigger_reason, created_at),
    )
    con.commit()
    return cur.rowcount == 1


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
