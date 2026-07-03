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
import os
import pathlib
import sqlite3
import tempfile
import time
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime

try:  # POSIX advisory file locks — the cross-process half of process_lock (Render/dev; see process_lock)
    import fcntl
except (
    ImportError
):  # pragma: no cover — non-POSIX (Windows dev): in-process guards only
    fcntl = None

from health_intelligence.config import CONFIG_VERSION
from health_intelligence.models import (
    SEVERITY_ORDER,
    Escalation,
    EscalationKind,
    EscalationLevel,
    Feedback,
    FeedbackRecord,
    HealthIntelligenceResponse,
    LabResult,
    MemberProfile,
    Note,
    Observation,
    PromptVersionRecord,
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
    # Concurrency hardening (sync routes run in anyio's threadpool, one connection per request — see the
    # check_same_thread note above): WAL lets a reader proceed against the last committed snapshot while a
    # writer commits (instead of blocking), and busy_timeout turns an immediate SQLITE_BUSY (→ a 500) under
    # write contention into a short wait-and-retry. Both are idempotent per connection; WAL is a silent
    # no-op on ``:memory:``. Together with the atomic reseed (``reseed_transaction``), a concurrent request
    # during a reseed/upload never sees a HALF-WIPED DB, and waits the write out rather than erroring — up to
    # the 5s busy_timeout (ample for the 15-member seed). This is a bounded-wait mitigation, NOT an absolute
    # guarantee: a write lock held past 5s (a far larger dataset / very slow disk) still surfaces SQLITE_BUSY.
    con.execute("PRAGMA busy_timeout = 5000")
    # The one-time delete->WAL transition (the FIRST connection ever made to a fresh DB file) needs an
    # exclusive lock, and SQLite returns SQLITE_BUSY for it IMMEDIATELY — without consulting the busy
    # handler (deadlock avoidance) — so the 5s busy_timeout above does NOT cover it. Under --workers 2
    # both workers' first connect() races exactly this switch on a fresh disk. Steady-state is
    # unaffected (a DB already in WAL answers the pragma with no exclusive lock), so a short bounded
    # retry rides out the sibling's millisecond transition. Retried, never skipped: silently serving in
    # rollback-journal mode would forfeit the reader/writer concurrency the multi-worker deploy rests on.
    for attempt in range(20):
        try:
            con.execute("PRAGMA journal_mode = WAL")
            break
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or attempt == 19:
                raise
            time.sleep(0.05)
    return con


def _process_lock_path(con: sqlite3.Connection, name: str) -> str:
    """The advisory-lock file path for THIS connection's database — sited next to the SQLite file so
    co-located worker processes (which all open that same file) contend on the same lock. Derived from
    the LIVE connection (``PRAGMA database_list``), not an env var, so it tracks whatever DB this process
    actually opened. An unfiled/in-memory DB (tests) has no path -> a PER-PROCESS temp-dir fallback: an
    in-memory SQLite DB is process-private by construction, so cross-process exclusion is meaningless for
    it — and a host-global fallback name made two INDEPENDENT test runs (parallel sessions both running
    ``make test``) spuriously contend, failing each other's /learn single-flight tests with LearnBusy.
    The PID scope keeps the documented same-process property (two fds within one process still contend)."""
    row = con.execute("PRAGMA database_list").fetchone()
    db_file = (row["file"] if row is not None else "") or ""
    if not db_file:
        return os.path.join(
            tempfile.gettempdir(), f"health_intelligence_{os.getpid()}_{name}.lock"
        )
    return f"{db_file}.{name}.lock"


#: Bound on a ``blocking=True`` acquire (see :func:`process_lock`): the guarded startup section is
#: sub-second on a warm restart and seconds on a fresh seed, so a minute means the holder is WEDGED
#: (a dead holder auto-releases its flock) — fail loudly rather than stall a SIGTERM-immune wait.
_BLOCKING_ACQUIRE_TIMEOUT_S = 60.0
_BLOCKING_ACQUIRE_POLL_S = 0.1


@contextmanager
def process_lock(con: sqlite3.Connection, name: str, *, blocking: bool):
    """Cross-PROCESS mutual exclusion between the workers sharing this connection's database — an OS
    advisory file lock (``fcntl.flock``) on a ``<db-file>.<name>.lock`` sibling. The multi-worker seam
    (§15): SQLite serializes individual transactions, but check-then-act sequences that span statements,
    model calls, or ``executescript`` (startup init+seed, a ``/learn`` run, the learning-state resets)
    need a lock that holds across worker processes. flock auto-releases when its holder dies (no
    stale-lock TTL to manage, unlike a DB lease); two ``os.open`` fds contend even within one process, so
    this also excludes threadpool peers. Scope is one HOST, which matches the single-instance deploy (a
    Render persistent disk pins the service to one instance).

    Yields ``True`` when the lock is held; with ``blocking=False`` yields ``False`` on contention instead
    of waiting (the caller maps that to its own busy signal, e.g. 409). ``blocking=True`` waits with a
    BOUNDED poll (``_BLOCKING_ACQUIRE_TIMEOUT_S``) — for short critical sections only (startup init),
    never around a model call. Bounded rather than an untimed ``flock`` deliberately: an untimed
    ``LOCK_EX`` is SIGTERM-immune (uvicorn's handler only sets a flag and PEP 475 transparently retries
    the syscall), so a WEDGED holder — a dead one auto-releases — would stall the waiting worker until
    the platform's SIGKILL; the poll turns that into a loud, bounded ``TimeoutError`` instead.

    Contention is the ONLY soft outcome: ``flock`` signals it as ``BlockingIOError`` (EWOULDBLOCK), and
    any OTHER ``OSError`` (ENOLCK / EACCES / EIO — the locking facility itself is broken, e.g. a network
    mount without flock support) PROPAGATES. Yielding ``False`` for those would fail OPEN — the blocking
    startup caller would run the fresh-DB init unserialized, and the non-blocking callers would misreport
    a broken environment as an endless 409-busy — so a loud error is the only safe degrade.

    Lock NAMES partition the exclusion domains: ``"startup"`` serializes lifespan init; ``"learn"`` is
    the learning-state lock shared by ``/learn``, ``/reset``, and ``/admin/reseed``; ``"data"`` is the
    dataset/member bulk-ingest lock shared by ``/members/upload``, ``POST /members``, and
    ``/admin/reseed``. On non-POSIX platforms (no ``fcntl``) it degrades to yielding ``True`` —
    single-process dev only, documented."""
    if fcntl is None:  # pragma: no cover — non-POSIX degrade (Windows dev)
        yield True
        return
    lock_fd = os.open(_process_lock_path(con, name), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if blocking:
            deadline = time.monotonic() + _BLOCKING_ACQUIRE_TIMEOUT_S
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"could not acquire the {name!r} process lock within "
                            f"{_BLOCKING_ACQUIRE_TIMEOUT_S:.0f}s — the holder looks wedged"
                        ) from None
                    time.sleep(_BLOCKING_ACQUIRE_POLL_S)
        else:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False  # held elsewhere — contention, the one legitimate soft outcome
                return
        yield True
    finally:
        os.close(lock_fd)  # closing the fd releases any flock held on it


def init_db(con: sqlite3.Connection, schema_path: pathlib.Path = SCHEMA_PATH) -> None:
    """Idempotently ensure the nine tables exist, then apply additive column migrations. schema.sql uses
    bare ``CREATE TABLE`` (no ``IF NOT EXISTS``), so guard on presence: if all nine are already there, skip
    creation; else run the script. schema.sql stays the FRESH-DB contract — never edited here — and
    :func:`_apply_migrations` brings a DB created under an OLDER schema.sql up to date (e.g. adds
    ``escalations.status``) non-destructively. Safe to call on every startup (Phase 8 relies on this).

    NOT internally race-safe on a FRESH database: the presence check + ``executescript`` is check-then-act,
    so two processes first-initializing the same file can both run the script and the loser crashes on
    "table already exists". Concurrent callers must serialize via ``process_lock(con, "startup")`` — the
    api.py lifespan (the one multi-process caller; --workers 2) does; the CLI (``make init-db``) and tests
    are single-process. On an ALREADY-initialized DB it is a pure read + guarded no-op migrations — safe."""
    existing = {
        r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    present = _EXPECTED_TABLES & existing
    if present == _EXPECTED_TABLES:
        _apply_migrations(
            con
        )  # existing DB: bring its columns up to the current schema
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
    _apply_migrations(
        con
    )  # fresh DB already has every column (no-op) — keeps the path uniform


def _add_column_if_missing(con: sqlite3.Connection, table: str, ddl: str) -> None:
    """One guarded additive migration: ``ALTER TABLE ADD COLUMN`` iff the column is absent. The column
    name is DERIVED from ``ddl``'s first token (the ADD COLUMN grammar guarantees it; this repo's DDL is
    bare snake_case identifiers only) rather than passed separately — a second parameter could silently
    disagree with the DDL, leaving the guard checking a name the ALTER never adds and the duplicate-column
    swallow masking the re-run on every boot. The ``PRAGMA table_info`` guard is the normal no-op path;
    the ``duplicate column`` catch is the check-then-act race closer — two processes migrating the same DB
    can both pass the guard, and the loser's ALTER must be a no-op, not a startup crash (defense-in-depth
    under the startup process_lock, and the guard for direct init_db callers — the CLI, tests — that
    don't hold it)."""
    column = ddl.split(None, 1)[0]
    cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    if column in cols:
        return
    try:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise  # a real DDL failure — surface it, don't mask it as the benign race
    con.commit()


def _apply_migrations(con: sqlite3.Connection) -> None:
    """Additive, idempotent column migrations for a DB created under an EARLIER schema.sql (which is the
    fresh-DB contract). Each guards on ``PRAGMA table_info`` so it no-ops once the column exists (a fresh DB,
    or a second startup), and tolerates the concurrent-migrator race (``_add_column_if_missing``).
    ``ADD COLUMN`` with a ``NOT NULL DEFAULT`` back-fills existing rows in place (verified) — never a
    destructive rewrite. Commits its own change (mirrors init_db's commit).

    Migrations (append-only — never reorder or remove one):
      - ``escalations.status`` ('open'|'superseded', default 'open') — the Phase-7 §720 escalation
        lifecycle: a re-scan supersedes an escalation whose finding a /feedback override cleared."""
    _add_column_if_missing(
        con,
        "escalations",
        "status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','superseded'))",
    )


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
    commit: bool = True,
) -> None:
    """Persist one member's normalized data, replacing any prior version in a single transaction.

    UPSERT the ``members`` row (never DELETE/REPLACE it): with FKs ON, a REPLACE would delete the
    parent first and FK-fail surviving children — or, with cascade, silently wipe clinician
    ``feedback`` and the escalation audit trail. UPSERT updates the profile in place, so
    ``interactions``/``observations``/``escalations``/``feedback`` survive a re-ingest. The owned
    children (``lab_results``/``notes``) are deleted then reinserted with deterministic PKs;
    ``reference_ranges`` are global (identical across members) and upserted by ``range_id``.

    ``commit=False`` runs the writes WITHOUT committing, so the caller's transaction owns the commit —
    used by the atomic reseed (``reseed_transaction``), where every member + the v0 seed must land in one
    all-or-nothing unit. The default keeps the per-member commit the normal seed/upload path relies on.
    """
    mid = profile.member_id
    with (
        con if commit else nullcontext()
    ):  # commit here, or defer to the caller's transaction
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


def write_observation(con: sqlite3.Connection, obs: Observation) -> bool:
    """Write one observation as an OVERWRITE-on-conflict (UPSERT) on its deterministic
    (data_version-keyed) ``observation_id``; return ``True`` iff the row was NEWLY created (its id was
    not in the table before this write — the signal ``pipeline.scan`` aggregates into the
    ``new_observations`` count). This is the observation set's *replace* discipline
    (architecture §48): a re-scan at the same ``data_version`` refreshes the derived projection
    (severity/title/trigger_reason/response_id) **in place** rather than keeping the
    first write — so a
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
    scan's transaction owns it.

    The created-check is a SELECT-then-UPSERT rather than a rowcount read because SQLite reports 1
    changed row for BOTH arms of ``ON CONFLICT DO UPDATE`` — insert and overwrite are indistinguishable
    after the fact. Safe un-atomically: writes serialize on the single connection inside the scan's
    transaction."""
    created = (
        con.execute(
            "SELECT 1 FROM observations WHERE observation_id = ?",
            (obs.observation_id,),
        ).fetchone()
        is None
    )
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
    return created


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


def reconcile_escalation_status(
    con: sqlite3.Connection,
    member_id: str,
    data_version: str,
    live_escalation_ids: set[str],
) -> None:
    """Reconcile the member's ``data_finding`` escalation statuses against THIS scan — the escalation-queue
    peer of :func:`prune_observations` (Phase-7 §720 lifecycle). ``live_escalation_ids`` is the set of
    escalation_ids the scan just emitted (one per currently-raised marker, at its CURRENT
    ``compute_marker_version``). For each escalation whose observation is at the CURRENT ``data_version``:
    ``open`` if the scan just re-emitted it (``escalation_id in live_escalation_ids``) OR it is ``urgent``
    (see SAFETY), else ``superseded`` — so a re-scan after a ``/feedback`` override that CLEARED a
    ``clinician_review`` flag drops that escalation off the active queue (:func:`get_all_escalations` shows
    ``open`` only) while the row is KEPT for audit. SYMMETRIC: remove the override, re-scan, the marker is
    raised again at its original ``marker_version`` → its (INSERT-OR-IGNORE deduped) row is emitted again →
    back to ``open``. Does NOT commit (runs inside the scan transaction).

    Keyed on the emitted ESCALATION identity (not the observation): a stale ``clinician_review`` twin the
    scan did NOT re-emit (a ``range_override`` that changed a still-raised marker's ``marker_version``, or a
    cleared finding) is superseded. Reconciled in PYTHON off the set (not a SQL ``IN``) so the headline
    empty-set case (an override cleared the member's ONLY finding → nothing emitted) is not a ``... IN ()``
    syntax error (the footgun :func:`prune_observations` also guards). Scoped to the current ``data_version``
    via the JOIN, so escalations pinned to a PRIOR version (genuine historical events) are untouched.

    SAFETY — an ``urgent`` (panic) escalation is NEVER superseded, full stop: a fired urgent stays on the
    queue until a HUMAN resolves it (the deferred manual-resolve step). Only the softer ``clinician_review``
    tier auto-clears. ACCEPTED consequence: a ``range_override`` that changes a STILL-urgent marker's
    ``marker_version`` leaves the old AND new urgent both ``open`` — a duplicate the human-resolve lifecycle
    will clean up; a safe-direction over-show (a clinician sees a finding twice, never MISSES one). A
    conditional supersede of the stale urgent twin was tried and REVERTED: keyed on emitted-obs-ids it could
    HIDE a downgraded urgent (panic → clinician_review via a partial re-bound fills the obs-id set) or
    resurrect a superseded twin when the marker later fully clears — both worse than the duplicate. Never
    trade 'urgent never hidden' for de-duplication (the reconcile-finder repro that proved it)."""
    rows = con.execute(
        "SELECT e.escalation_id, e.level FROM escalations e "
        "JOIN observations o ON o.observation_id = e.observation_id "
        "WHERE e.member_id = ? AND e.kind = 'data_finding' AND o.data_version = ?",
        (member_id, data_version),
    ).fetchall()
    for r in rows:
        # A stale clinician_review the scan didn't re-emit is superseded; an urgent is NEVER superseded (a
        # fired urgent stays until a human resolves it — never trade 'urgent never hidden' for de-duplication).
        keep_open = r["escalation_id"] in live_escalation_ids or r["level"] == "urgent"
        con.execute(
            "UPDATE escalations SET status = ? WHERE escalation_id = ?",
            ("open" if keep_open else "superseded", r["escalation_id"]),
        )


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
        "SELECT observation_id, member_id, response_id, severity, title, trigger_reason, "
        "data_version "
        "FROM observations WHERE member_id = ? AND data_version = ?",
        (member_id, data_version),
    ).fetchall()
    # member_explanation is NOT stored — it is derived at the /observations projection
    # (pipeline.observations); the Observation defaults it to "" here, filled in there.
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


#: Worst-first rank on the escalation-severity axis — the SQL mirror of ``models.LEVEL_TO_SEVERITY``
#: + ``SEVERITY_ORDER`` (rank explicitly, never string-compare: 'clinician_review' < 'urgent' happens
#: to sort right lexically, but only by accident — the models.py order-map rule applies in SQL too).
#: ONE home for BOTH escalation reads: every escalation surface leads highest-severity first, so the
#: two queries cannot drift apart on what "worst first" means.
_LEVEL_RANK_SQL = (
    "CASE level WHEN 'urgent' THEN 0 WHEN 'clinician_review' THEN 1 ELSE 2 END"
)

#: The escalation SELECT column list, in ``_escalation_from_row``'s read order — ONE home for the two reads
#: (:func:`get_escalations` + :func:`get_all_escalations`), so a new column (as ``status`` was) is a single
#: edit here + the row-mapper, never a per-query one that leaves the other read short a key. The
#: ``acknowledged`` overlay column is DERIVED here, not stored (the §720 'acknowledged' tier): TRUE iff an
#: ACTIVE ``escalation_accept`` feedback row targets this escalation — by its own id (a chat escalation, no
#: observation row) or by its ``observation_id`` (the finding dropdown's value for a data finding). Sharing
#: it here is what guarantees the member drill-in and the global queue can never disagree on it. Names are
#: ``escalations.``-qualified (no alias in either read, and ``feedback`` has its own ``member_id`` an
#: unqualified reference would capture). Deriving from the live feedback row means `/reset` (deactivate)
#: reverts the overlay symmetrically, and an escalation for CHANGED data (new ids) starts un-acknowledged.
#: SOURCE-FILTERED like ``resolve_overrides`` (clinician/system only — "the member experience cannot
#: lobby the safety logic", ui-ux §7): the docs promise the overlay derives from a CLINICIAN's accept, so
#: a member-sourced ``escalation_accept`` is stored for audit but INERT on the clinician queue — it must
#: not mark a triage item as already-reviewed or demote it within its severity tier.
_ESCALATION_COLUMNS = (
    "escalation_id, member_id, kind, dedup_key, level, observation_id, "
    "interaction_id, trigger_reason, created_at, status, "
    "EXISTS(SELECT 1 FROM feedback f WHERE f.kind = 'escalation_accept' AND f.active = 1 "
    "AND f.source IN ('clinician', 'system') "
    "AND f.member_id = escalations.member_id "
    "AND (f.target = escalations.escalation_id OR f.target = escalations.observation_id)"
    ") AS acknowledged"
)


def _escalation_from_row(r: sqlite3.Row) -> Escalation:
    """Row → ``Escalation`` (the two-layers mapping; shared by the per-member and global reads). The
    effective ``status`` folds in the read-time ``acknowledged`` overlay: only a stored 'open' can render
    as 'acknowledged' — a stored 'superseded' WINS over an accept row (the finding is gone from the data;
    dropping off the active queue is the stronger, safe fact), and re-opens as plain 'open' only via the
    scan's symmetric reconcile with the SAME ids (in which case the still-active accept row re-applies)."""
    status = r["status"]
    if status == "open" and r["acknowledged"]:
        status = "acknowledged"
    return Escalation(
        escalation_id=r["escalation_id"],
        member_id=r["member_id"],
        kind=r["kind"],
        dedup_key=r["dedup_key"],
        level=r["level"],
        observation_id=r["observation_id"],
        interaction_id=r["interaction_id"],
        trigger_reason=r["trigger_reason"],
        created_at=r["created_at"],
        status=status,
    )


def get_escalations(con: sqlite3.Connection, member_id: str) -> list[Escalation]:
    """One member's clinician-review record (read projection) — the per-member DRILL-IN shown when
    already viewing that member, NOT the queue. HIGHEST severity first (``_LEVEL_RANK_SQL``, the same
    worst-first lead as the global queue — no escalation surface may bury an urgent under older or
    softer rows), then oldest-first WITHIN a severity tier: the drill-in doubles as the audit view, so
    inside a tier it reads chronologically. Returns the full standing set across data_versions AND all
    lifecycle statuses ('open' / 'acknowledged' / 'superseded'): a re-scan RECONCILES an escalation's
    ``status`` (a clinician_review finding an override cleared flips to 'superseded'; an urgent one
    never does) but never DELETES the row. The cross-member triage queue — active ('open') only — is
    :func:`get_all_escalations`."""
    rows = con.execute(
        f"SELECT {_ESCALATION_COLUMNS} FROM escalations WHERE member_id = ? "
        f"ORDER BY {_LEVEL_RANK_SQL}, created_at, escalation_id",
        (member_id,),
    ).fetchall()
    return [_escalation_from_row(r) for r in rows]


def get_all_escalations(con: sqlite3.Connection) -> list[Escalation]:
    """The GLOBAL clinician-review queue across ALL members — the triage worklist (``GET /escalations``).
    Escalation exists to make sure a human sees something they didn't know to look for, so the queue is
    cross-member by design: a per-member read can only be opened by someone already on that patient, which
    is the one case escalation must not depend on.

    Ordered FOR triage — most-severe first (``urgent`` before ``clinician_review``), then un-acknowledged
    before 'acknowledged' WITHIN a tier (the worklist answers "what's outstanding?" and an acknowledged row
    is literally already-triaged — but severity still DOMINATES: an acknowledged urgent never sinks below an
    open clinician_review), then most-RECENT first, then ``escalation_id`` as a stable tie-break. Both
    escalation reads lead highest-severity first (the shared ``_LEVEL_RANK_SQL``); within a tier this
    worklist DELIBERATELY differs from the per-member :func:`get_escalations` (newest-first + the
    acknowledged demotion here vs. that view's chronological audit read) — do not "fix" one to match
    the other. A pure read, filtered to the STORED
    ``status='open'`` — the ACTIVE worklist (§720 lifecycle): a 'superseded' escalation (its
    clinician_review finding cleared by a /feedback override, per the scan's reconcile) drops OFF this queue
    but its row persists for audit (via :func:`get_escalations`); an 'acknowledged' one (the read-time
    overlay a clinician's active ``escalation_accept`` derives — stored 'open') STAYS listed, visibly
    triaged — acknowledging says "this was right", never "hide it". An 'urgent' escalation is never
    superseded, so a panic can never be cleared off this queue by a suppress. Rows are never deleted.

    Scope note: "all members" is the whole tenant here because the prototype has no clinician identity; in
    a real deployment this is a ``clinician_id``/panel scope, or it leaks other panels' members (§15)."""
    rows = con.execute(
        f"SELECT {_ESCALATION_COLUMNS} FROM escalations WHERE status = 'open' "
        f"ORDER BY {_LEVEL_RANK_SQL}, acknowledged, created_at DESC, escalation_id"
    ).fetchall()
    return [_escalation_from_row(r) for r in rows]


def escalation_acknowledged(
    con: sqlite3.Connection, member_id: str, target: str
) -> bool:
    """Whether the queue NOW shows an escalation matching ``target`` as 'acknowledged' — the
    ``POST /feedback`` response's honesty flag for an ``escalation_accept`` (the route only REPORTS it).
    Lives here, beside ``_ESCALATION_COLUMNS``, so the target-matching rule (escalation_id OR its
    observation_id) has ONE home next to the SQL overlay it mirrors — and it rides
    :func:`get_escalations`, the canonical read, so the source filter and superseded-wins semantics can
    never drift from what the queue actually renders."""
    return any(
        e.status == "acknowledged" and target in (e.escalation_id, e.observation_id)
        for e in get_escalations(con, member_id)
    )


# --------------------------------------------------------------------------------------------------
# Feedback-override resolution — the single seam where the deterministic half of self-improvement
# plugs in (Phase 7). Built now so analysis.py never reaches the DB. MUST return NEW lists (never
# mutate the caller's inputs in place). Phase 2: feedback is empty, so it is a faithful pass-through.
# --------------------------------------------------------------------------------------------------


def _resolve_range(
    ranges: list[ReferenceRange], marker: str, sex: str
) -> ReferenceRange | None:
    """The marker's reference range for a member of ``sex`` — the ``sex`` row, else the ``'any'`` fallback —
    mirroring ``analysis._range_for`` inline so db.py never imports ``analysis`` (the documented purity
    edge). The SINGLE home for a resolution both :func:`_latest_breaches_panic` (the panic-inert suppress
    guard) and the ``range_override`` inherit path in :func:`_apply_overrides` need; one function means a
    future change to the fallback tiers (e.g. an ``'other'``/``'unknown'`` rung, or age-banding) can't drift
    the two apart."""
    cands = [rg for rg in ranges if rg.marker == marker]
    return next((rg for rg in cands if rg.sex == sex), None) or next(
        (rg for rg in cands if rg.sex == "any"), None
    )


def _latest_breaches_panic(
    marker: str,
    results: list[LabResult],
    ranges: list[ReferenceRange],
    *,
    sex: str,
) -> bool:
    """Does ``marker``'s LATEST reading breach its panic bound? Replicates ``analysis._flags``' panic test
    EXACTLY — ``latest < panic_low`` / ``latest > panic_high`` on ``series[-1]`` (the reading with the max
    ``panel_date``, matching ``analysis._series``' ``sorted(key=panel_date)[-1]``), against the range
    resolved by the member's ``sex`` (then ``'any'``). Inline — the same no-``analysis``-import purity edge
    the range inheritance in :func:`_apply_overrides` uses — so ``suppress_marker`` can stay INERT against a
    panic floor without db.py importing the core. Err-SAFE: no results / no range -> ``False`` (analysis
    would produce no panic flag either), so an ambiguous case ALLOWS the suppress and never wrongly blocks
    one. A ``test_db`` regression pins this against the real ``safety.data_floor`` so it can't drift."""
    marker_results = [r for r in results if r.marker == marker]
    if not marker_results:
        return False
    latest = sorted(marker_results, key=lambda r: r.panel_date)[-1].value
    rng = _resolve_range(ranges, marker, sex)
    if rng is None:
        return False
    return (rng.panic_low is not None and latest < rng.panic_low) or (
        rng.panic_high is not None and latest > rng.panic_high
    )


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
      vanishes from the trajectory): quiets an expected-abnormal marker. INERT against a panic floor — a
      marker whose LATEST value breaches its panic bound is NOT dropped (:func:`_latest_breaches_panic`), so
      a suppress can never lower the deterministic floor urgent -> none; only a deliberate ``range_override``
      panic re-bound may clear a panic.

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
            # SAFETY — suppress is INERT against a panic floor (the invariant the operator UI + the
            # `suppress_marker` docstring below both promise). Suppress quiets an expected-abnormal marker's
            # range/trend flagging, but must NOT hide a PANIC-level value: dropping a panic marker removes it
            # from analyze(), so `safety.data_floor` would fall urgent -> none and a later /ask would not
            # escalate a life-threatening result. So skip the drop when the marker's LATEST value breaches
            # its panic bound. Only a DELIBERATE `range_override` panic re-bound may clear a panic (that path
            # is intended — `test_clearing_override_keeps_escalation_pinned_panic_observation` — and is left
            # untouched here). Checked against the CURRENT `out_ranges`, so a re-bound-then-suppress composes.
            if _latest_breaches_panic(fb.target, out_results, out_ranges, sex=sex):
                continue  # leave the panic marker in analysis -> the deterministic floor holds
            out_results = [r for r in out_results if r.marker != fb.target]
            out_ranges = [rg for rg in out_ranges if rg.marker != fb.target]
        elif fb.kind == "range_override" and fb.payload:
            # Inherit omitted bounds from the member's ACTUAL band: the (marker, sex) row, then the
            # (marker, 'any') fallback — the same resolution analysis._range_for uses, via the shared
            # _resolve_range helper (db.py's inline mirror, so it never imports analysis — the purity edge).
            existing = _resolve_range(out_ranges, fb.target, sex)
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


def get_feedback(con: sqlite3.Connection, member_id: str) -> list[FeedbackRecord]:
    """One member's full submitted-feedback trail, newest-first — the read-only AUDIT projection behind
    ``GET /members/{id}/feedback``. Returns EVERY row, ``active=0`` included (a ``/reset`` deactivates
    rather than deletes precisely so the trail survives — the flag shows the reviewer which rows still
    feed the live consumers). A pure read the clinician panel renders raw; the live paths consume active
    rows through their own seams (:func:`resolve_overrides` · :func:`get_active_preferences` ·
    :func:`get_active_signals`), never through this one."""
    rows = con.execute(
        "SELECT feedback_id, kind, target, payload_json, source, active, created_at "
        "FROM feedback WHERE member_id = ? "
        "ORDER BY created_at DESC, feedback_id DESC",
        (member_id,),
    ).fetchall()
    return [
        FeedbackRecord(
            feedback_id=r["feedback_id"],
            kind=r["kind"],
            target=r["target"],
            payload=json.loads(r["payload_json"]) if r["payload_json"] else None,
            source=r["source"],
            active=bool(r["active"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


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


def count_active_corrections(con: sqlite3.Connection) -> int:
    """Count active CORRECTION/preference rows (range_override/suppress_marker/preference) across all
    members — the input the DETERMINISTIC correction path consumes (:func:`resolve_overrides` /
    :func:`get_active_preferences`), which is a separate path from the /learn SIGNALS read by
    :func:`get_active_signals`. Lets ``learn.run_learn``'s empty-signals no-op tell the operator that
    feedback *exists* but is on the other path (it feeds every Scan/Ask, not /learn) — so an applied
    override no longer reads as "/learn picks up nothing"."""
    return con.execute(
        "SELECT COUNT(*) FROM feedback WHERE active = 1 "
        "AND kind IN ('range_override', 'suppress_marker', 'preference')"
    ).fetchone()[0]


def reset_learning(con: sqlite3.Connection) -> dict:
    """The ``POST /reset`` revert (architecture §9/§688): deactivate ALL feedback (``active=0``) and
    revert every learned prompt above the v0 baseline (``promoted`` OR ``rejected``) to
    ``status='reverted'`` — so the composer falls back to the latest remaining promoted version (v0, or
    the constant when none was ever seeded). Reverting the ``rejected`` rows too matters: a gate verdict
    is BASELINE-relative, and /reset changes the active baseline, so a candidate rejected against the old
    baseline must be re-gateable (not stuck cached as 'rejected') if its feedback is re-posted — the
    symmetric case to a reverted promotion. This is the learning-revert, NOT a data wipe: the feedback
    rows and prompt history are preserved (the trail stays), every member and the dataset untouched. The
    factory reset is the separate :func:`clear_all_data` (``POST /admin/reseed``).

    Returns affected-row counts PLUS ``affected_members`` — the members whose core overrides
    (``range_override`` / ``suppress_marker``, the analysis-affecting kinds) this call deactivated, which
    the reset route re-scans so their persisted artifacts actively revert. Captured by the deactivating
    ``UPDATE ... RETURNING`` itself (one statement, maps to Postgres), NOT by a separate read beforehand:
    a read-then-update pair left a gap where a core ``/feedback`` committing in the other worker between
    the two statements was deactivated but never revert-scanned — its superseded escalation / pruned
    observation then stayed stale with no active override and no self-heal trigger."""
    with con:
        rows = con.execute(
            "UPDATE feedback SET active = 0 WHERE active = 1 RETURNING member_id, kind"
        ).fetchall()
        pv = con.execute(
            "UPDATE prompt_versions SET status = 'reverted' "
            "WHERE version > 0 AND status IN ('promoted', 'rejected')"
        ).rowcount
    affected = sorted(
        {
            r["member_id"]
            for r in rows
            if r["kind"] in ("range_override", "suppress_marker")
        }
    )
    return {
        "feedback_deactivated": len(rows),
        "prompts_reverted": pv,
        "affected_members": affected,
    }


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


def _eval_summary(report_json: str | None) -> dict | None:
    """A compact digest of a stored ``eval_report_json`` for the Prompt-history projection — never the
    full report (it embeds the raw, re-scorable case set; inlining it per version would swamp the
    operator readout, and ``status`` already carries the gate's verdict). Parsed LENIENTLY with plain
    ``json``/.get (no ``eval.report.Report`` import — the serving library never pulls in ``eval``), so a
    report written under an older harness schema degrades to partial keys instead of a 500."""
    if not report_json:
        return None
    try:
        d = json.loads(report_json)
    except ValueError:
        return {"parse_error": True}
    if not isinstance(d, dict):
        # Valid JSON but not a report object ('null', a list, a scalar) — the same degrade as unparseable,
        # never an AttributeError that 500s the whole projection (the leniency promise above).
        return {"parse_error": True}
    cases = d.get("cases") or []
    if not isinstance(cases, list):
        cases = []
    return {
        "dataset": d.get("dataset"),
        "model_version": d.get("model_version"),
        "n_runs": d.get("n_runs"),
        "generated_at": d.get("generated_at"),
        "cases": len(cases),
        # Count scorer results that FIRED a never_event. Derived from the mode1/mode2 scorer lists because
        # that is what Report.to_json() actually serializes — CaseReport.never_events is a plain @property
        # the dump never writes, so reading a top-level "never_events" key would be structurally 0 on
        # every real stored report (a safety-rejected candidate would digest as zero blocking failures).
        "never_events": sum(
            1
            for c in cases
            if isinstance(c, dict)
            for s in [*(c.get("mode2") or []), *(c.get("mode1") or [])]
            if isinstance(s, dict) and s.get("never_event")
        ),
    }


def get_prompt_versions(con: sqlite3.Connection) -> list[PromptVersionRecord]:
    """The FULL composer-prompt history, newest version first — the read projection behind the
    operator's Prompt-history readout (``GET /prompts``). Every row is returned regardless of
    ``status`` (this is the audit trail of the learning loop: v0 = the seeded baseline; each later row
    a ``/learn`` candidate with its gate verdict — ``promoted``/``rejected`` — or ``reverted`` where a
    ``/reset`` rolled a promotion back). ``active`` flags the one row the composer resolves now — the
    latest ``promoted``, exactly :func:`get_active_prompt`'s pick, derived here from the same ordering
    so the two reads cannot disagree. A pure read; the consumption path stays
    :func:`get_active_prompt`, so this projection can never become a second serving seam."""
    rows = con.execute(
        "SELECT version, prompt_text, status, eval_report_json, created_at "
        "FROM prompt_versions ORDER BY version DESC"
    ).fetchall()
    active_version = next(
        (r["version"] for r in rows if r["status"] == "promoted"), None
    )
    return [
        PromptVersionRecord(
            version=r["version"],
            status=r["status"],
            active=r["version"] == active_version,
            created_at=r["created_at"],
            prompt_text=r["prompt_text"],
            eval_summary=_eval_summary(r["eval_report_json"]),
        )
        for r in rows
    ]


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
    """How many CANDIDATE prompt_versions rows (``version >= 1``) carry ``created_at`` on ``date_prefix``
    (``YYYY-MM-DD``) — the learn DAILY-CAP backstop. Debounced short-circuits write no row, so they don't
    count (free). ``version = 0`` (the v0 baseline seeded by startup / reseed) is EXCLUDED: it is not a
    ``/learn`` run, so on a fresh-DB / reseed / cold-start day it must not consume one of the day's slots
    (which silently dropped the cap 20 → 19)."""
    return con.execute(
        "SELECT COUNT(*) AS c FROM prompt_versions "
        "WHERE version >= 1 AND substr(created_at, 1, 10) = ?",
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


def write_baseline_prompt(
    con: sqlite3.Connection, *, prompt_text: str, commit: bool = True
) -> None:
    """Upsert the v0 baseline row (``version=0``, always ``status='promoted'``, no report) — the SINGLE
    writer of version 0, distinct from :func:`insert_prompt_version`'s ``version>=1`` candidates. This is
    the SEED path only (startup / reseed); the report is attached LAZILY and separately by the first
    ``/learn`` via :func:`set_prompt_report`, which is why this never takes a report.

    The ``DO UPDATE`` keeps v0's ``prompt_text`` synced to the incoming base (so a redeploy that edits
    ``BASE_COMPOSE_SYSTEM`` on a persisted disk re-syncs v0 — NOT a learned vN>0, which is a frozen
    snapshot of the base-at-assembly and only picks up a new base on a re-learn). It is text-aware about
    the report so v0 can never carry a report that measures a DIFFERENT prompt text: when the text is
    UNCHANGED an idempotent re-seed PRESERVES an already-attached report (must not clobber it to NULL);
    when the text CHANGED it DROPS the now-stale report (→ NULL) so the next ``/learn`` recomputes the
    baseline against the new text. ``created_at`` is set only on first insert. The caller passes
    ``prompt_text`` (``learn`` owns the ``llm.BASE_COMPOSE_SYSTEM`` constant) so this module stays free of
    an ``llm`` import. ``commit=False`` defers the commit to the caller's transaction (the atomic reseed)."""
    created_at = datetime.now(UTC).isoformat()
    with con if commit else nullcontext():
        con.execute(
            "INSERT INTO prompt_versions (version, prompt_text, status, eval_report_json, created_at) "
            "VALUES (0, ?, 'promoted', NULL, ?) "
            "ON CONFLICT(version) DO UPDATE SET "
            "  prompt_text = excluded.prompt_text, "
            "  status = 'promoted', "
            "  eval_report_json = CASE "
            "    WHEN excluded.prompt_text = prompt_versions.prompt_text "
            "      THEN prompt_versions.eval_report_json "  # unchanged text: keep any lazily-attached report
            "    ELSE NULL END",  # changed text: drop the now-stale report
            (prompt_text, created_at),
        )


def set_prompt_report(
    con: sqlite3.Connection, version: int, eval_report_json: str
) -> None:
    """Attach (or replace) the eval report on an existing ``prompt_versions`` row (commits). The
    report-attach seam ``learn._baseline_report`` uses to cache the lazily-computed baseline report onto
    the ACTIVE prompt's row (v0, or a learned vN) — version-general, unlike :func:`write_baseline_prompt`
    which only writes v0. It only touches ``eval_report_json`` (never ``prompt_text``/``status``), so it
    cannot introduce the text/report mismatch the seed path guards against. A no-op if ``version`` absent."""
    with con:
        con.execute(
            "UPDATE prompt_versions SET eval_report_json = ? WHERE version = ?",
            (eval_report_json, version),
        )


# --------------------------------------------------------------------------------------------------
# Factory reset (Phase 7) — the destructive clean-slate behind ``POST /admin/reseed`` (architecture
# §688/§756), DISTINCT from reset_learning's learning-only revert. The whole wipe + re-ingest + v0 seed
# runs as ONE atomic transaction (``reseed_transaction`` + ``clear_all_data``) so a concurrent request
# never reads a half-wiped DB; the back-edge to preprocessing (re-ingest) stays in api.py, not here.
# --------------------------------------------------------------------------------------------------


@contextmanager
def reseed_transaction(con: sqlite3.Connection):
    """One ATOMIC transaction for the whole ``POST /admin/reseed`` (truncate → re-ingest → v0 seed): a
    concurrent request on another connection sees the pre-reseed state or the post-reseed state, never a
    half-wiped DB (the old ``nuke_all`` committed mid-op, between DROP and re-ingest, exposing an empty DB
    — and DROP+recreate can't be made atomic here because ``init_db``'s ``executescript`` force-commits a
    pending transaction). FK enforcement is turned OFF on THIS connection only (per-connection; readers
    keep theirs) so truncate order is irrelevant, and everything commits once at the end / rolls back to
    the prior populated state on any error. The ``foreign_keys`` PRAGMA must be toggled with no active
    transaction, so it brackets the ``with`` body (which the caller fills with ``commit=False`` writes)."""
    con.execute(
        "PRAGMA foreign_keys = OFF"
    )  # per-connection; must be set with NO active transaction
    try:
        yield
        con.commit()  # the single all-or-nothing commit for the whole reseed
    except Exception:
        con.rollback()  # any failure reverts to the prior populated state, not an empty DB
        raise
    finally:
        con.execute(
            "PRAGMA foreign_keys = ON"
        )  # restore the per-connection invariant connect() sets


def clear_all_data(con: sqlite3.Connection) -> None:
    """TRUNCATE — ``DELETE`` every row from every table — the destructive half of the reseed, run INSIDE
    :func:`reseed_transaction` (which owns the commit and has FK enforcement off, so table order is
    irrelevant). Truncate, not DROP+recreate: the schema is already correct (``init_db`` owns it) and
    ``DELETE`` composes inside a transaction, whereas ``executescript`` (which recreating the schema needs)
    force-commits a pending transaction and would break the reseed's atomicity. Robust to ANY table that
    exists (names read live from ``sqlite_master``), like the old drop-list was."""
    for (name,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        con.execute(f'DELETE FROM "{name}"')  # noqa: S608 — names from sqlite_master, not input


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
