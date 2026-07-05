"""The data layer over persisted data — db.py + ingest.py wired to a real (in-memory) SQLite.

Covers the Phase-2 contract: the mandated idempotent escalation emit; the row<->model round-trip;
the safety-critical panic transcription; vitals folded as markers with config units; panel_id
survival; re-ingest replacing a member and bumping (only on change) the content-hash data_version;
the override seam never mutating its inputs; and the end-to-end floor — a curated potassium panic
reaching ``urgent`` through ingest -> load -> analyze, on both a synthetic series and the supplied
C07. The schema loads via db.init_db; nothing here touches SQLite except through db.py.
"""

import pytest
from builders import (
    fresh_con as _con,
)
from builders import (
    make_bundle as _bundle,
)
from builders import (
    make_panel as _panel,
)
from builders import (
    make_result as _r,
)

from health_intelligence import db
from health_intelligence.analysis import analyze
from health_intelligence.config import ANALYSIS_CONFIG, CONFIG_VERSION
from health_intelligence.models import LabResult, ReferenceRange
from preprocessing.ingest import ingest_bundle, ingest_dataset

# helpers (_con / _r / _panel / _bundle) now live in tests/builders.py — imported above


def _k_panels(values):
    """Potassium-only panels at 6-month spacing, range as the data prints it."""
    dates = ["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01"]
    return [
        _panel(f"M-P{i + 1}", dates[i], [_r("Potassium", v, "mmol/L", "3.5-5.1")])
        for i, v in enumerate(values)
    ]


# ---- mandated: idempotent escalation emit --------------------------------------------------------


def test_emit_escalation_is_idempotent():
    con = _con()
    # escalations.member_id is NOT NULL REFERENCES members, and connect() turns FKs ON, so the parent
    # must exist or the insert raises IntegrityError instead of writing "one row".
    con.execute("INSERT INTO members (member_id, sex) VALUES ('C07', 'male')")
    con.commit()
    key = "data:C07:Potassium:abc123"
    created1 = db.emit_escalation(
        con,
        member_id="C07",
        kind="data_finding",
        dedup_key=key,
        level="urgent",
        trigger_reason="K+ 6.1",
    )
    created2 = db.emit_escalation(
        con,
        member_id="C07",
        kind="data_finding",
        dedup_key=key,
        level="urgent",
        trigger_reason="K+ 6.1",
    )
    assert created1 is True and created2 is False
    n = con.execute(
        "SELECT COUNT(*) FROM escalations WHERE dedup_key = ?", (key,)
    ).fetchone()[0]
    assert n == 1


def test_insert_escalation_refresh_reason_heals_a_stale_data_finding_reason():
    """A re-emitted ``data_finding`` (same dedup_key, same level) with ``refresh_reason`` UPDATEs the
    kept-first row's ``trigger_reason`` to the current wording — so a Theil-Sen/tau format change
    deployed over a durable DB heals on the next scan instead of leaving the queue's reason diverged from
    the observation's re-derived one. Heals at BOTH data_finding levels: clinician_review AND urgent (the
    urgent tier must not be shadowed by the day-scoped urgent-UPGRADE branch, whose WHERE matches nothing
    on an already-urgent row). Without the flag the reason is kept-first (chat's behavior); neither path
    creates a second row or changes the level."""
    con = _con()
    con.execute("INSERT INTO members (member_id, sex) VALUES ('C07', 'male')")
    con.commit()

    def _emit(key, level, reason, *, refresh):
        return db._insert_escalation(
            con,
            member_id="C07",
            kind="data_finding",
            dedup_key=key,
            level=level,
            trigger_reason=reason,
            refresh_reason=refresh,
        )

    def _reason(key):
        return con.execute(
            "SELECT trigger_reason FROM escalations WHERE dedup_key = ?", (key,)
        ).fetchone()[0]

    for level, key in [
        ("clinician_review", "data:C07:HbA1c:v1"),
        (
            "urgent",
            "data:C07:Potassium:v1",
        ),  # the panic tier the shadowing bug silently skipped
    ]:
        new = "Mann-Kendall p=0.017, tau=1.00, n=5, Theil-Sen slope 0.03/day"
        assert (
            _emit(key, level, "old p/n-only wording", refresh=True) is True
        )  # created
        # a re-scan re-emits the same key at the same level with the new wording -> heals in place
        assert _emit(key, level, new, refresh=True) is False
        row = con.execute(
            "SELECT level, trigger_reason FROM escalations WHERE dedup_key = ?", (key,)
        ).fetchone()
        assert row["level"] == level, level  # level never moved
        assert "Theil-Sen slope 0.03/day" in row["trigger_reason"], level

        # refresh_reason=False (the chat default) keeps the first reason: a no-op re-emit does not overwrite
        assert _emit(key, level, "a different, later reason", refresh=False) is False
        assert _reason(key) == new, level


# ---- global queue ordering (GET /escalations) ----------------------------------------------------


def test_get_all_escalations_ranks_severity_then_recency_across_members():
    """The global triage worklist: most-severe first (urgent before clinician_review), then
    most-recent first, spanning members. created_at is passed explicitly so the ordering is asserted
    deterministically (not at the mercy of a wall clock). The per-member get_escalations leads
    worst-first too but is oldest-first WITHIN a tier — that difference is intentional (a worklist
    vs. a history)."""
    con = _con()
    for m in ("A", "B", "C"):
        con.execute("INSERT INTO members (member_id, sex) VALUES (?, 'male')", (m,))
    con.commit()
    # (member, level, created_at, dedup_key) — deliberately interleaved so neither member nor
    # insert order matches the expected output.
    rows = [
        (
            "A",
            "clinician_review",
            "2024-01-01T00:00:00+00:00",
            "data:A:m1:v1",
        ),  # oldest
        ("B", "urgent", "2024-02-01T00:00:00+00:00", "data:B:m2:v1"),  # only urgent
        ("C", "clinician_review", "2024-03-01T00:00:00+00:00", "data:C:m3:v1"),
        (
            "A",
            "clinician_review",
            "2024-04-01T00:00:00+00:00",
            "data:A:m4:v1",
        ),  # newest c_r
    ]
    for member, level, created_at, key in rows:
        db.emit_escalation(
            con,
            member_id=member,
            kind="data_finding",
            dedup_key=key,
            level=level,
            trigger_reason="t",
            created_at=created_at,
        )

    queue = db.get_all_escalations(con)
    # urgent first; then clinician_review newest→oldest. By dedup_key that is: B, A(04), C(03), A(01).
    assert [e.dedup_key for e in queue] == [
        "data:B:m2:v1",
        "data:A:m4:v1",
        "data:C:m3:v1",
        "data:A:m1:v1",
    ]
    assert queue[0].level == "urgent"  # worst-first
    assert {e.member_id for e in queue} == {"A", "B", "C"}  # genuinely cross-member
    # the clinician_review tail is strictly recency-descending
    cr = [e for e in queue if e.level == "clinician_review"]
    assert [e.created_at for e in cr] == sorted(
        (e.created_at for e in cr), reverse=True
    )
    # each item carries its severity on the shared observation axis (derived from level at read)
    assert [e.severity for e in queue] == [
        "urgent",
        "attention",
        "attention",
        "attention",
    ]


def test_get_escalations_drill_in_leads_worst_first_with_derived_severity():
    """The per-member drill-in ranks highest-severity first like the queue (no escalation surface may
    bury an urgent under older/softer rows), oldest-first WITHIN a tier (it doubles as the audit
    history). Each item carries the derived ``severity`` (models.LEVEL_TO_SEVERITY — computed from
    ``level`` at read, never stored): urgent → 'urgent', clinician_review → 'attention'."""
    con = _con()
    con.execute("INSERT INTO members (member_id, sex) VALUES ('A', 'male')")
    con.commit()
    # (level, created_at, dedup_key) — interleaved so insert order matches neither tier nor time.
    rows = [
        ("clinician_review", "2024-01-01T00:00:00+00:00", "data:A:m1:v1"),
        ("urgent", "2024-02-01T00:00:00+00:00", "data:A:m2:v1"),
        ("clinician_review", "2024-03-01T00:00:00+00:00", "data:A:m3:v1"),
        ("urgent", "2024-04-01T00:00:00+00:00", "data:A:m4:v1"),
    ]
    for level, created_at, key in rows:
        db.emit_escalation(
            con,
            member_id="A",
            kind="data_finding",
            dedup_key=key,
            level=level,
            trigger_reason="t",
            created_at=created_at,
        )
    drill = db.get_escalations(con, "A")
    # urgent tier first (oldest→newest inside it), then clinician_review (oldest→newest).
    assert [e.dedup_key for e in drill] == [
        "data:A:m2:v1",
        "data:A:m4:v1",
        "data:A:m1:v1",
        "data:A:m3:v1",
    ]
    assert [e.severity for e in drill] == ["urgent", "urgent", "attention", "attention"]


# ---- row <-> model round-trip --------------------------------------------------------------------


def test_member_round_trips_through_db():
    con = _con()
    bundle = _bundle(
        "M3",
        [
            _panel(
                "M3-P1",
                "2024-01-01",
                [
                    _r(
                        "Hemoglobin",
                        12.5,
                        "g/dL",
                        "13.5-17.5 (male) / 12.0-15.5 (female)",
                    )
                ],
            )
        ],
        sex="female",
        age=29,
        conditions=["iron-deficiency anemia"],
        medications=["ferrous sulfate"],
        family_history=["mother: hypothyroidism"],
        lifestyle={"diet": "vegetarian"},
        notes=[{"date": "2024-01-02", "source": "in-app", "text": "feeling tired"}],
    )
    ingest_bundle(con, bundle)

    p = db.get_member(con, "M3")
    assert p is not None
    assert p.sex == "female" and p.age == 29
    assert p.conditions == ["iron-deficiency anemia"] and p.medications == [
        "ferrous sulfate"
    ]
    assert p.family_history == ["mother: hypothyroidism"] and p.lifestyle == {
        "diet": "vegetarian"
    }

    notes = db.get_notes(con, "M3")
    assert (
        len(notes) == 1
        and notes[0].source == "in-app"
        and notes[0].text == "feeling tired"
    )


# ---- safety-critical: panic thresholds transcribed from config onto every range row --------------


def test_reseed_transaction_rolls_back_to_prior_state_on_failure():
    # B2: a mid-reseed failure (after the truncate, before the re-ingest completes) must leave the PRIOR
    # populated DB, never a half-wiped one — the atomicity the old nuke_all lacked (it committed mid-op).
    con = _con()
    ingest_dataset(con)  # the supplied 15-member bundle
    assert con.execute("SELECT COUNT(*) FROM members").fetchone()[0] == 15
    with pytest.raises(RuntimeError, match="boom"):
        with db.reseed_transaction(con):
            db.clear_all_data(con)  # truncate every table (inside the transaction)
            assert (
                con.execute("SELECT COUNT(*) FROM members").fetchone()[0] == 0
            )  # gone mid-transaction...
            raise RuntimeError("boom mid-reseed")
    # ...but the rollback restores them — the DB is never left empty for a concurrent reader.
    assert con.execute("SELECT COUNT(*) FROM members").fetchone()[0] == 15


def test_apply_migrations_backfills_escalation_status_on_an_old_db():
    # §720 lifecycle: a DB created under the OLD schema.sql (escalations with no `status` column) must gain
    # it on the next init_db, back-filling existing rows to 'open' — an additive ALTER, never a destructive
    # rewrite. Simulate the pre-lifecycle shape, then run the migration directly.
    import sqlite3

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute(  # the pre-lifecycle escalations shape: no `status`
        "CREATE TABLE escalations (escalation_id TEXT PRIMARY KEY, member_id TEXT, kind TEXT, "
        "dedup_key TEXT, level TEXT, observation_id TEXT, interaction_id TEXT, trigger_reason TEXT, "
        "created_at TEXT)"
    )
    con.execute(
        "INSERT INTO escalations (escalation_id, member_id, kind, dedup_key, level, trigger_reason, "
        "created_at) VALUES ('e1','C01','data_finding','k1','urgent','why','2026-01-01')"
    )
    con.commit()
    assert "status" not in {r[1] for r in con.execute("PRAGMA table_info(escalations)")}

    db._apply_migrations(con)  # additive: ADD COLUMN status NOT NULL DEFAULT 'open'

    assert "status" in {r[1] for r in con.execute("PRAGMA table_info(escalations)")}
    assert (  # existing row back-filled in place (not rewritten / lost)
        con.execute(
            "SELECT status FROM escalations WHERE escalation_id='e1'"
        ).fetchone()[0]
        == "open"
    )
    db._apply_migrations(
        con
    )  # idempotent: a second run is a clean no-op (column already present)
    assert "status" in {r[1] for r in con.execute("PRAGMA table_info(escalations)")}
    assert (  # column + back-filled data survive the no-op second run (not just "didn't raise")
        con.execute(
            "SELECT status FROM escalations WHERE escalation_id='e1'"
        ).fetchone()[0]
        == "open"
    )


def test_count_prompt_versions_on_excludes_the_v0_baseline():
    # C6: the seeded v0 baseline (version 0) is NOT a /learn run and must not consume one of the day's
    # DAILY_LEARN_CAP slots (which silently dropped the cap 20 -> 19 on a reseed / cold-start day); only
    # real candidates (version >= 1) count.
    from health_intelligence import learn

    con = _con()
    learn.seed_baseline_prompt(con)  # v0 row, created_at = now
    day = con.execute(
        "SELECT substr(created_at, 1, 10) FROM prompt_versions WHERE version = 0"
    ).fetchone()[0]
    assert db.count_prompt_versions_on(con, day) == 0  # v0 excluded
    con.execute(
        "INSERT INTO prompt_versions (version, prompt_text, status, eval_report_json, created_at) "
        "VALUES (1, 'x', 'rejected', NULL, ?)",
        (day + "T12:00:00",),
    )
    con.commit()
    assert db.count_prompt_versions_on(con, day) == 1  # a real candidate DOES count


def test_reference_ranges_carry_config_panic():
    con = _con()
    ingest_dataset(con)  # the supplied 15-member bundle
    ranges = {(r.marker, r.sex): r for r in db.get_ranges(con)}

    k = ranges[("Potassium", "any")]
    assert (k.ref_low, k.ref_high) == (3.5, 5.1)  # parsed from the data
    assert (k.panic_low, k.panic_high) == (2.8, 6.0)  # transcribed from config

    g = ranges[("Fasting glucose", "any")]
    assert (g.panic_low, g.panic_high) == (50.0, 500.0)

    # sex-split marker -> both rows carry the (sex-agnostic) panic threshold
    assert ranges[("Hemoglobin", "male")].panic_low == 7.0
    assert ranges[("Hemoglobin", "female")].panic_low == 7.0

    # a marker with no curated panic keeps None (typed absence, not a fabricated 0)
    assert ranges[("Total cholesterol", "any")].panic_low is None
    assert ranges[("Total cholesterol", "any")].panic_high is None


# ---- vitals folded as markers with config-supplied units -----------------------------------------


def test_vitals_folded_as_markers_with_config_units():
    con = _con()
    ingest_dataset(con)
    mid = db.list_members(con)[0]
    results = db.get_results(con, mid)
    markers = {r.marker for r in results}
    assert {"systolic_bp", "diastolic_bp", "bmi"} <= markers

    units = {r.marker: r.unit for r in results}
    assert units["systolic_bp"] == "mmHg" and units["diastolic_bp"] == "mmHg"
    assert units["bmi"] == "kg/m2"

    ranges = {(r.marker, r.sex): r for r in db.get_ranges(con)}
    assert (ranges[("bmi", "any")].ref_low, ranges[("bmi", "any")].ref_high) == (
        18.5,
        25.0,
    )


# ---- panel_id survives the flatten ---------------------------------------------------------------


def test_panel_id_preserved_on_flatten():
    con = _con()
    ingest_dataset(con)
    results = db.get_results(con, "C01")
    assert any(r.panel_id == "C01-P1" for r in results)
    # every result keeps a non-empty panel id (the trajectory's grouping key)
    assert all(r.panel_id for r in results)


# ---- re-ingest replaces the member and bumps data_version only on change -------------------------


def test_reingest_replaces_member_and_bumps_data_version():
    con = _con()
    panels = [
        _panel(f"M4-P{i}", d, [_r("HbA1c", v, "%", "<5.7")])
        for i, (d, v) in enumerate(
            [
                ("2024-01-01", 5.3),
                ("2024-07-01", 5.4),
                ("2025-01-01", 5.3),
                ("2025-07-01", 5.4),
            ],
            start=1,
        )
    ]
    ingest_bundle(con, _bundle("M4", panels))
    v1 = db.compute_data_version(con, "M4")
    n1 = len(db.get_results(con, "M4"))

    # identical re-ingest -> same fingerprint, no duplicated rows (replaced in place)
    ingest_bundle(con, _bundle("M4", panels))
    assert db.compute_data_version(con, "M4") == v1
    assert len(db.get_results(con, "M4")) == n1

    # one changed value -> the fingerprint bumps, still no duplication
    changed = [
        _panel(f"M4-P{i}", d, [_r("HbA1c", v, "%", "<5.7")])
        for i, (d, v) in enumerate(
            [
                ("2024-01-01", 5.3),
                ("2024-07-01", 5.4),
                ("2025-01-01", 5.3),
                ("2025-07-01", 6.4),
            ],
            start=1,
        )
    ]
    ingest_bundle(con, _bundle("M4", changed))
    assert db.compute_data_version(con, "M4") != v1
    assert len(db.get_results(con, "M4")) == n1


# ---- override seam returns new lists and never mutates its inputs ---------------------------------


def test_resolve_overrides_returns_new_lists_without_mutating():
    con = _con()
    con.execute("INSERT INTO members (member_id, sex) VALUES ('M5', 'male')")
    con.commit()
    results = [
        LabResult(
            marker="HbA1c", value=5.3, unit="%", panel_id="p1", panel_date="2024-01-01"
        )
    ]
    ranges = [
        ReferenceRange(
            marker="HbA1c",
            sex="any",
            unit="%",
            ref_high=5.7,
            config_version=CONFIG_VERSION,
        )
    ]

    out_results, out_ranges = db.resolve_overrides(
        con, "M5", results, ranges, sex="male"
    )
    assert out_results is not results and out_ranges is not ranges  # new list objects
    assert len(results) == 1 and len(ranges) == 1  # inputs untouched
    assert [r.marker for r in out_results] == [
        "HbA1c"
    ]  # no overrides -> pass-through content


# ---- end-to-end safety floor: curated potassium panic reaches urgent over the DB -----------------


def test_potassium_61_reaches_urgent_over_the_db():
    con = _con()
    ingest_bundle(con, _bundle("M1", _k_panels([4.3, 4.2, 4.4, 4.3, 6.1]), age=58))
    member, results, ranges, age, data_version = db.load_for_analysis(con, "M1")
    analysis = analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version
    )
    assert analysis.overall_floor == "urgent"  # panic_high (6.0, from config) < 6.1
    k = next(m for m in analysis.markers if m.marker == "Potassium")
    assert "panic_high" in k.flags


def test_real_member_c07_potassium_reaches_urgent():
    # the architecture's named happy-path, against the actual supplied data (C07 ends at K+ 6.1)
    con = _con()
    ingest_dataset(con)
    member, results, ranges, age, data_version = db.load_for_analysis(con, "C07")
    analysis = analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version
    )
    assert analysis.overall_floor == "urgent"
    k = next(m for m in analysis.markers if m.marker == "Potassium")
    assert "panic_high" in k.flags


def test_all_supplied_members_analyze_over_the_db():
    # the phase's headline runnable: the pure core runs over EVERY persisted member, not just the two
    # spot-checked ones. The sparse member (C12, n=3) still abstains on the *trend* ("too short to call"
    # — no Mann-Kendall verdict), but its floor now escalates via the sub-n_min sparse rule (Fix 2:
    # monotonic adverse change that clears RCV), exercised here through persistence.
    con = _con()
    ingest_dataset(con)
    floors = {}
    for mid in db.list_members(con):
        member, results, ranges, age, dv = db.load_for_analysis(con, mid)
        floors[mid] = analyze(
            member, results, ranges, age, ANALYSIS_CONFIG, data_version=dv
        ).overall_floor
    assert len(floors) == 15
    assert floors["C07"] == "urgent"  # the one curated panic in the supplied data

    member, results, ranges, age, dv = db.load_for_analysis(con, "C12")
    c12 = analyze(member, results, ranges, age, ANALYSIS_CONFIG, data_version=dv)
    assert len({r.panel_id for r in results}) == 3  # the sparse member (n=3 < n_min=4)
    assert all(
        mk.trend is None for mk in c12.markers
    )  # no MK/Theil-Sen verdict at n=3 (trend abstains)
    egfr = next(mk for mk in c12.markers if mk.marker == "eGFR")
    assert (
        egfr.severity == "attention"
    )  # ...yet the monotonic RCV-clearing decline escalates (Fix 2)
    assert c12.overall_floor == "clinician_review"  # matching eval E12


def test_sex_split_panic_reaches_unknown_sex_member():
    # Harden (architecture §6, "rather over-escalate than miss"): a sex-split marker carrying a config
    # panic also gets a panic-only ('any') row, so an 'other'/'unknown'-sex member — with no male/female
    # row to match — still hits the panic floor instead of silently getting `no_reference`. Without it,
    # Hemoglobin's panic_low=7.0 would never fire for such a member (the exact panic-doesn't-fire risk).
    con = _con()
    panels = [
        _panel(
            f"U-P{i}",
            d,
            [_r("Hemoglobin", v, "g/dL", "13.5-17.5 (male) / 12.0-15.5 (female)")],
        )
        for i, (d, v) in enumerate(
            [
                ("2024-01-01", 11.0),
                ("2024-07-01", 9.5),
                ("2025-01-01", 8.0),
                ("2025-07-01", 6.5),
            ],
            start=1,
        )
    ]
    ingest_bundle(con, _bundle("U1", panels, sex="unknown"))

    # the panic-only fallback exists, ref bounds stay None (no sex-agnostic normal range invented),
    # and the sex-specific rows are untouched
    ranges = {(r.marker, r.sex): r for r in db.get_ranges(con)}
    assert ranges[("Hemoglobin", "any")].panic_low == 7.0
    assert (
        ranges[("Hemoglobin", "any")].ref_low is None
        and ranges[("Hemoglobin", "any")].ref_high is None
    )
    assert ranges[("Hemoglobin", "male")].ref_low == 13.5

    member, results, ranges_l, age, dv = db.load_for_analysis(con, "U1")
    a = analyze(member, results, ranges_l, age, ANALYSIS_CONFIG, data_version=dv)
    hb = next(m for m in a.markers if m.marker == "Hemoglobin")
    assert "panic_low" in hb.flags  # 6.5 < 7.0 fires even with no sex-specific range
    assert a.overall_floor == "urgent"


def test_marker_name_collision_with_vital_is_rejected():
    # labs + vitals share one marker namespace; a lab analyte named 'bmi' collides with the folded
    # vital on result_id. The firewall must reject it with a clear error, not an opaque PK crash.
    con = _con()
    panels = [_panel("X-P1", "2024-01-01", [_r("bmi", 27.0, "kg/m2", "18.5-25.0")])]
    with pytest.raises(ValueError, match="duplicate marker"):
        ingest_bundle(con, _bundle("X1", panels))


def test_divergent_reference_range_across_members_is_rejected():
    # reference_ranges is global + constant per (marker, sex, config); a second member printing a
    # different range for the same marker must fail loud, not silently clobber the shared row.
    con = _con()
    ingest_bundle(
        con,
        _bundle("A1", [_panel("A-P1", "2024-01-01", [_r("HbA1c", 5.3, "%", "<5.7")])]),
    )
    with pytest.raises(ValueError, match="diverges from the stored definition"):
        ingest_bundle(
            con,
            _bundle(
                "B1",
                [_panel("B-P1", "2024-01-01", [_r("HbA1c", 5.3, "%", "<6.0")])],
                sex="female",
            ),
        )


# ---- DB-driven analysis reproduces an in-memory (Phase-1 style) construction ----------------------


def test_load_for_analysis_matches_in_memory_fixture():
    con = _con()
    series = [
        ("2024-01-05", 96.0),
        ("2024-07-05", 95.0),
        ("2025-01-05", 97.0),
        ("2025-07-05", 96.0),
    ]
    panels = [
        _panel(f"M2-P{i}", d, [_r("eGFR", v, "mL/min/1.73m2", ">=90")])
        for i, (d, v) in enumerate(series, start=1)
    ]
    ingest_bundle(con, _bundle("M2", panels, sex="male", age=50))

    member, results, ranges, age, dv = db.load_for_analysis(con, "M2")
    db_analysis = analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=dv
    )

    # the same eGFR inputs assembled directly in memory must yield the same per-marker verdict
    mem_results = [
        LabResult(
            marker="eGFR",
            value=v,
            unit="mL/min/1.73m2",
            panel_id=f"M2-P{i}",
            panel_date=d,
        )
        for i, (d, v) in enumerate(series, start=1)
    ]
    mem_ranges = [
        ReferenceRange(
            marker="eGFR",
            sex="any",
            unit="mL/min/1.73m2",
            ref_low=90.0,
            ref_high=None,
            config_version=CONFIG_VERSION,
        )
    ]
    mem_analysis = analyze(
        member, mem_results, mem_ranges, 50, ANALYSIS_CONFIG, data_version=dv
    )

    db_egfr = next(m for m in db_analysis.markers if m.marker == "eGFR")
    mem_egfr = next(m for m in mem_analysis.markers if m.marker == "eGFR")
    assert db_egfr.severity == mem_egfr.severity
    assert db_egfr.flags == mem_egfr.flags
    assert db_egfr.latest.value == 96.0  # last reading round-tripped through SQLite
    assert (
        db_analysis.overall_floor == "none"
    )  # eGFR + default vitals all in-range/stable


# --------------------------------------------------------------------------------------------------
# process_lock — the cross-process mutual-exclusion seam the --workers 2 deploy rests on (startup init
# + the learning-state lock). flock locks the open file DESCRIPTION, so a second fd in THIS process
# models a second worker faithfully; the subprocess test exercises two genuinely separate processes.
# --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(db.fcntl is None, reason="flock is POSIX-only")
def test_process_lock_contends_and_releases(tmp_path):
    """Non-blocking contention yields False (never blocks, never raises); exit releases the lock."""
    con = db.connect(str(tmp_path / "health.db"))
    with db.process_lock(con, "startup", blocking=False) as first:
        assert first is True
        with db.process_lock(con, "startup", blocking=False) as second:
            assert second is False  # a second holder (fd == another worker) is refused
        # names partition the exclusion domains: a different name is a different lock file
        with db.process_lock(con, "learn", blocking=False) as other_domain:
            assert other_domain is True
    with db.process_lock(con, "startup", blocking=False) as reacquired:
        assert reacquired is True  # released with the fd — no leak, no stale lock
    con.close()


@pytest.mark.skipif(db.fcntl is None, reason="flock is POSIX-only")
def test_process_lock_raises_on_a_real_flock_failure(tmp_path, monkeypatch):
    """A non-contention OSError from flock (ENOLCK/EIO — the locking FACILITY is broken, not busy) must
    PROPAGATE, never yield False: a soft False fails OPEN for the blocking startup caller (init would run
    unserialized — the exact race the lock closes) and misreports an infra fault as an endless 409 for
    the non-blocking callers. Contention (BlockingIOError) stays the one soft outcome."""
    import errno

    con = db.connect(str(tmp_path / "health.db"))

    def broken_flock(fd, op):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(db.fcntl, "flock", broken_flock)
    with pytest.raises(OSError):
        with db.process_lock(con, "startup", blocking=True):
            pass  # pragma: no cover — the acquire raises before the body runs
    with pytest.raises(OSError):
        with db.process_lock(con, "learn", blocking=False):
            pass  # pragma: no cover — the acquire raises before the body runs
    con.close()


@pytest.mark.skipif(db.fcntl is None, reason="flock is POSIX-only")
def test_process_lock_blocking_acquire_is_bounded(tmp_path, monkeypatch):
    """blocking=True polls with a deadline: a WEDGED holder (a dead one auto-releases its flock) turns
    into a loud TimeoutError, instead of an untimed flock wait that is SIGTERM-immune (the handler runs
    but PEP 475 retries the syscall) and would stall the losing worker until the platform's SIGKILL."""
    import os as _os

    monkeypatch.setattr(db, "_BLOCKING_ACQUIRE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(db, "_BLOCKING_ACQUIRE_POLL_S", 0.02)
    con = db.connect(str(tmp_path / "health.db"))
    lock_path = db._process_lock_path(con, "startup")
    fd = _os.open(lock_path, _os.O_CREAT | _os.O_RDWR, 0o600)
    db.fcntl.flock(fd, db.fcntl.LOCK_EX | db.fcntl.LOCK_NB)  # the wedged holder
    try:
        with pytest.raises(TimeoutError):
            with db.process_lock(con, "startup", blocking=True):
                pass  # pragma: no cover — the acquire times out before the body runs
    finally:
        db.fcntl.flock(fd, db.fcntl.LOCK_UN)
        _os.close(fd)
    con.close()


@pytest.mark.skipif(db.fcntl is None, reason="flock is POSIX-only")
def test_concurrent_first_init_serializes_under_the_startup_lock(tmp_path):
    """The --workers 2 boot race, for real: two PROCESSES first-initialize the same fresh DB file at
    once. init_db's presence-check + executescript is check-then-act (bare CREATE TABLE), so without
    the startup lock the loser crashes on "table already exists"; under the lock both must exit 0 and
    leave one intact schema."""
    import subprocess
    import sys

    db_path = str(tmp_path / "health.db")
    code = (
        "import sys\n"
        "from health_intelligence import db\n"
        "con = db.connect(sys.argv[1])\n"
        "with db.process_lock(con, 'startup', blocking=True):\n"
        "    db.init_db(con)\n"
        "con.close()\n"
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, db_path],
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    for p in procs:
        _, err = p.communicate(timeout=60)
        assert p.returncode == 0, f"concurrent first-init crashed: {err}"
    con = db.connect(db_path)
    tables = {
        r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert db._EXPECTED_TABLES <= tables  # one intact schema, no partial wreckage
    con.close()


def test_add_column_if_missing_tolerates_the_check_then_act_race(tmp_path):
    """The migration race closer: a racer whose PRAGMA guard read STALE state (column looks absent,
    then the ALTER finds it already added by the winner) must no-op, not crash the losing worker's
    startup. A non-duplicate DDL failure must still surface."""
    con = db.connect(str(tmp_path / "health.db"))
    con.execute("CREATE TABLE t (a TEXT)")
    db._add_column_if_missing(con, "t", "b TEXT NOT NULL DEFAULT ''")
    assert {r[1] for r in con.execute("PRAGMA table_info(t)")} == {"a", "b"}

    class StaleGuardCon:
        """Delegates to the real connection but reports the column ABSENT — the loser's stale read."""

        def __init__(self, real):
            self._real = real

        def execute(self, sql, *args):
            if sql.startswith("PRAGMA table_info"):
                return iter(
                    []
                )  # stale: no columns visible -> guard passes -> ALTER runs
            return self._real.execute(sql, *args)

        def commit(self):
            self._real.commit()

    db._add_column_if_missing(
        StaleGuardCon(con), "t", "b TEXT NOT NULL DEFAULT ''"
    )  # no crash
    with pytest.raises(
        db.sqlite3.OperationalError
    ):  # a REAL DDL failure is never masked
        db._add_column_if_missing(
            StaleGuardCon(con), "t", "c BOGUSTYPE NOT NULL DEFAULT"
        )
    con.close()


# ---- prompt-history projection (GET /prompts) -----------------------------------------------------


def test_get_prompt_versions_history_newest_first_with_active_flag():
    """The Prompt-history read projection (db.get_prompt_versions): every prompt_versions row,
    newest version first, regardless of status — the audit trail of the learn loop (prompts are
    LEARNT from feedback and VALIDATED by /learn's eval gate; `status` records each verdict).
    `active` marks exactly get_active_prompt's pick — the LATEST 'promoted' row — so a later
    rejected candidate never steals it. `eval_summary` is the compact digest, never the stored
    report: None for the report-less seeded v0; counts (cases, never_events) for a gated row."""
    con = _con()
    db.write_baseline_prompt(con, prompt_text="BASE")  # v0: promoted, report-less
    # The stored report is a REAL harness serialization (Report.to_json), not hand-crafted JSON: the
    # digest's never_events count must read what the serializer actually writes (the mode1/mode2 scorer
    # results — CaseReport.never_events is a plain @property the dump never emits), and a hand-written
    # fixture once masked exactly that mismatch (the digest read a key no real report carries -> always 0).
    from eval.report import CaseReport, Report
    from eval.types import ScorerResult

    report = Report(
        dataset="training_data",
        model_version="m",
        config_version="c",
        n_runs=1,
        generated_at="2026-01-01T00:00:00+00:00",
        cases=[
            CaseReport(
                case_id="E07",
                category="c",
                tags=[],
                mode1_covered=False,
                mode2=[ScorerResult(dimension="grounding", passed=True)],
            ),
            CaseReport(
                case_id="A10",
                category="c",
                tags=[],
                mode1_covered=False,
                mode2=[
                    ScorerResult(
                        dimension="grounding",
                        passed=False,
                        never_event="fabricated_value",
                    )
                ],
            ),
        ],
    ).to_json()
    db.insert_prompt_version(
        con,
        version=1,
        prompt_text="BASE + exemplar",
        status="promoted",
        eval_report_json=report,
        created_at="2026-01-02T00:00:00+00:00",
    )
    db.insert_prompt_version(
        con,
        version=2,
        prompt_text="BASE + exemplar 2",
        status="rejected",
        eval_report_json=report,
        created_at="2026-01-03T00:00:00+00:00",
    )

    hist = db.get_prompt_versions(con)
    assert [h.version for h in hist] == [2, 1, 0]  # newest first
    assert [h.status for h in hist] == ["rejected", "promoted", "promoted"]
    # active == the latest PROMOTED (v1): the rejected v2 never steals it, and the flag agrees with
    # the composer's own resolution seam by construction.
    assert [h.active for h in hist] == [False, True, False]
    assert db.get_active_prompt(con)[0] == 1
    # full prompt_text IS returned (the inspectable artifact) ...
    assert hist[1].prompt_text == "BASE + exemplar"
    # ... but the report is digested, never inlined raw
    assert (
        hist[2].eval_summary is None
    )  # v0 seeds report-less (/learn attaches it lazily)
    assert (
        hist[0].eval_summary
        == {
            "dataset": "training_data",
            "model_version": "m",
            "n_runs": 1,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "cases": 2,
            "never_events": 1,  # counted from the serialized scorer results a REAL report carries
        }
    )


def test_eval_summary_degrades_on_non_object_json():
    """LENIENT means lenient on every shape: valid JSON that is not a report OBJECT ('null', a list, a
    scalar) must degrade to the parse_error sentinel exactly like unparseable text — an AttributeError
    here would 500 the whole GET /prompts projection over one bad row (a hand-edited durable-disk DB or
    a foreign writer; every in-app writer stores Report.to_json())."""
    for bad in ("null", "[]", '"x"', "3", "not json {"):
        assert db._eval_summary(bad) == {"parse_error": True}
    assert db._eval_summary(None) is None
    assert db._eval_summary("") is None
