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


# ---- global queue ordering (GET /escalations) ----------------------------------------------------


def test_get_all_escalations_ranks_severity_then_recency_across_members():
    """The global triage worklist: most-severe first (urgent before clinician_review), then
    most-recent first, spanning members. created_at is passed explicitly so the ordering is asserted
    deterministically (not at the mercy of a wall clock). Contrast the per-member get_escalations,
    which is oldest-first — this difference is intentional (a worklist, not a history)."""
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
