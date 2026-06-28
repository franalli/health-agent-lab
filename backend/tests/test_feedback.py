"""Phase 7 — the deterministic correction path (``/feedback`` overrides + ``/reset``).

These prove the architecture's headline self-correction claim at the seam that matters: an override
re-resolves into ``analysis.analyze``'s inputs (so a flag changes), and ``reset_learning`` reverts it —
all WITHOUT the core ever reaching the DB (the override is folded in by ``db.resolve_overrides`` before
``analyze`` runs). LDL cholesterol is the test marker: it has no band-cutpoints/graded-bands, so a
range widen cleanly flips its single ``above_range`` flag with no residual band signal.
"""

import sqlite3

import pytest
from builders import fresh_con, make_bundle, make_panel, make_result

from health_intelligence import db, pipeline
from health_intelligence.analysis import analyze
from health_intelligence.config import ANALYSIS_CONFIG
from health_intelligence.models import Feedback
from preprocessing.ingest import ingest_bundle


def _analysis(con, member_id):
    """Load (override-resolved) + analyze — the same path scan/ask run."""
    member, results, ranges, age, dv = db.load_for_analysis(con, member_id)
    return analyze(member, results, ranges, age, ANALYSIS_CONFIG, data_version=dv)


def _ldl(analysis):
    return next(m for m in analysis.markers if m.marker == "LDL cholesterol")


def _seed(con):
    ingest_bundle(
        con,
        make_bundle(
            "M1",
            [
                make_panel(
                    "M1-P1",
                    "2024-01-15",
                    [
                        make_result("LDL cholesterol", 160.0, "mg/dL", "<100"),
                        make_result("HbA1c", 5.3, "%", "<5.7"),
                    ],
                )
            ],
        ),
    )


def test_range_override_clears_a_flag_then_reset_restores_it():
    con = fresh_con()
    _seed(con)

    # Before: LDL 160 is above its <100 range.
    assert "above_range" in _ldl(_analysis(con, "M1")).flags

    # A clinician re-bounds LDL (the override re-resolves into analyze's inputs, both modes).
    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="range_override",
            target="LDL cholesterol",
            payload={"ref_high": 200.0},
            source="clinician",
        ),
    )
    after = _ldl(_analysis(con, "M1"))
    assert (
        "above_range" not in after.flags
    )  # the flag changed — no DB surgery, no rule change

    # /reset reverts: the override deactivates and the flag returns.
    db.reset_learning(con)
    assert "above_range" in _ldl(_analysis(con, "M1")).flags


def test_range_override_is_per_member_and_resolves_for_any_sex():
    # The override row is written sex='any'; _range_for must resolve it for a male member.
    con = fresh_con()
    _seed(con)  # M1 is male (builders default)
    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="range_override",
            target="LDL cholesterol",
            payload={"ref_high": 200.0},
            source="clinician",
        ),
    )
    assert "above_range" not in _ldl(_analysis(con, "M1")).flags


def test_suppress_marker_drops_it_from_the_analysis():
    con = fresh_con()
    _seed(con)
    db.insert_feedback(
        con,
        "M1",
        Feedback(kind="suppress_marker", target="LDL cholesterol", source="clinician"),
    )
    a = _analysis(con, "M1")
    assert all(m.marker != "LDL cholesterol" for m in a.markers)  # gone entirely
    assert any(m.marker == "HbA1c" for m in a.markers)  # siblings untouched


def test_latest_active_range_override_wins():
    con = fresh_con()
    _seed(con)
    # Two overrides for the same marker, applied in created_at order — the later one wins.
    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="range_override",
            target="LDL cholesterol",
            payload={"ref_high": 120.0},  # still below 160 -> would stay flagged
            source="clinician",
        ),
        created_at="2024-01-01T00:00:00+00:00",
    )
    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="range_override",
            target="LDL cholesterol",
            payload={"ref_high": 200.0},  # the later, wider bound clears the flag
            source="clinician",
        ),
        created_at="2024-02-01T00:00:00+00:00",
    )
    assert "above_range" not in _ldl(_analysis(con, "M1")).flags


def test_preference_is_not_an_analysis_input_but_reaches_the_composer_hint():
    con = fresh_con()
    _seed(con)
    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="preference", payload={"text": "keep answers brief"}, source="member"
        ),
    )
    # preference must NOT touch analysis (LDL still flagged) ...
    assert "above_range" in _ldl(_analysis(con, "M1")).flags
    # ... but is surfaced as a composer hint.
    assert db.get_active_preferences(con, "M1") == ["keep answers brief"]
    db.reset_learning(con)
    assert db.get_active_preferences(con, "M1") == []


def test_get_active_signals_excludes_overrides():
    con = fresh_con()
    _seed(con)
    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="incorrect",
            target="f:x",
            payload={"question": "q", "corrected_answer": "a"},
            source="clinician",
        ),
    )
    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="range_override",
            target="LDL cholesterol",
            payload={"ref_high": 200.0},
            source="clinician",
        ),
    )
    signals = db.get_active_signals(con)
    assert [s.kind for s in signals] == [
        "incorrect"
    ]  # the override is not a learn signal


# ---- the scan reconciles its observation set after an override (the prune) ------------------------


def _scan_titles(con, member_id):
    return [o.title for o in pipeline.scan(con, member_id)]


def test_override_clearing_a_flag_prunes_the_observation_on_rescan():
    # Overrides don't bump data_version, so a re-scan must still drop a now-cleared marker's
    # observation (the §48 set-reconciliation), or the panel strands a stale finding.
    con = fresh_con()
    panels = [
        make_panel(
            f"M1-P{i}",
            f"{2021 + i}-01-01",
            [make_result("LDL cholesterol", v, "mg/dL", "<100")],
        )
        for i, v in enumerate([150, 155, 160, 165, 170], start=1)
    ]
    ingest_bundle(con, make_bundle("M1", panels))
    assert any("LDL" in t for t in _scan_titles(con, "M1"))  # raised before

    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="range_override",
            target="LDL cholesterol",
            payload={"ref_high": 500.0},
            source="clinician",
        ),
    )
    assert not any(
        "LDL" in t for t in _scan_titles(con, "M1")
    )  # pruned after the override
    db.reset_learning(con)
    assert any("LDL" in t for t in _scan_titles(con, "M1"))  # restored on reset


def test_override_keeps_an_escalation_pinned_observation():
    # A panic K+ escalates; an override that clears the analytical flag must NOT delete the observation
    # an escalation points at (the RESTRICT FK + the durable clinician task) — the prune keeps it.
    con = fresh_con()
    ingest_bundle(
        con,
        make_bundle(
            "M1",
            [
                make_panel(
                    "M1-P1",
                    "2024-01-15",
                    [make_result("Potassium", 6.1, "mmol/L", "3.5-5.0")],
                )
            ],
        ),
    )
    assert any("Potassium" in t for t in _scan_titles(con, "M1"))
    assert db.get_escalations(con, "M1")  # an urgent data-finding was queued

    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="range_override",
            target="Potassium",
            payload={"ref_high": 10.0, "panic_high": 10.0},
            source="clinician",
        ),
    )
    # the analytical flag is gone, but the escalation-pinned observation + the durable task remain
    assert any("Potassium" in t for t in _scan_titles(con, "M1"))
    assert db.get_escalations(con, "M1")


# ---- review-fix regressions ----------------------------------------------------------------------

_HB_SPLIT = "13.5-17.5 (male) / 12.0-15.5 (female)"  # a sex-split reference range


def test_one_sided_override_on_sex_split_marker_inherits_the_members_own_band():
    # A one-sided range_override (omitting ref_low) on a sex-split marker must inherit the MEMBER'S sex
    # band, not an arbitrary row. A male member's Hemoglobin override keeps the male ref_low (13.5).
    con = fresh_con()
    ingest_bundle(
        con,
        make_bundle(
            "MALE",
            [
                make_panel(
                    "p1",
                    "2024-01-15",
                    [make_result("Hemoglobin", 14.0, "g/dL", _HB_SPLIT)],
                )
            ],
            sex="male",
        ),
    )
    db.insert_feedback(
        con,
        "MALE",
        Feedback(
            kind="range_override",
            target="Hemoglobin",
            payload={
                "ref_high": 18.0
            },  # one-sided: ref_low must be inherited from the MALE band
            source="clinician",
        ),
    )
    _, _, ranges, _, _ = db.load_for_analysis(con, "MALE")
    hb = next(r for r in ranges if r.marker == "Hemoglobin")
    assert hb.ref_low == 13.5  # male band inherited (not the female 12.0)


def test_member_sourced_override_is_inert_clinician_applies():
    # A member must not be able to change what's flagged (only clinician/system overrides feed analysis).
    con = fresh_con()
    _seed(con)  # LDL 160, above its <100 range
    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="range_override",
            target="LDL cholesterol",
            payload={"ref_high": 500.0},
            source="member",
        ),
    )
    assert (
        "above_range" in _ldl(_analysis(con, "M1")).flags
    )  # member override ignored for analysis
    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="range_override",
            target="LDL cholesterol",
            payload={"ref_high": 500.0},
            source="clinician",
        ),
    )
    assert (
        "above_range" not in _ldl(_analysis(con, "M1")).flags
    )  # clinician override applies


def test_override_to_one_marker_does_not_duplicate_anothers_escalation():
    # The data-finding dedup keys on the PER-MARKER analysis state, so an override to Potassium must not
    # re-fire Hemoglobin's already-queued escalation on the next scan.
    con = fresh_con()
    ingest_bundle(
        con,
        make_bundle(
            "M1",
            [
                make_panel(
                    "M1-P1",
                    "2024-01-15",
                    [
                        make_result(
                            "Potassium", 6.1, "mmol/L", "3.5-5.0"
                        ),  # panic_high
                        make_result(
                            "Hemoglobin", 6.0, "g/dL", _HB_SPLIT
                        ),  # panic_low (7.0)
                    ],
                )
            ],
            sex="male",
        ),
    )

    def _hb_escalations():
        return [
            e for e in db.get_escalations(con, "M1") if ":Hemoglobin:" in e.dedup_key
        ]

    pipeline.scan(con, "M1")
    assert len(_hb_escalations()) == 1  # both markers escalated; Hemoglobin fired once

    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="range_override",
            target="Potassium",  # override ONLY Potassium
            payload={"ref_high": 99.0, "panic_high": 99.0},
            source="clinician",
        ),
    )
    pipeline.scan(con, "M1")
    assert (
        len(_hb_escalations()) == 1
    )  # NOT duplicated by the Potassium override (per-marker dedup)


def test_insert_feedback_same_timestamp_gets_distinct_ids():
    # Two same-(member,kind,target,created_at) posts must not collide on the PK / 500 — salted + retried.
    con = fresh_con()
    _seed(con)
    ts = "2024-01-01T00:00:00+00:00"
    fb = Feedback(kind="helpful", target="f:1", source="member")
    id1 = db.insert_feedback(con, "M1", fb, created_at=ts)
    id2 = db.insert_feedback(con, "M1", fb, created_at=ts)
    assert id1 != id2
    assert len(db.get_active_signals(con)) == 2


def test_insert_feedback_surfaces_a_real_constraint_error_not_an_id_error():
    # The salt/retry must ONLY catch the feedback_id PK collision — an FK violation (unknown member) must
    # surface as IntegrityError immediately, not spin 1000× into a misleading "couldn't synthesize an id".
    con = fresh_con()  # no members exist
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_feedback(
            con, "GHOST", Feedback(kind="helpful", target="f:1", source="member")
        )
