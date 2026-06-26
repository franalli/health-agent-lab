"""Phase 3a — the proactive deterministic spine: safety floor/validator + the scan over persisted data.

Covers the §48 "Review" surface and the runnable check: the validator enforces ``escalation >= floor``
(and never lowers it); the scan emits ranked observations and persists each finding as its own
interaction (the FK it hangs off); finding text and its evidence stat describe the SAME signal; the
curated potassium panic (C07, K+ 6.1) forces an ``urgent`` floor and writes exactly one escalation; a
re-scan is idempotent (no duplicate rows); the negative control (C02) stays calm — no escalations; and
the data_finding dedup is finding-stable: a notes-only edit (which moves ``data_version`` but not the
analysis) does not mint a second escalation. In-memory SQLite throughout; nothing touches the store
except through db.py.
"""

import pytest

from health_intelligence import db, pipeline, safety
from health_intelligence.analysis import analyze
from health_intelligence.config import ANALYSIS_CONFIG, CONFIG_VERSION, MODEL_VERSION_DETERMINISTIC
from health_intelligence.models import (
    HealthIntelligenceResponse,
    Note,
    ResponseMetadata,
    SEVERITY_ORDER,
)
from preprocessing.ingest import ingest_bundle, ingest_dataset


# ---- helpers (match test_db.py style) ------------------------------------------------------------

def _con():
    con = db.connect(":memory:")
    db.init_db(con)
    return con


def _r(analyte, value, unit, reference_range):
    return {"analyte": analyte, "value": value, "unit": unit, "reference_range": reference_range}


def _panel(panel_id, date, results, vitals=None):
    return {
        "panel_id": panel_id, "collected_date": date, "results": results,
        "vitals": vitals or {"systolic_bp": 118, "diastolic_bp": 76, "bmi": 22.5},
    }


def _bundle(member_id, panels, *, sex="male", age=50, notes=None):
    from health_intelligence.models import MemberBundle
    return MemberBundle.model_validate({
        "member_id": member_id,
        "profile": {"member_id": member_id, "age": age, "sex": sex,
                    "conditions": [], "medications": [], "family_history": [], "lifestyle": {}},
        "panels": panels,
        "notes": notes or [],
    })


def _resp(escalation):
    """A minimal HealthIntelligenceResponse at a given escalation level (for validator tests)."""
    return HealthIntelligenceResponse(
        answer="x", answer_disposition="answered", escalation=escalation,
        metadata=ResponseMetadata(response_id="r", data_version="d",
                                  model_version=MODEL_VERSION_DETERMINISTIC, config_version=CONFIG_VERSION),
    )


# ---- safety: floor projection + validator --------------------------------------------------------

def test_data_floor_is_the_cores_projection_verbatim():
    con = _con()
    ingest_bundle(con, _bundle("M1", [_panel("M1-P1", "2024-01-01",
                  [_r("Potassium", 6.1, "mmol/L", "3.5-5.1")])]))
    member, results, ranges, age, dv = db.load_for_analysis(con, "M1")
    analysis = analyze(member, results, ranges, age, ANALYSIS_CONFIG, data_version=dv)
    # panic -> urgent floor, and data_floor returns exactly what the pure core projected (no re-derive).
    assert analysis.overall_floor == "urgent"
    assert safety.data_floor(analysis) == analysis.overall_floor


def test_validator_rejects_under_floor_and_passes_at_or_above():
    # below the floor -> rejected (a softer answer can't sit on a harder floor)
    with pytest.raises(safety.FloorViolation):
        safety.validate(_resp("none"), "clinician_review")
    with pytest.raises(safety.FloorViolation):
        safety.validate(_resp("clinician_review"), "urgent")
    # at or above the floor -> passes through unchanged (ranked, not string-compared)
    assert safety.validate(_resp("clinician_review"), "clinician_review").escalation == "clinician_review"
    assert safety.validate(_resp("urgent"), "clinician_review").escalation == "urgent"
    assert safety.validate(_resp("none"), "none").escalation == "none"


def test_severity_to_level_mirrors_the_floor_projection():
    assert safety.severity_to_level("urgent") == "urgent"
    assert safety.severity_to_level("attention") == "clinician_review"
    assert safety.severity_to_level("notable") is None
    assert safety.severity_to_level("info") is None


def test_per_marker_levels_max_to_the_overall_floor():
    # the structural guarantee: max over per-marker escalation levels == the core's whole-member floor
    # (both derive from SEVERITY_TO_FLOOR), so the scan can't queue an escalation below the response floor
    from health_intelligence.analysis import _floor
    from health_intelligence.models import FLOOR_ORDER
    for combo in (["info"], ["notable"], ["attention"], ["urgent"],
                  ["notable", "attention"], ["attention", "urgent"], ["info", "notable", "urgent"]):
        levels = [lvl for s in combo if (lvl := safety.severity_to_level(s)) is not None]
        max_level = max(levels, key=lambda l: FLOOR_ORDER[l], default="none")
        assert max_level == _floor(combo)


# ---- scan: ranked observations + persisted interaction (the FK) ----------------------------------

def test_scan_emits_ranked_finding_and_persists_one_interaction_each():
    con = _con()
    # a panic potassium (urgent) alongside an out-of-range LDL (notable) -> ranking must put urgent first
    ingest_bundle(con, _bundle("M2", [_panel("M2-P1", "2024-01-01", [
        _r("Potassium", 6.1, "mmol/L", "3.5-5.1"),
        _r("LDL cholesterol", 180.0, "mg/dL", "0-100"),
    ])]))
    obs = pipeline.scan(con, "M2")
    assert [o.severity for o in obs] == sorted(
        (o.severity for o in obs), key=lambda s: -SEVERITY_ORDER[s]
    )
    assert obs[0].severity == "urgent" and obs[0].title.startswith("Potassium")

    # per-finding (architecture §48): one interaction per observation, each driver='scan', ids distinct,
    # and observations FK 1:1 onto them
    rows = con.execute("SELECT response_id, driver FROM interactions WHERE member_id='M2'").fetchall()
    assert len(rows) == len(obs) and all(r["driver"] == "scan" for r in rows)
    resp_ids = {r["response_id"] for r in rows}
    assert {o.response_id for o in obs} == resp_ids and len(resp_ids) == len(obs)


def test_finding_text_and_evidence_describe_the_same_signal():
    con = _con()
    # diastolic_bp: carries reference bounds but no CVa/CVi, and a strictly-rising 5-point series -> it is
    # both above_range AND FDR-significant-but-uncounted. The unified classifier must make the Finding
    # text and its evidence stat agree (not 'above range' text with a Mann-Kendall stat).
    dates = ["2022-01-01", "2022-07-01", "2023-01-01", "2023-07-01", "2024-01-01"]
    dia = [82, 84, 86, 88, 92]
    panels = [_panel(f"M4-P{i+1}", dates[i],
                     [_r("Potassium", 4.2, "mmol/L", "3.5-5.1")],
                     vitals={"systolic_bp": 118, "diastolic_bp": dia[i], "bmi": 22.5})
              for i in range(5)]
    ingest_bundle(con, _bundle("M4", panels))
    pipeline.scan(con, "M4")
    rows = con.execute("SELECT response_json FROM interactions WHERE member_id='M4'").fetchall()
    import json
    for r in rows:
        resp = json.loads(r["response_json"])
        for f in resp["findings"]:
            text, stat = f["text"], (f["evidence"][0]["stat"] or "")
            trendy_text = "rising" in text or "falling" in text
            trendy_stat = "Mann-Kendall" in stat
            assert trendy_text == trendy_stat, f"text/evidence signal mismatch: {text!r} vs {stat!r}"


def test_scan_with_no_signal_writes_nothing():
    con = _con()
    ingest_bundle(con, _bundle("M3", [_panel("M3-P1", "2024-01-01",
                  [_r("Potassium", 4.2, "mmol/L", "3.5-5.1")])]))
    obs = pipeline.scan(con, "M3")
    # per-finding: no raised marker -> no findings -> no interactions/observations/escalations
    assert obs == []
    assert db.get_escalations(con, "M3") == []
    assert con.execute("SELECT COUNT(*) FROM interactions WHERE member_id='M3'").fetchone()[0] == 0


# ---- the supplied members: the happy path and the negative control -------------------------------

def test_c07_potassium_forces_urgent_and_writes_exactly_one_escalation_idempotently():
    con = _con()
    ingest_dataset(con)
    obs = pipeline.scan(con, "C07")
    esc = db.get_escalations(con, "C07")
    assert any(o.severity == "urgent" and o.title.startswith("Potassium") for o in obs)
    assert len(esc) == 1
    assert esc[0].level == "urgent" and esc[0].kind == "data_finding"
    assert "Potassium" in esc[0].dedup_key

    obs_count = con.execute("SELECT COUNT(*) FROM observations WHERE member_id='C07'").fetchone()[0]
    int_count = con.execute("SELECT COUNT(*) FROM interactions WHERE member_id='C07'").fetchone()[0]

    # a re-scan over identical data is idempotent: no new observation, interaction, or escalation rows
    pipeline.scan(con, "C07")
    assert con.execute("SELECT COUNT(*) FROM observations WHERE member_id='C07'").fetchone()[0] == obs_count
    assert con.execute("SELECT COUNT(*) FROM interactions WHERE member_id='C07'").fetchone()[0] == int_count
    assert len(db.get_escalations(con, "C07")) == 1


def test_c02_negative_control_raises_no_escalation():
    con = _con()
    ingest_dataset(con)
    member, results, ranges, age, dv = db.load_for_analysis(con, "C02")
    analysis = analyze(member, results, ranges, age, ANALYSIS_CONFIG, data_version=dv)
    assert analysis.overall_floor == "none"
    pipeline.scan(con, "C02")
    assert db.get_escalations(con, "C02") == []


def test_genuine_data_change_supersedes_the_prior_observation_set():
    con = _con()
    # four in-range panels, then a fifth re-ingest that pushes potassium into panic — a real data change
    base = [_panel(f"M5-P{i+1}", d, [_r("Potassium", 4.2, "mmol/L", "3.5-5.1")])
            for i, d in enumerate(["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"])]
    ingest_bundle(con, _bundle("M5", base))
    pipeline.scan(con, "M5")
    assert pipeline.scan(con, "M5") == [] or db.get_observations(con, "M5") == []  # calm at baseline

    panic = base + [_panel("M5-P5", "2026-01-01", [_r("Potassium", 6.1, "mmol/L", "3.5-5.1")])]
    ingest_bundle(con, _bundle("M5", panic))  # data_version AND analysis_version both move
    obs = pipeline.scan(con, "M5")

    # the version-scoped read returns ONLY the new set — prior-version rows are hidden, not accumulated
    assert all(o.severity == "urgent" for o in obs) and obs
    current_dv = db.compute_data_version(con, "M5")
    assert all(o.data_version == current_dv for o in obs)


def test_data_finding_dedup_is_stable_across_a_notes_only_edit():
    con = _con()
    ingest_dataset(con)
    pipeline.scan(con, "C07")
    assert len(db.get_escalations(con, "C07")) == 1
    dv_before = db.compute_data_version(con, "C07")

    # Re-persist C07 with an added note: data_version moves, but analyze() reads no notes, so the
    # analysis_version (and thus the data_finding dedup_key) is unchanged -> no second escalation.
    member = db.get_member(con, "C07")
    assert member is not None
    results = db.get_results(con, "C07")
    ranges = db.get_ranges(con, markers={r.marker for r in results})
    notes = db.get_notes(con, "C07") + [Note(date="2026-01-01", source="in-app", text="added note")]
    db.replace_member(con, profile=member, results=results, ranges=ranges, notes=notes)

    assert db.compute_data_version(con, "C07") != dv_before  # the full-record hash did move
    pipeline.scan(con, "C07")
    assert len(db.get_escalations(con, "C07")) == 1  # but the finding-stable dedup held
