"""Phase 0 contract guard — the runnable bar made repeatable.

Proves the three Phase-0 acceptance criteria (schema loads, models round-trip over the real data)
plus the config-curation contract (what is anchored vs curated vs deliberately deferred as typed
absence). The curation test tracks forward — Phase 1 fills the eval-forced subset, so it now asserts
both the anchors and the surviving skip-paths; the analysis suite itself lives in test_analysis.py.
"""

import json
import pathlib
import sqlite3
from typing import get_args

import pytest
from pydantic import ValidationError

from health_intelligence import config
from preprocessing.datasets import members_path
from health_intelligence.models import (
    FLOOR_ORDER,
    SEVERITY_ORDER,
    FloorLevel,
    HealthIntelligenceResponse,
    MarkerTrajectory,
    MemberBundle,
    Reading,
    ResponseMetadata,
    Severity,
    TrajectoryAnalysis,
)

BACKEND = pathlib.Path(__file__).resolve().parent.parent
SCHEMA = BACKEND / "schema.sql"
MEMBERS = members_path()  # active dataset's bundle (defaults to data/training_data/)

EXPECTED_TABLES = {
    "escalations", "feedback", "interactions", "lab_results", "members",
    "notes", "observations", "prompt_versions", "reference_ranges",
}


def test_schema_loads_the_nine_tables():
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA.read_text())
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert tables == EXPECTED_TABLES


def test_every_supplied_member_validates():
    members = json.loads(MEMBERS.read_text())
    bundles = [MemberBundle.model_validate(m) for m in members]
    assert len(bundles) == 15
    # the sparse member is present (drives the Phase-1 n=3 abstention case)
    sparse = next(b for b in bundles if b.member_id == "C12")
    assert len(sparse.panels) == 3
    # vitals folded as required floats on every panel
    assert all(p.vitals.systolic_bp > 0 for b in bundles for p in b.panels)


def test_computed_contracts_round_trip():
    analysis = TrajectoryAnalysis(
        member_id="C01", data_version="d1", overall_floor="none",
        markers=[MarkerTrajectory(
            marker="HbA1c", unit="%",
            latest=Reading(value=5.3, date="2024-02-12"), severity="info",
        )],
    )
    assert TrajectoryAnalysis.model_validate_json(analysis.model_dump_json()) == analysis

    resp = HealthIntelligenceResponse(
        answer="Your HbA1c is within range.", answer_disposition="answered",
        metadata=ResponseMetadata(
            response_id="r1", data_version="d1",
            model_version="deterministic", config_version=config.CONFIG_VERSION,
        ),
    )
    assert HealthIntelligenceResponse.model_validate_json(resp.model_dump_json()) == resp


def test_config_anchors_and_typed_absence():
    markers = config.ANALYSIS_CONFIG.markers
    assert len(markers) == 19  # 16 labs + 3 vitals
    assert config.ANALYSIS_CONFIG.stats.n_min == 4  # so n=3 abstains

    # Anchored in the supplied materials -> filled now:
    assert markers["HbA1c"].band_cutpoints == (5.7, 6.5)
    assert len(markers["Vitamin D (25-OH)"].graded_bands) == 3
    assert markers["Potassium"].panic_high == 6.0  # eval E07: K+ 6.1 -> urgent
    # unit is config-supplied for vitals only (data prints none); labs get their unit from the data
    assert markers["systolic_bp"].unit == "mmHg"
    assert markers["HbA1c"].unit is None

    # Phase-1 curation (eval-forced + demo-prominent), each sourced inline in config.py:
    assert markers["HbA1c"].adverse_direction == "up"
    assert markers["HDL cholesterol"].adverse_direction == "down"   # protective -> lower is adverse
    assert markers["HbA1c"].cva == 0.6 and markers["HbA1c"].cvi == 1.2  # EFLM
    # Spot-check more RCV-driving CVa/CVi so a transposed value on any escalation-relevant marker fails here:
    assert markers["CRP"].cva == 21.0 and markers["CRP"].cvi == 42.0
    assert markers["TSH"].cva == 9.85 and markers["TSH"].cvi == 19.7
    assert markers["eGFR"].cva == 2.65 and markers["eGFR"].cvi == 5.3
    assert markers["Hemoglobin"].cva == 1.4 and markers["Hemoglobin"].cvi == 2.8
    assert markers["systolic_bp"].ref_low == 90.0 and markers["systolic_bp"].ref_high == 120.0  # AHA
    assert markers["bmi"].ref_high == 25.0  # WHO
    panic_markers = {k for k, v in markers.items() if v.panic_low is not None or v.panic_high is not None}
    assert panic_markers == {"Potassium", "Fasting glucose", "Hemoglobin"}
    assert markers["Potassium"].panic_low == 2.8  # critical-low side now curated

    # The skip-path discipline is preserved where no case exercises a constant (typed absence, not guess):
    assert markers["Potassium"].adverse_direction is None              # bidirectional -> no single adverse trend
    assert markers["Potassium"].cva is None and markers["Potassium"].cvi is None  # RCV skip-path
    assert markers["systolic_bp"].cvi == 5.7                           # demo-prominent -> RCV-gated (Option C)
    assert markers["diastolic_bp"].cvi is None and markers["bmi"].cvi is None  # RCV-free -> trends cap at notable
    assert markers["Total cholesterol"].panic_low is None and markers["Total cholesterol"].panic_high is None


def _valid_bundle_dict() -> dict:
    """A minimal well-formed bundle, mutated by the strictness tests below."""
    return {
        "member_id": "C01",
        "profile": {"member_id": "C01", "age": 46, "sex": "male"},
        "panels": [{
            "panel_id": "C01-P1", "collected_date": "2024-02-12",
            "results": [{"analyte": "HbA1c", "value": 5.3, "unit": "%", "reference_range": "<5.7"}],
            "vitals": {"systolic_bp": 130, "diastolic_bp": 83, "bmi": 27.8},
        }],
        "notes": [],
    }


def test_ingest_is_a_strict_format_check():
    MemberBundle.model_validate(_valid_bundle_dict())  # the happy path validates

    # extra='forbid': a mistyped optional key errors instead of silently dropping context
    typo = _valid_bundle_dict()
    typo["profile"]["medication"] = ["statin"]  # singular typo for "medications"
    with pytest.raises(ValidationError):
        MemberBundle.model_validate(typo)

    # the two member_ids must agree, or storage (root id) and narration (profile id) cross-wire
    mismatch = _valid_bundle_dict()
    mismatch["profile"]["member_id"] = "C99"
    with pytest.raises(ValidationError):
        MemberBundle.model_validate(mismatch)


def test_safety_axes_are_ordinally_ranked():
    # every Literal value is ranked (a new level can't be added without a rank)...
    assert set(FLOOR_ORDER) == set(get_args(FloorLevel))
    assert set(SEVERITY_ORDER) == set(get_args(Severity))
    # ...and ranks follow clinical severity, NOT string order. The trap a raw >=/max() hits:
    # lexicographically "none" > "clinician_review" and "notable" > "attention".
    assert FLOOR_ORDER["none"] < FLOOR_ORDER["clinician_review"] < FLOOR_ORDER["urgent"]
    assert (
        SEVERITY_ORDER["info"] < SEVERITY_ORDER["notable"]
        < SEVERITY_ORDER["attention"] < SEVERITY_ORDER["urgent"]
    )
