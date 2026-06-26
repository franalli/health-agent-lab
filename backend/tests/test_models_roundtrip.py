"""Phase 0 contract guard — the runnable bar made repeatable.

Proves the three Phase-0 acceptance criteria (schema loads, models round-trip over the real data)
plus the config-curation contract (what's anchored vs deliberately deferred as typed absence).
This is the ONLY test Phase 0 ships; the analysis/pipeline suites arrive with their phases.
"""

import json
import pathlib
import sqlite3
from typing import get_args

import pytest
from pydantic import ValidationError

from health_intelligence import config
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
MEMBERS = BACKEND / "data" / "members.json"

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

    # Externally sourced -> typed absence (curated in Phase 1 beside their tests):
    assert all(m.cva is None and m.cvi is None for m in markers.values())
    assert all(m.adverse_direction is None for m in markers.values())
    panic_anchored = [k for k, v in markers.items() if v.panic_low is not None or v.panic_high is not None]
    assert panic_anchored == ["Potassium"]


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
