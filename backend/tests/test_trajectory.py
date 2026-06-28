"""Phase 7 — the read-only ``GET /trajectory`` projection (``pipeline.trajectory``).

A UI/operator read: the member's per-marker series + the analysis pass's verdict + the drawable
Theil–Sen line. The boundary holds — this is the one place the RAW series is exposed, and only to the
UI/operator; the LLM still consumes only the collapsed ``TrajectoryAnalysis`` (the composer reads
``ctx.analysis``, never this projection).
"""

import pytest
from builders import fresh_con, make_bundle, make_panel, make_result

from health_intelligence import pipeline
from preprocessing.ingest import ingest_bundle


def _ldl_panels(values, start_year=2021):
    return [
        make_panel(
            f"M1-P{i}",
            f"{start_year + i}-01-01",
            [make_result("LDL cholesterol", v, "mg/dL", "<100")],
        )
        for i, v in enumerate(values, start=1)
    ]


def test_trajectory_projects_readings_range_and_theil_sen_line():
    con = fresh_con()
    # A strong, clean 5-panel decline (n >= n_min=4) so Theil–Sen signs the direction.
    ingest_bundle(con, make_bundle("M1", _ldl_panels([200, 180, 160, 140, 120])))

    traj = pipeline.trajectory(con, "M1")
    ldl = next(t for t in traj if t["marker"] == "LDL cholesterol")

    assert [r["value"] for r in ldl["readings"]] == [200, 180, 160, 140, 120]
    assert ldl["reference_range"]["ref_high"] == 100.0
    assert ldl["trend"] is not None  # n >= n_min, a trend is computed
    # The drawable line tracks the computed slope: present iff the slope was signed.
    if ldl["trend"]["slope"] is not None:
        line = ldl["theil_sen"]
        assert line is not None and len(line) == 2
        assert line[0]["date"] == "2022-01-01" and line[1]["date"] == "2026-01-01"
        assert line[1]["value"] < line[0]["value"]  # decreasing
    else:
        assert ldl["theil_sen"] is None


def test_trajectory_no_line_when_too_few_readings():
    con = fresh_con()
    # n < n_min -> no trend claimed -> no Theil–Sen line (honest abstention).
    ingest_bundle(con, make_bundle("M1", _ldl_panels([150, 140])))
    ldl = next(
        t for t in pipeline.trajectory(con, "M1") if t["marker"] == "LDL cholesterol"
    )
    assert ldl["trend"] is None
    assert ldl["theil_sen"] is None
    assert [r["value"] for r in ldl["readings"]] == [150, 140]  # series still surfaced


def test_trajectory_filters_to_one_marker():
    con = fresh_con()
    panels = [
        make_panel(
            "M1-P1",
            "2024-01-15",
            [
                make_result("LDL cholesterol", 120, "mg/dL", "<100"),
                make_result("HbA1c", 5.3, "%", "<5.7"),
            ],
        )
    ]
    ingest_bundle(con, make_bundle("M1", panels))
    only = pipeline.trajectory(con, "M1", marker="HbA1c")
    assert [t["marker"] for t in only] == ["HbA1c"]


def test_trajectory_unknown_member_raises_keyerror():
    con = fresh_con()
    with pytest.raises(KeyError):
        pipeline.trajectory(con, "NOPE")
