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

from health_intelligence import db, pipeline, safety, templates
from health_intelligence.analysis import analyze
from health_intelligence.config import (
    ANALYSIS_CONFIG,
    COMPOSE_MODEL,
    CONFIG_VERSION,
    GATE_MODEL,
    MODEL_VERSION_DETERMINISTIC,
)
from health_intelligence.gate import GateClassification
from health_intelligence.llm import LLMParseError, LLMUnavailable
from health_intelligence.models import (
    SEVERITY_ORDER,
    ComposeDraft,
    HealthIntelligenceResponse,
    Note,
    ResponseMetadata,
)
from preprocessing.ingest import ingest_bundle, ingest_dataset


def _analysis(con, member_id):
    member, results, ranges, age, dv = db.load_for_analysis(con, member_id)
    return analyze(member, results, ranges, age, ANALYSIS_CONFIG, data_version=dv)


def _raised_markers(analysis):
    return [t.marker for t in analysis.markers if pipeline._is_raised(t)]


# helpers (_con / _r / _panel / _bundle) now live in tests/builders.py — imported above


def _resp(escalation):
    """A minimal HealthIntelligenceResponse at a given escalation level (for validator tests)."""
    return HealthIntelligenceResponse(
        answer="x",
        answer_disposition="answered",
        escalation=escalation,
        metadata=ResponseMetadata(
            response_id="r",
            data_version="d",
            model_version=MODEL_VERSION_DETERMINISTIC,
            config_version=CONFIG_VERSION,
        ),
    )


# ---- safety: floor projection + validator --------------------------------------------------------


def test_data_floor_is_the_cores_projection_verbatim():
    con = _con()
    ingest_bundle(
        con,
        _bundle(
            "M1",
            [
                _panel(
                    "M1-P1", "2024-01-01", [_r("Potassium", 6.1, "mmol/L", "3.5-5.1")]
                )
            ],
        ),
    )
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
    assert (
        safety.validate(_resp("clinician_review"), "clinician_review").escalation
        == "clinician_review"
    )
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

    for combo in (
        ["info"],
        ["notable"],
        ["attention"],
        ["urgent"],
        ["notable", "attention"],
        ["attention", "urgent"],
        ["info", "notable", "urgent"],
    ):
        levels = [
            lvl for s in combo if (lvl := safety.severity_to_level(s)) is not None
        ]
        max_level = max(levels, key=lambda lvl: FLOOR_ORDER[lvl], default="none")
        assert max_level == _floor(combo)


# ---- scan: ranked observations + persisted interaction (the FK) ----------------------------------


def test_scan_emits_ranked_finding_and_persists_one_interaction_each():
    con = _con()
    # a panic potassium (urgent) alongside an out-of-range LDL (notable) -> ranking must put urgent first
    ingest_bundle(
        con,
        _bundle(
            "M2",
            [
                _panel(
                    "M2-P1",
                    "2024-01-01",
                    [
                        _r("Potassium", 6.1, "mmol/L", "3.5-5.1"),
                        _r("LDL cholesterol", 180.0, "mg/dL", "0-100"),
                    ],
                )
            ],
        ),
    )
    obs = pipeline.scan(con, "M2")
    assert [o.severity for o in obs] == sorted(
        (o.severity for o in obs), key=lambda s: -SEVERITY_ORDER[s]
    )
    assert obs[0].severity == "urgent" and obs[0].title.startswith("Potassium")

    # per-finding (architecture §48): one interaction per observation, each driver='scan', ids distinct,
    # and observations FK 1:1 onto them
    rows = con.execute(
        "SELECT response_id, driver FROM interactions WHERE member_id='M2'"
    ).fetchall()
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
    panels = [
        _panel(
            f"M4-P{i + 1}",
            dates[i],
            [_r("Potassium", 4.2, "mmol/L", "3.5-5.1")],
            vitals={"systolic_bp": 118, "diastolic_bp": dia[i], "bmi": 22.5},
        )
        for i in range(5)
    ]
    ingest_bundle(con, _bundle("M4", panels))
    pipeline.scan(con, "M4")
    rows = con.execute(
        "SELECT response_json FROM interactions WHERE member_id='M4'"
    ).fetchall()
    import json

    for r in rows:
        resp = json.loads(r["response_json"])
        for f in resp["findings"]:
            text, stat = f["text"], (f["evidence"][0]["stat"] or "")
            trendy_text = "rising" in text or "falling" in text
            trendy_stat = "Mann-Kendall" in stat
            assert trendy_text == trendy_stat, (
                f"text/evidence signal mismatch: {text!r} vs {stat!r}"
            )


def test_observation_summary_surfaces_status_on_a_flagged_trending_marker():
    """B1 regression: a marker that is BOTH significantly trending AND out-of-range / band-crossed must
    have its current status stated in the title and trigger_reason, not be narrated by trend alone
    (CLAUDE.md "every narrator surfaces the core flags"). This locks the ``observation_summary`` trend /
    sparse-trend surfaces (scan trigger_reason, Mode-1 pivot, overview) — previously only
    ``_change_narrative`` was covered, so a trend-only "X rising" on a flagged value slipped through."""
    from health_intelligence.models import (
        ClinicalChange,
        MarkerTrajectory,
        Reading,
        TrendResult,
    )

    # counted adverse trend (severity 'attention') that has ALSO breached range and crossed a band
    counted = MarkerTrajectory(
        marker="HbA1c",
        unit="%",
        latest=Reading(value=6.1, date="2024-01-01"),
        trend=TrendResult(
            direction="increasing",
            tau=0.9,
            p_value=0.017,
            slope=0.001,
            n=5,
            significant=True,
        ),
        clinical_change=ClinicalChange(rcv=0.3, net_change=0.8, exceeds_rcv=True),
        flags=["above_range", "band_cross"],
        severity="attention",
        n_readings=5,
    )
    title, trigger = templates.observation_summary(counted)
    assert "rising" in title and "above range" in title, (
        title
    )  # trend AND status, both stated
    assert "above range" in trigger, trigger

    # sub-n_min sparse adverse decline that is below range (the C12-eGFR shape)
    sparse = MarkerTrajectory(
        marker="eGFR",
        unit="mL/min/1.73m2",
        latest=Reading(value=61, date="2024-01-01"),
        trend=None,  # no MK verdict at n<n_min
        clinical_change=ClinicalChange(rcv=5.0, net_change=-22.0, exceeds_rcv=True),
        flags=["below_range"],
        severity="attention",
        n_readings=3,
    )
    s_title, s_trigger = templates.observation_summary(sparse)
    assert "falling" in s_title and "below range" in s_title, s_title
    assert (
        "limited history" in s_title and "significant" not in s_title
    )  # honest, not overstated
    assert "below range" in s_trigger, s_trigger

    # an in-range trending marker (no value flag) is correctly NOT given a spurious status phrase
    inrange = MarkerTrajectory(
        marker="ALT",
        unit="U/L",
        latest=Reading(value=30, date="2024-01-01"),
        trend=TrendResult(
            direction="increasing",
            tau=0.9,
            p_value=0.02,
            slope=0.01,
            n=5,
            significant=True,
        ),
        clinical_change=ClinicalChange(rcv=5.0, net_change=8.0, exceeds_rcv=True),
        flags=[],
        severity="attention",
        n_readings=5,
    )
    i_title, _ = templates.observation_summary(inrange)
    assert "range" not in i_title and "threshold" not in i_title, i_title


def test_sparse_decline_escalates_with_grounded_limited_history_narration():
    con = _con()
    # 3 panels only (sub-n_min): eGFR 78 -> 70 -> 61, a strictly monotonic decline below the >=90 range
    # that clears RCV. Fix 2 escalates it to clinician_review; the Finding's evidence stat must cite the
    # SAME signal as its title (grounding contract), and the narration must convey the decline AND the
    # limited base WITHOUT overstating certainty (E12 must_not). The render is the only place the
    # surface-core-flags invariant is enforced, so it is asserted here, not just severity/floor.
    dates = ["2023-01-01", "2023-07-01", "2024-01-01"]
    egfr = [78, 70, 61]
    panels = [
        _panel(
            f"S-P{i + 1}",
            dates[i],
            [
                _r("eGFR", egfr[i], "mL/min/1.73m2", ">=90"),
                _r("Potassium", 4.2, "mmol/L", "3.5-5.1"),
            ],
            vitals={"systolic_bp": 118, "diastolic_bp": 78, "bmi": 22.5},
        )
        for i in range(3)
    ]
    ingest_bundle(con, _bundle("S1", panels))
    pipeline.scan(con, "S1")

    # 1) the sparse monotonic decline reaches clinician_review (never urgent — panic only)
    egfr_escs = con.execute(
        "SELECT level FROM escalations WHERE member_id='S1' AND dedup_key LIKE '%eGFR%'"
    ).fetchall()
    assert len(egfr_escs) == 1 and egfr_escs[0]["level"] == "clinician_review"

    # 2) grounding parity: the eGFR Finding's text and its evidence stat cite the SAME (sparse) signal —
    #    not 'falling' text paired with a 'latest reading' / 'outside range' stat.
    import json

    findings = [
        f
        for r in con.execute(
            "SELECT response_json FROM interactions WHERE member_id='S1'"
        ).fetchall()
        for f in json.loads(r["response_json"])["findings"]
        if f["evidence"][0]["marker"] == "eGFR"
    ]
    assert findings, "expected an eGFR finding"
    text, stat = findings[0]["text"], findings[0]["evidence"][0]["stat"]
    assert (
        "falling" in text and "limited history" in text
    )  # title: decline + sparse base
    assert "panels" in stat and "reference-change" in stat  # stat cites the SAME signal
    assert (
        "Mann-Kendall" not in stat
    )  # there is no MK verdict at n=3 — must not fake one
    assert "significant" not in text and "significant" not in stat  # no overstatement

    # 3) the full change narrative surfaces the out-of-range status AND the limited-history caveat,
    #    honestly (lowers, not overstates, certainty).
    egfr_traj = next(t for t in _analysis(con, "S1").markers if t.marker == "eGFR")
    narr = templates._change_narrative(egfr_traj)
    assert (
        "below the reference range" in narr
    )  # surface-core-flags invariant (not trend-only)
    assert (
        "limited history" in narr and "panels" in narr
    )  # honest about the sparse base
    assert "lowers certainty" in narr  # does not overstate from a short series


def test_scan_with_no_signal_writes_nothing():
    con = _con()
    ingest_bundle(
        con,
        _bundle(
            "M3",
            [
                _panel(
                    "M3-P1", "2024-01-01", [_r("Potassium", 4.2, "mmol/L", "3.5-5.1")]
                )
            ],
        ),
    )
    obs = pipeline.scan(con, "M3")
    # per-finding: no raised marker -> no findings -> no interactions/observations/escalations
    assert obs == []
    assert db.get_escalations(con, "M3") == []
    assert (
        con.execute(
            "SELECT COUNT(*) FROM interactions WHERE member_id='M3'"
        ).fetchone()[0]
        == 0
    )


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

    obs_count = con.execute(
        "SELECT COUNT(*) FROM observations WHERE member_id='C07'"
    ).fetchone()[0]
    int_count = con.execute(
        "SELECT COUNT(*) FROM interactions WHERE member_id='C07'"
    ).fetchone()[0]

    # a re-scan over identical data is idempotent: no new observation, interaction, or escalation rows
    pipeline.scan(con, "C07")
    assert (
        con.execute(
            "SELECT COUNT(*) FROM observations WHERE member_id='C07'"
        ).fetchone()[0]
        == obs_count
    )
    assert (
        con.execute(
            "SELECT COUNT(*) FROM interactions WHERE member_id='C07'"
        ).fetchone()[0]
        == int_count
    )
    assert len(db.get_escalations(con, "C07")) == 1


def test_rescan_overwrites_stale_observation_narration_without_fk_failure():
    # Regression (architecture §48): write_observation is OVERWRITE-on-conflict, not keep-first
    # INSERT OR IGNORE. A re-scan at an unchanged data_version must REFRESH the persisted projection
    # (so a templates/display-name change self-heals on the next scan) and must do so WITHOUT a blanket
    # delete — escalations.observation_id is a RESTRICT FK, so deleting an escalated marker's observation
    # FK-fails. C07 carries a data_finding escalation, so it is exactly the case keep-first strands and
    # delete-then-insert breaks.
    con = _con()
    ingest_dataset(con)
    obs = pipeline.scan(con, "C07")
    pot = next(o for o in obs if o.title.startswith("Potassium"))

    # simulate stale narration persisted by an earlier templating version
    with con:
        con.execute(
            "UPDATE observations SET title='STALE', trigger_reason='STALE' WHERE observation_id=?",
            (pot.observation_id,),
        )

    # re-scan over identical data (same data_version → same observation_id, still escalation-referenced):
    # must not FK-fail, must restore current narration, must keep exactly one row for the marker.
    refreshed = pipeline.scan(con, "C07")
    pot2 = next(o for o in refreshed if o.observation_id == pot.observation_id)
    assert pot2.title == pot.title != "STALE"  # overwrite, not keep-first
    assert pot2.trigger_reason == pot.trigger_reason != "STALE"
    assert (
        con.execute(
            "SELECT COUNT(*) FROM observations WHERE observation_id=?",
            (pot.observation_id,),
        ).fetchone()[0]
        == 1
    )
    assert (
        len(db.get_escalations(con, "C07")) == 1
    )  # the referencing escalation survived intact


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
    base = [
        _panel(f"M5-P{i + 1}", d, [_r("Potassium", 4.2, "mmol/L", "3.5-5.1")])
        for i, d in enumerate(["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"])
    ]
    ingest_bundle(con, _bundle("M5", base))
    pipeline.scan(con, "M5")
    assert (
        pipeline.scan(con, "M5") == [] or db.get_observations(con, "M5") == []
    )  # calm at baseline

    panic = base + [
        _panel("M5-P5", "2026-01-01", [_r("Potassium", 6.1, "mmol/L", "3.5-5.1")])
    ]
    ingest_bundle(
        con, _bundle("M5", panic)
    )  # data_version AND analysis_version both move
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
    notes = db.get_notes(con, "C07") + [
        Note(date="2026-01-01", source="in-app", text="added note")
    ]
    db.replace_member(con, profile=member, results=results, ranges=ranges, notes=notes)

    assert (
        db.compute_data_version(con, "C07") != dv_before
    )  # the full-record hash did move
    pipeline.scan(con, "C07")
    assert len(db.get_escalations(con, "C07")) == 1  # but the finding-stable dedup held


# ---- Phase 3b: Mode 1 reactive suggestions (the navigable preset loop) ----------------------------
# Mirrors the §50 "likely to break" surface: byte-identical preset answers, the loop never dead-ending
# (anchors always present), grounding (chips only resurface the member's own data), the mechanism
# boundary (no "why" chips), the floor standing on every chip, and the reactive path reusing the 3a
# response builder. `/suggestions` is a PURE READ — these assert no rows are written.

_ANCHORS = [
    "What's changed since my last results?",
    "Give me a quick overview of my results.",
]


def test_suggestions_are_byte_identical_across_calls():
    con = _con()
    ingest_dataset(con)
    a = pipeline.suggestions(con, "C01")
    b = pipeline.suggestions(con, "C01")
    assert a and [sp.prompt for sp in a] == [sp.prompt for sp in b]
    # the WHOLE payload (prose + evidence + deterministic ids) is identical — no clock, no model
    assert [sp.response.model_dump_json() for sp in a] == [
        sp.response.model_dump_json() for sp in b
    ]


def test_suggestions_is_a_pure_read_writes_no_rows():
    con = _con()
    ingest_dataset(con)
    before = {
        t: con.execute(f"SELECT COUNT(*) FROM {t} WHERE member_id='C07'").fetchone()[0]
        for t in ("interactions", "observations", "escalations")
    }
    pipeline.suggestions(con, "C07", focus="Potassium")
    after = {
        t: con.execute(f"SELECT COUNT(*) FROM {t} WHERE member_id='C07'").fetchone()[0]
        for t in ("interactions", "observations", "escalations")
    }
    assert before == after == {"interactions": 0, "observations": 0, "escalations": 0}


def test_anchors_are_present_even_when_every_finding_is_asked():
    # the §50 dead-end failure: with every pivot filtered out (all asked) and one focused, the
    # ever-present anchors must still offer a next step — the loop never empties.
    con = _con()
    ingest_dataset(con)
    raised = _raised_markers(_analysis(con, "C01"))
    assert raised, "C01 is the trend member; it must raise at least one finding"
    chips = pipeline.suggestions(con, "C01", focus=raised[0], asked=tuple(raised))
    prompts = [sp.prompt for sp in chips]
    assert all(anchor in prompts for anchor in _ANCHORS)
    assert not any(
        p.startswith("Tell me about my") for p in prompts
    )  # every pivot was filtered
    assert (
        prompts[0] == f"How has my {raised[0]} changed over time?"
    )  # the focus drill-down still leads


def test_focus_opens_a_grounded_status_drilldown_for_that_marker():
    con = _con()
    ingest_dataset(con)
    chips = pipeline.suggestions(con, "C01", focus="HbA1c")
    assert (
        chips[0].prompt == "How has my HbA1c changed over time?"
    )  # drill-downs lead the loop
    drill = chips[0].response
    ev = drill.findings[0].evidence
    # the member's ACTUAL readings, one verbatim Evidence each (C01 has 5 HbA1c panels), all grounded
    assert len(ev) >= 2 and all(e.marker == "HbA1c" for e in ev)
    assert (
        ev[-1].stat and "Mann-Kendall" in ev[-1].stat
    )  # the change stat sits on the latest reading
    assert "why" not in drill.answer.lower()  # status/data, never causal


def test_chip_count_stays_within_the_documented_two_to_five():
    # C01 raises many findings; a turn must still be a calm 2–5 chips (§145), not a wall of buttons.
    con = _con()
    ingest_dataset(con)
    assert (
        len(_raised_markers(_analysis(con, "C01"))) > 3
    )  # more findings than the pivot budget
    assert 2 <= len(pipeline.suggestions(con, "C01")) <= 5  # landing turn
    assert (
        2 <= len(pipeline.suggestions(con, "C01", focus="HbA1c")) <= 5
    )  # focused turn (+drill-down)


def test_capped_findings_stay_reachable_through_the_overview_anchor():
    # the cap must not hide a finding: the overview anchor always carries the FULL raised set, and the
    # `asked` filter pages lower-severity findings into the pivot slots over turns — so nothing is lost.
    con = _con()
    ingest_dataset(con)
    raised = set(_raised_markers(_analysis(con, "C01")))
    chips = pipeline.suggestions(con, "C01")
    pivots = [sp for sp in chips if sp.prompt.startswith("Tell me about my")]
    assert len(pivots) < len(raised)  # genuinely capped — fewer pivots than findings
    overview = next(sp for sp in chips if sp.prompt == _ANCHORS[0])
    overview_markers = {
        ev.marker for f in overview.response.findings for ev in f.evidence
    }
    assert (
        raised <= overview_markers
    )  # every raised finding remains reachable in the overview


def test_asked_filters_that_pivot_but_never_the_anchors():
    con = _con()
    ingest_dataset(con)
    target = _raised_markers(_analysis(con, "C01"))[0]
    assert f"Tell me about my {target}." in [
        sp.prompt for sp in pipeline.suggestions(con, "C01")
    ]
    after = [sp.prompt for sp in pipeline.suggestions(con, "C01", asked=(target,))]
    assert f"Tell me about my {target}." not in after  # the visited pivot is gone
    assert all(anchor in after for anchor in _ANCHORS)  # the anchors survive the filter


def test_every_chip_resurfaces_only_the_members_own_markers():
    con = _con()
    ingest_dataset(con)
    for mid in ("C01", "C02", "C07"):
        own = {t.marker for t in _analysis(con, mid).markers}
        for sp in pipeline.suggestions(con, mid, focus=next(iter(own))):
            for f in sp.response.findings:
                for ev in f.evidence:
                    assert (
                        ev.marker in own
                    )  # grounding: never a marker absent from the record


def test_no_chip_is_a_mechanism_question_and_every_chip_answers():
    con = _con()
    ingest_dataset(con)
    for mid in ("C01", "C02", "C07"):
        for sp in pipeline.suggestions(con, mid, focus="HbA1c"):
            assert (
                "why" not in sp.prompt.lower()
            )  # mechanism handed to Mode 2, not templated
            assert not sp.prompt.lower().startswith(("what caus", "what's caus"))
            assert sp.response.answer_disposition == "answered"


def test_floor_is_standing_on_every_chip_for_an_urgent_member():
    con = _con()
    ingest_dataset(con)
    analysis = _analysis(con, "C07")
    assert analysis.overall_floor == "urgent"
    non_panic = next(t.marker for t in analysis.markers if t.marker != "Potassium")
    chips = pipeline.suggestions(
        con, "C07", focus=non_panic
    )  # focus an unrelated, non-panic marker
    assert chips and all(
        sp.response.escalation == "urgent" for sp in chips
    )  # floor always on
    for sp in chips:  # and each passes the same validator the scan/ask path uses
        assert safety.validate(sp.response, "urgent").escalation == "urgent"


def test_calm_member_gets_the_no_findings_overview_not_a_manufactured_concern():
    # a guaranteed-calm member (all in-range, n=1 → no trend) must hit the calm overview branch: the
    # anchors only, no findings, no escalation — "honest before reassuring" without inventing a problem.
    con = _con()
    ingest_bundle(
        con,
        _bundle(
            "CALM",
            [
                _panel(
                    "CALM-P1", "2024-01-01", [_r("Potassium", 4.2, "mmol/L", "3.5-5.1")]
                )
            ],
        ),
    )
    chips = pipeline.suggestions(con, "CALM")
    assert [
        sp.prompt for sp in chips
    ] == _ANCHORS  # only anchors — no pivots, no dead-end
    overview = chips[0].response
    assert overview.findings == [] and overview.escalation == "none"
    assert "within their normal ranges" in overview.answer
    assert all(sp.response.escalation == "none" for sp in chips)


def test_c02_negative_control_loop_is_calm_and_never_dead_ends():
    con = _con()
    ingest_dataset(con)
    assert _analysis(con, "C02").overall_floor == "none"
    chips = pipeline.suggestions(con, "C02")
    assert all(anchor in [sp.prompt for sp in chips] for anchor in _ANCHORS)
    assert all(
        sp.response.escalation == "none" for sp in chips
    )  # never escalates a calm member


def test_pivot_answer_reuses_the_3a_render_finding_builder():
    con = _con()
    ingest_dataset(con)
    raised = sorted(
        (t for t in _analysis(con, "C01").markers if pipeline._is_raised(t)),
        key=lambda t: (-SEVERITY_ORDER[t.severity], t.marker),
    )
    assert raised
    marker, traj = raised[0].marker, raised[0]
    sp = next(
        s
        for s in pipeline.suggestions(con, "C01")
        if s.prompt == f"Tell me about my {marker}."
    )
    # the pivot's Finding is byte-for-byte the per-marker finding the scan builds (same title +
    # evidence), wrapped by the 3a render_finding (its boilerplate answer is the tell).
    assert sp.response.findings[0].text == templates.observation_summary(traj)[0]
    assert sp.response.findings[0].evidence[0].marker == marker
    assert "drawn directly from your own readings" in sp.response.answer


def test_suggestions_unknown_member_raises_keyerror():
    con = _con()
    ingest_dataset(con)
    with pytest.raises(KeyError):
        pipeline.suggestions(con, "NOPE")


# ---- Phase 3b regressions (the code-review findings — locked so they can't recur) -----------------


def test_change_drilldown_never_narrates_a_flagged_value_as_normal_variation():
    # render_change was trend-only and called an out-of-range / panic value 'within normal variation'
    # (false reassurance the deterministic core contradicts). It must now state the flag status, in
    # prose AND the evidence stat, consistent with the same marker's pivot chip.
    con = _con()
    ingest_dataset(con)
    seen_flagged = False
    for mid in ("C07", "C14", "C01", "C05"):
        for t in _analysis(con, mid).markers:
            if not templates._is_flagged(t):
                continue
            seen_flagged = True
            ans = pipeline.suggestions(con, mid, focus=t.marker)[0].response.answer
            assert "within normal variation" not in ans, (mid, t.marker, ans)
    assert seen_flagged  # the loop actually exercised flagged markers
    pot = pipeline.suggestions(con, "C07", focus="Potassium")[0].response
    assert "critical-high" in pot.answer  # panic stated in prose
    assert "critical-high" in (
        pot.findings[0].evidence[-1].stat or ""
    )  # and in the evidence stat


def test_vitals_surface_member_friendly_names_not_storage_keys():
    con = _con()
    ingest_dataset(con)
    chips = pipeline.suggestions(con, "C07", focus="systolic_bp")  # C07 raises vitals
    blob = " | ".join(sp.prompt for sp in chips) + " | " + chips[0].response.answer
    assert "systolic_bp" not in blob and "systolic blood pressure" in chips[0].prompt
    # the structured evidence field keeps the canonical key (grounding/matching is unaffected)
    assert chips[0].response.findings[0].evidence[0].marker == "systolic_bp"


def test_all_clear_branches_do_not_claim_in_range_for_an_unreferenced_marker():
    # a no_reference marker (an 'other'-sex member with a sex-split marker → no applicable range) must
    # not be asserted 'within normal ranges' by EITHER all-clear anchor — honest before reassuring.
    con = _con()
    ingest_bundle(
        con,
        _bundle(
            "OTH",
            [
                _panel(
                    "OTH-P1",
                    "2024-01-01",
                    [
                        _r(
                            "Creatinine",
                            1.0,
                            "mg/dL",
                            "0.74-1.35 (male) / 0.59-1.04 (female)",
                        )
                    ],
                )
            ],
            sex="other",
        ),
    )
    assert any("no_reference" in t.flags for t in _analysis(con, "OTH").markers)
    chips = pipeline.suggestions(con, "OTH")  # m == 0: overview calm branch + summary
    for sp in chips:  # both the overview AND the summary anchor
        assert "within their normal ranges" not in sp.response.answer
    assert all(sp.response.escalation == "none" for sp in chips)


def test_summary_with_findings_does_not_overclaim_an_unreferenced_marker():
    # render_summary's m>0 branch ('the rest are within their normal ranges') must also soften when a
    # non-raised marker has no reference range — the second surface the finding spanned.
    con = _con()
    ingest_bundle(
        con,
        _bundle(
            "OTH2",
            [
                _panel(
                    "OTH2-P1",
                    "2024-01-01",
                    [
                        _r(
                            "Potassium", 6.1, "mmol/L", "3.5-5.1"
                        ),  # raised (panic) -> m>0
                        _r(
                            "Creatinine",
                            1.0,
                            "mg/dL",
                            "0.74-1.35 (male) / 0.59-1.04 (female)",
                        ),  # no_reference for sex=other
                    ],
                )
            ],
            sex="other",
        ),
    )
    summary = next(
        sp
        for sp in pipeline.suggestions(con, "OTH2")
        if sp.prompt == "Give me a quick overview of my results."
    ).response
    assert (
        "worth a closer look" in summary.answer
        and "within their normal ranges" not in summary.answer
    )


# ---- Phase 4: Mode 2 — the ask path (gate -> compose/template -> validate -> escalate) ------------
# Mirrors §52's runnable + "likely to break": free-form answers grounded in CODE-BUILT evidence, a
# typed emergency escalating on normal labs, the LLM never naming a number or touching the floor, the
# validator/degradation scaffold firing on LLMUnavailable, and substance identical across re-runs. The
# one network seam (llm.Provider) is faked, so the whole path runs offline and deterministically.


def _draft(answer, *, disposition="answered", cited=(), uncertainty=None):
    return ComposeDraft(
        answer=answer,
        uncertainty=uncertainty,
        answer_disposition=disposition,
        cited_markers=list(cited),
    )


def test_ask_compose_attaches_code_built_evidence_for_cited_markers(fake_provider):
    con = _con()
    ingest_dataset(con)
    draft = _draft(
        "Your HbA1c has been rising over your panels.",
        cited=("HbA1c",),
        uncertainty="Based on your recorded panels.",
    )
    prov = fake_provider(GateClassification(route="none"), draft)
    resp = pipeline.ask(con, "C01", "what's happening with my HbA1c?", provider=prov)

    assert resp.answer == draft.answer  # the model's prose is the answer
    # the evidence is built from the analysis, NOT the draft — numbers come from code (the one law)
    f = next(f for f in resp.findings if f.evidence and f.evidence[0].marker == "HbA1c")
    hba1c = next(t for t in _analysis(con, "C01").markers if t.marker == "HbA1c")
    assert f.evidence[0].value == hba1c.latest.value  # exact, code-sourced
    assert resp.escalation == _analysis(con, "C01").overall_floor  # deterministic floor
    assert resp.metadata.model_version == COMPOSE_MODEL
    # both the gate and the composer ran -> tokens/cost reflect the turn
    assert resp.metadata.tokens and resp.metadata.cost_usd is not None


def test_ask_drops_an_unknown_cited_marker_and_never_fabricates_evidence(fake_provider):
    con = _con()
    ingest_dataset(con)
    # the model cites a marker the member never measured — it must be dropped, never invented
    draft = _draft("...", cited=("HbA1c", "Selenium"))
    prov = fake_provider(GateClassification(route="none"), draft)
    resp = pipeline.ask(con, "C01", "...", provider=prov)
    markers = {ev.marker for f in resp.findings for ev in f.evidence}
    assert "HbA1c" in markers and "Selenium" not in markers


def test_ask_absent_marker_answer_carries_no_fabricated_evidence(fake_provider):
    con = _con()
    ingest_dataset(con)
    # an honest "not measured" answer cites nothing -> zero findings (no evidence conjured for it)
    draft = _draft("That hasn't been measured in your records.", cited=())
    prov = fake_provider(GateClassification(route="none"), draft)
    resp = pipeline.ask(con, "C01", "what's my vitamin B12?", provider=prov)
    assert resp.findings == [] and resp.answer == draft.answer


def test_ask_escalation_is_code_set_to_the_floor_the_llm_cannot_under_escalate(
    fake_provider,
):
    con = _con()
    ingest_dataset(con)
    assert _analysis(con, "C07").overall_floor == "urgent"  # C07's potassium panic
    # a breezy composed answer that mentions nothing alarming STILL carries the deterministic floor —
    # under approach A the LLM never touches escalation, so it literally cannot under-escalate
    draft = _draft("Everything looks pretty stable.", cited=())
    prov = fake_provider(GateClassification(route="none"), draft)
    resp = pipeline.ask(con, "C07", "how am I doing overall?", provider=prov)
    assert resp.escalation == "urgent"
    assert (
        safety.validate(resp, "urgent").escalation == "urgent"
    )  # the shared guard passes


def test_ask_typed_emergency_escalates_even_on_normal_labs(fake_provider):
    con = _con()
    ingest_dataset(con)
    assert _analysis(con, "C02").overall_floor == "none"  # the labs themselves are calm
    prov = fake_provider(
        GateClassification(route="acute_medical")
    )  # gate routes -> no compose call
    # a phrase-free acute message, so escalation==urgent ISOLATES the gate route's floor contribution
    # (it does not also trip the deterministic emergency-phrase backstop — that path is its own test)
    resp = pipeline.ask(
        con,
        "C02",
        "my left arm went numb and my vision suddenly blurred",
        provider=prov,
    )

    assert resp.escalation == "urgent" and resp.answer_disposition == "answered"
    assert (
        "seek medical care now" in resp.answer
    )  # the fixed seek-care template, no labs to dilute it
    esc = db.get_escalations(con, "C02")
    assert len(esc) == 1 and esc[0].kind == "chat" and esc[0].level == "urgent"
    assert (
        resp.metadata.model_version == GATE_MODEL
    )  # the gate produced this routing decision


def test_ask_emergency_phrase_floor_escalates_even_when_the_gate_says_none(
    fake_provider,
):
    con = _con()
    ingest_dataset(con)
    # the gate (faked) misses a self-harm message as `none` -> the answer is composed (the floor-only
    # gap), but the deterministic emergency-phrase floor still forces urgent and the escalation fires
    prov = fake_provider(
        GateClassification(route="none"),
        _draft("Here's a look at your data.", cited=()),
    )
    resp = pipeline.ask(con, "C02", "I don't want to live anymore", provider=prov)
    assert resp.escalation == "urgent"
    assert len(db.get_escalations(con, "C02")) == 1


def test_ask_compose_unavailable_degrades_to_a_grounded_deterministic_answer(
    fake_provider,
):
    con = _con()
    ingest_dataset(con)
    # gate clears the message (none) but the composer is down -> Mode 2 degrades to a grounded Mode-1
    # answer with no model call (architecture §6), not an error or an unguarded reply
    prov = fake_provider(
        GateClassification(route="none"), LLMUnavailable("provider down")
    )
    resp = pipeline.ask(con, "C01", "what's changed?", provider=prov)
    assert resp.answer_disposition == "answered"
    assert (
        resp.metadata.model_version == MODEL_VERSION_DETERMINISTIC
    )  # no model authored the prose
    assert (
        resp.escalation == _analysis(con, "C01").overall_floor
    )  # still floor-respecting
    assert (
        resp.findings and "worth a closer look" in resp.answer
    )  # grounded overview, not an error


def test_ask_compose_parse_failure_is_repaired_by_one_bounded_retry(fake_provider):
    con = _con()
    ingest_dataset(con)
    # the first compose attempt is a parse glitch (a truncated/refused tool call); the one bounded retry
    # at temp 0 succeeds, so the member still gets the normal composed answer — no degradation
    good = _draft("Your HbA1c is rising.", cited=("HbA1c",))
    prov = fake_provider(
        GateClassification(route="none"), LLMParseError("truncated"), good
    )
    resp = pipeline.ask(con, "C01", "how's my HbA1c?", provider=prov)
    assert resp.answer == good.answer
    assert resp.metadata.model_version == COMPOSE_MODEL
    assert any(ev.marker == "HbA1c" for f in resp.findings for ev in f.evidence)


def test_ask_compose_parse_failure_degrades_never_propagates_a_500(fake_provider):
    con = _con()
    ingest_dataset(con)
    # a malformed/refused structured output that SURVIVES the one retry (e.g. max_tokens truncation,
    # stop_reason='refusal') must degrade to the grounded deterministic answer — Mode 2 never lets a
    # parse failure reach the member as an unhandled error (architecture §6, "fails safe by construction")
    prov = fake_provider(
        GateClassification(route="none"),
        LLMParseError("truncated"),
        LLMParseError("still malformed"),
    )
    resp = pipeline.ask(con, "C01", "what's changed?", provider=prov)
    assert resp.answer_disposition == "answered"  # an answer, not a raised exception
    assert resp.metadata.model_version == MODEL_VERSION_DETERMINISTIC
    assert resp.escalation == _analysis(con, "C01").overall_floor  # floor-respecting
    assert resp.findings  # the grounded overview, not an error


def test_ask_gate_down_routes_to_couldnt_route_holding_the_clinician_review_floor(
    fake_provider,
):
    con = _con()
    ingest_dataset(con)
    assert _analysis(con, "C02").overall_floor == "none"  # data floor is none
    prov = fake_provider(
        LLMUnavailable("gate down")
    )  # the gate's only call fails -> couldnt_route
    resp = pipeline.ask(con, "C02", "qwerty asdf ???", provider=prov)
    assert (
        resp.escalation == "clinician_review"
    )  # the fail-closed floor holds above none
    assert (
        "rephras" in resp.answer.lower()
    )  # invites a rephrase without dropping the floor
    assert (
        resp.metadata.model_version == MODEL_VERSION_DETERMINISTIC
    )  # the gate produced NO usable output -> deterministic, not GATE_MODEL
    assert (
        len(db.get_escalations(con, "C02")) == 1
    )  # clinician_review still queues a chat task


def test_ask_out_of_scope_refuses_without_escalating_a_calm_member(fake_provider):
    con = _con()
    ingest_dataset(con)
    prov = fake_provider(GateClassification(route="out_of_scope"))
    resp = pipeline.ask(con, "C02", "can you change my metformin dose?", provider=prov)
    assert resp.answer_disposition == "out_of_scope" and resp.escalation == "none"
    assert (
        db.get_escalations(con, "C02") == []
    )  # out-of-scope on calm labs -> no clinician task
    assert resp.metadata.model_version == GATE_MODEL


def test_ask_substance_is_identical_across_reruns(fake_provider):
    con = _con()
    ingest_dataset(con)

    def run():
        draft = _draft("Your HbA1c is rising.", cited=("HbA1c",))
        return pipeline.ask(
            con,
            "C01",
            "how's my HbA1c?",
            provider=fake_provider(GateClassification(route="none"), draft),
        )

    a, b = run(), run()
    # the load-bearing substance — findings (evidence), escalation, disposition — is byte-identical;
    # only metadata (a per-turn response_id, latency) differs, which the architecture allows (§7)
    assert [f.model_dump() for f in a.findings] == [f.model_dump() for f in b.findings]
    assert a.escalation == b.escalation and a.answer_disposition == b.answer_disposition


def test_ask_appends_a_distinct_interaction_per_turn(fake_provider):
    con = _con()
    ingest_dataset(con)

    def ask_once(q):
        return pipeline.ask(
            con,
            "C02",
            q,
            provider=fake_provider(GateClassification(route="none"), _draft("ok")),
        )

    r1, r2 = ask_once("question one"), ask_once("question two")
    rows = con.execute(
        "SELECT response_id, driver FROM interactions WHERE member_id='C02'"
    ).fetchall()
    assert len(rows) == 2 and all(
        r["driver"] == "ask" for r in rows
    )  # appends (not idempotent no-op)
    assert {r["response_id"] for r in rows} == {
        r1.metadata.response_id,
        r2.metadata.response_id,
    }
    assert r1.metadata.response_id != r2.metadata.response_id


def test_ask_chat_escalation_is_one_per_member_per_day(fake_provider):
    con = _con()
    ingest_dataset(con)
    # C07's data floor is urgent, so every ask warrants a chat escalation — but deduped to one per day
    for q in ("first question", "second question"):
        pipeline.ask(
            con,
            "C07",
            q,
            provider=fake_provider(GateClassification(route="none"), _draft("ok")),
        )
    chat = [e for e in db.get_escalations(con, "C07") if e.kind == "chat"]
    assert (
        len(chat) == 1
    )  # INSERT OR IGNORE on chat:{member}:{day} — fire once per member per day


def test_ask_chat_escalation_upgrades_clinician_review_to_urgent_same_day(
    fake_provider,
):
    con = _con()
    ingest_dataset(con)
    # C02 is calm (data floor none). Morning: gate down -> couldnt_route -> a clinician_review chat row.
    pipeline.ask(
        con, "C02", "qwerty asdf", provider=fake_provider(LLMUnavailable("down"))
    )
    morning = [e for e in db.get_escalations(con, "C02") if e.kind == "chat"]
    assert len(morning) == 1 and morning[0].level == "clinician_review"
    # Afternoon, SAME UTC day: a typed acute emergency -> urgent. The day-scoped row must UPGRADE, not be
    # dropped by INSERT OR IGNORE (which would leave the queue masking the acute event at the lower level).
    pipeline.ask(
        con,
        "C02",
        "my left arm went numb",
        provider=fake_provider(GateClassification(route="acute_medical")),
    )
    afternoon = [e for e in db.get_escalations(con, "C02") if e.kind == "chat"]
    assert len(afternoon) == 1  # still one clinician task for the day...
    assert (
        afternoon[0].level == "urgent"
    )  # ...but upgraded to the true urgency, not masked
    assert (
        "acute" in afternoon[0].trigger_reason
    )  # and repointed at the acute turn's reason


def test_ask_dedups_repeated_cited_markers_into_one_finding(fake_provider):
    con = _con()
    ingest_dataset(con)
    # the model lists the same key twice (easy under tool-use when one marker is discussed twice); the
    # response must carry ONE finding with a stable finding_id, not a duplicate chip / ambiguous id
    draft = _draft("Your HbA1c keeps coming up.", cited=("HbA1c", "HbA1c"))
    prov = fake_provider(GateClassification(route="none"), draft)
    resp = pipeline.ask(con, "C01", "tell me about my HbA1c", provider=prov)
    hba1c_findings = [f for f in resp.findings if f.finding_id == "f:HbA1c"]
    assert len(hba1c_findings) == 1  # deduped, not two findings sharing one finding_id


def test_ask_unknown_member_raises_keyerror(fake_provider):
    con = _con()
    ingest_dataset(con)
    # load_for_analysis raises before any provider call -> the route maps it to 404
    with pytest.raises(KeyError):
        pipeline.ask(
            con, "NOPE", "hi", provider=fake_provider(GateClassification(route="none"))
        )
