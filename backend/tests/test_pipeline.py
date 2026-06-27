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

from health_intelligence import db, pipeline, safety, templates
from health_intelligence.analysis import analyze
from health_intelligence.config import ANALYSIS_CONFIG, CONFIG_VERSION, MODEL_VERSION_DETERMINISTIC
from health_intelligence.models import (
    HealthIntelligenceResponse,
    Note,
    ResponseMetadata,
    SEVERITY_ORDER,
)
from preprocessing.ingest import ingest_bundle, ingest_dataset


def _analysis(con, member_id):
    member, results, ranges, age, dv = db.load_for_analysis(con, member_id)
    return analyze(member, results, ranges, age, ANALYSIS_CONFIG, data_version=dv)


def _raised_markers(analysis):
    return [t.marker for t in analysis.markers if pipeline._is_raised(t)]


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


# ---- Phase 3b: Mode 1 reactive suggestions (the navigable preset loop) ----------------------------
# Mirrors the §50 "likely to break" surface: byte-identical preset answers, the loop never dead-ending
# (anchors always present), grounding (chips only resurface the member's own data), the mechanism
# boundary (no "why" chips), the floor standing on every chip, and the reactive path reusing the 3a
# response builder. `/suggestions` is a PURE READ — these assert no rows are written.

_ANCHORS = ["What's changed since my last results?", "Give me a quick overview of my results."]


def test_suggestions_are_byte_identical_across_calls():
    con = _con(); ingest_dataset(con)
    a = pipeline.suggestions(con, "C01")
    b = pipeline.suggestions(con, "C01")
    assert a and [sp.prompt for sp in a] == [sp.prompt for sp in b]
    # the WHOLE payload (prose + evidence + deterministic ids) is identical — no clock, no model
    assert [sp.response.model_dump_json() for sp in a] == [sp.response.model_dump_json() for sp in b]


def test_suggestions_is_a_pure_read_writes_no_rows():
    con = _con(); ingest_dataset(con)
    before = {t: con.execute(f"SELECT COUNT(*) FROM {t} WHERE member_id='C07'").fetchone()[0]
              for t in ("interactions", "observations", "escalations")}
    pipeline.suggestions(con, "C07", focus="Potassium")
    after = {t: con.execute(f"SELECT COUNT(*) FROM {t} WHERE member_id='C07'").fetchone()[0]
             for t in ("interactions", "observations", "escalations")}
    assert before == after == {"interactions": 0, "observations": 0, "escalations": 0}


def test_anchors_are_present_even_when_every_finding_is_asked():
    # the §50 dead-end failure: with every pivot filtered out (all asked) and one focused, the
    # ever-present anchors must still offer a next step — the loop never empties.
    con = _con(); ingest_dataset(con)
    raised = _raised_markers(_analysis(con, "C01"))
    assert raised, "C01 is the trend member; it must raise at least one finding"
    chips = pipeline.suggestions(con, "C01", focus=raised[0], asked=tuple(raised))
    prompts = [sp.prompt for sp in chips]
    assert all(anchor in prompts for anchor in _ANCHORS)
    assert not any(p.startswith("Tell me about my") for p in prompts)  # every pivot was filtered
    assert prompts[0] == f"How has my {raised[0]} changed over time?"  # the focus drill-down still leads


def test_focus_opens_a_grounded_status_drilldown_for_that_marker():
    con = _con(); ingest_dataset(con)
    chips = pipeline.suggestions(con, "C01", focus="HbA1c")
    assert chips[0].prompt == "How has my HbA1c changed over time?"  # drill-downs lead the loop
    drill = chips[0].response
    ev = drill.findings[0].evidence
    # the member's ACTUAL readings, one verbatim Evidence each (C01 has 5 HbA1c panels), all grounded
    assert len(ev) >= 2 and all(e.marker == "HbA1c" for e in ev)
    assert ev[-1].stat and "Mann-Kendall" in ev[-1].stat  # the change stat sits on the latest reading
    assert "why" not in drill.answer.lower()  # status/data, never causal


def test_chip_count_stays_within_the_documented_two_to_five():
    # C01 raises many findings; a turn must still be a calm 2–5 chips (§145), not a wall of buttons.
    con = _con(); ingest_dataset(con)
    assert len(_raised_markers(_analysis(con, "C01"))) > 3  # more findings than the pivot budget
    assert 2 <= len(pipeline.suggestions(con, "C01")) <= 5                    # landing turn
    assert 2 <= len(pipeline.suggestions(con, "C01", focus="HbA1c")) <= 5     # focused turn (+drill-down)


def test_capped_findings_stay_reachable_through_the_overview_anchor():
    # the cap must not hide a finding: the overview anchor always carries the FULL raised set, and the
    # `asked` filter pages lower-severity findings into the pivot slots over turns — so nothing is lost.
    con = _con(); ingest_dataset(con)
    raised = set(_raised_markers(_analysis(con, "C01")))
    chips = pipeline.suggestions(con, "C01")
    pivots = [sp for sp in chips if sp.prompt.startswith("Tell me about my")]
    assert len(pivots) < len(raised)  # genuinely capped — fewer pivots than findings
    overview = next(sp for sp in chips if sp.prompt == _ANCHORS[0])
    overview_markers = {ev.marker for f in overview.response.findings for ev in f.evidence}
    assert raised <= overview_markers  # every raised finding remains reachable in the overview


def test_asked_filters_that_pivot_but_never_the_anchors():
    con = _con(); ingest_dataset(con)
    target = _raised_markers(_analysis(con, "C01"))[0]
    assert f"Tell me about my {target}." in [sp.prompt for sp in pipeline.suggestions(con, "C01")]
    after = [sp.prompt for sp in pipeline.suggestions(con, "C01", asked=(target,))]
    assert f"Tell me about my {target}." not in after          # the visited pivot is gone
    assert all(anchor in after for anchor in _ANCHORS)          # the anchors survive the filter


def test_every_chip_resurfaces_only_the_members_own_markers():
    con = _con(); ingest_dataset(con)
    for mid in ("C01", "C02", "C07"):
        own = {t.marker for t in _analysis(con, mid).markers}
        for sp in pipeline.suggestions(con, mid, focus=next(iter(own))):
            for f in sp.response.findings:
                for ev in f.evidence:
                    assert ev.marker in own  # grounding: never a marker absent from the record


def test_no_chip_is_a_mechanism_question_and_every_chip_answers():
    con = _con(); ingest_dataset(con)
    for mid in ("C01", "C02", "C07"):
        for sp in pipeline.suggestions(con, mid, focus="HbA1c"):
            assert "why" not in sp.prompt.lower()              # mechanism handed to Mode 2, not templated
            assert not sp.prompt.lower().startswith(("what caus", "what's caus"))
            assert sp.response.answer_disposition == "answered"


def test_floor_is_standing_on_every_chip_for_an_urgent_member():
    con = _con(); ingest_dataset(con)
    analysis = _analysis(con, "C07")
    assert analysis.overall_floor == "urgent"
    non_panic = next(t.marker for t in analysis.markers if t.marker != "Potassium")
    chips = pipeline.suggestions(con, "C07", focus=non_panic)  # focus an unrelated, non-panic marker
    assert chips and all(sp.response.escalation == "urgent" for sp in chips)  # floor always on
    for sp in chips:  # and each passes the same validator the scan/ask path uses
        assert safety.validate(sp.response, "urgent").escalation == "urgent"


def test_calm_member_gets_the_no_findings_overview_not_a_manufactured_concern():
    # a guaranteed-calm member (all in-range, n=1 → no trend) must hit the calm overview branch: the
    # anchors only, no findings, no escalation — "honest before reassuring" without inventing a problem.
    con = _con()
    ingest_bundle(con, _bundle("CALM", [_panel("CALM-P1", "2024-01-01",
                  [_r("Potassium", 4.2, "mmol/L", "3.5-5.1")])]))
    chips = pipeline.suggestions(con, "CALM")
    assert [sp.prompt for sp in chips] == _ANCHORS            # only anchors — no pivots, no dead-end
    overview = chips[0].response
    assert overview.findings == [] and overview.escalation == "none"
    assert "within their normal ranges" in overview.answer
    assert all(sp.response.escalation == "none" for sp in chips)


def test_c02_negative_control_loop_is_calm_and_never_dead_ends():
    con = _con(); ingest_dataset(con)
    assert _analysis(con, "C02").overall_floor == "none"
    chips = pipeline.suggestions(con, "C02")
    assert all(anchor in [sp.prompt for sp in chips] for anchor in _ANCHORS)
    assert all(sp.response.escalation == "none" for sp in chips)  # never escalates a calm member


def test_pivot_answer_reuses_the_3a_render_finding_builder():
    con = _con(); ingest_dataset(con)
    raised = sorted((t for t in _analysis(con, "C01").markers if pipeline._is_raised(t)),
                    key=lambda t: (-SEVERITY_ORDER[t.severity], t.marker))
    assert raised
    marker, traj = raised[0].marker, raised[0]
    sp = next(s for s in pipeline.suggestions(con, "C01") if s.prompt == f"Tell me about my {marker}.")
    # the pivot's Finding is byte-for-byte the per-marker finding the scan builds (same title +
    # evidence), wrapped by the 3a render_finding (its boilerplate answer is the tell).
    assert sp.response.findings[0].text == templates.observation_summary(traj)[0]
    assert sp.response.findings[0].evidence[0].marker == marker
    assert "drawn directly from your own readings" in sp.response.answer


def test_suggestions_unknown_member_raises_keyerror():
    con = _con(); ingest_dataset(con)
    with pytest.raises(KeyError):
        pipeline.suggestions(con, "NOPE")


# ---- Phase 3b regressions (the code-review findings — locked so they can't recur) -----------------

def test_change_drilldown_never_narrates_a_flagged_value_as_normal_variation():
    # render_change was trend-only and called an out-of-range / panic value 'within normal variation'
    # (false reassurance the deterministic core contradicts). It must now state the flag status, in
    # prose AND the evidence stat, consistent with the same marker's pivot chip.
    con = _con(); ingest_dataset(con)
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
    assert "critical-high" in pot.answer                                   # panic stated in prose
    assert "critical-high" in (pot.findings[0].evidence[-1].stat or "")    # and in the evidence stat


def test_vitals_surface_member_friendly_names_not_storage_keys():
    con = _con(); ingest_dataset(con)
    chips = pipeline.suggestions(con, "C07", focus="systolic_bp")  # C07 raises vitals
    blob = " | ".join(sp.prompt for sp in chips) + " | " + chips[0].response.answer
    assert "systolic_bp" not in blob and "systolic blood pressure" in chips[0].prompt
    # the structured evidence field keeps the canonical key (grounding/matching is unaffected)
    assert chips[0].response.findings[0].evidence[0].marker == "systolic_bp"


def test_all_clear_branches_do_not_claim_in_range_for_an_unreferenced_marker():
    # a no_reference marker (an 'other'-sex member with a sex-split marker → no applicable range) must
    # not be asserted 'within normal ranges' by EITHER all-clear anchor — honest before reassuring.
    con = _con()
    ingest_bundle(con, _bundle("OTH", [_panel("OTH-P1", "2024-01-01",
                  [_r("Creatinine", 1.0, "mg/dL", "0.74-1.35 (male) / 0.59-1.04 (female)")])], sex="other"))
    assert any("no_reference" in t.flags for t in _analysis(con, "OTH").markers)
    chips = pipeline.suggestions(con, "OTH")               # m == 0: overview calm branch + summary
    for sp in chips:  # both the overview AND the summary anchor
        assert "within their normal ranges" not in sp.response.answer
    assert all(sp.response.escalation == "none" for sp in chips)


def test_summary_with_findings_does_not_overclaim_an_unreferenced_marker():
    # render_summary's m>0 branch ('the rest are within their normal ranges') must also soften when a
    # non-raised marker has no reference range — the second surface the finding spanned.
    con = _con()
    ingest_bundle(con, _bundle("OTH2", [_panel("OTH2-P1", "2024-01-01", [
        _r("Potassium", 6.1, "mmol/L", "3.5-5.1"),                               # raised (panic) -> m>0
        _r("Creatinine", 1.0, "mg/dL", "0.74-1.35 (male) / 0.59-1.04 (female)"),  # no_reference for sex=other
    ])], sex="other"))
    summary = next(sp for sp in pipeline.suggestions(con, "OTH2")
                   if sp.prompt == "Give me a quick overview of my results.").response
    assert "worth a closer look" in summary.answer and "within their normal ranges" not in summary.answer
