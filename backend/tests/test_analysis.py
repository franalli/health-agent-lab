"""Phase 1 — known-answer tests for the deterministic core.

Fixtures are hand-built domain objects (``MemberProfile`` / ``LabResult`` / ``ReferenceRange``), NOT
raw bundles: the five range-string shapes are parsed by ``ingest.py`` (Phase 2), so wiring real members
into ``analyze`` waits for that firewall. Each fixture mirrors a labeled eval case so Phase 1 de-risks
Phase 5 directly. The supplied members are unlabeled on purpose; these give the analysis a ground truth.
"""

import ast
import pathlib
from datetime import date

import pytest

from health_intelligence import analysis, config
from health_intelligence.config import AnalysisConfig, StatConfig
from health_intelligence.models import (
    LabResult,
    MemberProfile,
    ReferenceRange,
    TrendResult,
)

CFG = config.ANALYSIS_CONFIG


# --------------------------------------------------------------------------------------------------
# Builders — construct the post-ingest domain objects directly (no firewall in Phase 1).
# --------------------------------------------------------------------------------------------------


def _member(member_id="T01", sex="male", age=50):
    return MemberProfile(member_id=member_id, age=age, sex=sex)


def _dates(n, start="2024-01-15"):
    """n monthly ISO dates from `start` — irregular real spacing (28-31 days) Theil-Sen handles."""
    y, m, d = (int(x) for x in start.split("-"))
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}-{d:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _results(marker, unit, values, dates=None):
    dates = dates or _dates(len(values))
    return [
        LabResult(marker=marker, value=v, unit=unit, panel_id=f"P{i}", panel_date=dt)
        for i, (dt, v) in enumerate(zip(dates, values, strict=True), 1)
    ]


def _range(
    marker,
    sex="any",
    unit="",
    ref_low=None,
    ref_high=None,
    panic_low=None,
    panic_high=None,
):
    return ReferenceRange(
        marker=marker,
        sex=sex,
        unit=unit,
        ref_low=ref_low,
        ref_high=ref_high,
        panic_low=panic_low,
        panic_high=panic_high,
        config_version="v0",
    )


def _marker(analysis_out, marker):
    return next(m for m in analysis_out.markers if m.marker == marker)


# --------------------------------------------------------------------------------------------------
# Trend detection + rate recovery (E01/E05-shaped seeded slope)
# --------------------------------------------------------------------------------------------------


def test_seeded_slope_is_detected_and_its_rate_recovered():
    # eGFR on a perfect line (slope k per day) over 12 months: Mann-Kendall flags it, Theil-Sen
    # recovers k exactly (every pairwise slope == k), the CI excludes zero, RCV clears, and the
    # adverse-down direction escalates to clinician_review.
    dates = _dates(12)
    o0 = date.fromisoformat(dates[0]).toordinal()
    k = -0.05
    values = [100.0 + k * (date.fromisoformat(d).toordinal() - o0) for d in dates]
    out = analysis.analyze(
        _member(),
        _results("eGFR", "mL/min/1.73m2", values, dates),
        [_range("eGFR", ref_low=90.0)],
        age=50,
        cfg=CFG,
        data_version="d1",
    )
    egfr = _marker(out, "eGFR")
    assert egfr.trend is not None
    assert egfr.trend.direction == "decreasing"
    assert egfr.trend.p_value < 0.05
    assert egfr.trend.significant is True
    assert egfr.trend.slope == pytest.approx(k)  # Theil-Sen recovers the rate exactly
    assert egfr.trend.slope_ci[1] < 0  # CI excludes zero -> direction is assured
    assert (
        egfr.clinical_change.exceeds_rcv is True
    )  # 100 -> ~78 (~-22%) clears eGFR RCV
    assert egfr.severity == "attention"
    assert out.overall_floor == "clinician_review"


# --------------------------------------------------------------------------------------------------
# Graceful abstention at n=3 (C12 / E12)
# --------------------------------------------------------------------------------------------------


def test_n3_abstains_on_the_trend_but_escalates_a_monotonic_rcv_clearing_decline():
    # C12 / E12: only 3 panels, eGFR 75 -> 68 -> 61 (below the >=90 range). n=3 < n_min=4, so NO trend
    # is claimed (MK's exact p floors at 0.33 there — FDR unreachable). But the decline is strictly
    # monotonic and clears RCV, so the sub-n_min rule (Fix 2) escalates it to clinician_review.
    out = analysis.analyze(
        _member("C12"),
        _results("eGFR", "mL/min/1.73m2", [75.0, 68.0, 61.0]),
        [_range("eGFR", ref_low=90.0)],
        age=58,
        cfg=CFG,
        data_version="d1",
    )
    egfr = _marker(out, "eGFR")
    assert (
        egfr.trend is None
    )  # honest abstention on the trend — no MK/Theil-Sen verdict at n=3
    assert "below_range" in egfr.flags  # range flags still compute at n<n_min
    assert (
        egfr.clinical_change.exceeds_rcv is True
    )  # the 2-point RCV test IS valid at low n
    assert (
        egfr.severity == "attention"
    )  # monotonic + adverse + RCV -> escalates (Fix 2)
    assert out.overall_floor == "clinician_review"  # matching eval E12


def test_sparse_rule_refuses_volatile_within_rcv_improving_and_uncurated():
    # The §6 stress matrix: the sub-n_min rule must fire ONLY on a consistent, beyond-noise, adverse
    # move — and refuse everything else. This is what makes it a generalizing rule, not a C12 patch.
    def floor_of(values, ref_low=90.0):
        out = analysis.analyze(
            _member("T"),
            _results("eGFR", "mL/min/1.73m2", values),
            [_range("eGFR", ref_low=ref_low)],
            age=58,
            cfg=CFG,
            data_version="d1",
        )
        return _marker(out, "eGFR").severity, out.overall_floor

    assert floor_of([75.0, 68.0, 61.0]) == (
        "attention",
        "clinician_review",
    )  # consistent decline: fires
    assert floor_of([75.0, 90.0, 61.0]) == (
        "notable",
        "none",
    )  # volatile V-shape: monotonicity blocks
    assert floor_of([75.0, 74.0, 73.0]) == (
        "notable",
        "none",
    )  # mild: within RCV, RCV gate blocks
    # rising eGFR is the GOOD direction (adverse is "down") -> not adverse -> never escalates
    assert floor_of([61.0, 68.0, 75.0], ref_low=90.0)[1] == "none"


def test_sparse_rule_does_not_fire_without_cva_cvi():
    # A marker with no curated CVa/CVi cannot confirm "beyond noise" (exceeds_rcv is None), so a
    # monotonic adverse sparse change is surfaced (notable) but NEVER escalated — same discipline as the
    # n>=n_min path. "Ferritin" carries CV data in config; use an uncurated synthetic marker instead.
    out = analysis.analyze(
        _member("T"),
        _results(
            "ZZZ unknown marker", "u", [50.0, 40.0, 30.0]
        ),  # monotonic down, no config entry
        [_range("ZZZ unknown marker", ref_low=60.0)],
        age=58,
        cfg=CFG,
        data_version="d1",
    )
    zzz = _marker(out, "ZZZ unknown marker")
    assert zzz.trend is None
    assert zzz.clinical_change.exceeds_rcv is None  # no CVa/CVi -> unconfirmable
    assert zzz.severity == "notable"  # surfaced (below range) but NOT escalated
    assert out.overall_floor == "none"


# --------------------------------------------------------------------------------------------------
# Noise vs signal (C08 / E08) — the RCV + CI-spans-zero gate
# --------------------------------------------------------------------------------------------------


def test_in_range_noise_reads_as_noise_not_a_trend():
    # eGFR jumps around inside the normal range with no monotonic drift: not significant, CI spans
    # zero (-> "flat"), net change within RCV, and no floor is raised.
    out = analysis.analyze(
        _member("C08"),
        _results("eGFR", "mL/min/1.73m2", [95.0, 92.0, 99.0, 91.0, 99.0, 93.0, 95.0]),
        [_range("eGFR", ref_low=90.0)],
        age=44,
        cfg=CFG,
        data_version="d1",
    )
    egfr = _marker(out, "eGFR")
    assert egfr.trend is not None  # a trend was computed (n>=n_min)...
    assert egfr.trend.significant is False  # ...but it is not significant (p >= alpha)
    assert (
        egfr.trend.direction == "flat"
    )  # Theil-Sen CI spans zero -> too noisy to sign
    assert egfr.clinical_change.exceeds_rcv is False
    assert egfr.severity == "info"
    assert out.overall_floor == "none"


# --------------------------------------------------------------------------------------------------
# Direction resolved by the significance test when a tie flattens the Theil-Sen CI (C11 / E11)
# --------------------------------------------------------------------------------------------------


def _trend_obj(direction, tau, *, significant, p_value=0.02):
    """A bare TrendResult to exercise `_resolve_direction` in isolation (the core ethos: unit-test the
    rule, not just the member)."""
    return TrendResult(
        direction=direction, tau=tau, p_value=p_value, n=5, significant=significant
    )


def test_resolve_direction_signs_a_significant_flat_trend_and_only_that():
    # The whole truth table for the post-FDR direction refinement, in isolation.
    t = _trend_obj("flat", 0.8, significant=True)
    analysis._resolve_direction(t)
    assert t.direction == "increasing"  # significant + flat + tau>0 -> signed up

    t = _trend_obj("flat", -0.8, significant=True)
    analysis._resolve_direction(t)
    assert t.direction == "decreasing"  # significant + flat + tau<0 -> signed down

    t = _trend_obj("increasing", 0.9, significant=True)
    analysis._resolve_direction(t)
    assert t.direction == "increasing"  # already signed -> untouched

    t = _trend_obj("flat", 0.8, significant=False)
    analysis._resolve_direction(t)
    assert (
        t.direction == "flat"
    )  # NOT significant -> the CI's abstention stands (no MK override)

    t = _trend_obj("flat", 0.0, significant=True)
    analysis._resolve_direction(t)
    assert (
        t.direction == "flat"
    )  # tau==0 (S==0) -> no direction to sign (and never significant anyway)


def test_significant_trend_flattened_by_a_tie_is_signed_by_mann_kendall():
    # C11/E11: HbA1c 5.6 -> 5.9 with a repeated 5.7 — a zero-slope pair pins the Theil-Sen CI's lower
    # bound at 0, so `_trend` alone calls it "flat" and the adverse trend is silently dropped. MK
    # certifies the trend (p < alpha), so direction resolves to "increasing" and, being adverse-up and
    # RCV-clearing, it escalates — the floor C11 was missing.
    out = analysis.analyze(
        _member("C11"),
        _results("HbA1c", "%", [5.6, 5.7, 5.7, 5.8, 5.9]),
        [_range("HbA1c", ref_high=5.6)],
        age=61,
        cfg=CFG,
        data_version="d1",
    )
    hba1c = _marker(out, "HbA1c")
    assert hba1c.trend.significant is True
    lo, hi = hba1c.trend.slope_ci
    assert (
        lo <= 0 <= hi
    )  # the Theil-Sen CI genuinely spans zero (the tie) — pre-fix this meant "flat"
    assert hba1c.trend.direction == "increasing"  # ...yet MK signs it (Fix 1)
    assert (
        hba1c.trend.slope is None
    )  # rate stays unasserted: the *magnitude* CI does span zero
    assert hba1c.clinical_change.exceeds_rcv is True
    assert (
        hba1c.severity == "attention"
    )  # adverse-up + RCV + now-signed -> counts toward the floor
    assert out.overall_floor == "clinician_review"


def test_resolved_direction_is_direction_only_not_a_new_escalation_path():
    # Same tie-flattened significant shape, but *beneficial* (HbA1c falling, adverse_direction is "up").
    # `_resolve_direction` signs it "decreasing" — yet it must NOT escalate: the adverse criterion still
    # gates the floor. Proves Fix 1 only un-flattens direction; it never relaxes what escalates.
    out = analysis.analyze(
        _member("T-benign"),
        _results("HbA1c", "%", [6.4, 6.3, 6.3, 6.2, 6.1]),
        [
            _range("HbA1c", ref_high=5.6)
        ],  # above range (notable), but the move is the good direction
        age=55,
        cfg=CFG,
        data_version="d1",
    )
    hba1c = _marker(out, "HbA1c")
    assert hba1c.trend.significant is True
    assert (
        hba1c.trend.direction == "decreasing"
    )  # signed by MK, just like the adverse case
    assert (
        hba1c.severity == "notable"
    )  # ...but beneficial -> the trend adds nothing over the range flag
    assert (
        out.overall_floor == "none"
    )  # no escalation: direction resolution is not an escalation path


# --------------------------------------------------------------------------------------------------
# Direction-aware severity — a beneficial trend must NOT escalate (C03 / E03)
# --------------------------------------------------------------------------------------------------


def test_beneficial_trend_does_not_raise_severity():
    # LDL falling 168 -> 118 since a statin: a strong, significant trend, but adverse_direction is "up",
    # so a decreasing move is good news. It stays out-of-range (>100) -> notable, never attention.
    out = analysis.analyze(
        _member("C03"),
        _results("LDL cholesterol", "mg/dL", [168.0, 150.0, 138.0, 125.0, 118.0]),
        [_range("LDL cholesterol", ref_high=100.0)],
        age=52,
        cfg=CFG,
        data_version="d1",
    )
    ldl = _marker(out, "LDL cholesterol")
    assert ldl.trend.significant is True
    assert ldl.trend.direction == "decreasing"
    assert (
        ldl.severity == "notable"
    )  # above range, but the beneficial trend adds nothing
    assert out.overall_floor == "none"


# --------------------------------------------------------------------------------------------------
# Panic pins urgent (C07 / E07) — and doubles as the RCV skip-path (K+ has no CVa/CVi)
# --------------------------------------------------------------------------------------------------


def test_panic_value_forces_urgent_and_potassium_skips_rcv():
    out = analysis.analyze(
        _member("C07"),
        _results("Potassium", "mmol/L", [4.2, 4.3, 4.4, 4.5, 6.1]),
        [_range("Potassium", ref_low=3.5, ref_high=5.1, panic_high=6.0)],
        age=60,
        cfg=CFG,
        data_version="d1",
    )
    k = _marker(out, "Potassium")
    assert "panic_high" in k.flags
    assert k.severity == "urgent"
    assert out.overall_floor == "urgent"
    assert (
        k.clinical_change.exceeds_rcv is None
    )  # Potassium has no CVa/CVi -> RCV skip-path, not silent


def test_panic_low_pins_urgent():
    # The critical-LOW side (configured for K+/glucose/Hb but previously unexercised): a hypokalemic
    # value below panic_low must pin urgent, mirroring the panic_high path.
    out = analysis.analyze(
        _member("C07"),
        _results("Potassium", "mmol/L", [4.5, 4.2, 3.6, 3.0, 2.5]),
        [_range("Potassium", ref_low=3.5, ref_high=5.1, panic_low=2.8)],
        age=55,
        cfg=CFG,
        data_version="d1",
    )
    k = _marker(out, "Potassium")
    assert "panic_low" in k.flags
    assert k.severity == "urgent"
    assert out.overall_floor == "urgent"


def test_increasing_adverse_trend_escalates_end_to_end():
    # The "increasing" Theil-Sen branch + adverse_direction="up" -> attention -> clinician_review,
    # asserted end-to-end (every other significant-trend fixture is decreasing, and the only other
    # increasing input, Potassium panic, is adverse_direction=None so it can't reach attention).
    out = analysis.analyze(
        _member(),
        _results("HbA1c", "%", [5.6, 5.8, 6.0, 6.2, 6.4]),
        [_range("HbA1c", ref_high=5.7)],
        age=50,
        cfg=CFG,
        data_version="d1",
    )
    hba1c = _marker(out, "HbA1c")
    assert hba1c.trend.direction == "increasing"
    assert hba1c.trend.significant is True
    assert hba1c.severity == "attention"
    assert out.overall_floor == "clinician_review"


# --------------------------------------------------------------------------------------------------
# Graded band + adverse decline (C13 / E13)
# --------------------------------------------------------------------------------------------------


def test_vitamin_d_band_cross_with_adverse_decline_escalates():
    # 28 (insufficient) -> 12 (deficient): a band crossing, a significant decreasing trend in the
    # adverse direction that clears RCV, and below the 20 floor -> clinician_review.
    out = analysis.analyze(
        _member("C13"),
        _results("Vitamin D (25-OH)", "ng/mL", [28.0, 22.0, 18.0, 14.0, 12.0]),
        [_range("Vitamin D (25-OH)", ref_low=20.0)],
        age=39,
        cfg=CFG,
        data_version="d1",
    )
    vitd = _marker(out, "Vitamin D (25-OH)")
    assert "band_cross" in vitd.flags
    assert vitd.trend.direction == "decreasing" and vitd.trend.significant is True
    assert vitd.severity == "attention"
    assert out.overall_floor == "clinician_review"


# --------------------------------------------------------------------------------------------------
# Ties — "strict monotonic" must survive a repeated value (real labs have them)
# --------------------------------------------------------------------------------------------------


def test_kendall_s_treats_ties_as_sign_zero():
    # 5.5, 5.5, 5.6, 5.8, 6.0 -> the (5.5,5.5) pair contributes 0; the other 9 pairs are concordant.
    assert analysis._kendall_s([5.5, 5.5, 5.6, 5.8, 6.0]) == 9


def test_tied_series_yields_a_valid_pvalue_without_crashing():
    out = analysis.analyze(
        _member(),
        _results("HbA1c", "%", [5.5, 5.5, 5.6, 5.8, 6.0]),
        [_range("HbA1c", ref_high=5.7)],
        age=46,
        cfg=CFG,
        data_version="d1",
    )
    hba1c = _marker(out, "HbA1c")
    assert hba1c.trend is not None
    assert 0.0 < hba1c.trend.p_value <= 1.0  # exact tie-aware enumeration path (n<=8)


def test_exact_pvalue_engines_agree_at_their_boundary():
    # Real members are n<=5 (the enumeration path); a Phase-2 upload with >=9 panels reaches the
    # inversion-count DP, which nothing else cross-checks and which drives escalation. At n=8 (distinct)
    # both exact engines apply -> they must agree, pinning the Mahonian convolution and the
    # S = n0 - 2D tail mapping against an off-by-one.
    values = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
    s = analysis._kendall_s(values)
    assert analysis._exact_p_enum(values, s) == pytest.approx(
        analysis._exact_p_inversions(len(values), s)
    )


# --------------------------------------------------------------------------------------------------
# Cross-marker pass — the alpha trigger partitions; BH step-up (active only when q < alpha)
# --------------------------------------------------------------------------------------------------


def test_fdr_keeps_significant_candidates_and_drops_sub_alpha_trends():
    # Two markers: a strong 12-point decline (p well under alpha) and 7-point in-range noise (p high).
    egfr = _results("eGFR", "mL/min/1.73m2", [100.0 - 1.8 * i for i in range(12)])
    crp = _results("CRP", "mg/L", [1.0, 2.5, 1.2, 2.8, 1.1, 2.6, 1.3], dates=_dates(7))
    out = analysis.analyze(
        _member(),
        egfr + crp,
        [_range("eGFR", ref_low=90.0), _range("CRP", ref_high=3.0)],
        age=50,
        cfg=CFG,
        data_version="d1",
    )
    assert _marker(out, "eGFR").trend.significant is True
    assert (
        _marker(out, "CRP").trend.significant is False
    )  # p >= alpha -> never a candidate


def test_bh_rejection_branch_activates_only_when_q_below_alpha():
    # With the production config (fdr_q 0.10 > alpha 0.05) BH is a structural backstop that always
    # passes its candidates. Drive it with q < alpha to exercise the step-up rejection branch directly.
    trends = [
        ("a", TrendResult(direction="increasing", tau=0.9, p_value=0.001, n=12)),
        ("b", TrendResult(direction="increasing", tau=0.6, p_value=0.020, n=12)),
        ("c", TrendResult(direction="increasing", tau=0.5, p_value=0.040, n=12)),
    ]
    inert = AnalysisConfig(
        version="t", stats=StatConfig(alpha=0.05, fdr_q=0.10), markers={}
    )
    assert analysis._fdr(trends, inert) == {
        "a": True,
        "b": True,
        "c": True,
    }  # vacuous (q > alpha)

    active = AnalysisConfig(
        version="t", stats=StatConfig(alpha=0.05, fdr_q=0.01), markers={}
    )
    # BH at q=0.01: rank1 0.001<=0.0033 ok; rank2 0.020<=0.0067 no; rank3 0.040<=0.01 no -> only "a".
    assert analysis._fdr(trends, active) == {"a": True, "b": False, "c": False}


# --------------------------------------------------------------------------------------------------
# Edge cases — no_reference, sex fallback, floor projection
# --------------------------------------------------------------------------------------------------


def test_marker_without_a_range_gets_a_typed_no_reference_note():
    out = analysis.analyze(
        _member(),
        _results("ALT", "U/L", [20.0, 22.0]),
        ranges=[],
        age=50,
        cfg=CFG,
        data_version="d1",
    )
    alt = _marker(out, "ALT")
    assert alt.flags == ["no_reference"]  # never a silent pass
    assert alt.severity == "info"


def test_range_selection_prefers_sex_then_falls_back_to_any():
    sexed = [
        _range("Hemoglobin", "male", ref_low=13.5, ref_high=17.5),
        _range("Hemoglobin", "female", ref_low=12.0, ref_high=15.5),
    ]
    assert analysis._range_for("Hemoglobin", "female", 30, sexed).ref_low == 12.0
    assert analysis._range_for("Hemoglobin", "male", 30, sexed).ref_high == 17.5
    assert (
        analysis._range_for("Hemoglobin", "unknown", 30, sexed) is None
    )  # no 'any' to fall back to
    with_any = sexed + [_range("Hemoglobin", "any", ref_low=12.0, ref_high=17.5)]
    assert analysis._range_for("Hemoglobin", "unknown", 30, with_any).sex == "any"


def test_floor_is_the_max_severity_projected():
    assert analysis._floor(["info", "notable", "attention"]) == "clinician_review"
    assert analysis._floor(["notable", "urgent", "attention"]) == "urgent"
    assert analysis._floor(["info", "notable"]) == "none"
    assert analysis._floor([]) == "none"


def test_output_is_deterministic():
    args = (
        _member(),
        _results("eGFR", "mL/min/1.73m2", [100.0 - i for i in range(6)]),
        [_range("eGFR", ref_low=90.0)],
    )
    a = analysis.analyze(*args, age=50, cfg=CFG, data_version="d1")
    b = analysis.analyze(*args, age=50, cfg=CFG, data_version="d1")
    assert a.model_dump_json() == b.model_dump_json()


# --------------------------------------------------------------------------------------------------
# Purity — the load-bearing invariant: no DB / LLM / network / clock reachable from analysis.py
# --------------------------------------------------------------------------------------------------


def test_analysis_module_is_pure():
    src = pathlib.Path(analysis.__file__).read_text()
    tree = ast.parse(src)
    top_level, full = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level.add(alias.name.split(".")[0])
                full.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
            full.add(node.module)

    banned_top = {
        "sqlite3",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "http",
        "os",
        "sys",
        "time",
        "random",
        "secrets",
        "anthropic",
        "asyncio",
    }
    assert not (top_level & banned_top), f"impure import: {top_level & banned_top}"
    banned_full = {
        "health_intelligence.db",
        "health_intelligence.llm",
        "health_intelligence.gate",
        "health_intelligence.pipeline",
    }
    assert not (full & banned_full), (
        f"core imports a downstream layer: {full & banned_full}"
    )
    for token in (
        ".now(",
        ".today(",
        "time.time",
        "monotonic(",
        "perf_counter",
        "urandom",
    ):
        assert token not in src, f"impure clock/entropy call: {token}"


def test_no_rcv_marker_surfaces_trend_at_notable_not_attention():
    # The §352/§394 rule: a no-CVa/CVi marker (BMI) with a significant adverse rise that stays IN range
    # is surfaced at `notable` (a visible observation) but NOT escalated — RCV is the gate to attention.
    out = analysis.analyze(
        _member(),
        _results("bmi", "kg/m2", [21.0, 21.8, 22.5, 23.3, 24.0]),
        [_range("bmi", ref_low=18.5, ref_high=25.0)],
        age=40,
        cfg=CFG,
        data_version="d1",
    )
    bmi = _marker(out, "bmi")
    assert bmi.trend.significant is True
    assert bmi.clinical_change.exceeds_rcv is None  # RCV skip-path (no CVa/CVi)
    assert bmi.severity == "notable"  # surfaced, not escalated to attention
    assert out.overall_floor == "none"


def test_systolic_bp_is_rcv_gated_and_can_escalate():
    # Option C: systolic carries CVa/CVi, so a confirmed rising-BP trend clears RCV and escalates
    # first-class — the demo-prominent vital now treated like any RCV-gated lab.
    out = analysis.analyze(
        _member(),
        _results("systolic_bp", "mmHg", [120.0, 126.0, 132.0, 138.0, 145.0]),
        [_range("systolic_bp", ref_low=90.0, ref_high=120.0)],
        age=58,
        cfg=CFG,
        data_version="d1",
    )
    sbp = _marker(out, "systolic_bp")
    assert sbp.trend.direction == "increasing" and sbp.trend.significant is True
    assert sbp.clinical_change.exceeds_rcv is True
    assert sbp.severity == "attention"
    assert out.overall_floor == "clinician_review"


def test_normal_approx_path_detects_a_tied_high_n_trend():
    # The >=9-panel-with-ties path (a Phase-2 upload) routes MK significance through _normal_p, which no
    # other fixture reaches. A clear tied uptrend must still read significant — guards the continuity
    # correction and the tie-variance sign that would otherwise silently mis-gate escalation.
    values = [
        10.0,
        11.0,
        11.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
    ]  # n=11, a tie -> _normal_p
    out = analysis.analyze(
        _member(),
        _results("CRP", "mg/L", values),
        [_range("CRP", ref_high=3.0)],
        age=50,
        cfg=CFG,
        data_version="d1",
    )
    crp = _marker(out, "CRP")
    assert crp.trend.direction == "increasing"
    assert (
        crp.trend.p_value < 0.05
    )  # the normal approximation did not invert a clear trend


def test_single_reading_still_computes_flags():
    # n=1: no trend, no clinical_change, but range/panic flags must still evaluate — a lone critical
    # value escalates (smallest other fixtures are n=2 / n=3).
    out = analysis.analyze(
        _member(),
        _results("Potassium", "mmol/L", [6.5]),
        [_range("Potassium", ref_low=3.5, ref_high=5.1, panic_high=6.0)],
        age=50,
        cfg=CFG,
        data_version="d1",
    )
    k = _marker(out, "Potassium")
    assert k.trend is None and k.clinical_change is None
    assert "panic_high" in k.flags
    assert out.overall_floor == "urgent"


def test_analyze_selects_sex_specific_range_through_the_pipeline():
    # member.sex -> _range_for wiring exercised end-to-end (not just the _range_for unit test): Hb 12.5
    # is below the male floor (13.5) but inside the female band (12.0-15.5).
    ranges = [
        _range("Hemoglobin", "male", unit="g/dL", ref_low=13.5, ref_high=17.5),
        _range("Hemoglobin", "female", unit="g/dL", ref_low=12.0, ref_high=15.5),
    ]
    fem = analysis.analyze(
        _member("F1", sex="female"),
        _results("Hemoglobin", "g/dL", [12.6, 12.5]),
        ranges,
        age=30,
        cfg=CFG,
        data_version="d1",
    )
    assert "below_range" not in _marker(fem, "Hemoglobin").flags  # female band applied
    male = analysis.analyze(
        _member("M1", sex="male"),
        _results("Hemoglobin", "g/dL", [12.6, 12.5]),
        ranges,
        age=30,
        cfg=CFG,
        data_version="d1",
    )
    assert "below_range" in _marker(male, "Hemoglobin").flags  # male band applied
