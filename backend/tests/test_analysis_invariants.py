"""Out-of-sample tests for the deterministic core — randomized invariants + independent-oracle cases.

WHY THIS FILE EXISTS. The 15 supplied members are *simultaneously* the demo data and ``eval_set.jsonl``,
so a green eval cannot — by construction — detect overfitting of the statistical thresholds (``n_min``,
the RCV constants, the sub-``n_min`` monotonic rule). If a constant were ever nudged to make those 15
read "right", ``make eval`` would still pass 22/22 and report nothing. This module instruments against
that blind spot with data the thresholds were NEVER fitted to:

  * **Invariants** (``test_inv_*``) — the escalation *contract* asserted over hundreds of randomized
    series. Ground truth here is *structural* (a flat value never escalates; panic pins urgent), so it
    needs no oracle and cannot be a change-detector.
  * **Point cases** (``test_oos_*``) — a few fixed series asserted against an INDEPENDENT Mann-Kendall
    p / RCV computation (``_independent_mk_p`` / ``_independent_rcv_pct``, both hand-anchored below),
    never against ``analysis.py``'s own output. Asserting ``== what analyze() emits`` would ratify a
    wrong threshold; asserting against an independent oracle refutes one.

Scope: this is a peer of ``test_analysis.py`` (the pure verdict), NOT of ``eval_set.jsonl`` (the
ask-path cases) — a different artifact, graded one layer below the LLM. Nothing here touches the eval.

Note on the contract (vs. the intuitive "all in-range is safe"): an *in-range* marker CAN escalate —
a significant, adverse, RCV-clearing trajectory reaches ``attention`` even with both endpoints inside
the band (that is the point of trajectory monitoring; see ``test_oos_in_range_climb_can_escalate``). So
the law these invariants encode is the precise one — *out-of-range / band-cross ALONE is at most
``notable``; escalation requires panic, an RCV-clearing significant adverse trend, or the sparse path* —
not the looser "in-range ⇒ safe".

Randomness is a fixed-seed stdlib generator (reproducible — the core's determinism ethos), written as
predicates so it could lift to Hypothesis unchanged. ``random`` is fine in a TEST; the purity ban is on
``analysis.py``, which ``test_analysis.test_analysis_module_is_pure`` enforces against that source.
"""

import itertools
import math
import random

import pytest

from health_intelligence import analysis, config
from health_intelligence.models import (
    LabResult,
    MemberProfile,
    ReferenceRange,
)

CFG = config.ANALYSIS_CONFIG
ALPHA = CFG.stats.alpha
N_MIN = CFG.stats.n_min
TRIALS = 300  # randomized cases per invariant — cheap (pure core), enough to exercise each boundary


# --------------------------------------------------------------------------------------------------
# Builders (mirror test_analysis.py — post-ingest domain objects, no firewall here)
# --------------------------------------------------------------------------------------------------


def _member(member_id="X", sex="male", age=40):
    return MemberProfile(member_id=member_id, age=age, sex=sex)


def _dates(n, start="2024-01-15"):
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


def _range(marker, *, ref_low=None, ref_high=None, panic_low=None, panic_high=None):
    return ReferenceRange(
        marker=marker,
        sex="any",
        unit="",
        ref_low=ref_low,
        ref_high=ref_high,
        panic_low=panic_low,
        panic_high=panic_high,
        config_version="v0",
    )


def _analyze(marker, unit, values, rng):
    out = analysis.analyze(
        _member(),
        _results(marker, unit, values),
        [rng],
        age=40,
        cfg=CFG,
        data_version="d1",
    )
    return out, next(m for m in out.markers if m.marker == marker)


# --------------------------------------------------------------------------------------------------
# Independent oracles — derived from the mathematical definition, NOT from analysis.py. Each is pinned
# to a pure hand-computed value first, so a shared-bug-with-the-implementation cannot pass silently.
# --------------------------------------------------------------------------------------------------


def _independent_mk_p(values: list[float]) -> float:
    """Exact two-sided Mann-Kendall p by enumerating every ordering of the observed multiset: under the
    null all orderings are equally likely, so p = P(|S| >= |S_obs|). Written straight from that
    definition (clean-room; analysis._mk_pvalue uses optimized inversion-count / normal-approx paths the
    test must not borrow). n <= 8 only — the enumeration is 8! at most."""
    assert len(values) <= 8, "enumeration oracle is for small n only"

    def s_stat(vs):
        s = 0
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                s += (vs[j] > vs[i]) - (vs[j] < vs[i])
        return s

    target = abs(s_stat(values))
    perms = list(itertools.permutations(values))
    extreme = sum(1 for p in perms if abs(s_stat(list(p))) >= target)
    return extreme / len(perms)


def _independent_rcv_pct(cva: float, cvi: float, z: float) -> float:
    """RCV as a percent threshold, from the textbook formula sqrt(2)*Z*sqrt(CVa^2+CVi^2). Independent
    re-derivation of the gate in analysis._clinical_change (a bug in that formula would diverge here)."""
    return math.sqrt(2.0) * z * math.sqrt(cva**2 + cvi**2)


def _creatinine_rcv_pct() -> float:
    """Creatinine's RCV from the LIVE config CVs (not a literal) — so a config CV change reruns the gate
    cleanly here instead of producing a confusing oracle-vs-code mismatch. The literal 2.65/5.3 lives
    only in the hand-anchor test, which is where pinning the formula to a hand value belongs."""
    mc = CFG.markers["Creatinine"]
    assert mc.cva is not None and mc.cvi is not None  # curated for Creatinine
    return _independent_rcv_pct(mc.cva, mc.cvi, CFG.stats.rcv_z)


def test_oracles_match_hand_computed_anchors():
    # Pin the MK oracle to values computable WITHOUT any algorithm: a strictly monotonic run of n
    # distinct points has |S| maximal, reached by exactly 2 of the n! orderings (the sorted asc/desc),
    # so p = 2/n!. n=5 -> 2/120 = 1/60; n=3 -> 2/6 = 1/3 (the documented "MK floors at 0.33 at n=3").
    assert _independent_mk_p([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(2 / 120)
    assert _independent_mk_p([1.0, 2.0, 3.0]) == pytest.approx(2 / 6)
    # A TIED case by hand (so a tie bug shared with analysis._exact_p_enum can't pass the PC1 cross-check
    # silently): multiset {1,1,2} has 3 orderings -> S = +2, 0, -2; |S|>=2 in 2 of 3 -> p = 2/3.
    assert _independent_mk_p([1.0, 1.0, 2.0]) == pytest.approx(2 / 3)
    # And the RCV formula against a hand value for Creatinine's curated CVs (2.65, 5.3) at Z=1.96:
    #   sqrt(2)*1.96*sqrt(2.65^2 + 5.3^2) = sqrt(2)*1.96*sqrt(35.1125) = 16.42 (2 dp).
    assert _independent_rcv_pct(2.65, 5.3, 1.96) == pytest.approx(16.42, abs=0.01)


# --------------------------------------------------------------------------------------------------
# Invariant 1 — determinism over random multi-marker inputs (the byte-identical core, generalized past
# test_analysis.test_output_is_deterministic's single fixture).
# --------------------------------------------------------------------------------------------------


def test_inv_analyze_is_deterministic():
    rnd = random.Random(1)
    markers = [
        ("HbA1c", "%", _range("HbA1c", ref_high=5.7)),
        ("eGFR", "mL/min/1.73m2", _range("eGFR", ref_low=90.0)),
        ("LDL cholesterol", "mg/dL", _range("LDL cholesterol", ref_high=100.0)),
        (
            "Potassium",
            "mmol/L",
            _range("Potassium", ref_low=3.5, ref_high=5.1, panic_high=6.0),
        ),
    ]
    for _ in range(TRIALS):
        results, ranges = [], []
        for marker, unit, rng in markers:
            n = rnd.randint(1, 7)
            vals = [round(rnd.uniform(0.5, 200.0), 2) for _ in range(n)]
            results += _results(marker, unit, vals)
            ranges.append(rng)
        a = analysis.analyze(
            _member(), results, ranges, age=40, cfg=CFG, data_version="d1"
        )
        b = analysis.analyze(
            _member(), results, ranges, age=40, cfg=CFG, data_version="d1"
        )
        assert a.model_dump_json() == b.model_dump_json()


# --------------------------------------------------------------------------------------------------
# Invariant 2 — a FLAT trajectory never escalates, however extreme the value. Encodes "escalation !=
# out-of-range": a constant series (no trend, net change 0) on a no-panic marker stays <= notable. LDL
# has no panic threshold, so NO constant value — 50 or 5000 — may raise the floor.
# --------------------------------------------------------------------------------------------------


def test_inv_flat_series_never_escalates():
    rnd = random.Random(2)
    rng = _range("LDL cholesterol", ref_high=100.0)  # no panic on LDL
    for _ in range(TRIALS):
        value = round(rnd.uniform(1.0, 5000.0), 2)  # spans far past any range
        n = rnd.randint(1, 6)
        _, traj = _analyze("LDL cholesterol", "mg/dL", [value] * n, rng)
        out, _ = _analyze("LDL cholesterol", "mg/dL", [value] * n, rng)
        assert out.overall_floor == "none", f"flat LDL=={value} escalated"
        assert traj.severity in ("info", "notable")  # at most a visible observation


# --------------------------------------------------------------------------------------------------
# Invariant 3 — a latest value past a panic bound pins urgent, regardless of the history before it
# (panic is value-based, not trajectory-based; _flags reads series[-1]).
# --------------------------------------------------------------------------------------------------


def test_inv_panic_latest_pins_urgent():
    rnd = random.Random(3)
    rng = _range("Potassium", ref_low=3.5, ref_high=5.1, panic_low=2.8, panic_high=6.0)
    for _ in range(TRIALS):
        n = rnd.randint(1, 6)
        history = [
            round(rnd.uniform(3.6, 5.0), 2) for _ in range(n - 1)
        ]  # arbitrary in-range past
        if rnd.random() < 0.5:
            latest = round(rnd.uniform(6.01, 9.0), 2)  # above panic_high
        else:
            latest = round(rnd.uniform(0.5, 2.79), 2)  # below panic_low
        out, traj = _analyze("Potassium", "mmol/L", history + [latest], rng)
        assert out.overall_floor == "urgent", f"latest {latest} did not pin urgent"
        assert traj.severity == "urgent"


# --------------------------------------------------------------------------------------------------
# Invariant 4 — direction symmetry: reflecting a series about a constant negates every pairwise
# difference, so Mann-Kendall S (and the Theil-Sen sign) flip. increasing <-> decreasing; flat <-> flat.
# Catches any asymmetry in the direction logic (e.g. a sign error in _resolve_direction).
# --------------------------------------------------------------------------------------------------


def test_inv_reflection_flips_trend_direction():
    rnd = random.Random(4)
    opposite = {"increasing": "decreasing", "decreasing": "increasing", "flat": "flat"}
    rng = _range("eGFR", ref_low=90.0)
    for _ in range(TRIALS):
        n = rnd.randint(N_MIN, 8)  # n >= n_min so a trend is computed (not None)
        vals = [round(rnd.uniform(50.0, 150.0), 2) for _ in range(n)]
        k = 2 * (
            sum(vals) / len(vals)
        )  # reflect about the mean -> values stay positive
        reflected = [round(k - v, 2) for v in vals]
        _, t1 = _analyze("eGFR", "mL/min/1.73m2", vals, rng)
        _, t2 = _analyze("eGFR", "mL/min/1.73m2", reflected, rng)
        assert t1.trend is not None and t2.trend is not None
        assert t2.trend.direction == opposite[t1.trend.direction]


# --------------------------------------------------------------------------------------------------
# Invariant 5 — escalation REQUIRES a cause. The contrapositive of _severity's escalation branches:
# a marker at attention/urgent must carry a panic flag, OR a significant trend, OR be sparse (n<n_min,
# where the monotonic stand-in lives). It would FAIL the instant out-of-range ALONE escalated — the
# regression this whole law guards against — across randomized markers, ranges, lengths and values.
#
# Random uniform values almost never form a p<0.05 monotonic run, so a purely-random fuzzer would only
# ever escalate via panic / out-of-range and leave _severity's TREND path untested. So one branch
# deliberately manufactures a significant, adverse, RCV-clearing climb, and the loop asserts BOTH the
# panic and the trend escalation paths were actually reached (non-vacuity) — a future refactor that
# stopped producing escalating cases would turn this guard into a green no-op otherwise.
# --------------------------------------------------------------------------------------------------


def test_inv_escalation_requires_a_cause():
    rnd = random.Random(5)
    # (marker, unit, range, sampling span) — a mix with/without panic, ref_low/ref_high, and a banded one.
    space = [
        ("HbA1c", "%", _range("HbA1c", ref_high=5.7), (4.5, 9.0)),
        ("eGFR", "mL/min/1.73m2", _range("eGFR", ref_low=90.0), (40.0, 120.0)),
        (
            "LDL cholesterol",
            "mg/dL",
            _range("LDL cholesterol", ref_high=100.0),
            (50.0, 220.0),
        ),
        (
            "Creatinine",
            "mg/dL",
            _range("Creatinine", ref_low=0.74, ref_high=1.35),
            (0.4, 2.5),
        ),
        (
            "Potassium",
            "mmol/L",
            _range(
                "Potassium", ref_low=3.5, ref_high=5.1, panic_low=2.8, panic_high=6.0
            ),
            (2.0, 7.0),
        ),
    ]
    # Markers with a known adverse direction + a span wide enough that a sorted run clears RCV. "up" ->
    # ascending (adverse), "down" -> descending (adverse).
    adverse = [
        (
            "Creatinine",
            "mg/dL",
            _range("Creatinine", ref_low=0.74, ref_high=1.35),
            "up",
            (0.8, 3.0),
        ),
        ("eGFR", "mL/min/1.73m2", _range("eGFR", ref_low=90.0), "down", (40.0, 130.0)),
        ("HbA1c", "%", _range("HbA1c", ref_high=5.7), "up", (5.0, 9.0)),
    ]
    seen = {"panic": 0, "trend": 0}
    for _ in range(TRIALS * 2):
        if rnd.random() < 0.35:  # manufacture a significant adverse RCV-clearing climb
            marker, unit, rng, direction, (lo, hi) = rnd.choice(adverse)
            n = rnd.randint(
                5, 8
            )  # n>=5 so a clean monotonic run is significant (2/n! < alpha)
            vals = sorted(
                (round(rnd.uniform(lo, hi), 3) for _ in range(n)),
                reverse=(direction == "down"),
            )
            if len(set(vals)) < n:
                continue  # a chance tie could drop significance; skip (negligibly rare)
        else:
            marker, unit, rng, (lo, hi) = rnd.choice(space)
            n = rnd.randint(1, 8)
            vals = [round(rnd.uniform(lo, hi), 3) for _ in range(n)]
        out, _ = _analyze(marker, unit, vals, rng)
        for t in out.markers:
            if t.severity in ("attention", "urgent"):
                has_panic = "panic_low" in t.flags or "panic_high" in t.flags
                has_sig_trend = t.trend is not None and t.trend.significant
                assert (
                    t.n_readings is not None
                )  # analyze() always populates the series length
                is_sparse = t.n_readings < N_MIN  # sub-n_min monotonic path lives here
                assert has_panic or has_sig_trend or is_sparse, (
                    f"{marker} escalated to {t.severity} with no cause: "
                    f"flags={t.flags} n={t.n_readings} values={vals}"
                )
                if has_panic:
                    seen["panic"] += 1
                elif has_sig_trend:
                    seen["trend"] += 1
    # Non-vacuity: the guard actually exercised both escalation causes, not a green no-op.
    assert seen["panic"] > 0 and seen["trend"] > 0, f"vacuous coverage: {seen}"


# --------------------------------------------------------------------------------------------------
# Point case — the refutation made executable: an in-range climb DOES escalate (so "in-range ⇒ safe" is
# wrong, and these invariants must not assume it). Creatinine 0.75 -> 1.30, both inside [0.74, 1.35].
# --------------------------------------------------------------------------------------------------


def test_oos_in_range_climb_can_escalate():
    vals = [0.75, 0.88, 1.00, 1.13, 1.30]  # strictly inside [0.74, 1.35] throughout
    rng = _range("Creatinine", ref_low=0.74, ref_high=1.35)
    out, traj = _analyze("Creatinine", "mg/dL", vals, rng)
    assert (
        "above_range" not in traj.flags and "below_range" not in traj.flags
    )  # genuinely in-range
    assert traj.trend is not None
    # Independent oracle: monotonic distinct n=5 -> p = 2/120 < alpha (significant); net +73% >> RCV 16.42%.
    assert _independent_mk_p(vals) == pytest.approx(traj.trend.p_value)
    assert _independent_mk_p(vals) < ALPHA
    net_pct = (vals[-1] - vals[0]) / vals[0] * 100.0
    assert net_pct > _creatinine_rcv_pct()  # clears RCV by a wide margin
    assert traj.severity == "attention"
    assert out.overall_floor == "clinician_review"


# --------------------------------------------------------------------------------------------------
# Point case — anti-false-positive: in-range NOISE must not be manufactured into a trend. eGFR
# oscillating inside the range, net ~flat. Independent oracle confirms p >> alpha BEFORE we trust the
# verdict, so this refutes a too-eager significance threshold rather than ratifying the current one.
# --------------------------------------------------------------------------------------------------


def test_oos_in_range_noise_is_not_a_trend():
    vals = [95.0, 92.0, 99.0, 91.0, 99.0, 93.0, 96.0]
    rng = _range("eGFR", ref_low=90.0)
    out, traj = _analyze("eGFR", "mL/min/1.73m2", vals, rng)
    assert traj.trend is not None
    p_oracle = _independent_mk_p(vals)
    assert p_oracle == pytest.approx(
        traj.trend.p_value
    )  # implementation matches the clean-room p
    assert p_oracle >= ALPHA  # independently: NOT a significant trend
    assert traj.trend.significant is False
    assert traj.trend.direction == "flat"
    assert out.overall_floor == "none"


# --------------------------------------------------------------------------------------------------
# Point case — anti-false-positive: a REAL, significant, adverse, in-range climb that stays WITHIN
# biological noise (RCV) must not escalate. The C11-creatinine archetype (1.00 -> 1.16, ~+16% vs RCV
# 16.42%): the trend is genuine (oracle: p < alpha) yet the RCV gate correctly withholds escalation.
# This is the case a naive "significant adverse trend -> escalate" rule would over-flag.
# --------------------------------------------------------------------------------------------------


def test_oos_significant_in_range_climb_within_rcv_does_not_escalate():
    vals = [1.00, 1.04, 1.08, 1.12, 1.16]
    rng = _range("Creatinine", ref_low=0.74, ref_high=1.35)
    out, traj = _analyze("Creatinine", "mg/dL", vals, rng)
    assert traj.trend is not None and traj.clinical_change is not None
    assert _independent_mk_p(vals) == pytest.approx(traj.trend.p_value)
    assert _independent_mk_p(vals) < ALPHA  # independently: a real trend...
    assert traj.trend.significant is True and traj.trend.direction == "increasing"
    net_pct = (vals[-1] - vals[0]) / vals[0] * 100.0  # +16.0%
    assert net_pct < _creatinine_rcv_pct()  # ...but within RCV (16.42%)
    assert traj.clinical_change.exceeds_rcv is False
    assert traj.severity == "info"  # in-range + within-RCV -> not even notable
    assert out.overall_floor == "none"


# --------------------------------------------------------------------------------------------------
# Point case — the E15 / C15 benign-out-of-range archetype, made out-of-sample (the eval's prose treats
# a young athlete's creatinine as "flagged high"; the shipped data keeps it in-range). Both shapes here
# are asserted to be NON-escalating, with independent ground truth for the "flat" claim.
# --------------------------------------------------------------------------------------------------


def test_oos_mildly_elevated_flat_is_notable_not_escalation():
    # (a) E15 as the prose intends it: creatinine just ABOVE range, but flat -> notable, never escalates.
    above = [1.40, 1.38, 1.41, 1.39]
    rng = _range("Creatinine", ref_low=0.74, ref_high=1.35)
    out, traj = _analyze("Creatinine", "mg/dL", above, rng)
    assert "above_range" in traj.flags
    assert (
        _independent_mk_p(above) >= ALPHA
    )  # flat: not a significant trend (independent)
    net_pct = abs((above[-1] - above[0]) / above[0] * 100.0)
    assert net_pct < _creatinine_rcv_pct()  # within RCV
    assert (
        traj.severity == "notable"
    )  # mildly elevated, benign -> a visible note, not an alarm
    assert out.overall_floor == "none"  # matches E15's "escalation: low"

    # (b) The shipped C15 shape: creatinine in-range (1.31 < 1.35), so it is NOT flagged at all. Locks
    # the data/eval-prose gap in a test — the member's claim "flagged high" is not borne out by the data.
    shipped = [1.30, 1.32, 1.29, 1.31]
    out2, traj2 = _analyze("Creatinine", "mg/dL", shipped, rng)
    assert "above_range" not in traj2.flags
    assert traj2.severity == "info"
    assert out2.overall_floor == "none"
