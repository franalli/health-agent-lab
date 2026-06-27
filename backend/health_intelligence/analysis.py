"""analysis.py — the deterministic analytical core (Phase 1).

The module that makes D1 real: every number, trend verdict, flag, and the escalation floor is computed
here in pure Python, so the LLM never reasons numerically. **Pure** — no DB, no LLM, no network, no
clock: typed inputs in, a typed ``TrajectoryAnalysis`` out, byte-identical across re-runs. The one
``datetime`` use parses the *given* panel dates (a deterministic function of input) — never the wall
clock. This is the one module whose statistical correctness is fully unit-testable in isolation.

Per marker: build the dated series → select the sex-appropriate range → Mann-Kendall (drift? exact
small-sample p, Kendall's tau) + Theil-Sen (direction, rate, distribution-free CI) → RCV (is the change
bigger than the marker's own noise?) → range/panic/band flags. Then one cross-marker FDR pass, a
direction-aware severity per marker, and a single ``overall_floor`` projection.

Two deliberate deviations, documented at their site and bundled in the Phase-1 report:

  * **FDR family (see ``_fdr``).** The doc's "Benjamini-Hochberg across markers" cannot be taken as
    *all* measured markers: real members cap at 5 panels, so the exact MK two-sided p floors at
    2/120 = 0.0167 (n=5). BH over ~16 markers needs the rank-1 p <= (1/16)*q = 0.006, so single- and
    double-trend cases (E05 TSH, E10 CRP, E04 Hb+ferritin) could *never* be significant and the core
    would only ever escalate on panic. The family is therefore the markers passing the doc's explicit
    ``p < alpha`` trigger; BH then controls multiplicity *within* that candidate set. At the data's
    n (<=5) the operative gate is the discrete-alpha bar (p must be <= 0.0167, a 1/120 event — not the
    naive 1-in-20) plus RCV and adverse-direction; with ``fdr_q`` (0.10) > ``alpha`` (0.05) the BH step
    is structurally present but currently always-passes its candidates (it would bite only if a future
    config sets q < alpha or widens the family).

  * **E12 / sparse abstention.** The architecture says n < n_min abstains ("too short to call") and an
    out-of-range value is at most ``notable`` — so C12 (3 panels, eGFR 75->61 below range) projects to
    ``overall_floor = none`` here, while eval case E12 expects a routine clinician review. That is an
    architecture-vs-eval tension to settle in Phase 5 (a sub-n_min escalation policy is a *rule*, not a
    constant), NOT a Phase-1 bug: this module implements the architecture's abstention faithfully.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from datetime import date
from statistics import NormalDist, median
from typing import NamedTuple

from health_intelligence.config import AnalysisConfig, MarkerConfig
from health_intelligence.models import (
    FLOOR_ORDER,
    SEVERITY_ORDER,
    SEVERITY_TO_FLOOR,
    ClinicalChange,
    Flag,
    FloorLevel,
    LabResult,
    MarkerTrajectory,
    MemberProfile,
    Reading,
    ReferenceRange,
    Severity,
    TrajectoryAnalysis,
    TrendResult,
)

# Both significance Z's live in versioned config, never as literals here: the Theil-Sen CI Z derives
# from cfg.stats.ts_ci_level and the RCV Z from cfg.stats.rcv_z (so an RCV-gated escalation is
# reproducible against config_version).
# Exact MK p-value strategy by series length: enumerate all n! orderings (tie-exact) up to here...
_EXACT_ENUM_MAX = 8
# ...else the inversion-count DP (exact, distinct values only) up to here; above it, the tie-corrected
# normal approximation with continuity correction (acceptable only as n approaches where it calibrates).
_EXACT_DP_MAX = 25


class _MarkerWork(NamedTuple):
    """Per-marker signals from pass 1, carried across the cross-marker FDR pass into assembly — one
    record instead of five parallel dicts. ``trend`` is the same object the FDR pass mutates in place."""

    marker: str
    unit: str
    series: list[Reading]
    trend: TrendResult | None
    change: ClinicalChange | None
    flags: list[Flag]
    mcfg: MarkerConfig


# --------------------------------------------------------------------------------------------------
# Public entry point — pure: identical inputs give identical output.
# --------------------------------------------------------------------------------------------------


def analyze(
    member: MemberProfile,
    results: list[LabResult],
    ranges: list[ReferenceRange],
    age: int | None,
    cfg: AnalysisConfig,
    *,
    data_version: str,
) -> TrajectoryAnalysis:
    """Compute the whole deterministic verdict for one member.

    ``age`` is accepted per the architecture's clock-free contract (the caller resolves it); current
    ranges key on sex alone (no age bands in the data), so it is not yet load-bearing. ``data_version``
    is a keyword-only caller stamp — like ``age``, contextual state the pure core does not derive — so
    the positional signature stays exactly the architecture's ``(member, results, ranges, age, cfg)``.
    """
    # Pass 1 — one record of per-marker signals (severity waits on the cross-marker FDR pass).
    work: list[_MarkerWork] = []
    for marker in _ordered_markers(results):
        mcfg = cfg.markers.get(marker, MarkerConfig())
        series = _series(results, marker)
        work.append(
            _MarkerWork(
                marker=marker,
                unit=_unit(results, marker),
                series=series,
                trend=_trend(series, cfg),
                change=_clinical_change(series, mcfg, cfg.stats.rcv_z),
                flags=_flags(series, _range_for(marker, member.sex, age, ranges), mcfg),
                mcfg=mcfg,
            )
        )

    # Cross-marker pass — Benjamini-Hochberg over the p<alpha candidates sets `significant` (see _fdr).
    significant = _fdr([(w.marker, w.trend) for w in work if w.trend is not None], cfg)
    for w in work:
        if (
            w.trend is not None
        ):  # only trended markers are in `significant`; also narrows the Optional
            w.trend.significant = significant[w.marker]

    # Pass 2 — assemble each trajectory with its direction-aware severity, then project the floor.
    trajectories = [
        MarkerTrajectory(
            marker=w.marker,
            unit=w.unit,
            latest=w.series[-1],
            trend=w.trend,
            clinical_change=w.change,
            flags=w.flags,
            severity=_severity(w.trend, w.change, w.flags, w.mcfg),
        )
        for w in work
    ]
    return TrajectoryAnalysis(
        member_id=member.member_id,
        data_version=data_version,
        overall_floor=_floor([t.severity for t in trajectories]),
        markers=trajectories,
    )


# --------------------------------------------------------------------------------------------------
# Series + range selection
# --------------------------------------------------------------------------------------------------


def _ordered_markers(results: list[LabResult]) -> list[str]:
    """Distinct markers in first-seen order (stable output ordering, no set nondeterminism)."""
    seen: dict[str, None] = {}
    for r in results:
        seen.setdefault(r.marker, None)
    return list(seen)


def _series(results: list[LabResult], marker: str) -> list[Reading]:
    rs = sorted((r for r in results if r.marker == marker), key=lambda r: r.panel_date)
    return [Reading(value=r.value, date=r.panel_date) for r in rs]


def _unit(results: list[LabResult], marker: str) -> str:
    # Units are consistent per marker (conversion happened at ingest), so any reading's unit serves.
    return next((r.unit for r in results if r.marker == marker), "")


def _range_for(
    marker: str, sex: str, age: int | None, ranges: list[ReferenceRange]
) -> ReferenceRange | None:
    """The applicable band: the (marker, sex) row, else the sex-agnostic (marker, 'any') row, else
    None — so 'other'/'unknown' members and non-sex-split markers both resolve to 'any', and a marker
    with no band at all yields the typed ``no_reference`` flag downstream. ``age`` is in the
    architecture's signature for future age-banded ranges; the supplied data has none, so it is unused."""
    del (
        age
    )  # reserved (no age-banded ranges in the data); keeps the documented signature
    cands = [r for r in ranges if r.marker == marker]
    return next((r for r in cands if r.sex == sex), None) or next(
        (r for r in cands if r.sex == "any"), None
    )


# --------------------------------------------------------------------------------------------------
# Mann-Kendall (monotonic trend) — tie-aware S, exact small-sample p
# --------------------------------------------------------------------------------------------------


def _kendall_s(values: list[float]) -> int:
    """Kendall's S over the time-ordered values: sum of sign(v_j - v_i) for i<j; ties contribute 0."""
    s = 0
    n = len(values)
    for i in range(n):
        vi = values[i]
        for j in range(i + 1, n):
            d = values[j] - vi
            s += (d > 0) - (d < 0)
    return s


def _tau_b(values: list[float], s: int) -> float:
    """Kendall's tau-b (tie-corrected); time ties are absent (panel dates order the series)."""
    n = len(values)
    n0 = n * (n - 1) // 2
    n1 = sum(c * (c - 1) // 2 for c in Counter(values).values())
    denom = math.sqrt((n0 - n1) * n0)
    return s / denom if denom > 0 else 0.0


def _mk_pvalue(values: list[float], s: int) -> float:
    """Exact two-sided Mann-Kendall p-value. The normal approximation to S is miscalibrated below ~10
    points (every marker here), so small n is exact: the inversion-count distribution for distinct
    values up to n<=25 (fast), the full n!-ordering enumeration for ties up to n<=8, else a tie-corrected
    normal approximation with continuity correction."""
    n = len(values)
    if s == 0:
        return 1.0
    if len(set(values)) == n and n <= _EXACT_DP_MAX:
        return _exact_p_inversions(
            n, s
        )  # distinct: fast exact via the inversion-count distribution
    if n <= _EXACT_ENUM_MAX:
        return _exact_p_enum(
            values, s
        )  # ties: exact via full multiset-ordering enumeration
    return _normal_p(values, s)  # large n with ties: tie-corrected normal approximation


def _exact_p_enum(values: list[float], s_obs: int) -> float:
    """Exact p by enumerating every ordering of the (multiset of) values — handles ties exactly."""
    target = abs(s_obs)
    total = extreme = 0
    for perm in itertools.permutations(values):
        total += 1
        if abs(_kendall_s(list(perm))) >= target:
            extreme += 1
    return extreme / total


def _inversion_counts(n: int) -> list[int]:
    """counts[d] = number of permutations of n DISTINCT items with exactly d inversions (Mahonian)."""
    counts = [1]
    for i in range(2, n + 1):
        nxt = [0] * (len(counts) + i - 1)
        for d, c in enumerate(counts):
            for k in range(i):
                nxt[d + k] += c
        counts = nxt
    return counts


def _exact_p_inversions(n: int, s_obs: int) -> float:
    """Exact two-sided p for distinct values via the inversion-count distribution. S = n0 - 2*D, so
    |S| >= |s| <=> D <= (n0-|s|)/2 (S>=|s|) or D >= (n0+|s|)/2 (S<=-|s|)."""
    n0 = n * (n - 1) // 2
    counts = _inversion_counts(n)
    total = sum(counts)
    s = abs(s_obs)
    lo_cut = (n0 - s) / 2
    hi_cut = (n0 + s) / 2
    tail = sum(c for d, c in enumerate(counts) if d <= lo_cut or d >= hi_cut)
    return tail / total


def _normal_p(values: list[float], s: int) -> float:
    var = _s_variance(values)
    if var <= 0:
        return 1.0
    z = (s - math.copysign(1, s)) / math.sqrt(var)  # continuity correction
    return 2.0 * (1.0 - NormalDist().cdf(abs(z)))


def _s_variance(values: list[float]) -> float:
    """Var(S) under H0 with the standard tie correction."""
    n = len(values)
    ties = sum(t * (t - 1) * (2 * t + 5) for t in Counter(values).values())
    return (n * (n - 1) * (2 * n + 5) - ties) / 18.0


# --------------------------------------------------------------------------------------------------
# Theil-Sen (direction, rate, distribution-free CI)
# --------------------------------------------------------------------------------------------------


def _theil_sen(series: list[Reading]) -> tuple[float | None, list[float]]:
    """Median of pairwise slopes (per day) and the sorted slope list (for the CI)."""
    pts = [(date.fromisoformat(r.date).toordinal(), r.value) for r in series]
    slopes = [
        (pts[j][1] - pts[i][1]) / (pts[j][0] - pts[i][0])
        for i in range(len(pts))
        for j in range(i + 1, len(pts))
        if pts[j][0] != pts[i][0]
    ]
    if not slopes:
        return None, []
    slopes.sort()
    return median(slopes), slopes


def _theil_sen_ci(
    slopes: list[float], values: list[float], ci_level: float
) -> tuple[float, float]:
    """Gilbert (1987) distribution-free CI from Var(S): the limits are order statistics of the
    pairwise slopes at ranks (N -/+ C)/2, C = Z * sqrt(Var(S))."""
    nslopes = len(slopes)
    c = NormalDist().inv_cdf(1.0 - (1.0 - ci_level) / 2.0) * math.sqrt(
        _s_variance(values)
    )
    lo = _clamp(int(round((nslopes - c) / 2.0)) - 1, 0, nslopes - 1)
    hi = _clamp(int(round((nslopes + c) / 2.0)), 0, nslopes - 1)
    return slopes[lo], slopes[hi]


def _clamp(i: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, i))


# --------------------------------------------------------------------------------------------------
# _trend — Mann-Kendall + Theil-Sen assembled; None below n_min (no trend claimed, not a noisy one)
# --------------------------------------------------------------------------------------------------


def _trend(series: list[Reading], cfg: AnalysisConfig) -> TrendResult | None:
    n = len(series)
    if n < cfg.stats.n_min:
        return None
    values = [r.value for r in series]
    s = _kendall_s(values)
    med, slopes = _theil_sen(series)
    direction, slope, ci = "flat", None, None
    if slopes:
        lo, hi = _theil_sen_ci(slopes, values, cfg.stats.ts_ci_level)
        ci = (lo, hi)
        if lo > 0:
            direction, slope = "increasing", med
        elif hi < 0:
            direction, slope = "decreasing", med
        # else: CI spans zero -> "flat", slope stays None ("too noisy to sign")
    return TrendResult(
        direction=direction,
        tau=_tau_b(values, s),
        p_value=_mk_pvalue(values, s),
        slope=slope,
        slope_ci=ci,
        n=n,
        significant=False,  # set by the cross-marker FDR pass
    )


# --------------------------------------------------------------------------------------------------
# _clinical_change — Reference Change Value (clinical trend-vs-noise gate)
# --------------------------------------------------------------------------------------------------


def _clinical_change(
    series: list[Reading], mcfg: MarkerConfig, rcv_z: float
) -> ClinicalChange | None:
    """RCV = sqrt(2)*Z*sqrt(CVa^2 + CVi^2) (Z from cfg.stats.rcv_z); the net (latest-vs-earliest) percent
    change clears it or is noise. CVa or CVi absent -> typed skip-path (``exceeds_rcv = None``): the
    trend is judged on Mann-Kendall + Theil-Sen CI alone, never silently treated as cleared-or-not."""
    if len(series) < 2:
        return None
    baseline, latest = series[0].value, series[-1].value
    net = ((latest - baseline) / baseline * 100.0) if baseline != 0 else None
    if mcfg.cva is None or mcfg.cvi is None:
        return ClinicalChange(rcv=None, net_change=net, exceeds_rcv=None)
    rcv = math.sqrt(2.0) * rcv_z * math.sqrt(mcfg.cva**2 + mcfg.cvi**2)
    exceeds = (abs(net) >= rcv) if net is not None else None
    return ClinicalChange(rcv=rcv, net_change=net, exceeds_rcv=exceeds)


# --------------------------------------------------------------------------------------------------
# _flags — range / panic / band on the latest value (work at n=1; band-cross needs n>=2)
# --------------------------------------------------------------------------------------------------


def _flags(
    series: list[Reading], rng: ReferenceRange | None, mcfg: MarkerConfig
) -> list[Flag]:
    if rng is None:
        return ["no_reference"]
    latest = series[-1].value
    flags: list[Flag] = []
    if rng.ref_low is not None and latest < rng.ref_low:
        flags.append("below_range")
    if rng.ref_high is not None and latest > rng.ref_high:
        flags.append("above_range")
    if rng.panic_low is not None and latest < rng.panic_low:
        flags.append("panic_low")
    if rng.panic_high is not None and latest > rng.panic_high:
        flags.append("panic_high")
    if len(series) >= 2:
        first_band = _band_index(series[0].value, mcfg)
        last_band = _band_index(latest, mcfg)
        if first_band is not None and last_band is not None and first_band != last_band:
            flags.append("band_cross")
    return flags


def _band_index(value: float, mcfg: MarkerConfig) -> int | None:
    """Which discrete band the value sits in (cut-points: count thresholds cleared; graded: the [low,
    high) segment). ``None`` when the marker has no band concept OR the value falls outside every
    graded band — both mean 'no band to compare', so a band-cross is not inferred against it."""
    if mcfg.band_cutpoints:
        return sum(1 for cut in mcfg.band_cutpoints if value >= cut)
    if mcfg.graded_bands:
        for idx, band in enumerate(mcfg.graded_bands):
            if (band.low is None or value >= band.low) and (
                band.high is None or value < band.high
            ):
                return idx
        return None  # outside all graded bands -> no band (not a sentinel index that would fake a cross)
    return None


# --------------------------------------------------------------------------------------------------
# Cross-marker FDR — Benjamini-Hochberg over the p<alpha candidates (see module-docstring note)
# --------------------------------------------------------------------------------------------------


def _fdr(trends: list[tuple[str, TrendResult]], cfg: AnalysisConfig) -> dict[str, bool]:
    """Set per-marker significance. The doc's ``p < alpha`` trigger defines the candidate family; BH at
    ``fdr_q`` then controls multiplicity *within* it (step-up: the largest rank passing rescues lower
    ranks). Restricting the family to candidates is what lets a lone real trend survive at n<=5, where
    BH across all ~16 markers would reject everything (module docstring). Returns {marker: significant}
    for every trended marker."""
    alpha, q = cfg.stats.alpha, cfg.stats.fdr_q
    candidates = sorted(
        ((m, t) for m, t in trends if t.p_value < alpha), key=lambda mt: mt[1].p_value
    )
    m = len(candidates)
    max_rank = 0
    for rank, (_, t) in enumerate(candidates, start=1):
        if t.p_value <= (rank / m) * q:
            max_rank = rank
    survivors = {
        marker
        for rank, (marker, _) in enumerate(candidates, start=1)
        if rank <= max_rank
    }
    return {marker: (marker in survivors) for marker, _ in trends}


# --------------------------------------------------------------------------------------------------
# Severity (direction-aware) + floor projection
# --------------------------------------------------------------------------------------------------


def _is_adverse(direction: str, adverse: str | None) -> bool:
    """A signed trend moving the wrong way. ``adverse is None`` (uncurated / bidirectional, e.g.
    Potassium) -> not classifiable, so it never raises severity (panic, not trend, guards those)."""
    if adverse is None:
        return False
    return (direction == "increasing" and adverse == "up") or (
        direction == "decreasing" and adverse == "down"
    )


def _severity(
    trend: TrendResult | None,
    change: ClinicalChange | None,
    flags: list[Flag],
    mcfg: MarkerConfig,
) -> Severity:
    """Map the verdict set to one severity. An out-of-range / band-cross value is at most ``notable``
    (escalation != out-of-range). A *significant adverse* trend reaches ``attention`` only once it
    clears RCV — the triage rule (architecture §2 D3) counts a trend toward the floor only if it cleared
    *both* RCV and FDR. Without CVa/CVi the trend is still surfaced at ``notable`` (a visible
    observation, not silent) but is not escalated; a change within RCV noise is not counted at all. A
    panic flag pins ``urgent`` over everything."""
    severity: Severity = "info"

    def raise_to(level: Severity) -> None:
        nonlocal severity
        if SEVERITY_ORDER[level] > SEVERITY_ORDER[severity]:
            severity = level

    if any(f in flags for f in ("below_range", "above_range", "band_cross")):
        raise_to("notable")

    if (
        trend is not None
        and trend.significant
        and _is_adverse(trend.direction, mcfg.adverse_direction)
    ):
        if change is not None and change.exceeds_rcv is True:
            raise_to(
                "attention"
            )  # RCV + FDR confirmed adverse trend -> counts toward the floor
        elif change is None or change.exceeds_rcv is None:
            raise_to(
                "notable"
            )  # no CVa/CVi -> surfaced on MK + CI, not escalated without RCV
        # change.exceeds_rcv is False -> net change within biological noise (RCV) -> not a counted trend

    if "panic_low" in flags or "panic_high" in flags:
        severity = "urgent"
    return severity


def _floor(severities: list[Severity]) -> FloorLevel:
    """Pure projection of the max per-marker severity onto the escalation axis, via the one shared
    ``SEVERITY_TO_FLOOR`` table (so the whole-member floor and ``safety.severity_to_level``'s per-marker
    level can't drift). Nothing downstream can lower it; the model only reads it."""
    floor: FloorLevel = "none"
    for s in severities:
        if FLOOR_ORDER[SEVERITY_TO_FLOOR[s]] > FLOOR_ORDER[floor]:
            floor = SEVERITY_TO_FLOOR[s]
    return floor
