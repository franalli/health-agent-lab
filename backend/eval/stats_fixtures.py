"""eval/stats_fixtures.py — known-answer trend fixtures for ``score_stats`` (architecture §8).

The supplied case set carries no structured trend verdict, so ``score_stats`` grades ``analysis.py``
against authored fixtures instead — "two label sources, kept distinct". These are lifted from the
Phase-1 ``tests/test_analysis.py`` known-answer cases (the architecture's "promote those deterministic
tests into scorers"): a seeded decline that must register significant, in-range noise that must NOT, an
adverse uptrend that must, and the sparse n=3 series that must honestly abstain (n < ``StatConfig.n_min``
= 4). The last two are the trend-vs-noise and sparse cells the failure-mode table predicts.

``score_stats`` runs ``analysis.analyze`` on each fixture directly (no service), so this is the one
scorer that ignores the service ``responses``. The small ``_member``/``_dates``/``_results``/``_range``
builders deliberately parallel the ones in ``tests/test_analysis.py`` rather than sharing a module:
hoisting them would invert the layering (an ``analysis`` unit-test must not import ``eval``, and ``eval``
must not import test code). Drift is not silent — ``test_score_stats_passes_every_authored_fixture``
re-derives every labeled verdict from these inputs, so a builder change that broke them fails loudly.
"""

from __future__ import annotations

from eval.types import StatsFixture, TrendExpectation
from health_intelligence.models import (
    LabResult,
    MemberProfile,
    MemberSex,
    ReferenceRange,
)


def _member(
    member_id: str = "S01", sex: MemberSex = "male", age: int = 50
) -> MemberProfile:
    return MemberProfile(member_id=member_id, age=age, sex=sex)


def _dates(n: int, start: str = "2024-01-15") -> list[str]:
    """n monthly ISO dates from ``start`` (mirrors tests/test_analysis.py — irregular real spacing)."""
    y, m, d = (int(x) for x in start.split("-"))
    out: list[str] = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}-{d:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _results(
    marker: str, unit: str, values: list[float], dates: list[str] | None = None
) -> list[LabResult]:
    dates = dates or _dates(len(values))
    return [
        LabResult(marker=marker, value=v, unit=unit, panel_id=f"P{i}", panel_date=dt)
        for i, (dt, v) in enumerate(zip(dates, values, strict=True), 1)
    ]


def _range(marker: str, *, ref_low=None, ref_high=None) -> ReferenceRange:
    return ReferenceRange(
        marker=marker,
        sex="any",
        unit="",
        ref_low=ref_low,
        ref_high=ref_high,
        config_version="v0",
    )


STATS_FIXTURES: list[StatsFixture] = [
    # Seeded monotonic decline (12 points) — Mann-Kendall well under alpha, Theil-Sen CI excludes zero.
    StatsFixture(
        label="seeded_decline_significant",
        member=_member(),
        results=_results("eGFR", "mL/min/1.73m2", [100.0 - 1.8 * i for i in range(12)]),
        ranges=[_range("eGFR", ref_low=90.0)],
        age=50,
        expected=TrendExpectation(
            marker="eGFR", direction="decreasing", significant=True
        ),
    ),
    # Adverse uptrend (HbA1c 5.6→6.4) — increasing, significant; the escalating-direction branch.
    StatsFixture(
        label="adverse_uptrend_significant",
        member=_member(),
        results=_results("HbA1c", "%", [5.6, 5.8, 6.0, 6.2, 6.4]),
        ranges=[_range("HbA1c", ref_high=5.7)],
        age=50,
        expected=TrendExpectation(
            marker="HbA1c", direction="increasing", significant=True
        ),
    ),
    # In-range noise (C08/E08-shaped) — a trend is COMPUTED (n≥n_min) but not significant; CI spans zero
    # → "flat". The core must not manufacture a decline from noise (the trend-vs-noise failure cell).
    StatsFixture(
        label="in_range_noise_not_significant",
        member=_member("S08"),
        results=_results(
            "eGFR", "mL/min/1.73m2", [95.0, 92.0, 99.0, 91.0, 99.0, 93.0, 95.0]
        ),
        ranges=[_range("eGFR", ref_low=90.0)],
        age=44,
        expected=TrendExpectation(marker="eGFR", direction="flat", significant=False),
    ),
    # Sparse n=3 (C12/E12-shaped) — below n_min=4, so NO trend is claimed: honest abstention, a
    # first-class verdict (the sparse-series failure cell). Range flags still compute, but trend is None.
    StatsFixture(
        label="sparse_n3_abstains",
        member=_member("S12"),
        results=_results("eGFR", "mL/min/1.73m2", [75.0, 68.0, 61.0]),
        ranges=[_range("eGFR", ref_low=90.0)],
        age=58,
        expected=TrendExpectation(marker="eGFR", trend_is_none=True),
    ),
]
