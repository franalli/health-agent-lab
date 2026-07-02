"""eval/types.py — the harness's internal data shapes (architecture §8's ``Case`` / ``ScorerResult``).

These are EVAL-LOCAL types, deliberately NOT in ``health_intelligence/models.py``: the serving library
is pure domain/wire/persistence and must not import eval code, and the harness is a consumer of that
library, never the reverse (the two-layer invariant). Mirrors the §8 module sketch.

Naming note: the §8 sketch writes ``pass`` on a scorer result; ``pass`` is a Python keyword, so the
field here is ``passed``. ``escalation`` is an *acceptable-set* (a list), because the supplied free-text
labels include genuinely-ambiguous ones the architecture normalizes to a set ("clinician_review OR
urgent both pass, none fails") — over-escalation is the safe direction, so the pass test is
``resp >= min(set)`` on the ranked floor axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from health_intelligence.models import (
    FLOOR_ORDER,
    AnswerDisposition,
    Escalation,
    FloorLevel,
    HealthIntelligenceResponse,
    LabResult,
    MemberProfile,
    ReferenceRange,
    TrendDirection,
)

#: The observable route proxy — the same labels as ``gate.GateRoute``, but the gate's internal route is
#: NOT on the wire response, so the harness infers it from the served response's disposition + the fixed
#: safety-template fingerprints (architecture §8 grades *observable behavior*).
Route = Literal["none", "out_of_scope", "acute_medical", "crisis"]

#: The three blocking safety failures (architecture §8). Any one fails the run and is surfaced first.
NeverEvent = Literal["missed_escalation", "fabricated_value", "unrefused_directive"]

#: The Mode-2 dimensions that carry a pass/fail VERDICT — the gate-eligible set, owned here (the eval
#: layer owns the dimensions). The ONE source for both the report's per-dimension table and the /learn
#: gate's "no dimension regresses" check, so adding a pass/fail scorer can't leave one of them stale
#: (``latency_cost`` is deliberately excluded — it is informational and always 'passes', §8).
GATE_DIMENSIONS: tuple[str, ...] = ("escalation", "routing", "grounding", "consistency")

#: Whether Mode 1 (the deterministic preset surface) is expected to handle a case's category, or to
#: gracefully defer it (out-of-scope / gate routes have no Mode-1 answerer). Coverage = fraction handled.
Mode1Coverage = Literal["covered", "deferred"]


class TrendExpectation(BaseModel):
    """A known-answer trend verdict for a ``score_stats`` fixture. The supplied set carries no structured
    trend verdict (architecture §8), so these labels come from authored fixtures only. ``trend_is_none``
    marks the sparse abstention (n < ``StatConfig.n_min`` → no trend claimed, a first-class verdict)."""

    marker: str
    direction: TrendDirection | None = None
    significant: bool | None = None
    trend_is_none: bool = False


class CaseExpectation(BaseModel):
    """The labels for one case — the adapter fills these from the supplied fields + the category maps."""

    route: Route = "none"
    escalation: list[
        FloorLevel
    ]  # acceptable-set; pass iff resp >= min(set) on FLOOR_ORDER
    escalation_raw: str = (
        ""  # the supplied free-text phrasing, kept verbatim for the audit
    )
    disposition: list[AnswerDisposition] = Field(default_factory=lambda: ["answered"])
    must_include: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    must_cite: list[str] = Field(
        default_factory=list
    )  # PRESENT marker key(s) the answer MUST cite as evidence — an over-refusal ("not in your results" for
    # a marker named by a lay alias, e.g. "blood pressure"->systolic_bp) cites nothing and fails this. A
    # DETERMINISTIC guard for the compose rule-3 over-refusal risk; empty (the default) skips it — zero blast
    # radius on cases that don't set it. (must_include / must_not are semantic assertions for the deferred
    # LLM judge; must_cite is the deterministic slice a scorer CAN check today.)
    absent_marker: list[str] = Field(
        default_factory=list
    )  # marker(s) not in the member's data
    mode1_coverage: Mode1Coverage = "covered"

    @property
    def escalation_is_set(self) -> bool:
        """True when the label normalized to a genuinely-ambiguous acceptable-set (>1 distinct level) —
        the flag the report's acceptable-set audit reads (Card-4: the tolerance must be visible)."""
        return len(set(self.escalation)) > 1

    @property
    def min_escalation(self) -> FloorLevel:
        """The lowest level that still passes — below it is the under-call that fails the scorer."""
        return min(self.escalation, key=lambda f: FLOOR_ORDER[f])


class Case(BaseModel):
    """One evaluation case after the adapter has mapped the supplied/added fields onto the internal
    shape. ``tags`` separates ``"supplied"`` (the 17-case set) from ``"added"`` (the tagged gate cases),
    so the report can report them apart (architecture §8: added cases are noted, with why)."""

    id: str
    member_id: str
    category: str
    driver: Literal["ask", "scan"] = "ask"
    question: str
    tags: list[str] = Field(default_factory=list)
    expected: CaseExpectation


class StatsFixture(BaseModel):
    """A self-contained known-answer trend case for ``score_stats`` — the post-ingest domain inputs
    ``analysis.analyze`` consumes plus the expected verdict. ``score_stats`` runs ``analyze`` on these
    directly (NOT through the service), the one scorer that ignores ``responses`` and grades the pure
    core against the Phase-1 fixtures, keeping the two label sources distinct (architecture §8)."""

    label: str
    member: MemberProfile
    results: list[LabResult]
    ranges: list[ReferenceRange]
    age: int | None
    expected: TrendExpectation


class CaseResponses(BaseModel):
    """Everything the runner collected for one case — the input to the pure scorers (architecture §8:
    "scorers pure over ``(case, responses)``"). ``mode2`` is the N live ``/ask`` responses (the N is for
    the consistency/flip-rate measurement); ``mode1`` is the deterministic "what's changed" overview
    (``None`` when the category is deferred); ``mode1_repeat`` is a second Mode-1 call for the
    byte-identical check; ``escalations`` is the post-run clinician-queue DB snapshot (the escalation
    scorer's DB check + the idempotency proof); ``marker_values`` is the member's latest value per
    canonical marker (the analysis snapshot the number-tracer needs to spot a discussed-but-uncited
    value)."""

    mode2: list[HealthIntelligenceResponse] = Field(default_factory=list)
    mode1: HealthIntelligenceResponse | None = None
    mode1_repeat: HealthIntelligenceResponse | None = None
    escalations: list[Escalation] = Field(default_factory=list)
    marker_values: dict[str, float] = Field(default_factory=dict)


class ScorerResult(BaseModel):
    """One scorer's verdict for one case (architecture §8 ``ScorerResult{dimension, pass, score,
    detail, never_event?}``; ``pass`` → ``passed``). ``metrics`` carries dimension-specific numbers the
    report aggregates (latency/cost percentiles, flip-rate); ``observed``/``expected`` feed the routing
    & escalation confusion matrices; ``used_acceptable_set`` flags the Card-4 audit."""

    dimension: str
    passed: bool
    detail: str = ""
    score: float | None = None
    never_event: NeverEvent | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    observed: str | None = None
    expected: str | None = None
    used_acceptable_set: bool = False


@dataclass(frozen=True)
class EvalConfig:
    """Run knobs (architecture §8: ``cfg`` sets N, gates the judge). ``n_runs`` is the Mode-2 repeat
    count for the consistency measurement (Mode 1 is byte-identical, so it needs no repeats); ``dataset``
    selects the bundle (None → the ``DATASET`` env / ``training_data``); ``use_judge`` gates the two
    judge scorers (Phase 5b — off in 5a)."""

    n_runs: int = 3
    use_judge: bool = False
    dataset: str | None = None
