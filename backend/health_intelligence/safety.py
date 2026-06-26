"""safety.py — the deterministic safety guard: the data floor and the output validator.

This is the layer CLAUDE.md calls "always on". The escalation *floor* is computed once, by the pure
core (``TrajectoryAnalysis.overall_floor`` — a projection of the max per-marker severity); this module
does NOT recompute it (a second floor computation is exactly the drift the "floor always on" invariant
forbids). It only (1) reads the floor off the analysis and (2) enforces that no response sits below it.

The validator is the runtime half of an assertion the eval harness re-runs offline (architecture §8):
the *same* ``escalation >= floor`` check guards every live response and scores every eval case, so the
eval certifies the guard that ships rather than a parallel copy that can drift. In Mode 1 (Phase 3a)
the response is templated *at* the floor, so ``validate`` always passes; it becomes load-bearing for
the LLM in Phase 4, where it rejects an under-floor compose, triggers one bounded retry, and falls
through to a deterministic safety template. The ordering is ranked through ``FLOOR_ORDER`` — never a
string compare, since lexicographically "clinician_review" < "none" would invert the floor.
"""

from __future__ import annotations

from typing import Optional

from health_intelligence.models import (
    FLOOR_ORDER,
    SEVERITY_TO_FLOOR,
    EscalationLevel,
    FloorLevel,
    HealthIntelligenceResponse,
    Severity,
    TrajectoryAnalysis,
)


class FloorViolation(ValueError):
    """Raised when a response's ``escalation`` axis sits below the computed floor — a softer-toned
    answer on a harder floor. The deterministic layer wins: the caller repairs (Phase 4) or, in
    Mode 1, this never fires because the response is built at the floor."""


def data_floor(analysis: TrajectoryAnalysis) -> FloorLevel:
    """The escalation floor the data implies — returned verbatim from the pure core's projection.
    The single source of truth for "how alarming is this member's data"; nothing here re-derives it."""
    return analysis.overall_floor


def severity_to_level(severity: Severity) -> Optional[EscalationLevel]:
    """The escalation level a single marker's severity projects onto the clinician-review queue, or
    ``None`` when it doesn't escalate — derived from the SAME ``SEVERITY_TO_FLOOR`` table the core's
    whole-member ``_floor`` uses, so the max over a member's per-marker levels equals ``overall_floor``
    by construction (no drift into a per-marker escalation below the computed floor)."""
    floor = SEVERITY_TO_FLOOR[severity]
    return None if floor == "none" else floor


def meets_floor(escalation: FloorLevel, floor: FloorLevel) -> bool:
    """True iff ``escalation`` is at or above ``floor`` on the escalation axis (ranked, not strung)."""
    return FLOOR_ORDER[escalation] >= FLOOR_ORDER[floor]


def validate(resp: HealthIntelligenceResponse, floor: FloorLevel) -> HealthIntelligenceResponse:
    """Enforce ``resp.escalation >= floor`` and return the response unchanged when it passes. Raises
    :class:`FloorViolation` otherwise — the validator never silently *raises* the response's escalation
    to meet the floor (that would mask a generator that under-escalated); the caller owns the repair."""
    if not meets_floor(resp.escalation, floor):
        raise FloorViolation(
            f"response escalation {resp.escalation!r} is below the computed floor {floor!r}"
        )
    return resp
