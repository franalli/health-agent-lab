"""templates.py — deterministic narration: verdicts -> the ``HealthIntelligenceResponse`` contract.

The Mode-1 / no-LLM rendering half (architecture §2). Two jobs in Phase 3a:

  * ``render_scan`` turns a ``TrajectoryAnalysis`` (the pure verdicts) into the response contract for
    the proactive scan — one ``Finding`` per raised marker, each carrying verbatim ``Evidence`` (the
    numbers came from code, so they are shown with confidence), the deterministic ``escalation`` floor,
    and a terse factual summary. Phase 4 swaps *only* the prose (``answer``) for LLM narration; the
    evidence and the floor are unchanged — they are not the model's to set.
  * the three safety responders (``seek_care`` / ``crisis`` / ``refuse``) the Phase-4 gate ``match``
    routes to instead of free-composing an emergency or a crisis reply over someone's labs. Built here
    so the contract exists; not wired to a route in 3a.

Microcopy follows ``ui-ux.md``: calm, honest, a next step — never alarmist, never a diagnosis. No
number is ever computed here; every value is read off the verdict or the reference range.
"""

from __future__ import annotations

from typing import Optional

from health_intelligence.models import (
    Evidence,
    Finding,
    FloorLevel,
    HealthIntelligenceResponse,
    MarkerTrajectory,
    ReferenceRange,
    ResponseMetadata,
)

# --------------------------------------------------------------------------------------------------
# Per-marker narration — title, explainability (trigger_reason), and the evidence stat ALL derive from
# one classifier (_classify), which names the single SIGNAL that drove the marker's severity in priority
# order: panic > counted adverse trend > range/band flag > surfaced-but-uncounted trend. Deriving every
# string from the one decision is what guarantees a Finding's text and its evidence chip can never cite
# different signals (the grounding contract), and that observation.title == finding.text.
# --------------------------------------------------------------------------------------------------

_DIRECTION_WORD = {"increasing": "rising", "decreasing": "falling", "flat": "changing"}


def _classify(traj: MarkerTrajectory) -> str:
    """Which signal drove this marker — the single source of truth all narration reads. A trend is the
    headline only when it was COUNTED toward the floor (severity 'attention': significant + adverse +
    RCV-cleared) or, absent any range/band flag, as a surfaced-but-uncounted trend; otherwise a present
    range/band flag leads."""
    flags = traj.flags
    if "panic_high" in flags:
        return "panic_high"
    if "panic_low" in flags:
        return "panic_low"
    t = traj.trend
    if traj.severity == "attention" and t is not None and t.direction != "flat":
        return "trend"
    if "above_range" in flags:
        return "above_range"
    if "below_range" in flags:
        return "below_range"
    if "band_cross" in flags:
        return "band_cross"
    if t is not None and t.direction != "flat":
        return "trend"
    return "flagged"


def observation_summary(traj: MarkerTrajectory) -> tuple[str, str]:
    """(title, trigger_reason) for the ``observations`` row and the Finding text — derived from
    ``_classify`` so the title and the evidence stat always describe the same signal."""
    m, signal = traj.marker, _classify(traj)
    val = f"{traj.latest.value} {traj.unit}"
    if signal == "panic_high":
        return f"{m} critically high", f"{m} latest {val} above the critical-high threshold"
    if signal == "panic_low":
        return f"{m} critically low", f"{m} latest {val} below the critical-low threshold"
    if signal == "trend":
        t = traj.trend
        word = _DIRECTION_WORD.get(t.direction, "changing") if t else "changing"
        rcv = "; clears reference-change value" if traj.severity == "attention" else ""
        p = f"{t.p_value:.3f}" if t else "n/a"
        n = t.n if t else 0
        return f"{m} {word}", f"{m} {word} trend (Mann-Kendall p={p}, n={n}{rcv})"
    if signal == "above_range":
        return f"{m} above range", f"{m} latest {val} above the reference range"
    if signal == "below_range":
        return f"{m} below range", f"{m} latest {val} below the reference range"
    if signal == "band_cross":
        return f"{m} crossed a clinical threshold", f"{m} moved across a clinical band cut-point"
    return f"{m} flagged", f"{m} flagged ({', '.join(traj.flags) or 'no signal'})"


def _stat_string(traj: MarkerTrajectory, rng: Optional[ReferenceRange]) -> str:
    """The statistic/threshold the claim rests on (evidence chip) — switched on the SAME ``_classify``
    decision as the title, so the two never diverge."""
    signal, t = _classify(traj), traj.trend
    if signal == "panic_high" and rng is not None and rng.panic_high is not None:
        return f"above critical-high {rng.panic_high} {traj.unit}"
    if signal == "panic_low" and rng is not None and rng.panic_low is not None:
        return f"below critical-low {rng.panic_low} {traj.unit}"
    if signal == "trend" and t is not None:
        slope = f", Theil-Sen slope {t.slope:.4g}/day" if t.slope is not None else ""
        return f"Mann-Kendall p={t.p_value:.3f}, tau={t.tau:.2f}, n={t.n}{slope}"
    if signal in ("above_range", "below_range"):
        return "outside reference range"
    if signal == "band_cross":
        return "crossed a clinical band cut-point"
    return "latest reading"


def scan_finding(
    traj: MarkerTrajectory, rng: Optional[ReferenceRange], finding_id: str, title: str
) -> Finding:
    """One ``Finding`` (text + a single backing ``Evidence``) for a raised marker. ``title`` is passed in
    (computed once by the caller via ``observation_summary``) so finding.text == observation.title by
    construction; the evidence stat is derived from the same ``_classify`` signal."""
    evidence = Evidence(
        marker=traj.marker,
        value=traj.latest.value,
        unit=traj.unit,
        date=traj.latest.date,
        ref_low=rng.ref_low if rng else None,
        ref_high=rng.ref_high if rng else None,
        stat=_stat_string(traj, rng),
    )
    return Finding(finding_id=finding_id, text=title, evidence=[evidence])


# --------------------------------------------------------------------------------------------------
# render_finding — one raised marker as its own HealthIntelligenceResponse (architecture §48: each
# observation persisted with its deterministic response). ``escalation`` is the member's deterministic
# floor (a standing property carried on every response, validated >= floor), NOT this marker's own
# severity. Phase 4 replaces the terse ``answer`` prose with LLM narration; findings + floor stay.
# --------------------------------------------------------------------------------------------------

def render_finding(
    finding: Finding,
    *,
    escalation: FloorLevel,
    metadata: ResponseMetadata,
) -> HealthIntelligenceResponse:
    return HealthIntelligenceResponse(
        answer=f"{finding.text}. This is drawn directly from your own readings; a clinician can interpret it in context.",
        findings=[finding],
        uncertainty="Based on your recorded panels for this marker.",
        answer_disposition="answered",
        escalation=escalation,
        metadata=metadata,
    )


# --------------------------------------------------------------------------------------------------
# Safety responders (architecture §2 D4) — fixed templates the Phase-4 gate routes to. Stubs in 3a:
# defined so the contract and the calm microcopy exist; not wired to a route until the gate lands.
# --------------------------------------------------------------------------------------------------

def seek_care_template(metadata: ResponseMetadata) -> HealthIntelligenceResponse:
    """Acute-medical responder: the unmissable next step leads; no lab narration to dilute it."""
    return HealthIntelligenceResponse(
        answer=(
            "Based on what you've described, please seek medical care now — contact urgent care or "
            "emergency services. This can't wait for a routine review."
        ),
        answer_disposition="answered",
        escalation="urgent",
        metadata=metadata,
    )


def crisis_template(metadata: ResponseMetadata) -> HealthIntelligenceResponse:
    """Crisis responder: warm, present, resources — never a clinical frame, never disengages."""
    return HealthIntelligenceResponse(
        answer=(
            "I'm really glad you told me, and I don't want you to go through this alone. If you're in "
            "immediate danger please contact emergency services or a crisis line right now — they're "
            "there for exactly this, any time."
        ),
        answer_disposition="answered",
        escalation="urgent",
        metadata=metadata,
    )


def refuse_template(metadata: ResponseMetadata) -> HealthIntelligenceResponse:
    """Out-of-scope responder: a friendly redirect that names the limit and points to a clinician."""
    return HealthIntelligenceResponse(
        answer=(
            "That's outside what I can help with from your lab history. Your GP or care team is the "
            "right place for this — I can help you make sense of your own results any time."
        ),
        answer_disposition="out_of_scope",
        escalation="none",
        metadata=metadata,
    )
