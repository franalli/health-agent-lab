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

from collections.abc import Callable

from health_intelligence.models import (
    Evidence,
    Finding,
    FloorLevel,
    HealthIntelligenceResponse,
    MarkerTrajectory,
    Reading,
    ReferenceRange,
    ResponseMetadata,
    SuggestedPrompt,
    TrajectoryAnalysis,
    TrendResult,
)

# --------------------------------------------------------------------------------------------------
# Per-marker narration — title, explainability (trigger_reason), and the evidence stat ALL derive from
# one classifier (_classify), which names the single SIGNAL that drove the marker's severity in priority
# order: panic > counted adverse trend > range/band flag > surfaced-but-uncounted trend. Deriving every
# string from the one decision is what guarantees a Finding's text and its evidence chip can never cite
# different signals (the grounding contract), and that observation.title == finding.text.
# --------------------------------------------------------------------------------------------------

_DIRECTION_WORD = {"increasing": "rising", "decreasing": "falling", "flat": "changing"}

#: Member-friendly labels for the canonical marker keys that aren't already clean. Labs read as printed
#: ("HbA1c"); the three folded-in vitals carry snake_case STORAGE keys that must never surface in member
#: copy. This maps only the *prose / titles / prompts* — the structured ``Evidence.marker`` field keeps
#: the canonical key (grounding + cross-chip matching key on it), so display naming can't break either.
_DISPLAY_NAME = {
    "systolic_bp": "systolic blood pressure",
    "diastolic_bp": "diastolic blood pressure",
    "bmi": "BMI",
}


def _display_name(marker: str) -> str:
    return _DISPLAY_NAME.get(marker, marker)


def _trend_stat(t: TrendResult) -> str:
    """The Mann-Kendall + Theil-Sen evidence stat for a signed, significant trend. The ONE place this
    wording lives, so the scan/pivot stat (``_stat_string``) and the change drill-down stat
    (``_change_stat``) cannot drift apart on precision or naming."""
    slope = f", Theil-Sen slope {t.slope:.4g}/day" if t.slope is not None else ""
    return f"Mann-Kendall p={t.p_value:.3f}, tau={t.tau:.2f}, n={t.n}{slope}"


#: The value-status flags (out-of-range / panic / band) — distinct from ``no_reference`` (no range to
#: judge against). A marker carrying any of these is NOT "within normal variation", whatever its trend.
_VALUE_FLAGS = ("panic_high", "panic_low", "above_range", "below_range", "band_cross")


def _is_flagged(traj: MarkerTrajectory) -> bool:
    return any(f in traj.flags for f in _VALUE_FLAGS)


def _status_clause(traj: MarkerTrajectory) -> str:
    """A plain status sentence from the value flags, in the same panic > range > band priority as
    ``_classify`` — so the change drill-down states a marker's out-of-range / panic standing instead of
    silently narrating only its trend (which can read as false reassurance on a flagged value)."""
    f = traj.flags
    if "panic_high" in f:
        return "This is above the critical-high threshold."
    if "panic_low" in f:
        return "This is below the critical-low threshold."
    if "above_range" in f:
        return "This is above the reference range."
    if "below_range" in f:
        return "This is below the reference range."
    if "band_cross" in f:
        return "This has crossed a clinical threshold."
    return ""


def _status_phrase(traj: MarkerTrajectory) -> str:
    """The terse value-status phrase (range > band priority) appended to a TREND narration, so a flagged
    value's current standing is stated alongside its trend rather than replaced by it (CLAUDE.md "every
    narrator surfaces the core flags"). Empty when the value carries no range/band flag — an in-range
    trending marker has no current breach to surface. Panic is handled upstream (a panic value classifies
    as ``panic_*`` and never reaches a trend branch), so it is omitted here."""
    f = traj.flags
    if "above_range" in f:
        return "above range"
    if "below_range" in f:
        return "below range"
    if "band_cross" in f:
        return "past a clinical threshold"
    return ""


def _member_status_phrase(traj: MarkerTrajectory) -> str:
    """The member-worded value-status phrase ("above the normal range") embedded in a member explanation
    — the member-facing mirror of ``_status_phrase`` (the clinician-tilted scan copy says "above range").
    Member copy says "normal range", never "reference range". Empty when the value carries no range/band
    flag; panic is handled in its own ``member_explanation`` branch upstream."""
    f = traj.flags
    if "above_range" in f:
        return "above the normal range"
    if "below_range" in f:
        return "below the normal range"
    if "band_cross" in f:
        return "past a clinical threshold"
    return ""


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
    if traj.severity == "attention" and t is None:
        return "sparse_trend"  # sub-n_min monotonic adverse change cleared RCV (Fix 2): the escalation
        # reason, so it leads over the range/band flag the value also carries
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
    m, signal = _display_name(traj.marker), _classify(traj)
    val = f"{traj.latest.value} {traj.unit}"
    if signal == "panic_high":
        return (
            f"{m} critically high",
            f"{m} latest {val} above the critical-high threshold",
        )
    if signal == "panic_low":
        return (
            f"{m} critically low",
            f"{m} latest {val} below the critical-low threshold",
        )
    if signal == "trend":
        t = traj.trend
        word = _DIRECTION_WORD.get(t.direction, "changing") if t else "changing"
        rcv = "; clears reference-change value" if traj.severity == "attention" else ""
        p = f"{t.p_value:.3f}" if t else "n/a"
        n = t.n if t else 0
        # Surface the value's current standing alongside the trend: a marker that has already breached
        # its range / crossed a band must not be narrated as a future-tense trend alone (the value is
        # flagged NOW). CLAUDE.md "every narrator surfaces the core flags".
        status = _status_phrase(traj)
        title = f"{m} {word}, now {status}" if status else f"{m} {word}"
        trigger = f"{m} {word} trend (Mann-Kendall p={p}, n={n}{rcv})"
        return title, (f"{trigger}; latest {status}" if status else trigger)
    if (
        signal == "sparse_trend"
    ):  # sub-n_min monotonic adverse change cleared RCV (Fix 2)
        cc = traj.clinical_change
        net = cc.net_change if cc is not None and cc.net_change is not None else 0.0
        word = "rising" if net > 0 else "falling"
        span = f"{traj.n_readings} panels" if traj.n_readings else "a short series"
        status = _status_phrase(
            traj
        )  # state the value's standing alongside the sparse trend
        title = (
            f"{m} {word}, now {status} (limited history)"
            if status
            else f"{m} {word} (limited history)"
        )
        status_trig = f"; latest {status}" if status else ""
        return (
            title,
            f"{m} {word} ~{abs(net):.0f}% across {span}, clearing its reference-change "
            f"value{status_trig}; limited history — interpret with caution",
        )
    if signal == "above_range":
        return f"{m} above range", f"{m} latest {val} above the reference range"
    if signal == "below_range":
        return f"{m} below range", f"{m} latest {val} below the reference range"
    if signal == "band_cross":
        return (
            f"{m} crossed a clinical threshold",
            f"{m} moved across a clinical band cut-point",
        )
    return f"{m} flagged", f"{m} flagged ({', '.join(traj.flags) or 'no signal'})"


def _fmt_num(x: float) -> str:
    """A clinical value as member-friendly text — drops a trailing ``.0`` ("142", not "142.0"; "8.2"
    stays "8.2"). The clinician ``trigger_reason`` keeps the raw-float style (``observation_summary``);
    this trims it for member copy only. Lab magnitudes never reach scientific-notation territory."""
    return f"{x:g}"


def _format_reference_range(
    ref_low: float | None, ref_high: float | None, unit: str
) -> str:
    """A marker's normal range as member-facing text, in whichever of the three shapes its bounds allow:
    a band ("70–99 mg/dL"), an upper bound only ("under 100 mg/dL"), or a lower bound only ("above 30
    mg/dL"). Empty when neither bound is set (a ``no_reference`` marker has no range to state). The
    numbers are read off the resolved ``ReferenceRange`` — never computed — which is exactly why the
    deterministic layer MAY print them where the LLM composer may NOT (llm.py rule 1: "The system prints
    the exact bounds as evidence")."""
    u = f" {unit}" if unit else ""
    if ref_low is not None and ref_high is not None:
        return f"{_fmt_num(ref_low)}–{_fmt_num(ref_high)}{u}"  # en-dash band
    if ref_high is not None:
        return f"under {_fmt_num(ref_high)}{u}"
    if ref_low is not None:
        return f"above {_fmt_num(ref_low)}{u}"
    return ""


def observation_member_explanation(
    traj: MarkerTrajectory, rng: ReferenceRange | None
) -> str:
    """The member-facing plain-language explanation for an observation card — "what this means for you",
    derived from the SAME ``_classify`` signal as ``observation_summary`` so the headline, the clinician
    ``trigger_reason``, and this explanation can never describe different findings. Deterministic by
    REQUIREMENT, not merely for cost: it states the numeric reference range, which the LLM composer is
    forbidden to do (llm.py rule 1 — "never state a numeric cutoff ... The system prints the exact bounds
    as evidence"); only this layer, reading the resolved ``ReferenceRange``, may print the real bounds.

    Upholds two invariants (locked by tests): it SURFACES THE CORE FLAG — a trend on an out-of-range
    value states the out-of-range standing, never the trend alone (CLAUDE.md "every narrator surfaces the
    core flags") — and it LEAKS NO STATISTICS (no test name, p-value, tau, or RCV wording in member
    copy; "your last 5 tests" is a count, not a statistic). The referral nudge is severity-gated: panic
    prompts urgent attention, an attention-level (sparse-)trend suggests raising it with a doctor, and a
    bare range/band flag (at most 'notable' — Escalation ≠ out-of-range) carries no referral, mirroring
    the deterministic floor."""
    m, signal = _display_name(traj.marker), _classify(traj)
    val = f"{_fmt_num(traj.latest.value)} {traj.unit}".strip()
    low, high = (rng.ref_low, rng.ref_high) if rng else (None, None)
    rng_txt = _format_reference_range(low, high, traj.unit)
    rng_paren = f" ({rng_txt})" if rng_txt else ""

    if signal == "panic_high":
        return f"Your {m} is {val}, which is critically high and needs prompt medical attention."
    if signal == "panic_low":
        return f"Your {m} is {val}, which is critically low and needs prompt medical attention."
    if signal in ("trend", "sparse_trend"):
        if signal == "trend":
            t = traj.trend
            word = _DIRECTION_WORD.get(t.direction, "changing") if t else "changing"
            span = f"steadily {word} across your last {t.n if t else traj.n_readings or 0} tests"
        else:  # sub-n_min: no MK verdict, direction from the net change, honest "short history"
            cc = traj.clinical_change
            net = cc.net_change if cc is not None and cc.net_change is not None else 0.0
            span = f"{'rising' if net > 0 else 'falling'} across your last few tests"
        # Referral is severity-gated, not signal-gated: a COUNTED trend (attention) and the sub-n_min
        # sparse trend (always attention) earn it; a surfaced-but-uncounted trend (notable/info, the
        # final _classify "trend" branch) does not — it is not an escalation (Escalation ≠ out-of-range),
        # so it gets no "see a doctor", same as a bare range flag.
        if signal == "sparse_trend":
            tail = (
                " Based on a short history so far, but worth raising with your doctor."
            )
        elif traj.severity == "attention":
            tail = " Worth raising with your doctor."
        else:
            tail = ""
        status = _member_status_phrase(
            traj
        )  # surface the value's standing alongside the trend
        if status:
            return f"Your {m} has been {span}, and your latest is {val}, {status}{rng_paren}.{tail}"
        return f"Your {m} has been {span}, now at {val}.{tail}"
    if signal == "above_range":
        return f"Your {m} is {val}, above the normal range{rng_paren}."
    if signal == "below_range":
        return f"Your {m} is {val}, below the normal range{rng_paren}."
    if signal == "band_cross":
        return (
            f"Your {m} has moved across a clinical threshold worth keeping an eye on."
        )
    return f"Your {m} was flagged for review."


def qa_title(traj: MarkerTrajectory) -> str:
    """A neutral, factual ``Finding.text`` for a marker a Mode-2 question cited but the core did NOT
    raise (severity 'info'). ``observation_summary`` assumes a raised signal and would mislabel a benign
    value ("X flagged"); for an in-range, no-trend marker the honest label is just its latest reading.
    The number is read off the verdict, never computed (the one law). Raised markers keep the
    signal-aware ``observation_summary`` title instead (so a flagged value is never narrated as calm)."""
    return (
        f"{_display_name(traj.marker)}: latest {traj.latest.value} {traj.unit} "
        f"(recorded {traj.latest.date})"
    )


def _stat_string(traj: MarkerTrajectory, rng: ReferenceRange | None) -> str:
    """The statistic/threshold the claim rests on (evidence chip) — switched on the SAME ``_classify``
    decision as the title, so the two never diverge."""
    signal, t = _classify(traj), traj.trend
    if signal == "panic_high" and rng is not None and rng.panic_high is not None:
        return f"above critical-high {rng.panic_high} {traj.unit}"
    if signal == "panic_low" and rng is not None and rng.panic_low is not None:
        return f"below critical-low {rng.panic_low} {traj.unit}"
    if signal == "trend" and t is not None:
        return _trend_stat(t)
    if (
        signal == "sparse_trend"
    ):  # sub-n_min: the same signal the title cites (Fix 2 grounding parity)
        cc = traj.clinical_change
        net = cc.net_change if cc is not None and cc.net_change is not None else 0.0
        return f"~{abs(net):.0f}% over {traj.n_readings or 0} panels, clears reference-change value"
    if signal in ("above_range", "below_range"):
        return "outside reference range"
    if signal == "band_cross":
        return "crossed a clinical band cut-point"
    return "latest reading"


def scan_finding(
    traj: MarkerTrajectory, rng: ReferenceRange | None, finding_id: str, title: str
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


def couldnt_route_template(metadata: ResponseMetadata) -> HealthIntelligenceResponse:
    """The gate's fail-closed responder (architecture §98): when the input classifier returns an
    off-enum/unparseable result twice, or the provider is down, the turn lands here — the floor HOLDS at
    ``clinician_review`` (set by the caller, never below) while the copy blends both possibilities so
    clarification lives in the words without lowering the floor. It never free-composes a possibly-urgent
    message and never replies with a bare rephrase-ask and no floor: the glitched message is exactly the
    one the gate exists to catch. ``answer_disposition='answered'`` — it gives a real next step."""
    return HealthIntelligenceResponse(
        answer=(
            "I couldn't quite tell what you're asking. If you're worried about symptoms you're having "
            "right now, please don't wait on me — contact your GP or urgent care. Otherwise, try "
            "rephrasing and I'll help you make sense of your own results."
        ),
        answer_disposition="answered",
        escalation="clinician_review",
        metadata=metadata,
    )


# --------------------------------------------------------------------------------------------------
# Mode 1 reactive answering (Phase 3b) — ``suggest_prompts`` builds the navigable preset loop, each chip
# bound to a PRE-COMPUTED ``HealthIntelligenceResponse`` templated from the same verdicts (no model call,
# so byte-identical and ~free). The loop (architecture §147–164): drill-downs of the finding just opened
# + a few unexplored pivots + one or two ever-present anchors. Two boundaries keep it honest:
#   * Grounding — chips only resurface the member's OWN markers (every Evidence reads off a trajectory).
#   * Mechanism — NO chip is a causal/"why" question; those would hedge to action or hand to Mode 2
#     (§153). Every prompt here asks for status/data ("tell me about", "how has it changed"), and every
#     answer reports the movement the core measured, never its cause.
# The escalation floor is STANDING: every chip is built at the member's data floor, so even an unrelated
# chip carries the floor (asserted in pipeline) — "floor always on" (CLAUDE.md), demonstrated.
# --------------------------------------------------------------------------------------------------


def _join_markers(markers: list[str]) -> str:
    """ "a", "a and b", "a, b and c" — a calm, readable marker list for the overview prose."""
    if len(markers) == 1:
        return markers[0]
    return ", ".join(markers[:-1]) + " and " + markers[-1]


def _change_narrative(traj: MarkerTrajectory) -> str:
    """How a marker has moved over the recorded panels — latest value, its out-of-range / panic STATUS,
    the net change, and the trend verdict. Status/data only: it names what the core measured (value,
    flags, %, Mann-Kendall verdict), never a cause. FLAG-AWARE: a marker carrying a value flag is never
    narrated as 'within normal variation' (which would be false reassurance on an out-of-range or panic
    value — the same calm the deterministic core did not produce). Reads only off the trajectory."""
    latest = traj.latest
    parts = [
        f"Your most recent {_display_name(traj.marker)} is {latest.value} {traj.unit} (recorded {latest.date})."
    ]
    status = _status_clause(traj)
    if status:
        parts.append(status)
    net = traj.clinical_change.net_change if traj.clinical_change is not None else None
    if net is not None and abs(net) >= 1.0:
        parts.append(
            f"That's {'up' if net > 0 else 'down'} about {abs(net):.0f}% from your first recorded reading."
        )
    t = traj.trend
    if traj.severity == "attention" and t is None:
        # Sub-n_min escalation (Fix 2): a monotonic adverse change cleared RCV but n is below n_min, so
        # there is no Mann-Kendall verdict. State the consistent move AND the limited base — never
        # overstate certainty from a short series (E12 must_not).
        span = f"only {traj.n_readings} panels" if traj.n_readings else "a short series"
        parts.append(
            f"Across {span} this is a consistent move beyond normal variation; the limited history "
            "lowers certainty, but it warrants a clinician's review."
        )
    elif t is not None and t.direction != "flat" and t.significant:
        word = _DIRECTION_WORD.get(t.direction, "changing")
        parts.append(
            f"Across {t.n} readings this is a {word} trend (Mann-Kendall p={t.p_value:.3f})."
        )
    elif t is not None and not _is_flagged(traj):
        parts.append(
            f"Across {t.n} readings there's no clear trend — the movement is within normal variation."
        )
    elif t is not None:
        parts.append(
            f"Across {t.n} readings there's no significant trend."
        )  # flagged: don't claim 'normal variation'
    return " ".join(parts)


def _change_stat(traj: MarkerTrajectory, rng: ReferenceRange | None) -> str:
    """The statistic the change drill-down's latest-reading evidence chip rests on. FLAG-AWARE: when the
    value is out-of-range / panic / band, defer to ``_stat_string`` so the chip states the threshold —
    the SAME stat that marker's pivot/scan chip shows (no cross-chip divergence). Otherwise the change
    statistic: the Mann-Kendall verdict for a signed significant trend, else a material net % change
    (suppressed below 1%, matching the narrative — so the chip never reads a misleading '+0%')."""
    if _is_flagged(traj):
        return _stat_string(traj, rng)
    t = traj.trend
    if t is not None and t.direction != "flat" and t.significant:
        return _trend_stat(t)
    cc = traj.clinical_change
    if cc is not None and cc.net_change is not None and abs(cc.net_change) >= 1.0:
        return f"net change {cc.net_change:+.0f}% since baseline"
    return "latest reading"


def _marker_finding(traj: MarkerTrajectory, rng: ReferenceRange | None) -> Finding:
    """One raised marker's ``Finding`` — the per-marker unit shared by the pivot answer and the overview
    aggregate, so a marker reads identically whether shown singly or inside the 'what's changed' list
    (one ``observation_summary`` title + one ``scan_finding`` evidence, one ``f:{marker}`` id scheme)."""
    return scan_finding(traj, rng, f"f:{traj.marker}", observation_summary(traj)[0])


def render_change(
    traj: MarkerTrajectory,
    rng: ReferenceRange | None,
    series: list[Reading],
    *,
    escalation: FloorLevel,
    metadata: ResponseMetadata,
) -> HealthIntelligenceResponse:
    """A focus drill-down: how this marker has moved over the recorded panels. One ``Finding`` whose
    evidence is the member's ACTUAL readings — one verbatim ``Evidence`` each (the latest carrying the
    change statistic) — under a plain summary of the net change and trend verdict. Status/data only
    (never 'why' — the mechanism boundary, §153): it reports the movement the core measured, not its
    cause. ``series`` (the dated readings) is supplied by the caller and is non-empty (the sole caller
    only drills into a marker that exists in the record). Built AT the floor, like every Mode-1 response."""
    last = len(series) - 1
    finding = Finding(
        finding_id=f"chg:{traj.marker}",
        text=_change_narrative(traj),
        evidence=[
            Evidence(
                marker=traj.marker,
                value=rd.value,
                unit=traj.unit,
                date=rd.date,
                ref_low=rng.ref_low if rng else None,
                ref_high=rng.ref_high if rng else None,
                stat=_change_stat(traj, rng)
                if i == last
                else None,  # the change stat sits on the latest reading
            )
            for i, rd in enumerate(series)
        ],
    )
    return HealthIntelligenceResponse(
        answer=finding.text,
        findings=[finding],
        uncertainty=f"Based on {len(series)} recorded reading{'s' if len(series) != 1 else ''} for this marker.",
        answer_disposition="answered",
        escalation=escalation,
        metadata=metadata,
    )


def render_pivot(
    traj: MarkerTrajectory,
    rng: ReferenceRange | None,
    *,
    escalation: FloorLevel,
    metadata: ResponseMetadata,
) -> HealthIntelligenceResponse:
    """One raised finding as its own answer — the SAME per-marker ``Finding`` the proactive scan builds
    (``observation_summary`` title + ``scan_finding`` evidence), rendered by the 3a ``render_finding``.
    This is the "reactive path reuses the 3a response builder" of the phase, so a chip's answer for a
    marker is identical in substance to that marker's scan observation."""
    return render_finding(
        _marker_finding(traj, rng), escalation=escalation, metadata=metadata
    )


def render_overview(
    raised: list[MarkerTrajectory],
    rng_for: dict[str, ReferenceRange | None],
    *,
    escalation: FloorLevel,
    metadata: ResponseMetadata,
) -> HealthIntelligenceResponse:
    """The "what's changed" anchor — the aggregate the single-finding 3a builder can't produce. When
    findings are raised, one ``Finding`` per marker (reusing ``scan_finding``) wrapped in one response
    with summary prose; when nothing is raised, a calm, non-manufactured all-clear (the negative-control
    branch, ui-ux §1/§6, eval E02 — "honest before reassuring" still holds: it claims only in-range +
    no concerning trend, which is exactly what the core found). Built AT the floor."""
    if not raised:
        # Don't claim "within normal ranges" for a marker that HAS no reference range (no_reference, e.g.
        # an other/unknown-sex member with a sex-split marker) — say only what's true: nothing flagged.
        clause = (
            "your results sit within their normal ranges"
            if all(r is not None for r in rng_for.values())
            else "nothing in your readings is flagged"
        )
        return HealthIntelligenceResponse(
            answer=(
                f"Nothing in your recent panels stands out as needing attention — {clause}, with no "
                "concerning trend. I'll keep watching and flag anything that drifts."
            ),
            findings=[],
            uncertainty="Based on the panels currently on file.",
            answer_disposition="answered",
            escalation=escalation,
            metadata=metadata,
        )
    findings = [_marker_finding(t, rng_for.get(t.marker)) for t in raised]
    return HealthIntelligenceResponse(
        answer=(
            f"A few things are worth a closer look since your earlier panels: "
            f"{_join_markers([_display_name(t.marker) for t in raised])}. Here's what each shows — your "
            "clinician can interpret them in context."
        ),
        findings=findings,
        uncertainty="Based on your recorded panels for these markers.",
        answer_disposition="answered",
        escalation=escalation,
        metadata=metadata,
    )


def render_summary(
    analysis: TrajectoryAnalysis,
    raised: list[MarkerTrajectory],
    *,
    escalation: FloorLevel,
    metadata: ResponseMetadata,
) -> HealthIntelligenceResponse:
    """The "overview" anchor — a terse orientation roll-up (how many markers were looked at, how many
    are flagged), the calm landing for a vulnerable moment. No findings of its own; the detail lives in
    the overview and the pivots."""
    n, m = len(analysis.markers), len(raised)
    # As in render_overview: only claim "within their normal ranges" when every marker actually HAS a
    # reference range; otherwise fall back to the honest "not flagged" (no over-claim on a no_reference).
    all_ranged = all("no_reference" not in t.flags for t in analysis.markers)
    rest = "within their normal ranges" if all_ranged else "not flagged"
    if m == 0:
        answer = f"I looked at {n} markers from your panels, and they're all {rest} — nothing flagged right now."
    else:
        answer = (
            f"I looked at {n} markers from your panels. {m} {'is' if m == 1 else 'are'} worth a closer "
            f"look (shown on the right and in detail above); the rest are {rest}."
        )
    return HealthIntelligenceResponse(
        answer=answer,
        findings=[],
        uncertainty=None,
        answer_disposition="answered",
        escalation=escalation,
        metadata=metadata,
    )


#: The per-turn chip ceiling (architecture §145 / ui-ux §2: "2–5 data-derived preset prompts"). Mode 1
#: is the calm landing for a vulnerable moment, not a wall of buttons — so a busy member's many findings
#: are shown a few at a time, severity-ranked, with the rest reachable through the overview anchor and
#: paged up over turns as the member visits findings (§147's "walk the map", not "dump the map").
MAX_CHIPS = 5


def suggest_prompts(
    analysis: TrajectoryAnalysis,
    raised: list[MarkerTrajectory],
    focus: str | None,
    asked: frozenset[str],
    *,
    rng_for: dict[str, ReferenceRange | None],
    focus_series: list[Reading],
    escalation: FloorLevel,
    new_metadata: Callable[[str], ResponseMetadata],
) -> list[SuggestedPrompt]:
    """The navigable Mode-1 chip set for one turn, each prompt bound to its pre-computed answer.

    ``raised`` is the caller's raised-finding set as TRAJECTORIES, severity-sorted (pipeline owns the
    "raise an observation?" predicate, architecture §397) — it replaces the architecture sketch's
    ``observations`` argument because an ``Observation`` carries no marker field and none of the verdict
    data the evidence needs, so the trajectory is the correct, self-contained source. ``rng_for``
    (per-marker reference range), ``focus_series`` (the focused marker's dated readings), ``escalation``
    (the data floor), and ``new_metadata`` (the deterministic ``response_id`` factory) are keyword-only
    render context the pure narrator does not derive — the same shape as ``analyze(..., *, data_version)``.
    (The linked-GP-note drill-down of §149 would add a ``notes`` argument; it is deferred to Phase 7, so
    the parameter is not carried until the behavior that reads it lands — CLAUDE.md "phantom fields cut".)

    ``focus`` / ``asked`` are CANONICAL marker keys (the chip prompts show ``_display_name`` labels, but
    the loop state keys on the storage name). The chip order is the loop (§149): the opened finding's
    drill-down, then unexplored pivots, then the ever-present anchors — which are EXEMPT from the
    ``asked`` filter, so the loop never dead-ends (§153) and thins to anchors-only once every finding has
    been visited (the cue to offer Mode 2). The whole turn is bounded to ``MAX_CHIPS``: anchors are
    always shown, and the pivot slots that remain go to the most severe unexplored findings — applied
    AFTER the ``asked`` filter, so visited findings page out and the next-most-severe page in, and
    nothing becomes unreachable (the overview always lists all)."""
    by_marker = {t.marker: t for t in analysis.markers}

    # 1. Drill-down of the finding just opened (any marker in the member's record, raised or not).
    drilldowns: list[SuggestedPrompt] = []
    if focus is not None and focus in by_marker:
        drilldowns.append(
            SuggestedPrompt(
                prompt=f"How has my {_display_name(focus)} changed over time?",
                response=render_change(
                    by_marker[focus],
                    rng_for.get(focus),
                    focus_series,
                    escalation=escalation,
                    metadata=new_metadata(f"change:{focus}"),
                ),
            )
        )

    # 3. Ever-present anchors (built before the pivots so the pivot budget is what remains after them).
    anchors = [
        SuggestedPrompt(
            prompt="What's changed since my last results?",
            response=render_overview(
                raised,
                rng_for,
                escalation=escalation,
                metadata=new_metadata("overview"),
            ),
        ),
        SuggestedPrompt(
            prompt="Give me a quick overview of my results.",
            response=render_summary(
                analysis,
                raised,
                escalation=escalation,
                metadata=new_metadata("summary"),
            ),
        ),
    ]

    # 2. Unexplored pivots — severity-ranked (``raised`` is pre-sorted), filtered by ``asked`` and the
    #    current focus, then capped to whatever the turn budget leaves after the drill-down and anchors.
    pivot_cap = max(0, MAX_CHIPS - len(anchors) - len(drilldowns))
    eligible = [t for t in raised if t.marker != focus and t.marker not in asked]
    pivots = [
        SuggestedPrompt(
            prompt=f"Tell me about my {_display_name(t.marker)}.",
            response=render_pivot(
                t,
                rng_for.get(t.marker),
                escalation=escalation,
                metadata=new_metadata(f"marker:{t.marker}"),
            ),
        )
        for t in eligible[:pivot_cap]
    ]

    return drilldowns + pivots + anchors
