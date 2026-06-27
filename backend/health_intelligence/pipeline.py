"""pipeline.py — orchestration over the pure core (Phase 3a: the proactive scan, no LLM).

The scan is the deterministic half of the proactive behavior: load the member, run ``analyze``, decide
which markers are worth surfacing, narrate them into one ``HealthIntelligenceResponse``, and persist the
trace + observations + the clinician-review escalations. Every safety-relevant decision here is read off
the pure core (severity, the escalation floor) and only *acted on* — nothing is recomputed (CLAUDE.md:
the LLM, and equally this orchestration, is never the core).

Phase 3b adds the reactive half of Mode 1: ``suggestions`` — the deterministic preset loop. It is a
PURE READ (no DB writes): load the member, analyze, project the floor, and hand the raised set + render
context to ``templates.suggest_prompts``, which returns each chip bound to a pre-computed, byte-identical
``HealthIntelligenceResponse`` (no model call). The ask path (gate -> compose -> validate) is added in
Phase 4 alongside the LLM layer.

Determinism & idempotency: all persisted PKs key on an ``analysis_version`` (the hash of exactly the
inputs ``analyze`` reads — db.compute_analysis_version), so a re-scan of unchanged analysis inputs
UPSERTs the same rows and re-emits the same dedup_key (a no-op) rather than duplicating, and an edit
that doesn't touch the analysis (e.g. a note) cannot mint a second escalation. "Replace, not append"
across genuine data changes is delivered by the version-scoped observation read, not by deletes.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime

from health_intelligence import db, gate, llm, safety, templates

# _range_for / _series: reuse the core's single source of truth for sex/fallback range resolution and
# for the per-marker dated series (both pure fns) — no second copy of either to drift.
from health_intelligence.analysis import _range_for, _series, analyze
from health_intelligence.config import (
    ANALYSIS_CONFIG,
    COMPOSE_MODEL,
    CONFIG_VERSION,
    GATE_MODEL,
    MODEL_VERSION_DETERMINISTIC,
)
from health_intelligence.llm import LLMParseError, LLMUnavailable, LLMUsage
from health_intelligence.models import (
    SEVERITY_ORDER,
    ComposeDraft,
    FloorLevel,
    HealthIntelligenceResponse,
    MarkerTrajectory,
    MemberProfile,
    Observation,
    ReferenceRange,
    ResponseMetadata,
    SuggestedPrompt,
    TrajectoryAnalysis,
)
from health_intelligence.safety import severity_to_level


def _is_raised(traj: MarkerTrajectory) -> bool:
    """The "raise an observation?" predicate (architecture §397 leaves it to the caller — ``analyze``
    returns severity only). Surface anything the core judged above baseline: an out-of-range or
    band-crossing value, a trend, or a panic — i.e. severity beyond ``info``. A healthy in-range marker
    with no trend stays ``info`` and is not surfaced, so a calm member (negative control) raises nothing
    and escalates nothing — no false alarms."""
    return SEVERITY_ORDER[traj.severity] > SEVERITY_ORDER["info"]


def _raised_ranked(analysis: TrajectoryAnalysis) -> list[MarkerTrajectory]:
    """The raised markers, highest-severity first then marker name — the SINGLE ordering both halves of
    Mode 1 consume (the proactive scan and the reactive suggestions loop), so "which findings surface,
    and in what order" cannot drift between them. The marker-name tie-break keeps it deterministic
    (byte-identical output)."""
    return sorted(
        (t for t in analysis.markers if _is_raised(t)),
        key=lambda t: (-SEVERITY_ORDER[t.severity], t.marker),
    )


def _ranges_by_marker(
    analysis: TrajectoryAnalysis,
    sex: str,
    age: int | None,
    ranges: list[ReferenceRange],
) -> dict[str, ReferenceRange | None]:
    """Per-marker reference range (sex/age-resolved via the core's ``_range_for``) for every marker — the
    single source both the Mode-1 suggestions overview and the Mode-2 compose-down fallback render from,
    so the two can't silently diverge on how ranges are resolved."""
    return {t.marker: _range_for(t.marker, sex, age, ranges) for t in analysis.markers}


def _det_metadata(response_id: str, data_version: str) -> ResponseMetadata:
    """The reproducibility tuple stamped on a deterministic (no-LLM) response — model 'deterministic',
    prompt_version 0, no clock. Shared by the scan and the suggestions loop so the (data, config,
    template) stamp can't drift between the two write disciplines."""
    return ResponseMetadata(
        response_id=response_id,
        data_version=data_version,
        model_version=MODEL_VERSION_DETERMINISTIC,  # no LLM ran; tuple is (data, config, template)
        config_version=CONFIG_VERSION,
        prompt_version=0,
    )


def scan(con: sqlite3.Connection, member_id: str) -> list[Observation]:
    """Run the proactive scan for one member and persist its artifacts; return the current observations
    (ranked by severity). Raises ``KeyError`` if the member is absent.

    Each raised marker becomes one finding persisted as its own ``interactions`` row (architecture §48),
    keyed by a deterministic data_version-scoped ``response_id`` so an identical re-scan is a no-op and a
    genuine data change retains the prior audit row. The data_finding escalation dedup keys on
    ``analysis_version`` instead (the §48 finding-stable resolution): a notes-/profile-only edit moves
    data_version but not analysis_version, so it writes fresh audit rows without re-queuing a clinician
    task. All writes for the scan run in ONE transaction (atomic — no partial interaction/observation/
    escalation set on a crash)."""
    member, results, ranges, age, data_version = db.load_for_analysis(con, member_id)
    analysis = analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version
    )
    analysis_version = db.compute_analysis_version(
        results=results, ranges=ranges, sex=member.sex, age=age
    )
    floor = safety.data_floor(analysis)

    # Which markers to surface, ranked highest-severity first then by marker name (shared with suggestions).
    raised = _raised_ranked(analysis)

    with con:  # atomic: interaction + observation + escalation per finding commit (or roll back) together
        for traj in raised:
            response_id = db._det_id("scan:", member_id, traj.marker, data_version)
            obs_id = db._det_id(
                "obs:", member_id, traj.marker, data_version
            )  # finding_id == observation_id
            rng = _range_for(traj.marker, member.sex, age, ranges)
            title, trigger_reason = templates.observation_summary(traj)  # computed once
            finding = templates.scan_finding(traj, rng, obs_id, title)

            metadata = _det_metadata(response_id, data_version)
            # escalation = the member's deterministic floor, carried on every response and validated
            # >= floor (CLAUDE.md "floor always on"; §2 D4 routes the template path through the validator
            # before persist). Trivially passes here — the response is built AT the floor — which is why
            # asserting it is cheap insurance against any future drift in how the response sets escalation.
            resp = safety.validate(
                templates.render_finding(finding, escalation=floor, metadata=metadata),
                floor,
            )

            db.write_interaction(
                con, resp, member_id=member_id, driver="scan", question=None
            )
            db.write_observation(
                con,
                Observation(
                    observation_id=obs_id,
                    member_id=member_id,
                    response_id=response_id,
                    severity=traj.severity,
                    title=title,
                    trigger_reason=trigger_reason,
                    data_version=data_version,
                ),
            )
            level = severity_to_level(traj.severity)
            if (
                level is not None
            ):  # only attention/urgent reach the clinician-review queue
                db._insert_escalation(
                    con,
                    member_id=member_id,
                    kind="data_finding",
                    dedup_key=f"data:{member_id}:{traj.marker}:{analysis_version}",
                    level=level,
                    observation_id=obs_id,
                    trigger_reason=trigger_reason,
                )

    return db.get_observations(con, member_id, data_version=data_version)


def suggestions(
    con: sqlite3.Connection,
    member_id: str,
    *,
    focus: str | None = None,
    asked: tuple[str, ...] = (),
) -> list[SuggestedPrompt]:
    """The reactive Mode-1 preset loop for one member — a PURE READ (no DB writes). Raises ``KeyError``
    if the member is absent (the route maps it to 404, like ``scan``).

    Self-contained: the raised set is derived straight from ``analyze()`` (``_raised_ranked``, the same
    predicate + severity ranking the scan uses), so ``/suggestions`` has no ordering dependency on a
    prior ``/scan``. Each chip's ``response_id`` is deterministic (``_det_id`` over member + chip key +
    ``data_version``) and the responses carry no clock, so the whole payload is byte-identical across
    re-runs at the same ``data_version`` (architecture §7). ``driver='suggested'`` interaction logging
    (§151) is intentionally NOT done here — it serves feedback-attach (Phase 7) and an alternative to
    client-side loop state, neither due this phase; the deterministic ids mean a later write still
    lines up."""
    member, results, ranges, age, data_version = db.load_for_analysis(con, member_id)
    analysis = analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version
    )
    floor = safety.data_floor(analysis)

    raised = _raised_ranked(analysis)
    rng_for = _ranges_by_marker(analysis, member.sex, age, ranges)
    # The focus drill-down shows the member's actual readings, so build that one marker's dated series
    # (the same pure ``_series`` the core uses); empty when no/unknown focus, where it goes unused.
    focus_series = _series(results, focus) if focus is not None else []

    def new_metadata(chip_key: str) -> ResponseMetadata:
        """A deterministic response_id per chip — keyed like the scan's (member · key · data_version),
        so re-fetching the same turn yields a byte-identical payload and a later persist would UPSERT."""
        return _det_metadata(
            db._det_id("suggest:", member_id, chip_key, data_version), data_version
        )

    prompts = templates.suggest_prompts(
        analysis,
        raised,
        focus,
        frozenset(asked),
        rng_for=rng_for,
        focus_series=focus_series,
        escalation=floor,
        new_metadata=new_metadata,
    )
    # Floor always on: every chip is templated AT the floor, so this validate is a no-op — but running
    # it is cheap insurance that no chip (even an unrelated one) ever sits below the member's floor, the
    # same guard the scan applies to its findings (CLAUDE.md "the deterministic floor is always on").
    for sp in prompts:
        safety.validate(sp.response, floor)
    return prompts


# --------------------------------------------------------------------------------------------------
# Phase 4 — the ask path (Mode 2): the one pipeline that swaps a single step. retrieve -> analyze ->
# floor -> render -> validate -> escalate is shared with the scan; only `render` differs (LLM compose
# vs template) and the GATE fires only on free-form input. Every safety decision is read off the
# deterministic core and the gate, never invented here: the LLM renders ground truth and routes open
# language, while the floor, the validator, and the escalation logic sit AROUND it and can override it.
# --------------------------------------------------------------------------------------------------


def _llm_metadata(
    response_id: str,
    data_version: str,
    *,
    model_version: str,
    usages: list[LLMUsage],
    latency_ms: int,
) -> ResponseMetadata:
    """The reproducibility tuple + cost/latency instrumentation for an /ask turn. ``model_version`` names
    the model that authored the PROSE (compose model when composed, gate model when a gate-routed
    template, ``deterministic`` when an LLM-down fallback templated it); ``tokens``/``cost_usd`` aggregate
    every LLM call the turn actually made (so a gate-routed template still shows the gate's cost).
    ``prompt_version`` is 0 — the ``prompt_versions`` table is Phase 7; the composer runs a pinned prompt."""
    return ResponseMetadata(
        response_id=response_id,
        data_version=data_version,
        model_version=model_version,
        config_version=CONFIG_VERSION,
        prompt_version=0,
        latency_ms=latency_ms,
        tokens=sum(u.total_tokens for u in usages) if usages else None,
        cost_usd=round(sum(u.cost_usd for u in usages), 8) if usages else None,
    )


def _compose_response(
    draft: ComposeDraft,
    *,
    member: MemberProfile,
    analysis: TrajectoryAnalysis,
    ranges: list[ReferenceRange],
    age: int | None,
    floor: FloorLevel,
    metadata: ResponseMetadata,
) -> HealthIntelligenceResponse:
    """Approach A — lift the model's draft into the full response by attaching CODE-BUILT evidence. For
    each cited marker key, build one ``Finding`` + ``Evidence`` from its trajectory (numbers come from
    the analysis, never the model — the one law); a key not in the member's data is DROPPED, so a
    hallucinated marker can never produce an evidence chip. A raised marker keeps its signal-aware
    ``observation_summary`` title (so a flagged value is never narrated as calm); a benign one gets the
    neutral ``qa_title``. ``escalation`` is set to the deterministic floor — the model never touched it."""
    by_marker = {t.marker: t for t in analysis.markers}
    findings = []
    seen: set[str] = set()
    for key in draft.cited_markers:
        traj = by_marker.get(key)
        if traj is None or key in seen:
            continue  # unknown marker -> never fabricate; repeated key -> one chip, one stable finding_id
        seen.add(key)
        rng = _range_for(key, member.sex, age, ranges)
        title = (
            templates.observation_summary(traj)[0]
            if _is_raised(traj)
            else templates.qa_title(traj)
        )
        findings.append(templates.scan_finding(traj, rng, f"f:{key}", title))
    return HealthIntelligenceResponse(
        answer=draft.answer,
        findings=findings,
        uncertainty=draft.uncertainty,
        answer_disposition=draft.answer_disposition,
        escalation=floor,
        metadata=metadata,
    )


def _grounded_fallback(
    analysis: TrajectoryAnalysis,
    member: MemberProfile,
    ranges: list[ReferenceRange],
    age: int | None,
    *,
    floor: FloorLevel,
    metadata: ResponseMetadata,
) -> HealthIntelligenceResponse:
    """The auto-degradation answer when the gate cleared the message (``none``) but the composer is down
    (``LLMUnavailable``): a fully-grounded, floor-respecting answer with NO model call (architecture §6) —
    the deterministic overview the Mode-1 ``what's changed`` anchor already produces, stamped
    ``model_version='deterministic'``. Mode 2 degrades to Mode 1 rather than to an error or an unguarded
    reply."""
    raised = _raised_ranked(analysis)
    rng_for = _ranges_by_marker(analysis, member.sex, age, ranges)
    return templates.render_overview(
        raised, rng_for, escalation=floor, metadata=metadata
    )


def _chat_trigger_reason(
    route: str, floor: FloorLevel, emergency_floor: FloorLevel
) -> str:
    """A human-readable ``trigger_reason`` for the chat escalation — the queue reader disambiguates acute
    vs crisis vs phrase-floored vs data-driven from this free text (architecture §323, no structured
    sub-type in v1). The emergency-phrase check sits ABOVE the route label so a couldnt_route turn that a
    self-harm/acute phrase floored to ``urgent`` reports the urgency, never the lower 'held at clinician
    review' — i.e. the reason can't contradict the structured ``level`` it accompanies."""
    if route == "crisis":
        return "crisis/self-harm language detected in chat"
    if route == "acute_medical":
        return "acute medical concern described in chat"
    if emergency_floor == "urgent":
        return "emergency-phrase (self-harm/acute) language raised this chat turn"
    if route == "couldnt_route":
        return "message could not be classified; held at clinician review"
    return f"member's data floor is {floor} at chat time"


#: The non-``none`` routes -> their fixed responder (architecture §2 D4: you do not free-compose an
#: emergency, a crisis reply over someone's labs, or a fail-closed clarification). The ONE place the
#: gate's route set maps to templates, so adding/renaming a route is a single edit (the dispatch and the
#: model-version rule both read from here).
_SAFETY_TEMPLATES = {
    "out_of_scope": templates.refuse_template,
    "acute_medical": templates.seek_care_template,
    "crisis": templates.crisis_template,
    "couldnt_route": templates.couldnt_route_template,
}


def ask(
    con: sqlite3.Connection,
    member_id: str,
    message: str,
    *,
    provider: llm.Provider | None = None,
) -> HealthIntelligenceResponse:
    """One grounded Mode-2 turn. ``gate`` classifies the raw message and sets a message floor; the turn
    runs against ``floor = max(data, message)``; the gate's intent routes to open compose (``none``) or a
    fixed safety template; the response is validated at the floor, persisted (interactions APPENDS for
    ``driver='ask'``), and a chat escalation fires once-per-member-per-day when the floor warrants it.
    Fails SAFE: gate down -> ``couldnt_route`` template at ``clinician_review``; compose down -> a
    deterministic grounded answer at the data floor. Raises ``KeyError`` if the member is absent.

    ``provider`` is the LLM seam — defaults to the Anthropic provider; tests inject a fake so the whole
    path runs offline. A clock is used (created_at, the day-scoped dedup key, latency) — fine here; only
    ``analysis.py`` is clock-free."""
    start = time.perf_counter()
    now = datetime.now(UTC)
    now_iso = now.isoformat()
    today = now.date().isoformat()

    member, results, ranges, age, data_version = db.load_for_analysis(con, member_id)
    analysis = analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version
    )
    data_fl = safety.data_floor(analysis)

    g = gate.classify(message, provider=provider)
    floor = safety.max_floor(data_fl, g.msg_floor)

    usages: list[LLMUsage] = [g.usage] if g.usage is not None else []
    response_id = db._det_id("ask:", member_id, message, now_iso)

    # Render: compose only on the `none` route (the sole open generation); the safety branches and the
    # fail-closed couldnt_route are fixed templates (_SAFETY_TEMPLATES). compose() owns the one bounded
    # retry (via llm.call_structured); a parse failure that survives it, or a provider-down, degrades.
    draft: ComposeDraft | None = None
    if g.route == "none":
        prose_model = (
            MODEL_VERSION_DETERMINISTIC  # set to COMPOSE_MODEL only if compose succeeds
        )
        try:
            ctx = llm.ComposeContext(
                profile=member,
                analysis=analysis,
                notes=db.get_notes(con, member_id),
                observations=db.get_observations(
                    con, member_id, data_version=data_version
                ),
                floor=floor,
                message=message,
            )
            draft, compose_usage = llm.compose(ctx, provider=provider)
            usages.append(compose_usage)
            prose_model = COMPOSE_MODEL
        except LLMParseError as e:
            # malformed/refused/truncated output that survived the retry -> grounded fallback below.
            # Count the tokens those attempts still billed so the cost stamp isn't an undercount.
            if e.usage is not None:
                usages.append(e.usage)
            draft = None
        except LLMUnavailable:
            draft = None  # provider down -> grounded fallback. Mode 2 NEVER 500s; it fails safe (§6).
    elif g.route != "couldnt_route":
        # out_of_scope / acute_medical / crisis -> a fixed template the GATE model routed us to.
        prose_model = GATE_MODEL
    else:
        # couldnt_route -> the gate produced NO usable output (down, or off-schema twice); a deterministic
        # template authors the prose, so model_version is `deterministic` (the gate's billed tokens, if
        # any, still show in `usages`). Not GATE_MODEL: no gate call successfully decided this turn.
        prose_model = MODEL_VERSION_DETERMINISTIC

    latency_ms = int((time.perf_counter() - start) * 1000)
    metadata = _llm_metadata(
        response_id,
        data_version,
        model_version=prose_model,
        usages=usages,
        latency_ms=latency_ms,
    )

    if g.route == "none":
        resp = (
            _compose_response(
                draft,
                member=member,
                analysis=analysis,
                ranges=ranges,
                age=age,
                floor=floor,
                metadata=metadata,
            )
            if draft is not None
            else _grounded_fallback(  # compose unavailable -> grounded deterministic answer
                analysis, member, ranges, age, floor=floor, metadata=metadata
            )
        )
    else:
        resp = _SAFETY_TEMPLATES[g.route](metadata)

    # Deterministic floor wins: code sets escalation = floor (the LLM cannot touch it under approach A),
    # then the validator confirms escalation >= floor. The validate -> retry -> template scaffold stays
    # load-bearing — it is the same guard the Phase-5 harness re-runs offline, and the fallback above
    # genuinely fires on LLMUnavailable — even though, escalation being code-set, this assertion is
    # trivially true here (the LLM literally cannot under-escalate).
    resp.escalation = floor
    resp = safety.validate(resp, floor)

    # Atomic: the audit interaction + the clinician-queue escalation commit (or roll back) together.
    with con:
        db.write_interaction(
            con,
            resp,
            member_id=member_id,
            driver="ask",
            question=message,
            created_at=now_iso,
        )
        if safety.meets_floor(floor, "clinician_review"):
            db._insert_escalation(
                con,
                member_id=member_id,
                kind="chat",
                dedup_key=f"chat:{member_id}:{today}",  # one clinician task per member per day (§321)
                level=floor,  # guarded >= clinician_review, so a valid EscalationLevel  # type: ignore[arg-type]
                interaction_id=resp.metadata.response_id,
                trigger_reason=_chat_trigger_reason(g.route, floor, g.emergency_floor),
                created_at=now_iso,
            )
    return resp
