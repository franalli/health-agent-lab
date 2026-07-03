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

Determinism & idempotency: the scan's observation/audit PKs key on ``data_version`` and the data_finding
escalation dedup keys on a PER-MARKER hash (``db.compute_marker_version`` — exactly that marker's
override-resolved inputs), so a re-scan of unchanged inputs UPSERTs the same rows and re-emits the same
dedup_key (a no-op) rather than duplicating; an edit that doesn't touch a marker's analysis (a note, or
an override to a DIFFERENT marker) cannot mint a second escalation for it. "Replace, not append"
across genuine data changes is delivered by the version-scoped observation read, not by deletes — with
one targeted exception (Phase 7): a ``/feedback`` override changes the analysis WITHOUT bumping
``data_version``, so the scan also prunes a now-cleared marker's observation at the unchanged
``data_version`` (``db.prune_observations`` — escalation-pinned rows kept, never a blanket delete).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import UTC, date, datetime
from statistics import median
from typing import NamedTuple

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
    ConversationTurn,
    FloorLevel,
    HealthIntelligenceResponse,
    MarkerTrajectory,
    MemberProfile,
    Observation,
    ReferenceRange,
    ResponseMetadata,
    ScanResult,
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


def scan(con: sqlite3.Connection, member_id: str) -> ScanResult:
    """Run the proactive scan for one member and persist its artifacts; return a :class:`ScanResult` —
    the current observations (ranked by severity) plus ``new_observations``, how many of them this run
    NEWLY persisted (``db.write_observation``'s created signal; see ``ScanResult`` for the exact
    newness semantics — 0 on an idempotent re-scan). Raises ``KeyError`` if the member is absent.

    Each raised marker becomes one finding persisted as its own ``interactions`` row (architecture §48),
    keyed by a deterministic data_version-scoped ``response_id`` so an identical re-scan is a no-op and a
    genuine data change retains the prior audit row. The data_finding escalation dedup keys on the
    PER-MARKER ``compute_marker_version`` (the §48 finding-stable resolution): a notes-/profile-only edit,
    or an override to a DIFFERENT marker, leaves this marker's version unmoved, so it writes fresh audit
    rows without re-queuing a clinician task. All writes for the scan run in ONE transaction (atomic — no
    partial interaction/observation/escalation set on a crash)."""
    member, results, ranges, age, data_version = db.load_for_analysis(con, member_id)
    analysis = analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version
    )
    floor = safety.data_floor(analysis)

    # Which markers to surface, ranked highest-severity first then by marker name (shared with suggestions).
    raised = _raised_ranked(analysis)

    kept_ids: set[str] = (
        set()
    )  # the observation rows this scan keeps (the rest are pruned below)
    live_esc_ids: set[str] = (
        set()
    )  # the escalation rows this scan (re-)emitted — the live findings the reconcile keeps 'open'
    new_count = 0  # rows write_observation newly created (vs refreshed in place) — the ScanResult count
    with con:  # atomic: interaction + observation + escalation per finding commit (or roll back) together
        for traj in raised:
            response_id = db._det_id("scan:", member_id, traj.marker, data_version)
            obs_id = db._det_id(
                "obs:", member_id, traj.marker, data_version
            )  # finding_id == observation_id
            kept_ids.add(obs_id)
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
            new_count += db.write_observation(
                con,
                Observation(
                    observation_id=obs_id,
                    member_id=member_id,
                    response_id=response_id,
                    severity=traj.severity,
                    title=title,
                    trigger_reason=trigger_reason,
                    data_version=data_version,
                    # member_explanation is NOT stored; it is re-derived at read by `observations()`.
                ),
            )
            level = severity_to_level(traj.severity)
            if (
                level is not None
            ):  # only attention/urgent reach the clinician-review queue
                # dedup on THIS marker's own analysis state (compute_marker_version), not the whole-member
                # analysis_version: an override/edit to another marker must not shift this key and re-fire a
                # finding that never changed (§48, finding-stable per marker).
                marker_version = db.compute_marker_version(
                    marker=traj.marker,
                    results=results,
                    marker_range=rng,
                    sex=member.sex,
                    age=age,
                )
                dedup_key = f"data:{member_id}:{traj.marker}:{marker_version}"
                # Record the escalation identity this scan emits, so the reconcile below keeps the CURRENT
                # findings 'open' and supersedes a stale-marker_version clinician_review twin (urgent is
                # never superseded regardless — see reconcile_escalation_status).
                live_esc_ids.add(db._det_id("esc:", dedup_key))
                db._insert_escalation(
                    con,
                    member_id=member_id,
                    kind="data_finding",
                    dedup_key=dedup_key,
                    level=level,
                    observation_id=obs_id,
                    trigger_reason=trigger_reason,
                )

        # Reconcile the set: drop observations at THIS data_version that are no longer raised. Normally
        # a no-op (at a fixed data_version the raised set is constant), but a Phase-7 OVERRIDE changes the
        # analysis WITHOUT bumping data_version (it keys on the raw record), so a re-scan after a /feedback
        # override that cleared a flag would otherwise strand the prior observation. Escalation-referenced
        # rows are kept (the RESTRICT FK + the durable clinician task), so this is the §48 targeted prune,
        # never a blanket delete.
        db.prune_observations(con, member_id, data_version, kept_ids)

        # Escalation-queue peer of the prune (§720 lifecycle): a CLINICIAN_REVIEW data_finding this scan did
        # NOT re-emit — its marker was cleared by an override, or it's a stale-marker_version twin — flips to
        # 'superseded' (off the active GET /escalations queue, row KEPT for audit). Symmetric: removing the
        # override re-raises the marker and re-emits (re-opens) it. An 'urgent' escalation is NEVER superseded
        # (a fired urgent stays queued until a human resolves it — never trade that for de-duplicating a
        # still-urgent range_override twin, which stays as a safe-direction over-show).
        db.reconcile_escalation_status(con, member_id, data_version, live_esc_ids)

    # Return the same read projection GET /observations serves (member_explanation derived at read),
    # so a client that renders the scan response directly gets the member-facing prose too — the raw
    # stored rows carry only trigger_reason, and an empty member_explanation renders as a blank card.
    # Wrapped with the run's newly-created count (the UI's "found N new" signal; 0 on a re-scan).
    return ScanResult(
        observations=observations(con, member_id), new_observations=new_count
    )


def observations(con: sqlite3.Connection, member_id: str) -> list[Observation]:
    """The member-facing ``GET /observations`` projection: the member's persisted findings, each with its
    ``member_explanation`` DERIVED at read (it is not stored). The third read-time-derive peer to
    ``trajectory`` and ``suggestions``: re-``analyze()`` the member's current data, then attach the
    deterministic member-facing prose (``templates.observation_member_explanation``) to each stored
    observation, matched by its ``data_version``-keyed ``observation_id`` — which the analysis reproduces
    exactly (the id keys on ``(member, marker, data_version)`` and ``analyze`` is deterministic), so no
    fragile marker-parsing is needed. The explanation is therefore composed server-side in ``templates``
    (never in JS — preserving the no-stats-leak / surface-the-flags invariant) and always agrees with the
    value+range the Trajectory tab shows for that marker. Raises ``KeyError`` if the member is absent.

    An unscanned member (or one whose last scan raised nothing) has no rows at the current
    ``data_version`` -> returns ``[]`` without composing anything, so the panel is empty exactly as the
    scan-state gating intends. A stored row whose id the current analysis doesn't reproduce (only
    possible if data changed since the scan, which would also change ``data_version`` and so return no
    rows here) keeps its default ``""`` rather than a wrong explanation."""
    member, results, ranges, age, data_version = db.load_for_analysis(con, member_id)
    rows = db.get_observations(con, member_id, data_version=data_version)
    if not rows:
        return rows
    analysis = analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version
    )
    # Rebuild the obs_id -> (traj, rng) map exactly as `scan` minted it, so each finding matches its
    # trajectory by id. Same construction as the scan write-loop (db._det_id + _range_for).
    by_id = {
        db._det_id("obs:", member_id, traj.marker, data_version): (
            traj,
            _range_for(traj.marker, member.sex, age, ranges),
        )
        for traj in _raised_ranked(analysis)
    }
    # A stored observation whose marker is NO LONGER raised (its obs_id isn't in the live raised set) is a
    # STALE finding kept only by the escalation RESTRICT FK after a /feedback override cleared it (Finder E2):
    # HIDE it, so the member panel matches the live analysis + the Trajectory tab (and the clinician queue,
    # whose escalation the reconcile superseded) instead of a card with a blank explanation. The DB row is
    # still KEPT (FK + audit); only this member-facing projection drops it. A currently-raised finding is
    # always in `by_id`, so this never hides an active finding (safe direction).
    return [
        o.model_copy(
            update={
                "member_explanation": templates.observation_member_explanation(
                    *by_id[o.observation_id]
                )
            }
        )
        for o in rows
        if o.observation_id in by_id
    ]


class ScanSweep(NamedTuple):
    """``scan_members``' aggregate: how many members scanned cleanly, and how many observation rows the
    sweep NEWLY persisted across all of them (the sum of each member's ``ScanResult.new_observations`` —
    same row-identity newness semantics; a failed member contributes to neither count)."""

    scanned: int
    new_observations: int


def scan_members(con: sqlite3.Connection, member_ids: list[str]) -> ScanSweep:
    """Scan several members BEST-EFFORT; return a :class:`ScanSweep` — how many scanned cleanly plus how
    many observation rows the sweep newly persisted (the count the ingest-path responses surface, so an
    upload/reseed readout answers "did the sweep find anything new?"). The orchestration behind the
    auto-scan after ``POST /members/upload`` (so a freshly uploaded member's Observations match its live
    Trajectory at once) — kept in the library, not the route, per "logic lives in pipeline, routes stay
    thin". Best-effort by design: the upload's ingest has already committed, so one member's scan failing
    must not fail the others or the request — it's logged and skipped. (The natural home, too, for a
    future server-side 'scan all'.)"""
    scanned = 0
    new_observations = 0
    for member_id in member_ids:
        try:
            new_observations += scan(con, member_id).new_observations
            scanned += 1
        except Exception:
            logging.exception("scan_members: scan failed for member %s", member_id)
    return ScanSweep(scanned=scanned, new_observations=new_observations)


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


def _theil_sen_line(series, slope: float | None) -> list[dict] | None:
    """The two drawable endpoints of the marker's Theil–Sen line (architecture §751 "the already-computed
    Theil–Sen line"), or ``None`` when the slope was too noisy to sign (no line is honest there). Reuses
    the core's computed ``slope`` (per day) and pairs it with the standard Theil–Sen intercept
    ``median(yᵢ − slope·xᵢ)`` — presentation geometry for the sparkline, NOT a re-derived trend verdict
    (the verdict stays analysis.py's). ``x`` is the panel-date ordinal, matching ``analysis._theil_sen``."""
    if slope is None or not series:
        return None
    xs = [date.fromisoformat(r.date).toordinal() for r in series]
    ys = [r.value for r in series]
    intercept = median(y - slope * x for x, y in zip(xs, ys, strict=True))
    return [
        {"date": series[0].date, "value": round(slope * xs[0] + intercept, 4)},
        {"date": series[-1].date, "value": round(slope * xs[-1] + intercept, 4)},
    ]


def trajectory(
    con: sqlite3.Connection, member_id: str, *, marker: str | None = None
) -> list[dict]:
    """Full per-marker series for inspection/plotting (architecture §13/§751) — a UI/operator READ.
    Projects the member's ``lab_results`` + the analysis pass into ``{marker, unit, readings[], trend,
    clinical_change, flags, severity, reference_range, theil_sen}``; optional ``marker`` narrows to one.
    Raises ``KeyError`` if the member is absent.

    Boundary-preserving by construction: this is the one place the RAW per-marker series is exposed, and
    it is exposed ONLY to the UI/operator for charting + human verification of a finding — the **LLM still
    consumes only the collapsed ``TrajectoryAnalysis`` verdict** (§4/§207), never this. Plain dicts, not a
    model (architecture §13 keeps read projections off the contract surface, like ``list_member_summaries``)."""
    member, results, ranges, age, data_version = db.load_for_analysis(con, member_id)
    analysis = analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version
    )
    out: list[dict] = []
    for traj in analysis.markers:
        if marker is not None and traj.marker != marker:
            continue
        series = _series(
            results, traj.marker
        )  # the same pure dated series the core uses
        rng = _range_for(traj.marker, member.sex, age, ranges)
        slope = traj.trend.slope if traj.trend else None
        out.append(
            {
                "marker": traj.marker,
                "unit": traj.unit,
                "readings": [{"date": r.date, "value": r.value} for r in series],
                "trend": traj.trend.model_dump() if traj.trend else None,
                "clinical_change": (
                    traj.clinical_change.model_dump() if traj.clinical_change else None
                ),
                "flags": list(traj.flags),
                "severity": traj.severity,
                "reference_range": (
                    {
                        "ref_low": rng.ref_low,
                        "ref_high": rng.ref_high,
                        "panic_low": rng.panic_low,
                        "panic_high": rng.panic_high,
                    }
                    if rng is not None
                    else None
                ),
                "theil_sen": _theil_sen_line(series, slope),
            }
        )
    return out


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
    prompt_version: int = 0,
) -> ResponseMetadata:
    """The reproducibility tuple + cost/latency instrumentation for an /ask turn. ``model_version`` names
    the model that authored the PROSE (compose model when composed, gate model when a gate-routed
    template, ``deterministic`` when an LLM-down fallback templated it); ``tokens``/``cost_usd`` aggregate
    every LLM call the turn actually made (so a gate-routed template still shows the gate's cost).
    ``prompt_version`` is the promoted ``prompt_versions.version`` the composer ran under (Phase 7), or 0
    when the prose came from a template/fallback (no composer prompt produced it) or the v0 baseline
    constant is in force — so the stamp tracks exactly which learned prompt, if any, authored the answer."""
    return ResponseMetadata(
        response_id=response_id,
        data_version=data_version,
        model_version=model_version,
        config_version=CONFIG_VERSION,
        prompt_version=prompt_version,
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
#: model-version rule both read from here). ``out_of_scope`` is NOT in this map: its responder
#: (``templates.refuse_template``) is FLOOR-AWARE — it takes the turn's computed floor so a refusal
#: under a standing urgent/review floor restates the next step instead of reading flat — and a uniform
#: metadata-only dispatch would silently render the flat copy. It is dispatched explicitly in ``ask``.
_SAFETY_TEMPLATES = {
    "acute_medical": templates.seek_care_template,
    "crisis": templates.crisis_template,
    "couldnt_route": templates.couldnt_route_template,
}


#: Cap on replayed prior turns fed to the composer — the last 10 member+assistant exchanges (20 turns).
#: The history is UNTRUSTED client-authored prose (no server conversation store); this bounds the
#: prompt cost/storage of the field regardless of what the client sends. The gate never sees it.
MAX_HISTORY_TURNS = 20


def ask(
    con: sqlite3.Connection,
    member_id: str,
    message: str,
    *,
    history: list[ConversationTurn] | None = None,
    provider: llm.Provider | None = None,
) -> HealthIntelligenceResponse:
    """One grounded Mode-2 turn. ``gate`` classifies the raw message and sets a message floor; the turn
    runs against ``floor = max(data, message)``; the gate's intent routes to open compose (``none``) or a
    fixed safety template; the response is validated at the floor, persisted (interactions APPENDS for
    ``driver='ask'``), and a chat escalation fires once-per-member-per-day when the floor warrants it.
    Fails SAFE: gate down -> ``couldnt_route`` template at ``clinician_review``; compose down -> a
    deterministic grounded answer at the data floor. Raises ``KeyError`` if the member is absent.

    ``history`` is the prior turns of this chat (client-replayed; there is no server conversation store),
    threaded into the composer ONLY — capped to the most-recent ``MAX_HISTORY_TURNS`` and rendered as
    `<conversation_history>` so a short follow-up resolves against what was said. It is deliberately NOT
    fed to the gate: safety is classified per-message so a benign follow-up after a crisis turn can only
    RAISE the floor via its own analysis, never inherit a diluted route. History never touches the
    analysis or the floor — the one law holds (documented gap: a crisis stated only in a *prior* turn is
    not re-detected on a benign follow-up).

    ``provider`` is the LLM seam — defaults to the Anthropic provider; tests inject a fake so the whole
    path runs offline. A clock is used (created_at, the day-scoped dedup key, latency) — fine here; only
    ``analysis.py`` is clock-free."""
    history = list(history or ())[-MAX_HISTORY_TURNS:]
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

    # The composer prompt_version the prose ran under — 0 unless the composer actually authors it under a
    # promoted version (set in the compose branch below); a template/fallback answer carries 0.
    prompt_version = 0

    # Render: compose only on the `none` route (the sole open generation); the safety branches and the
    # fail-closed couldnt_route are fixed templates (_SAFETY_TEMPLATES). compose() owns the one bounded
    # retry (via llm.call_structured); a parse failure that survives it, or a provider-down, degrades.
    draft: ComposeDraft | None = None
    if g.route == "none":
        prose_model = (
            MODEL_VERSION_DETERMINISTIC  # set to COMPOSE_MODEL only if compose succeeds
        )
        # Resolve the active composer prompt ONLY here, the one path that uses it (Phase 7): the latest
        # PROMOTED prompt_version, else the v0 baseline constant at version 0 (pipeline owns the fallback
        # so db.py never imports llm). Read per request, never cached, so a promotion takes effect on the
        # next turn and only on a promotion event (§7).
        active_prompt = db.get_active_prompt(con)
        active_version, prompt_text = (
            (active_prompt[0], active_prompt[1])
            if active_prompt is not None
            else (0, llm.BASE_COMPOSE_SYSTEM)
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
                preferences=db.get_active_preferences(con, member_id),
                history=history,
            )
            draft, compose_usage = llm.compose(
                ctx, prompt_text=prompt_text, provider=provider
            )
            usages.append(compose_usage)
            prose_model = COMPOSE_MODEL
            prompt_version = active_version  # the prose ran under this promoted version
        except LLMParseError as e:
            # malformed/refused/truncated output that survived the retry -> grounded fallback below.
            # Count the tokens those attempts still billed so the cost stamp isn't an undercount.
            # Log it: the degrade is otherwise invisible — a fallback answer is indistinguishable on
            # screen from a real compose, so a silent throw reads as a quality regression (see the
            # "follow-up loses context" investigation — it was a transient degrade, not context loss).
            logging.warning(
                "ask: compose parse-failed for member %s -> grounded fallback: %s",
                member_id,
                e,
            )
            if e.usage is not None:
                usages.append(e.usage)
            draft = None
        except LLMUnavailable as e:
            # provider down -> grounded fallback. Mode 2 NEVER 500s; it fails safe (§6). A parse-fail-
            # then-unavailable retry carries the first attempt's billed tokens here -> still count them.
            logging.warning(
                "ask: compose unavailable for member %s -> grounded fallback: %s",
                member_id,
                e,
            )
            if e.usage is not None:
                usages.append(e.usage)
            draft = None
    elif g.route != "couldnt_route":
        # out_of_scope / acute_medical / crisis -> a fixed template the GATE model routed us to.
        prose_model = GATE_MODEL
    else:
        # couldnt_route -> the gate produced NO usable output (down, or off-schema twice); a deterministic
        # template authors the prose, so model_version is `deterministic` (the gate's billed tokens, if
        # any, still show in `usages`). Not GATE_MODEL: no gate call successfully decided this turn.
        prose_model = MODEL_VERSION_DETERMINISTIC

    latency_ms = int((time.perf_counter() - start) * 1000)
    # ``prompt_version`` is the composer version that authored the prose, or 0 for a template/fallback
    # (it is only set non-zero on a successful compose above) — parallel to how ``model_version`` already
    # tracks the prose's true author.
    metadata = _llm_metadata(
        response_id,
        data_version,
        model_version=prose_model,
        usages=usages,
        latency_ms=latency_ms,
        prompt_version=prompt_version,
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
        # out_of_scope renders FLOOR-AWARE: the refusal must restate a standing urgent/review next step,
        # never read flat mid-emergency (the floor is already computed; the template only renders it).
        # acute/crisis/couldnt_route stay floor-fixed templates.
        resp = (
            templates.refuse_template(metadata, floor=floor)
            if g.route == "out_of_scope"
            else _SAFETY_TEMPLATES[g.route](metadata)
        )

    # Deterministic floor wins: code sets escalation = floor (the LLM cannot touch it under approach A),
    # then the validator confirms escalation >= floor. The validate -> retry -> template scaffold stays
    # load-bearing — it is the same guard the Phase-5 harness re-runs offline, and the fallback above
    # genuinely fires on LLMUnavailable — even though, escalation being code-set, this assertion is
    # trivially true here (the LLM literally cannot under-escalate).
    resp.escalation = floor
    resp = safety.validate(resp, floor)

    # Atomic: the audit interaction + the clinician-queue escalation commit (or roll back) together.
    # DELIBERATE: only the current ``message`` is persisted, NOT ``history``. History is client-replayed
    # (the "no server conversation store" design), so it is untrusted and ephemeral — storing it would put
    # unbounded, unverified client text in the audit trail and contradict that design. Accepted tradeoff:
    # the stored answer reflects context (history) that isn't reconstructable server-side. The safety-
    # relevant inputs ARE durable and reproducible — analysis (``data_version``), floor, gate route
    # (``trigger_reason``), and prompt (``prompt_version``) — none of which history can influence.
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
