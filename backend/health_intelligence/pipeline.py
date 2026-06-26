"""pipeline.py — orchestration over the pure core (Phase 3a: the proactive scan, no LLM).

The scan is the deterministic half of the proactive behavior: load the member, run ``analyze``, decide
which markers are worth surfacing, narrate them into one ``HealthIntelligenceResponse``, and persist the
trace + observations + the clinician-review escalations. Every safety-relevant decision here is read off
the pure core (severity, the escalation floor) and only *acted on* — nothing is recomputed (CLAUDE.md:
the LLM, and equally this orchestration, is never the core). The ask path (gate -> compose -> validate)
is added in Phase 4 alongside the LLM layer.

Determinism & idempotency: all persisted PKs key on an ``analysis_version`` (the hash of exactly the
inputs ``analyze`` reads — db.compute_analysis_version), so a re-scan of unchanged analysis inputs
UPSERTs the same rows and re-emits the same dedup_key (a no-op) rather than duplicating, and an edit
that doesn't touch the analysis (e.g. a note) cannot mint a second escalation. "Replace, not append"
across genuine data changes is delivered by the version-scoped observation read, not by deletes.
"""

from __future__ import annotations

import sqlite3

from health_intelligence import db, safety, templates
# _range_for: reuse the core's single source of truth for sex/fallback range resolution (pure fn).
from health_intelligence.analysis import _range_for, analyze
from health_intelligence.config import ANALYSIS_CONFIG, CONFIG_VERSION, MODEL_VERSION_DETERMINISTIC
from health_intelligence.models import (
    MarkerTrajectory,
    Observation,
    ResponseMetadata,
    SEVERITY_ORDER,
)
from health_intelligence.safety import severity_to_level


def _is_raised(traj: MarkerTrajectory) -> bool:
    """The "raise an observation?" predicate (architecture §397 leaves it to the caller — ``analyze``
    returns severity only). Surface anything the core judged above baseline: an out-of-range or
    band-crossing value, a trend, or a panic — i.e. severity beyond ``info``. A healthy in-range marker
    with no trend stays ``info`` and is not surfaced, so a calm member (negative control) raises nothing
    and escalates nothing — no false alarms."""
    return SEVERITY_ORDER[traj.severity] > SEVERITY_ORDER["info"]


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
    analysis = analyze(member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version)
    analysis_version = db.compute_analysis_version(
        results=results, ranges=ranges, sex=member.sex, age=age
    )
    floor = safety.data_floor(analysis)

    # Which markers to surface, ranked highest-severity first (stable: then by marker name).
    raised = sorted(
        (t for t in analysis.markers if _is_raised(t)),
        key=lambda t: (-SEVERITY_ORDER[t.severity], t.marker),
    )

    with con:  # atomic: interaction + observation + escalation per finding commit (or roll back) together
        for traj in raised:
            response_id = db._det_id("scan:", member_id, traj.marker, data_version)
            obs_id = db._det_id("obs:", member_id, traj.marker, data_version)  # finding_id == observation_id
            rng = _range_for(traj.marker, member.sex, age, ranges)
            title, trigger_reason = templates.observation_summary(traj)  # computed once
            finding = templates.scan_finding(traj, rng, obs_id, title)

            metadata = ResponseMetadata(
                response_id=response_id,
                data_version=data_version,
                model_version=MODEL_VERSION_DETERMINISTIC,  # no LLM ran; tuple is (data, config, template)
                config_version=CONFIG_VERSION,
                prompt_version=0,
            )
            # escalation = the member's deterministic floor, carried on every response and validated
            # >= floor (CLAUDE.md "floor always on"; §2 D4 routes the template path through the validator
            # before persist). Trivially passes here — the response is built AT the floor — which is why
            # asserting it is cheap insurance against any future drift in how the response sets escalation.
            resp = safety.validate(
                templates.render_finding(finding, escalation=floor, metadata=metadata), floor
            )

            db.write_interaction(con, resp, member_id=member_id, driver="scan", question=None)
            db.write_observation(con, Observation(
                observation_id=obs_id, member_id=member_id, response_id=response_id,
                severity=traj.severity, title=title, trigger_reason=trigger_reason,
                data_version=data_version,
            ))
            level = severity_to_level(traj.severity)
            if level is not None:  # only attention/urgent reach the clinician-review queue
                db._insert_escalation(
                    con, member_id=member_id, kind="data_finding",
                    dedup_key=f"data:{member_id}:{traj.marker}:{analysis_version}",
                    level=level, observation_id=obs_id, trigger_reason=trigger_reason,
                )

    return db.get_observations(con, member_id, data_version=data_version)
