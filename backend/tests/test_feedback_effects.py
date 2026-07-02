"""Cohort integration sweep — `/feedback` corrections move `/scan` observations for EVERY applicable
seeded member, and the safety-shaped exceptions hold across the real 15-member training dataset.

The UNIT behaviors (a range_override clears a flag, suppress drops a marker, member-sourced is inert,
the escalation-pinned prune, the signal→prompt assembly split) are covered on synthetic data in
``test_feedback.py`` / ``test_learn.py``. This file is the gap the 2026-06-30 audit flagged: those are
*safety invariants* with zero standing coverage over the actual seeded cohort, so they should fail a
build rather than live in a one-off script. Deterministic, no LLM (scan/observations/analyze only).

Targets are DATA-DRIVEN (picked per member from its real marker state, asserted by the deterministic
``observation_id`` = ``db._det_id("obs:", member, marker, data_version)``) — an override does NOT bump
``data_version``, so the id is stable across the override and lets us assert exact presence/absence.
"""

import pytest
from builders import fresh_con

from health_intelligence import db, pipeline, safety
from health_intelligence.analysis import analyze
from health_intelligence.config import ANALYSIS_CONFIG
from health_intelligence.models import Feedback
from preprocessing.ingest import ingest_dataset

PANIC = {"panic_high", "panic_low"}
OOR = {"above_range", "below_range"}

# One seeded in-memory DB for the module; the autouse fixture reverts learning between tests so each
# correction is isolated (observations/escalations persist but every assertion is per-member-per-id).
_CON = fresh_con()
ingest_dataset(_CON)
_MEMBERS = db.list_members(_CON)


def _markers(mid):
    """Override-resolved marker verdicts + the (stable) data_version — the path scan/ask run."""
    member, results, ranges, age, dv = db.load_for_analysis(_CON, mid)
    return analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=dv
    ).markers, dv


# Classify each member ONCE (at import) for parametrization: which correction each member can exercise.
_NOTABLE_OOR, _NOTABLE_ANY, _PANIC, _ATTENTION = {}, {}, {}, {}
for _m in _MEMBERS:
    _ms, _ = _markers(_m)
    _NOTABLE_OOR[_m] = [
        t.marker for t in _ms if t.severity == "notable" and set(t.flags) & OOR
    ]
    _NOTABLE_ANY[_m] = [t.marker for t in _ms if t.severity == "notable"]
    _PANIC[_m] = [t.marker for t in _ms if set(t.flags) & PANIC]
    # attention-severity markers -> a clinician_review escalation on scan (the tier that auto-supersedes)
    _ATTENTION[_m] = [t.marker for t in _ms if t.severity == "attention"]

_HAS_OOR = [m for m in _MEMBERS if _NOTABLE_OOR[m]]
_HAS_NOTABLE = [m for m in _MEMBERS if _NOTABLE_ANY[m]]
_HAS_PANIC = [m for m in _MEMBERS if _PANIC[m]]
_HAS_ATTENTION = [m for m in _MEMBERS if _ATTENTION[m]]


@pytest.fixture(autouse=True)
def _clean_learning():
    db.reset_learning(_CON)  # deactivate any override left by a prior test
    yield
    db.reset_learning(_CON)


def _oid(mid, marker, dv):
    return db._det_id("obs:", mid, marker, dv)


def _scan_ids(mid):
    return {o.observation_id for o in pipeline.scan(_CON, mid)}


def _fb(mid, **kw):
    db.insert_feedback(_CON, mid, Feedback(**kw))


def _traj(mid, marker):
    return next((t for t in _markers(mid)[0] if t.marker == marker), None)


# ---- tighten range_override CREATES an observation — provable for EVERY member -----------------------


@pytest.mark.parametrize("mid", _MEMBERS)
def test_tighten_override_creates_observation_for_every_member(mid):
    ms, dv = _markers(mid)
    in_range = [
        t for t in ms if t.severity == "info" and not (set(t.flags) & (OOR | PANIC))
    ]
    assert in_range, f"{mid} unexpectedly has no clean in-range marker"
    t = in_range[0]
    tid = _oid(mid, t.marker, dv)
    assert tid not in _scan_ids(mid)  # in-range -> not raised at baseline
    # tighten ref_high below the latest value -> above_range (notable, not panic)
    _fb(
        mid,
        kind="range_override",
        target=t.marker,
        payload={"ref_high": t.latest.value - abs(t.latest.value) * 0.1 - 1},
        source="clinician",
    )
    assert tid in _scan_ids(mid)
    created = _traj(mid, t.marker)
    assert created is not None and "above_range" in created.flags


# ---- preference is INERT on observations but reaches the composer hint — EVERY member ----------------


@pytest.mark.parametrize("mid", _MEMBERS)
def test_preference_inert_on_observations_but_reaches_hint(mid):
    before = _scan_ids(mid)
    _fb(mid, kind="preference", payload={"text": "keep it brief"}, source="member")
    assert _scan_ids(mid) == before  # analysis untouched
    assert "keep it brief" in db.get_active_preferences(_CON, mid)


# ---- widen range_override CLEARS a notable out-of-range observation; /reset restores it --------------


@pytest.mark.parametrize("mid", _HAS_OOR)
def test_widen_override_clears_notable_out_of_range_observation(mid):
    ms, dv = _markers(mid)
    t = next(t for t in ms if t.severity == "notable" and set(t.flags) & OOR)
    tid = _oid(mid, t.marker, dv)
    assert tid in _scan_ids(mid)
    payload = {"ref_high": 1e9} if "above_range" in t.flags else {"ref_low": -1e9}
    _fb(
        mid, kind="range_override", target=t.marker, payload=payload, source="clinician"
    )
    assert tid not in _scan_ids(
        mid
    )  # flag cleared -> obs pruned (notable = non-escalated)
    cleared = _traj(mid, t.marker)
    assert cleared is not None and not (set(cleared.flags) & OOR)
    db.reset_learning(_CON)
    assert tid in _scan_ids(mid)  # reset restores the finding


# ---- suppress_marker drops the marker from analysis and prunes its observation ----------------------


@pytest.mark.parametrize("mid", _HAS_NOTABLE)
def test_suppress_removes_notable_observation(mid):
    ms, dv = _markers(mid)
    # prefer a notable out-of-range (non-escalated -> clean prune); else any notable
    t = next(
        (t for t in ms if t.severity == "notable" and set(t.flags) & OOR), None
    ) or next(t for t in ms if t.severity == "notable")
    tid = _oid(mid, t.marker, dv)
    assert tid in _scan_ids(mid)
    _fb(mid, kind="suppress_marker", target=t.marker, source="clinician")
    assert tid not in _scan_ids(mid)
    assert _traj(mid, t.marker) is None  # gone from the analysis entirely


# ---- member-sourced override is INERT for analysis (the safety property) ----------------------------


@pytest.mark.parametrize("mid", _HAS_OOR)
def test_member_sourced_override_is_inert(mid):
    ms, dv = _markers(mid)
    t = next(t for t in ms if t.severity == "notable" and set(t.flags) & OOR)
    tid = _oid(mid, t.marker, dv)
    assert tid in _scan_ids(mid)
    payload = {"ref_high": 1e9} if "above_range" in t.flags else {"ref_low": -1e9}
    _fb(mid, kind="range_override", target=t.marker, payload=payload, source="member")
    assert tid in _scan_ids(mid)  # member can't lobby the safety logic -> still flagged


# ---- a clearing override on a PANIC marker keeps the escalation-pinned observation (RESTRICT FK) -----


@pytest.mark.parametrize("mid", _HAS_PANIC)
def test_clearing_override_keeps_escalation_pinned_panic_observation(mid):
    ms, dv = _markers(mid)
    p = next(t for t in ms if set(t.flags) & PANIC)
    pid = _oid(mid, p.marker, dv)
    assert pid in _scan_ids(
        mid
    )  # first scan queues the urgent data-finding (panic raised -> shown)
    # the urgent escalation is OPEN on the ACTIVE queue — asserted via get_all_escalations, not the
    # all-status get_escalations (which never empties and can't fail: H2).
    assert any(
        e.observation_id == pid and e.level == "urgent" and e.status == "open"
        for e in db.get_all_escalations(_CON)
    )
    _fb(
        mid,
        kind="range_override",
        target=p.marker,
        payload={
            "ref_high": 1e9,
            "panic_high": 1e9,
            "ref_low": -1e9,
            "panic_low": -1e9,
        },
        source="clinician",
    )
    ids = _scan_ids(mid)
    cleared = _traj(mid, p.marker)
    assert cleared is not None and not (
        set(cleared.flags) & PANIC
    )  # analytical flag cleared
    # The observation ROW is KEPT by the escalation RESTRICT FK (checked at the DB layer) — but the
    # member-facing projection now HIDES the cleared finding (E2), so it is NOT in the scan return.
    assert any(
        o.observation_id == pid for o in db.get_observations(_CON, mid)
    )  # row kept
    assert pid not in ids  # member panel drops the cleared finding
    # SAFETY: the urgent task is NEVER superseded — still OPEN on the active queue until a human resolves it.
    assert any(
        e.observation_id == pid and e.status == "open"
        for e in db.get_all_escalations(_CON)
    )


# ---- SAFETY: suppress is INERT against a panic floor (must not drop the deterministic floor) ----------


@pytest.mark.parametrize("mid", _HAS_PANIC)
def test_clinician_suppress_cannot_lower_a_panic_floor(mid):
    """SAFETY (the FLOOR, not the queue): a clinician ``suppress_marker`` on a PANIC marker must NOT drop
    ``safety.data_floor`` from urgent to none — suppress is inert against a panic floor. Asserted through the
    REAL ``analyze`` + ``data_floor`` (not the inline predicate), so it catches ``db._latest_breaches_panic``
    drifting from ``analysis._flags``. This previously DID drop the floor to none — the bug the guard fixes."""
    p = next(t for t in _markers(mid)[0] if set(t.flags) & PANIC)

    def _floor():
        member, results, ranges, age, dv = db.load_for_analysis(_CON, mid)
        return safety.data_floor(
            analyze(member, results, ranges, age, ANALYSIS_CONFIG, data_version=dv)
        )

    assert _floor() == "urgent"  # the panic sets the urgent floor
    _fb(mid, kind="suppress_marker", target=p.marker, source="clinician")
    assert (
        _floor() == "urgent"
    )  # inert against the panic floor (the bug dropped it to 'none')


@pytest.mark.parametrize("mid", _HAS_PANIC)
def test_range_override_never_hides_a_still_urgent_escalation(mid):
    """SAFETY: a ``range_override`` re-bounding ONLY the ref range (not the panic threshold) leaves the
    marker STILL panic → a new ``marker_version`` fires a fresh urgent escalation while the old lingers. An
    urgent finding is NEVER hidden: it stays on the ACTIVE queue before AND after the override. We do NOT
    auto-dedup the stale urgent twin — a conditional supersede was tried and REVERTED (keyed on emitted
    obs-ids it could HIDE a downgraded urgent, panic→clinician_review, or resurrect a superseded twin, both
    worse than the duplicate). The duplicate is an accepted safe-direction over-show (a clinician sees the
    finding twice, never MISSES it) that the deferred human-resolve lifecycle cleans up."""
    p = next(t for t in _markers(mid)[0] if set(t.flags) & PANIC)
    _scan_ids(mid)

    def _open_urgent():
        return [
            e
            for e in db.get_all_escalations(_CON)
            if e.member_id == mid and e.level == "urgent" and p.marker in e.dedup_key
        ]

    assert len(_open_urgent()) >= 1  # the panic marker's urgent is on the active queue
    _fb(  # widen the REF range only — panic threshold untouched, so the marker stays panic
        mid,
        kind="range_override",
        target=p.marker,
        payload={"ref_high": 1e9, "ref_low": -1e9},
        source="clinician",
    )
    _scan_ids(mid)
    still = _traj(mid, p.marker)
    assert still is not None and (set(still.flags) & PANIC)  # precondition: STILL panic
    assert (
        len(_open_urgent()) >= 1
    )  # SAFETY: urgent never hidden (a duplicate twin is accepted over-show)


# ---- escalation lifecycle (§720): a cleared clinician_review finding leaves the ACTIVE queue -----------


def _esc_for(mid, marker, dv):
    """The member's escalation pinned to (marker, dv)'s observation, or None."""
    oid = _oid(mid, marker, dv)
    return next(
        (e for e in db.get_escalations(_CON, mid) if e.observation_id == oid), None
    )


@pytest.mark.parametrize("mid", _HAS_ATTENTION)
def test_override_supersedes_clinician_review_escalation_and_reopens_symmetrically(mid):
    """A clinician override that CLEARS an attention (clinician_review) finding drops its escalation off the
    ACTIVE queue (``get_all_escalations`` shows 'open' only) while KEEPING the row for audit
    (``get_escalations`` returns it, status='superseded'). Removing the override and re-scanning re-opens it
    — the reconcile is symmetric. The §720 escalation lifecycle."""
    _, dv = _markers(mid)
    marker = _ATTENTION[mid][0]
    _scan_ids(mid)  # first scan queues the clinician_review data-finding
    esc = _esc_for(mid, marker, dv)
    assert esc is not None and esc.level == "clinician_review" and esc.status == "open"
    eid = esc.escalation_id
    assert any(
        e.escalation_id == eid for e in db.get_all_escalations(_CON)
    )  # on the active queue

    _fb(
        mid, kind="suppress_marker", target=marker, source="clinician"
    )  # clears the finding
    _scan_ids(mid)
    kept = {e.escalation_id: e.status for e in db.get_escalations(_CON, mid)}
    assert kept[eid] == "superseded"  # row KEPT for audit, marked superseded
    assert all(
        e.escalation_id != eid for e in db.get_all_escalations(_CON)
    )  # OFF the active queue

    db.reset_learning(_CON)  # remove the override
    _scan_ids(mid)
    assert any(  # marker raised again -> escalation re-opened (symmetric)
        e.escalation_id == eid and e.status == "open"
        for e in db.get_all_escalations(_CON)
    )


@pytest.mark.parametrize("mid", _HAS_PANIC)
def test_urgent_panic_escalation_is_never_superseded(mid):
    """SAFETY: an 'urgent' panic escalation is NEVER auto-superseded. A DELIBERATE ``range_override`` that
    re-bounds the panic threshold is the ONLY path that can clear a panic flag (``suppress_marker`` is now
    panic-inert), yet the already-fired urgent task must stay on the active queue until a HUMAN resolves it.
    Exercises the level-guard DIRECTLY: with the flag cleared the marker is no longer raised, so only the
    ``level=='urgent'`` branch (not the raised-set branch) keeps the escalation open."""
    ms, _ = _markers(mid)
    p = next(t for t in ms if set(t.flags) & PANIC)
    _scan_ids(mid)  # first scan queues the urgent data-finding
    urgent = {
        e.escalation_id for e in db.get_escalations(_CON, mid) if e.level == "urgent"
    }
    assert urgent
    _fb(  # a deliberate clinician re-bound of BOTH the range and the panic threshold clears the flag
        mid,
        kind="range_override",
        target=p.marker,
        payload={
            "ref_high": 1e9,
            "panic_high": 1e9,
            "ref_low": -1e9,
            "panic_low": -1e9,
        },
        source="clinician",
    )
    _scan_ids(mid)
    cleared = _traj(mid, p.marker)
    # HONESTY PRECONDITION: the marker must be fully UNRAISED after the re-bound (severity 'info'). A widened
    # range kills the panic/range FLAGS but NOT a Mann-Kendall/RCV TREND — a marker that stayed trend-raised
    # would keep the escalation open via the raised-set branch, leaving the level-guard UNTESTED (vacuous).
    # Potassium is panic-gated with no trend, so it drops to 'info'; asserting it fails LOUDLY if a future
    # panic marker on this data stays trend-raised.
    assert (
        cleared is not None
        and cleared.severity == "info"
        and not (set(cleared.flags) & PANIC)
    )
    after = {e.escalation_id: e.status for e in db.get_escalations(_CON, mid)}
    assert all(after[i] == "open" for i in urgent)  # never superseded
    assert urgent <= {
        e.escalation_id for e in db.get_all_escalations(_CON)
    }  # still on the active queue


@pytest.mark.parametrize("mid", _HAS_ATTENTION)
def test_member_sourced_override_cannot_supersede_an_escalation(mid):
    """SAFETY: a MEMBER-sourced override is inert (``resolve_overrides`` honors clinician/system only), so it
    can never supersede a clinician's queue item — the marker stays raised, the escalation stays 'open'."""
    _, dv = _markers(mid)
    marker = _ATTENTION[mid][0]
    _scan_ids(mid)
    esc = _esc_for(mid, marker, dv)
    assert esc is not None
    eid = esc.escalation_id
    _fb(
        mid, kind="suppress_marker", target=marker, source="member"
    )  # member-sourced -> inert
    _scan_ids(mid)
    assert any(  # untouched: still open, still queued
        e.escalation_id == eid and e.status == "open"
        for e in db.get_all_escalations(_CON)
    )
