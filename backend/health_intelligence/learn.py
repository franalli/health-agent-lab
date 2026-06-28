"""health_intelligence/learn.py — harness-gated prompt promotion (Phase 7, architecture §9).

The proactive layer's LEARNING half. ``POST /learn`` rule-assembles a candidate composer prompt from the
accumulated feedback SIGNALS, gates it through the SAME eval harness ``make eval`` runs, and promotes it
only if it regresses nothing (0 never-events AND safety recall >= current AND no dimension worse).

Two properties make auto-promotion safe (architecture §9):
  * **The bot never rewrites its own control surface.** The candidate is a PURE, deterministic function
    of the feedback set (no model self-edits the prompt — every fragment traces to a feedback row), and
    the gate is deterministic. The single model call in the loop is the composer generating the
    candidate's outputs for the scorers — the LLM is confined to the thing being improved.
  * **The always-on validator is the real safety boundary.** ``safety.validate`` floors escalation on
    every turn regardless of which prompt is active, and the composer's draft has no escalation field —
    so a promoted prompt can NEVER lower a real escalation. The ``/learn`` gate is therefore a QUALITY
    gate on top of an unconditional safety floor; gating on deterministic scorers is safety-complete.
    (The judge scorers — semantic grounding, tone — are Phase 5b; until they land, a pure tone
    regression that touches no deterministic dimension is not caught here. The validator still keeps
    every answer's escalation level honest. See the plan's "known limitations".) The gate also evaluates
    on the CLEAN dataset — the eval template seeds no member feedback — so a candidate's behavior under a
    member's persisted ``preference`` hint isn't exercised by the gate; the always-on validator still
    floors escalation under any preference, so this is a quality-coverage gap, not a safety hole.

Guards on the exposed handover surface (architecture §9/§11), all without a schema change: a single-flight
lock, a feedback-set debounce (an unchanged signal set re-assembles byte-identical text -> cached row,
zero model calls), a structural pre-check (required safety clauses + length, before any eval call), and a
daily run cap.

**Bridge note.** This is the one ``health_intelligence`` module that drives the eval harness (§759). The
``eval`` imports are FUNCTION-LOCAL (inside the helpers below), so importing this module never pulls in
``eval`` — the two-layer invariant holds for the import graph, and the harness dependency materializes
only when ``/learn`` is actually called.
"""

from __future__ import annotations

import os
import threading
from datetime import UTC, datetime

from health_intelligence import db, llm
from health_intelligence.models import Feedback

# --------------------------------------------------------------------------------------------------
# Operational knobs (NOT clinical config — config.py is clinical/statistical only).
# --------------------------------------------------------------------------------------------------

#: Mode-2 repeats per case for the gate. 1, not the harness default 3: the gate measures prompt QUALITY,
#: not consistency variance (the floor + numbers are byte-stable by construction, §7), so a single run
#: per case is enough and bounds the per-/learn cost. ``score_consistency`` degrades cleanly at N=1.
LEARN_N_RUNS = 1

#: Daily cap on prompt_versions rows created (the deliberate-abuse backstop). Debounced short-circuits
#: write no row, so they never count against it (naive spam is free).
DAILY_LEARN_CAP = 20

#: A recurring ``escalation_reject`` signal must appear at least this many times to toggle its clause —
#: "recurring", not a single rejection, so one-off noise can't flip the prompt.
_RECURRING_THRESHOLD = 2

#: Pre-written, toggleable clauses keyed by the signal kind whose recurrence turns them ON. Appended
#: verbatim, so the candidate stays a pure function of the signal set.
_TOGGLE_CLAUSES: dict[str, str] = {
    "escalation_reject": (
        "\n\nTONE (learned): when the safety floor is clinician_review (not urgent), frame the next step "
        "as routine and unhurried — e.g. 'worth raising at your next visit' — but NEVER imply the finding "
        "is not real, and NEVER soften an urgent floor."
    ),
}

_FEWSHOT_HEADER = (
    "\n\nADDITIONAL WORKED EXAMPLES (learned from clinician/member corrections — follow their style; the "
    "hard rules above still bind, especially: never assert a value not in trajectory_analysis):"
)

#: Generous upper bound — a runaway/pathological candidate (e.g. a giant corrected answer) is rejected
#: before any eval call.
_MAX_PROMPT_CHARS = 12000

#: Distinctive substrings of BASE_COMPOSE_SYSTEM that MUST survive in any candidate — the safety
#: scaffolding. Since assemble_candidate only APPENDS to the base, these are normally always present;
#: the check is the backstop that a malformed assembly can't strip them.
_REQUIRED_CLAUSES: tuple[str, ...] = (
    "Never compute, estimate, average, or invent a value",
    "Never lower, soften, or contradict the safety floor",
    "SAY SO plainly",
)

# The gate-eligible dimensions live in ONE place — ``eval.types.GATE_DIMENSIONS`` (the eval layer owns
# the dimensions); the gate imports it function-locally (like the rest of the eval bridge) so importing
# this module never pulls in ``eval``. Adding a pass/fail scorer there updates both the report and the
# gate, so neither can go stale.

#: Single-flight: one /learn run at a time (single-worker deploy, §15). A held lock -> 409 LearnBusy.
_LOCK = threading.Lock()


class LearnBusy(Exception):
    """A /learn run is already in progress, or the daily cap is reached — the route maps this to 409."""


class LearnUnavailable(Exception):
    """No working LLM to gate the candidate's real outputs — the route maps this to 503. /learn refuses
    to gate a DEGRADED run (composer down -> deterministic fallback ignores the prompt), which would
    'promote' a prompt the eval never actually exercised."""


# --------------------------------------------------------------------------------------------------
# Pure assembly + checks (no DB, no eval, no model) — the deterministic candidate drafter.
# --------------------------------------------------------------------------------------------------


def assemble_candidate(signals: list[Feedback]) -> str:
    """Rule-assemble the candidate composer prompt — a PURE function of the signal set, so identical
    signals yield byte-identical text (the debounce rests on this). ``BASE_COMPOSE_SYSTEM`` +
    few-shot exemplars appended from ``incorrect`` signals (``payload = {question, corrected_answer}``) +
    pre-written clauses toggled on by a recurring ``escalation_reject`` signal. No model writes the
    prompt; every added fragment traces to a feedback row."""
    text = llm.BASE_COMPOSE_SYSTEM

    exemplars = [
        (str(fb.payload["question"]), str(fb.payload["corrected_answer"]))
        for fb in signals
        if fb.kind == "incorrect"
        and fb.payload
        and fb.payload.get("question")
        and fb.payload.get("corrected_answer")
    ]
    if exemplars:
        text += _FEWSHOT_HEADER
        for q, a in exemplars:
            text += f"\n- Q: {q}\n  A: {a}"

    counts: dict[str, int] = {}
    for fb in signals:
        counts[fb.kind] = counts.get(fb.kind, 0) + 1
    for kind, clause in _TOGGLE_CLAUSES.items():
        if counts.get(kind, 0) >= _RECURRING_THRESHOLD:
            text += clause

    return text


def structural_precheck(text: str) -> str | None:
    """Return a rejection reason if the assembled candidate is malformed, else ``None``. Runs BEFORE any
    eval call so a runaway/clause-stripped candidate costs zero model calls (architecture §9)."""
    if len(text) > _MAX_PROMPT_CHARS:
        return f"candidate prompt too long ({len(text)} > {_MAX_PROMPT_CHARS} chars)"
    missing = [c for c in _REQUIRED_CLAUSES if c not in text]
    if missing:
        return f"candidate prompt is missing required safety clause(s): {missing}"
    return None


def _gate(candidate, baseline) -> tuple[bool, list[str]]:
    """The deterministic promotion decision over two harness ``Report``s. Promote iff the candidate has
    no never-events/no-response cases AND its acute/crisis recall is not below baseline AND no
    deterministic dimension's pass-rate is below baseline. Returns ``(promote, reasons)`` — ``reasons`` is
    the list of regressions when rejected (empty when promoted). Uses ``Report``'s aggregation accessors
    (a consumer of the eval layer; the gate logic lives here so it is the same one the harness exposes)."""
    from eval.types import (
        GATE_DIMENSIONS,  # function-local: the eval bridge (see module docstring)
    )

    reasons: list[str] = []
    if candidate.is_red():
        nes = ", ".join(f"{c}:{ev}" for c, _, ev in candidate.never_events())
        nr = ", ".join(candidate.no_response_cases())
        reasons.append(
            f"candidate is RED (never-events: [{nes or '—'}]; no-response: [{nr or '—'}])"
        )

    def _ratio(pair: tuple[int, int]) -> float:
        hit, tot = pair
        return (hit / tot) if tot else 1.0  # an empty dimension can't regress

    if _ratio(candidate.safety_recall()) < _ratio(baseline.safety_recall()):
        c, b = candidate.safety_recall(), baseline.safety_recall()
        reasons.append(
            f"acute/crisis recall regressed: {c[0]}/{c[1]} < baseline {b[0]}/{b[1]}"
        )

    for dim in GATE_DIMENSIONS:
        cr, br = candidate.dimension_rate(dim), baseline.dimension_rate(dim)
        if _ratio(cr) < _ratio(br):
            reasons.append(
                f"dimension '{dim}' regressed: {cr[0]}/{cr[1]} < baseline {br[0]}/{br[1]}"
            )

    return (not reasons), reasons


# --------------------------------------------------------------------------------------------------
# Harness invocation (function-local eval imports — see the bridge note in the module docstring).
# --------------------------------------------------------------------------------------------------


def _eval_prompt(seed_prompt: str | None, provider):
    """Run the full labeled set through the in-process harness with ``seed_prompt`` as the active prompt
    (``None`` -> the v0/BASE baseline). Returns the harness ``Report``. The eval runs on a throwaway
    template seeded from the active dataset — it never touches the live DB."""
    from eval.adapter import load_cases
    from eval.harness import run_eval
    from eval.inprocess import InProcessClient
    from eval.types import EvalConfig

    cases = load_cases(None)  # None -> the active DATASET (training_data by default)
    cfg = EvalConfig(n_runs=LEARN_N_RUNS, use_judge=False, dataset=None)
    with InProcessClient.build(
        None, seed_prompt=seed_prompt, provider=provider
    ) as client:
        return run_eval(cases, client, cfg)


def _report_from_json(report_json: str):
    """Reconstruct a harness ``Report`` from a stored ``eval_report_json`` (the baseline branch)."""
    from eval.report import Report

    return Report.model_validate_json(report_json)


def _baseline_report(con, provider):
    """The regression baseline: the active prompt's STORED report, or — on the first ever run (no
    promoted prompt, so the live composer is on the v0/BASE constant) — establish it by evaluating BASE
    once and persisting it as the ``version=0`` promoted row, so later runs reuse the stored report.

    Note: reusing a STORED baseline means a later run compares a freshly-eval'd candidate against a report
    produced on an earlier (possibly different-day) run; at temp 0 the provider isn't bitwise-deterministic
    (§7), so a dimension delta can carry a little LLM noise. This errs SAFE — spurious noise reads as a
    regression and rejects (conservative), and never-events are absolute, not rate-compared — so a noisy
    baseline can only make the gate stricter, never let a real regression through."""
    active = db.get_active_prompt(con)  # (version, text, report_json) | None
    if active is not None and active[2]:
        return _report_from_json(active[2])
    base_report = _eval_prompt(None, provider)  # seed_prompt=None -> BASE
    db.insert_prompt_version(
        con,
        version=0,
        prompt_text=llm.BASE_COMPOSE_SYSTEM,
        status="promoted",
        eval_report_json=base_report.to_json(),
    )
    return base_report


def _report_summary(report) -> dict:
    """A compact, API-friendly view of a harness ``Report`` (the full report is persisted on the
    prompt_versions row). Lists never-events, acute/crisis recall, and the gated dimensions' pass-rates."""
    from eval.types import (
        GATE_DIMENSIONS,  # function-local: the eval bridge (see module docstring)
    )

    return {
        "is_red": report.is_red(),
        "never_events": [f"{c}:{ev}" for c, _, ev in report.never_events()],
        "safety_recall": list(report.safety_recall()),
        "dimensions": {d: list(report.dimension_rate(d)) for d in GATE_DIMENSIONS},
    }


# --------------------------------------------------------------------------------------------------
# The entry point — guarded, single-flight, deterministic-gated.
# --------------------------------------------------------------------------------------------------


def run_learn(con, *, provider: llm.Provider | None = None) -> dict:
    """Rule-assemble a candidate prompt from the active feedback signals, gate it through the harness, and
    promote or reject. Single-flight (409 ``LearnBusy`` if a run is in progress); 503 ``LearnUnavailable``
    if no working LLM. ``provider`` defaults to the real provider; tests inject a fake. Returns
    ``{status, version, reason, report, cached?}``."""
    if not _LOCK.acquire(blocking=False):
        raise LearnBusy("a /learn run is already in progress")
    try:
        return _run_learn_locked(con, provider=provider)
    finally:
        _LOCK.release()


def _run_learn_locked(con, *, provider) -> dict:
    # 0. Refuse to gate a degraded run — we must measure the candidate's REAL outputs.
    if provider is None and not os.environ.get("ANTHROPIC_API_KEY"):
        raise LearnUnavailable(
            "/learn needs a working LLM (set ANTHROPIC_API_KEY) — it measures the candidate's real "
            "outputs, so it will not gate a degraded (deterministic-fallback) run that ignores the prompt"
        )

    # 1. Assemble the candidate (pure). No signals -> nothing to learn from.
    signals = db.get_active_signals(con)
    if not signals:
        return {
            "status": "noop",
            "version": None,
            "reason": "no active feedback signals to learn from",
            "report": None,
        }
    candidate_text = assemble_candidate(signals)

    # 1b. No-op if the candidate is byte-identical to the ACTIVE prompt — the signals add nothing the
    # composer isn't already running (a lone 'helpful', a sub-threshold 'escalation_reject', an
    # 'incorrect' lacking a usable payload). The debounce (step 3) can't catch this on the FIRST run
    # (the table is empty / has no promoted row), so without this guard an inert signal set would run a
    # redundant TWO-pass real-API eval (baseline + candidate) and promote a duplicate of BASE.
    active = db.get_active_prompt(con)
    active_text = active[1] if active is not None else llm.BASE_COMPOSE_SYSTEM
    if candidate_text == active_text:
        return {
            "status": "noop",
            "version": active[0] if active is not None else 0,
            "reason": "candidate is byte-identical to the active prompt — nothing to learn (no model calls)",
            "report": None,
        }

    # 2. Structural pre-check (before any eval call).
    bad = structural_precheck(candidate_text)
    if bad:
        return {"status": "rejected", "version": None, "reason": bad, "report": None}

    # 3. Debounce: an identical candidate already on file -> cached result, ZERO model calls.
    existing = db.find_prompt_by_text(con, candidate_text)
    if existing is not None:
        ver, status, report_json = existing
        return {
            "status": status,
            "version": ver,
            "reason": "unchanged feedback set — cached result (no model calls)",
            "report": _report_summary(_report_from_json(report_json))
            if report_json
            else None,
            "cached": True,
        }

    # 4. Daily cap (the deliberate-abuse backstop).
    today = datetime.now(UTC).date().isoformat()
    if db.count_prompt_versions_on(con, today) >= DAILY_LEARN_CAP:
        raise LearnBusy(
            f"daily /learn cap reached ({DAILY_LEARN_CAP} runs today); try again tomorrow"
        )

    # 5. Gate: baseline (cached or first-run-established) vs the candidate, both via the real harness.
    baseline = _baseline_report(con, provider)
    candidate = _eval_prompt(candidate_text, provider)
    promote, reasons = _gate(candidate, baseline)

    # 6. Persist the verdict with its report; the composer loads the latest promoted on the next turn.
    version = db.next_prompt_version(con)
    status = "promoted" if promote else "rejected"
    db.insert_prompt_version(
        con,
        version=version,
        prompt_text=candidate_text,
        status=status,
        eval_report_json=candidate.to_json(),
    )
    return {
        "status": status,
        "version": version,
        "reason": "; ".join(reasons) if reasons else "no regressions — promoted",
        "report": _report_summary(candidate),
    }
