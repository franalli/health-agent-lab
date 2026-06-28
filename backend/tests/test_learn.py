"""Phase 7 — the harness-gated prompt-promotion loop (``learn.py``).

Three layers, fast to slow:
  * the PURE drafter + checks (``assemble_candidate`` / ``structural_precheck``) — no DB, no model;
  * the deterministic GATE decision over constructed ``Report``s — the promote/reject logic in isolation;
  * one end-to-end ``run_learn`` over the real in-process harness with a constant fake provider — proof
    the loop wires together (InProcessClient -> run_eval -> gate -> db) with no network.
"""

import pytest
from builders import fresh_con, make_bundle, make_panel, make_result

from eval.report import CaseReport, Report
from eval.types import ScorerResult
from health_intelligence import db, learn, llm
from health_intelligence.gate import GateClassification
from health_intelligence.models import ComposeDraft, Feedback
from preprocessing.ingest import ingest_bundle

# --------------------------------------------------------------------------------------------------
# Pure assembly + checks
# --------------------------------------------------------------------------------------------------


def _incorrect(q, a):
    return Feedback(
        kind="incorrect",
        target="f:x",
        payload={"question": q, "corrected_answer": a},
        source="clinician",
    )


def test_assemble_candidate_is_a_pure_function_of_the_signal_set():
    sigs = [_incorrect("q1", "a1")]
    assert learn.assemble_candidate(sigs) == learn.assemble_candidate(
        sigs
    )  # byte-identical


def test_assemble_candidate_with_no_signals_is_the_base_prompt():
    assert learn.assemble_candidate([]) == llm.BASE_COMPOSE_SYSTEM


def test_assemble_candidate_appends_incorrect_exemplars():
    text = learn.assemble_candidate(
        [_incorrect("why is my ferritin low", "see your GP")]
    )
    assert text.startswith(llm.BASE_COMPOSE_SYSTEM)
    assert "why is my ferritin low" in text and "see your GP" in text


def test_toggle_clause_requires_recurrence():
    one = [Feedback(kind="escalation_reject", source="clinician")]
    assert (
        learn.assemble_candidate(one) == llm.BASE_COMPOSE_SYSTEM
    )  # a single signal is noise
    two = [Feedback(kind="escalation_reject", source="clinician") for _ in range(2)]
    assert (
        learn.assemble_candidate(two) != llm.BASE_COMPOSE_SYSTEM
    )  # recurring -> toggled on


def test_structural_precheck_accepts_base_and_rejects_malformed():
    assert learn.structural_precheck(llm.BASE_COMPOSE_SYSTEM) is None
    assert learn.structural_precheck("a candidate with no safety clauses") is not None
    assert learn.structural_precheck(llm.BASE_COMPOSE_SYSTEM + "x" * 20_000) is not None


# --------------------------------------------------------------------------------------------------
# The deterministic gate (constructed Reports — no harness, no model)
# --------------------------------------------------------------------------------------------------


def _sr(dim, passed, never=None):
    return ScorerResult(dimension=dim, passed=passed, never_event=never)


def _case(scorers, *, exp_route="none", obs_route="none"):
    return CaseReport(
        case_id="X",
        category="c",
        tags=[],
        mode1_covered=False,
        mode2=scorers,
        expected_route=exp_route,
        observed_route=obs_route,
    )


def _report(cases):
    return Report(
        dataset="d",
        model_version="m",
        config_version="v",
        n_runs=1,
        generated_at="t",
        cases=cases,
        stats=[],
        raw=[],
    )


_ALL_PASS = lambda: [  # noqa: E731
    _sr("escalation", True),
    _sr("routing", True),
    _sr("grounding", True),
    _sr("consistency", True),
]


def test_gate_promotes_when_nothing_regresses():
    ok, reasons = learn._gate(
        _report([_case(_ALL_PASS())]), _report([_case(_ALL_PASS())])
    )
    assert ok and reasons == []


def test_gate_rejects_a_candidate_with_a_never_event():
    base = _report([_case([_sr("grounding", True)])])
    cand = _report([_case([_sr("grounding", False, never="fabricated_value")])])
    ok, reasons = learn._gate(cand, base)
    assert not ok and any("RED" in r for r in reasons)


def test_gate_rejects_a_dimension_regression():
    base = _report([_case([_sr("grounding", True)])])
    cand = _report([_case([_sr("grounding", False)])])  # 0/1 < 1/1, no never-event
    ok, reasons = learn._gate(cand, base)
    assert not ok and any("grounding" in r for r in reasons)


def test_gate_rejects_a_safety_recall_regression():
    base = _report(
        [_case([_sr("routing", True)], exp_route="crisis", obs_route="crisis")]
    )
    cand = _report(
        [_case([_sr("routing", False)], exp_route="crisis", obs_route="none")]
    )
    ok, reasons = learn._gate(cand, base)
    assert not ok and any("recall" in r for r in reasons)


def test_gate_allows_an_improvement():
    base = _report([_case([_sr("grounding", False)])])
    cand = _report([_case([_sr("grounding", True)])])
    ok, _ = learn._gate(cand, base)
    assert ok


# --------------------------------------------------------------------------------------------------
# End-to-end run_learn (real in-process harness, constant fake provider — no network)
# --------------------------------------------------------------------------------------------------


class _ConstFake:
    """A provider that returns a valid instance for ANY number of calls — unlike the queue-based
    conftest FakeProvider, which would overflow on a full eval. Routes every message 'none' and composes
    a benign answer; the deterministic emergency-phrase floor + data floor still drive escalation, so the
    gate behaves identically on baseline and candidate runs (the point: prove the loop, not the LLM)."""

    def __init__(self):
        self.calls = 0

    def structured(
        self, *, model, system, user, schema, max_tokens, tool_name, tool_description
    ):
        self.calls += 1
        if schema is GateClassification:
            obj: object = GateClassification(route="none")
        elif schema is ComposeDraft:
            obj = ComposeDraft(
                answer="Here is a summary based on your own results.",
                answer_disposition="answered",
                cited_markers=[],
            )
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected schema {schema}")
        return obj, llm.LLMUsage(model=model, input_tokens=10, output_tokens=5)


def _seed_member_with_signal(con):
    ingest_bundle(
        con,
        make_bundle(
            "M1",
            [
                make_panel(
                    "M1-P1", "2024-01-15", [make_result("HbA1c", 5.3, "%", "<5.7")]
                )
            ],
        ),
    )
    db.insert_feedback(con, "M1", _incorrect("q", "a benign learned clarification"))


def test_run_learn_rejects_via_a_real_never_event_end_to_end():
    # The const fake routes EVERY message 'none', so the labeled E17 ("change my dose") is answered
    # instead of refused -> the REAL score_routing fires `unrefused_directive` -> candidate is RED ->
    # the gate rejects. This exercises the full chain (InProcessClient -> run_eval -> real scorers ->
    # _gate -> db) and proves the reject path has teeth through the real harness, not just synthetic
    # Reports. (The fake ignores the prompt, so baseline and candidate are identical; the rejection is
    # the harness surfacing a never-event, which is exactly what must block a promotion.)
    con = fresh_con()
    _seed_member_with_signal(con)
    fake = _ConstFake()

    result = learn.run_learn(con, provider=fake)

    assert result["status"] == "rejected"
    assert fake.calls > 0  # the harness actually drove the composer/gate
    assert result["report"][
        "never_events"
    ]  # a real never-event surfaced and blocked it
    # The v0 baseline row was established lazily and the rejected candidate row was written.
    versions = [
        r["version"]
        for r in con.execute("SELECT version FROM prompt_versions ORDER BY version")
    ]
    assert 0 in versions and result["version"] in versions
    # find_prompt_by_text sees the candidate text (the debounce key).
    candidate_text = learn.assemble_candidate(db.get_active_signals(con))
    assert db.find_prompt_by_text(con, candidate_text) is not None


def test_run_learn_promotes_and_the_composer_loads_it_when_the_gate_passes(monkeypatch):
    # The promote BRANCH + the version-aware read, isolated from the LLM: stub the harness to return a
    # clean report for both baseline and candidate so _gate passes. run_learn must persist the candidate
    # as promoted and db.get_active_prompt must then return it (what the composer resolves next turn).
    con = fresh_con()
    _seed_member_with_signal(con)
    clean = _report([_case(_ALL_PASS())])
    monkeypatch.setattr(learn, "_baseline_report", lambda con, provider: clean)
    monkeypatch.setattr(learn, "_eval_prompt", lambda seed_prompt, provider: clean)

    result = learn.run_learn(con, provider=_ConstFake())

    assert result["status"] == "promoted" and result["version"] >= 1
    active = db.get_active_prompt(con)  # (version, prompt_text, eval_report_json)
    assert active is not None and active[0] == result["version"]
    assert (
        "a benign learned clarification" in active[1]
    )  # the version-aware composer would load this


def test_run_learn_debounces_an_unchanged_signal_set():
    con = fresh_con()
    _seed_member_with_signal(con)
    learn.run_learn(con, provider=_ConstFake())  # first run does the work

    fake2 = _ConstFake()
    again = learn.run_learn(con, provider=fake2)  # identical signals -> cached
    assert again.get("cached") is True
    assert fake2.calls == 0  # zero model calls on a debounced run


def test_run_learn_is_a_noop_without_signals():
    con = fresh_con()
    ingest_bundle(
        con,
        make_bundle(
            "M1",
            [
                make_panel(
                    "M1-P1", "2024-01-15", [make_result("HbA1c", 5.3, "%", "<5.7")]
                )
            ],
        ),
    )
    result = learn.run_learn(con, provider=_ConstFake())
    assert result["status"] == "noop"


def test_run_learn_refuses_a_degraded_run_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    con = fresh_con()
    _seed_member_with_signal(con)
    with pytest.raises(learn.LearnUnavailable):
        learn.run_learn(
            con
        )  # provider=None + no key -> refuse, don't gate a degraded run


def test_find_prompt_by_text_ignores_reverted_so_reset_then_relearn_works():
    # The debounce must cache only a GATE VERDICT (promoted/rejected), never a reverted row — else a
    # post-/reset re-learn of identical feedback short-circuits to 'reverted' and never re-promotes.
    con = fresh_con()
    db.insert_prompt_version(
        con, version=1, prompt_text="X", status="promoted", eval_report_json="{}"
    )
    assert db.find_prompt_by_text(con, "X") is not None  # a promoted verdict is cached
    db.reset_learning(con)  # /reset flips v1 promoted -> reverted
    assert (
        db.find_prompt_by_text(con, "X") is None
    )  # reverted is NOT a cached verdict -> re-learnable


def test_reset_reverts_a_rejected_verdict_so_it_re_gates():
    # A rejected verdict is baseline-relative; /reset changes the baseline, so a previously-rejected
    # candidate must be re-gateable afterward (not stuck cached as 'rejected').
    con = fresh_con()
    db.insert_prompt_version(
        con, version=1, prompt_text="Y", status="rejected", eval_report_json="{}"
    )
    assert (
        db.find_prompt_by_text(con, "Y") is not None
    )  # rejected cached within a session (debounce)
    db.reset_learning(con)
    assert (
        db.find_prompt_by_text(con, "Y") is None
    )  # after /reset, re-gateable vs the new baseline
