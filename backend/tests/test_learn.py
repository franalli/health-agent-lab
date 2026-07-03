"""Phase 7 — the harness-gated prompt-promotion loop (``learn.py``).

Three layers, fast to slow:
  * the PURE drafter + checks (``assemble_candidate`` / ``structural_precheck``) — no DB, no model;
  * the deterministic GATE decision over constructed ``Report``s — the promote/reject logic in isolation;
  * one end-to-end ``run_learn`` over the real in-process harness with a constant fake provider — proof
    the loop wires together (InProcessClient -> run_eval -> gate -> db) with no network.
"""

import os

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
    sigs = [_incorrect("q1", "a valid clarification about your ferritin result")]
    assert learn.assemble_candidate(sigs) == learn.assemble_candidate(
        sigs
    )  # byte-identical (screening is deterministic, so purity holds)


def test_assemble_candidate_with_no_signals_is_the_base_prompt():
    assert learn.assemble_candidate([]) == llm.BASE_COMPOSE_SYSTEM


def test_assemble_candidate_appends_incorrect_exemplars():
    text = learn.assemble_candidate(
        [
            _incorrect(
                "why is my ferritin low",
                "please raise it with your GP at your next visit",
            )
        ]
    )
    assert text.startswith(llm.BASE_COMPOSE_SYSTEM)
    assert "why is my ferritin low" in text and "raise it with your GP" in text


def test_assemble_candidate_drops_a_length_degenerate_exemplar_at_assembly():
    # assemble_candidate stays a PURE function of the signal set, so its re-screen is length-only (the
    # SEMANTIC judge is submission-only, an LLM call a pure function can't make). A bypass row that is
    # OVERSIZED (a pre-bar / direct-insert row) is still dropped here by the length bound; a
    # semantically-bad-but-length-OK bypass row is NOT caught here — that is POST /reset's job (documented).
    marker = "OVERSIZED-JUNK-MARKER"
    oversized = marker + " " + ("padding " * 500)  # > _MAX_CORRECTION_CHARS
    junk = _incorrect("Tell me about my Fasting glucose", oversized)
    good = _incorrect(
        "why is my ferritin low", "please raise it with your GP at your next visit"
    )
    text = learn.assemble_candidate([junk, good])
    assert (
        marker not in text
    )  # the oversized exemplar was dropped by the length bound at assembly
    assert "raise it with your GP" in text  # the fit exemplar still folds


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


def test_classify_feedback_channels_only_learn_kinds_to_learn():
    # The authoritative per-kind classification the /feedback response exposes, so the UI can't claim a kind
    # "feeds /learn" when it doesn't. channel=="learn" IS "changes the composer prompt" (no redundant
    # `learnable` field), and it must agree with assemble_candidate: incorrect-with-both-halves -> exemplar,
    # a _TOGGLE_CLAUSES kind -> clause; everything else feeds the core/runtime or is advisory.
    def cls(kind, payload=None):
        return learn.classify_feedback(
            Feedback(kind=kind, target="f:x", payload=payload, source="clinician")
        )

    # channel 'learn' — the only channel that changes the composer prompt; no `learnable` field is emitted.
    inc = cls("incorrect", {"question": "q", "corrected_answer": "a valid one"})
    assert inc["channel"] == "learn" and "learnable" not in inc
    assert "Run learn" in inc["effect"]  # immediate call-to-action for incorrect
    rej = cls("escalation_reject")
    assert rej["channel"] == "learn"
    # CONDITIONAL CTA (not a premature "press Run learn"): the effect states the recurrence requirement.
    assert (
        "recurs" in rej["effect"] and str(learn._RECURRING_THRESHOLD) in rej["effect"]
    )

    # Derives from _TOGGLE_CLAUSES, not a hardcoded literal — every toggle-clause kind classifies to 'learn',
    # so a future 2nd toggle clause can't silently drift to 'advisory'.
    for kind in learn._TOGGLE_CLAUSES:
        assert cls(kind)["channel"] == "learn"

    # Advisory signals — recorded, but NOT a learn channel (the bug the copy overstated).
    assert cls("helpful")["channel"] == "advisory"
    assert cls("escalation_accept")["channel"] == "advisory"
    assert "does not feed /learn" in cls("escalation_accept")["effect"]

    # Corrections — feed the deterministic core / runtime composer, never the promoted prompt.
    assert cls("range_override", {"ref_high": 5})["channel"] == "core"
    assert cls("suppress_marker")["channel"] == "core"
    assert cls("preference", {"text": "be brief"})["channel"] == "composer"

    # An 'incorrect' missing a half can't become an exemplar -> advisory (matches assemble_candidate).
    assert cls("incorrect", {"question": "q"})["channel"] == "advisory"
    assert cls("incorrect", None)["channel"] == "advisory"


# --------------------------------------------------------------------------------------------------
# Input bar — deterministic bounds + the Haiku input-judge (the "learns too literally" fix). A
# deterministic-regex attempt at the cutoff/softening checks was abandoned (three review passes: it could
# not tell a cutoff from an age, or a reassurance from a negated one — "is this a fit exemplar?" is a
# SEMANTIC judgment). So the judge is an LLM call: these tests inject a FAKE provider to script its
# verdict, and pin the deterministic parts (bounds / dispatch / fail-closed) exactly. The judge's real
# judgment quality is, by construction, not deterministically testable.
# --------------------------------------------------------------------------------------------------


class _FakeJudge:
    """A provider whose ``structured()`` returns a scripted ``FeedbackJudgment`` (the judge's only call)."""

    def __init__(self, fit, reason=""):
        self._j = learn.FeedbackJudgment(fit=fit, reason=reason)
        self.calls = 0

    def structured(self, **kw):
        self.calls += 1
        return self._j, llm.LLMUsage(model=kw["model"], input_tokens=5, output_tokens=3)


def test_validate_feedback_rejects_when_the_judge_says_unfit():
    # The motivating failure — {question: "...Fasting glucose", corrected_answer: "makes no sense"}. The
    # judge classifies it unfit; validate_feedback surfaces the judge's reason (the route -> 422).
    fb = _incorrect("Tell me about my Fasting glucose", "makes no sense")
    reason = learn.validate_feedback(
        fb,
        provider=_FakeJudge(
            False, "incoherent placeholder — not an answer to the member"
        ),
    )
    assert reason == "incoherent placeholder — not an answer to the member"


def test_validate_feedback_accepts_when_the_judge_says_fit():
    fb = _incorrect(
        "why is my ferritin low",
        "It is mildly below the usual range; mention it to your GP.",
    )
    assert learn.validate_feedback(fb, provider=_FakeJudge(True)) is None


def test_validate_feedback_bounds_reject_before_any_model_call():
    # The deterministic bounds (empty / oversized) run BEFORE the judge — a provider that would blow up is
    # never reached, so the cheap rejects cost no Haiku call.
    class _Boom:
        def structured(self, **kw):
            raise AssertionError(
                "the judge must not be called for a deterministic bounds reject"
            )

    assert learn.validate_feedback(_incorrect("q", "   "), provider=_Boom()) is not None
    reason = learn.validate_feedback(
        _incorrect("q", "word " * 600), provider=_Boom()
    )  # ~3000 chars
    assert reason is not None and "too long" in reason


def test_validate_feedback_rejects_a_whitespace_padded_corrected_answer():
    # S1: the oversize bound measures the RAW length, not the stripped one — leading/trailing whitespace
    # padding must not smuggle a >cap payload past the 2000-char cap into the Haiku prompt or the stored row.
    class _Boom:
        def structured(self, **kw):
            raise AssertionError(
                "a padded oversize answer must reject before any model call"
            )

    padded = (
        (" " * 5000) + "See your GP about this soon."
    )  # stripped is short; raw >> _MAX_CORRECTION_CHARS
    reason = learn.validate_feedback(_incorrect("q", padded), provider=_Boom())
    assert reason is not None and "too long" in reason


def test_validate_feedback_bounds_the_question_field_too():
    # Both untrusted halves are length-bounded, not just corrected_answer: an oversized QUESTION (a direct
    # POST /feedback bypassing the UI could send megabytes) is rejected deterministically BEFORE it reaches
    # the Haiku judge's prompt or the stored row — the _Boom provider proves no model call happened.
    class _Boom:
        def structured(self, **kw):
            raise AssertionError(
                "the judge must not be called for an oversized question"
            )

    reason = learn.validate_feedback(
        _incorrect("x" * 9000, "See your GP about this soon."), provider=_Boom()
    )
    assert reason is not None and "too long" in reason and "question" in reason
    # A normal-length question still reaches the judge (fit -> accepted), proving the bound isn't over-tight.
    assert (
        learn.validate_feedback(
            _incorrect("is my HbA1c ok?", "It is mildly high; mention it to your GP."),
            provider=_FakeJudge(True),
        )
        is None
    )


def test_validate_feedback_only_judges_a_full_incorrect_exemplar():
    # Only an 'incorrect' with BOTH a question and a corrected_answer can become an exemplar; every other
    # shape is inert and must pass through WITHOUT a judge call (the fake would reject if wrongly hit).
    j = _FakeJudge(False, "would reject if called")
    assert (
        learn.validate_feedback(
            Feedback(kind="preference", payload={"text": "brief"}, source="member"),
            provider=j,
        )
        is None
    )
    assert (
        learn.validate_feedback(
            Feedback(kind="helpful", target="o:1", source="clinician"), provider=j
        )
        is None
    )
    assert (
        learn.validate_feedback(
            Feedback(kind="incorrect", target="o:1", payload=None, source="clinician"),
            provider=j,
        )
        is None
    )
    # corrected_answer present but NO question -> can't fold -> inert, not judged.
    assert (
        learn.validate_feedback(
            Feedback(
                kind="incorrect",
                payload={"corrected_answer": "a full answer here"},
                source="clinician",
            ),
            provider=j,
        )
        is None
    )
    assert (
        j.calls == 0
    )  # nothing above was ever an exemplar, so the judge was never called


def test_validate_feedback_bounds_an_escalation_reject_reason_without_the_judge():
    # The reject 'reason' is an AUDIT descriptor, never prompt text (the /learn tone clause is pre-written),
    # so its WHOLE screen is deterministic bounds — the judge must never be called for this kind.
    j = _FakeJudge(False, "would reject if called")
    assert (
        learn.validate_feedback(
            Feedback(
                kind="escalation_reject",
                target="o:1",
                payload={
                    "reason": "member is on supervised treatment; this trend is expected"
                },
                source="clinician",
            ),
            provider=j,
        )
        is None
    )
    # A payload-less programmatic reject stays storable — /learn's toggle counts recurrence by KIND, and
    # only the FORM requires the descriptor.
    assert (
        learn.validate_feedback(
            Feedback(kind="escalation_reject", target="o:1", source="clinician"),
            provider=j,
        )
        is None
    )
    # Present-but-blank: a filed false alarm that says nothing -> rejected, not silently stored.
    blank = learn.validate_feedback(
        Feedback(
            kind="escalation_reject", payload={"reason": "   "}, source="clinician"
        ),
        provider=j,
    )
    assert blank is not None and "empty" in blank
    # Oversized on the RAW length — whitespace padding can't smuggle past the cap (_screen_length's lesson).
    big = learn.validate_feedback(
        Feedback(
            kind="escalation_reject",
            payload={"reason": "x" + " " * (learn._MAX_REASON_CHARS + 100)},
            source="clinician",
        ),
        provider=j,
    )
    assert big is not None and "too long" in big
    # Non-string reasons are rejected outright (the old str() coercion bounds-checked the REPR of any
    # JSON type while the stored payload kept the non-string — a shape no screen ever validated).
    non_string = learn.validate_feedback(
        Feedback(kind="escalation_reject", payload={"reason": 42}, source="clinician"),
        provider=j,
    )
    assert non_string is not None and "must be text" in non_string
    assert j.calls == 0  # deterministic-only: no reason shape ever reaches the judge


def test_validate_feedback_fails_closed_when_the_judge_is_unavailable(monkeypatch):
    # No injected provider + no key -> default_provider raises -> LearnUnavailable (the route 503s). The
    # bar is the ONLY thing catching off-eval junk, so it fails CLOSED (retry), never storing unjudged text.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(learn.LearnUnavailable):
        learn.validate_feedback(
            _incorrect("q", "a real replacement answer for the member")
        )


# --------------------------------------------------------------------------------------------------
# The deterministic gate (constructed Reports — no harness, no model)
# --------------------------------------------------------------------------------------------------


def _sr(dim, passed, never=None):
    return ScorerResult(dimension=dim, passed=passed, never_event=never)


def _case(scorers, *, exp_route="none", obs_route="none", case_id="X"):
    return CaseReport(
        case_id=case_id,
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
    conftest FakeProvider, which would overflow on a full eval. Routes every message 'none'; the
    deterministic emergency-phrase floor + data floor still drive escalation, so the gate behaves
    identically on baseline and candidate runs (the point: prove the loop, not the LLM). The composed
    answer deliberately FABRICATES a number beside the two absent-marker traps' names (E16's B12, A10's
    insulin — both in the /learn gate subset), so every REAL-harness run with this fake is RED via the
    `fabricated_value` never-event: the e2e reject path keeps its teeth, and a first run never promotes
    (the debounce test depends on the active prompt staying v0). Before the gate ran on the critical
    composer subset, this role was played by E17's `unrefused_directive` (route-everything-'none'
    unrefuses the dose directive) — E17 measures the message gate, which /learn can't move, so it left
    the subset and the teeth moved to the grounding trap."""

    def __init__(self):
        self.calls = 0

    def structured(
        self,
        *,
        model,
        system,
        user,
        schema,
        max_tokens,
        tool_name,
        tool_description,
        repair=None,
    ):
        self.calls += 1
        if schema is GateClassification:
            obj: object = GateClassification(route="none")
        elif schema is ComposeDraft:
            obj = ComposeDraft(
                answer=(
                    "Here is a summary based on your own results. Your vitamin B12 looks to be "
                    "around 250, and your insulin is 12."
                ),
                answer_disposition="answered",
                cited_markers=[],
            )
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected schema {schema}")
        return obj, llm.LLMUsage(model=model, input_tokens=10, output_tokens=5)


def _seed_member_with_signal(con):
    learn.seed_baseline_prompt(
        con
    )  # a realistically-initialized DB: startup/reseed has seeded v0
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
    # The const fake states a number beside the absent-marker traps' names (E16's vitamin B12 / A10's
    # insulin — both IN the /learn gate's composer subset), so the REAL score_grounding fires
    # `fabricated_value` -> candidate is RED -> the gate rejects. This exercises the full chain
    # (InProcessClient -> run_eval -> real scorers -> _gate -> db) and proves the reject path has teeth
    # through the real harness, not just synthetic Reports. (The fake ignores the prompt, so baseline and
    # candidate are identical; the rejection is the harness surfacing a never-event, which is exactly
    # what must block a promotion. E17's unrefused-directive used to play this role — it left the subset
    # because it measures the message gate, which a composer prompt can't move.)
    con = fresh_con()
    _seed_member_with_signal(con)  # seeds v0 (init state) + a member + a signal
    fake = _ConstFake()

    # v0 pre-exists (the seed wrote it); /learn must NOT mint it — exactly one row before the run.
    assert con.execute("SELECT COUNT(*) FROM prompt_versions").fetchone()[0] == 1

    result = learn.run_learn(con, provider=fake)

    assert result["status"] == "rejected"
    assert fake.calls > 0  # the harness actually drove the composer/gate
    assert result["report"][
        "never_events"
    ]  # a real never-event surfaced and blocked it
    # v0 pre-existed; the run added EXACTLY its one rejected candidate row — never a second v0 (the
    # "reseed -> 1 row, each gating /learn -> +1" invariant; the old "2 rows from one run" is impossible).
    versions = [
        r["version"]
        for r in con.execute("SELECT version FROM prompt_versions ORDER BY version")
    ]
    assert versions == [0, result["version"]] and result["version"] >= 1
    # find_prompt_by_text sees the candidate text (the debounce key).
    candidate_text = learn.assemble_candidate(db.get_active_signals(con))
    assert db.find_prompt_by_text(con, candidate_text) is not None


def test_run_learn_self_heals_a_missing_baseline(caplog):
    # Resilience (review finding): the startup v0 seed is NON-fatal, so if it was swallowed (a transient
    # boot error) the table can be empty when /learn runs. Rather than bricking with a 503 for the rest of
    # the process, _baseline_report SELF-HEALS by seeding v0 — logging a WARNING so the anomaly is surfaced,
    # not masked — and completes the run. The "reseed -> 1 row, gating -> +1" invariant still holds on the
    # normal (v0-seeded) path; this only covers the failure path.
    con = fresh_con()  # NO seed_baseline_prompt -> prompt_versions empty (simulates a swallowed startup seed)
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

    with caplog.at_level("WARNING"):
        result = learn.run_learn(con, provider=_ConstFake())  # does NOT raise

    assert result["status"] in ("promoted", "rejected")  # the run completed
    assert "self-healing" in caplog.text  # the seeding anomaly was surfaced, not masked
    # v0 was self-healed and the candidate landed on top of it (exactly two rows: v0 + the candidate).
    versions = [
        r["version"]
        for r in con.execute("SELECT version FROM prompt_versions ORDER BY version")
    ]
    assert versions[0] == 0 and len(versions) == 2


def test_run_learn_promotes_and_the_composer_loads_it_when_the_gate_passes(monkeypatch):
    # The promote BRANCH + the version-aware read, isolated from the LLM: stub the harness to return a
    # clean report for both baseline and candidate so _gate passes. run_learn must persist the candidate
    # as promoted and db.get_active_prompt must then return it (what the composer resolves next turn).
    con = fresh_con()
    _seed_member_with_signal(con)
    clean = _report([_case(_ALL_PASS())])
    monkeypatch.setattr(learn, "_baseline_report", lambda con, provider, active: clean)
    monkeypatch.setattr(
        learn, "_eval_prompt", lambda seed_prompt, provider, **_kw: clean
    )

    result = learn.run_learn(con, provider=_ConstFake())

    assert result["status"] == "promoted" and result["version"] >= 1
    active = db.get_active_prompt(con)  # (version, prompt_text, eval_report_json)
    assert active is not None and active[0] == result["version"]
    assert (
        "a benign learned clarification" in active[1]
    )  # the version-aware composer would load this


def test_seed_baseline_prompt_materializes_v0_idempotently():
    # A freshly-seeded DB (the reseed / startup path) gets the v0 baseline row explicitly — promoted,
    # BASE text, no report yet — and re-seeding is a no-op (idempotent, never a duplicate or a second row).
    con = fresh_con()
    learn.seed_baseline_prompt(con)
    learn.seed_baseline_prompt(con)  # idempotent
    rows = con.execute(
        "SELECT version, status, prompt_text, eval_report_json FROM prompt_versions"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["version"] == 0 and rows[0]["status"] == "promoted"
    assert rows[0]["prompt_text"] == llm.BASE_COMPOSE_SYSTEM
    assert rows[0]["eval_report_json"] is None  # report is attached lazily by /learn
    # The composer resolves it as the v0 baseline (== the implicit BASE fallback).
    active = db.get_active_prompt(con)
    assert (
        active is not None and active[0] == 0 and active[1] == llm.BASE_COMPOSE_SYSTEM
    )


def test_write_baseline_prompt_preserves_report_on_unchanged_reseed_drops_on_change():
    # write_baseline_prompt's report handling (locks the ON CONFLICT ... CASE so the phantom-param cleanup
    # stays behavior-preserving): an idempotent re-seed with UNCHANGED text PRESERVES a lazily-attached
    # report (must not clobber it to NULL); a re-seed with CHANGED text DROPS the now-stale report AND
    # re-syncs the text (a persisted-disk redeploy that edits BASE re-syncs v0, dropping its stale report so
    # the next /learn recomputes the baseline against the new text).
    con = fresh_con()
    db.write_baseline_prompt(con, prompt_text="BASE-A")
    assert db.get_active_prompt(con) == (0, "BASE-A", None)
    db.set_prompt_report(con, 0, '{"r": 1}')  # attach a report, as /learn does lazily
    db.write_baseline_prompt(
        con, prompt_text="BASE-A"
    )  # unchanged text -> keep the report
    assert db.get_active_prompt(con) == (0, "BASE-A", '{"r": 1}')
    db.write_baseline_prompt(
        con, prompt_text="BASE-B"
    )  # changed text -> drop stale report + resync text
    assert db.get_active_prompt(con) == (0, "BASE-B", None)


def test_baseline_report_recomputes_when_the_stored_report_config_is_stale(monkeypatch):
    # S4: write_baseline_prompt drops v0's report only on a TEXT change, so a durable-disk redeploy that
    # bumps COMPOSE_MODEL/CONFIG_VERSION (text unchanged) leaves a report measured under the OLD config.
    # _baseline_report must RECOMPUTE it rather than gate a new-config candidate against an old-config
    # baseline. The _report helper stamps model_version="m" (!= the live COMPOSE_MODEL), i.e. a stale report.
    con = fresh_con()
    learn.seed_baseline_prompt(con)
    db.set_prompt_report(
        con, 0, _report([_case(_ALL_PASS())]).to_json()
    )  # stale-config report
    fresh = _report([_case(_ALL_PASS())])
    calls: list = []
    monkeypatch.setattr(
        learn,
        "_eval_prompt",
        lambda seed_prompt, provider, **_kw: (calls.append(1), fresh)[1],
    )
    learn._baseline_report(con, _ConstFake(), db.get_active_prompt(con))
    assert (
        calls
    )  # recomputed, because the stored report's model/config != the live config


def test_baseline_report_reuses_a_current_config_report(monkeypatch):
    # The complement: a stored report stamped with the LIVE model+config AND measured over the CURRENT
    # gate case set is reused as-is (no wasted re-eval).
    from eval.adapter import LEARN_GATE_CASE_IDS
    from health_intelligence.config import COMPOSE_MODEL, CONFIG_VERSION

    con = fresh_con()
    learn.seed_baseline_prompt(con)
    current = _report(
        [_case(_ALL_PASS(), case_id=i) for i in sorted(LEARN_GATE_CASE_IDS)]
    ).model_copy(
        update={"model_version": COMPOSE_MODEL, "config_version": CONFIG_VERSION}
    )
    db.set_prompt_report(con, 0, current.to_json())
    monkeypatch.setattr(
        learn,
        "_eval_prompt",
        lambda seed_prompt, provider, **_kw: (_ for _ in ()).throw(
            AssertionError("must not recompute a current-config baseline")
        ),
    )
    out = learn._baseline_report(con, _ConstFake(), db.get_active_prompt(con))
    assert (
        out.model_version == COMPOSE_MODEL
    )  # the stored current-config report was reused


def test_baseline_report_recomputes_when_the_case_set_differs(monkeypatch):
    # The case-set half of the like-for-like guard: a stored baseline with the LIVE model+config but
    # measured over a DIFFERENT case set (a pre-subset full-set report, or an older curation) must be
    # recomputed — comparing full-set dimension rates against subset rates is apples-to-oranges. This is
    # also the migration path for every DB whose v0 baseline predates LEARN_GATE_CASE_IDS.
    from health_intelligence.config import COMPOSE_MODEL, CONFIG_VERSION

    con = fresh_con()
    learn.seed_baseline_prompt(con)
    stale_ids = _report([_case(_ALL_PASS(), case_id="X")]).model_copy(
        update={"model_version": COMPOSE_MODEL, "config_version": CONFIG_VERSION}
    )
    db.set_prompt_report(con, 0, stale_ids.to_json())
    fresh = _report([_case(_ALL_PASS())])
    calls: list = []
    monkeypatch.setattr(
        learn,
        "_eval_prompt",
        lambda seed_prompt, provider, **_kw: (calls.append(1), fresh)[1],
    )
    learn._baseline_report(con, _ConstFake(), db.get_active_prompt(con))
    assert (
        calls
    )  # recomputed: same config, but the stored case set != the current gate subset


def test_learn_gate_subset_is_the_composer_movable_surface():
    # Pins the /learn gate's case surface: exactly LEARN_GATE_CASE_IDS on the shipped dataset, and every
    # case in it routes 'none' (the compose path) — the gate-routing/crisis/refusal cases are excluded
    # BY PRINCIPLE, since /learn changes only the composer prompt and cannot move the message gate or the
    # deterministic templates (their inclusion added cost + temp-0 noise, no detection power).
    from eval.adapter import LEARN_GATE_CASE_IDS, load_gate_cases

    subset = load_gate_cases(None)
    assert {c.id for c in subset} == LEARN_GATE_CASE_IDS
    assert 8 <= len(subset) <= 10  # small and critical — the whole point
    assert all(c.expected.route == "none" for c in subset)  # composer-surface only
    # The axes that must stay covered: a grounding trap, the urgent floor, and a negative control.
    by_id = {c.id: c for c in subset}
    assert by_id[
        "E16"
    ].expected.absent_marker  # fabricated_value never-event stays armed
    assert (
        "urgent" in by_id["E07"].expected.escalation
    )  # the urgent floor stays observed
    assert by_id["E02"].expected.escalation == ["none"]  # the negative control stays


def test_learn_gate_subset_falls_back_to_full_for_a_foreign_dataset(monkeypatch):
    # A hold-out dataset ships its OWN case ids; the curated id list encodes the shipped training_data
    # set, so when no supplied case matches, the gate runs the dataset's FULL set (more coverage, never
    # less). The added A-cases alone matching by id does NOT count — they reference training_data members
    # a foreign bundle may not carry.
    from eval import adapter

    foreign = [
        adapter._to_case(
            {
                "id": f"H{i:02d}",
                "member_id": "H01",
                "category": "grounded_qa",
                "input": "How are my results?",
                "expected_behavior": "answers",
                "must_include": [],
                "must_not": [],
                "escalation_expected": "none",
            }
        )
        for i in range(3)
    ]
    monkeypatch.setattr(adapter, "load_supplied_cases", lambda dataset=None: foreign)
    cases = adapter.load_gate_cases(None)
    assert {c.id for c in cases} >= {"H00", "H01", "H02"}  # the full foreign set runs
    assert len(cases) == 3 + len(
        adapter.ADDED_CASES
    )  # plus every added case — nothing filtered


def test_learn_gate_subset_falls_back_on_a_partial_id_collision(monkeypatch):
    # A hold-out reusing SOME generic ids (here one supplied 'E01' with foreign content) must NOT gate on
    # the tiny intersection: an ANY-overlap predicate was defeatable exactly this way — one colliding id
    # silently gated /learn on that case plus the mis-membered A10/A11. The fallback requires the FULL
    # expected supplied slice, so a single missing id runs the dataset's whole set instead.
    from eval import adapter

    foreign = [
        adapter._to_case(
            {
                "id": case_id,
                "member_id": "H01",
                "category": "grounded_qa",
                "input": "How are my results?",
                "expected_behavior": "answers",
                "must_include": [],
                "must_not": [],
                "escalation_expected": "none",
            }
        )
        for case_id in ("E01", "H01", "H02")  # one colliding id among foreign ids
    ]
    monkeypatch.setattr(adapter, "load_supplied_cases", lambda dataset=None: foreign)
    cases = adapter.load_gate_cases(None)
    assert {c.id for c in cases} >= {"E01", "H01", "H02"}  # the FULL foreign set runs
    assert len(cases) == 3 + len(
        adapter.ADDED_CASES
    )  # never the E01+A10+A11 intersection


def test_learn_after_a_seeded_v0_does_not_collide_and_appends_a_candidate(monkeypatch):
    # The correctness hinge of pre-seeding v0: a seeded v0 (no report) must NOT make _baseline_report
    # PK-collide on version 0. After seed -> run_learn, v0 exists exactly once (now carrying its lazily
    # attached report) and the promoted candidate is a NEW row at version >= 1 — the "reseed -> 1 v0,
    # learn -> +1 row" invariant.
    con = fresh_con()
    learn.seed_baseline_prompt(con)  # simulate the post-reseed / startup state
    _seed_member_with_signal(con)
    clean = _report([_case(_ALL_PASS())])
    monkeypatch.setattr(
        learn, "_eval_prompt", lambda seed_prompt, provider, **_kw: clean
    )

    result = learn.run_learn(con, provider=_ConstFake())

    assert result["status"] == "promoted" and result["version"] >= 1
    v0 = con.execute(
        "SELECT eval_report_json FROM prompt_versions WHERE version = 0"
    ).fetchall()
    assert len(v0) == 1  # exactly one v0 — no duplicate, no IntegrityError
    assert (
        v0[0]["eval_report_json"] is not None
    )  # the baseline report was attached lazily
    candidate = con.execute(
        "SELECT version FROM prompt_versions WHERE version = ?", (result["version"],)
    ).fetchone()
    assert candidate is not None  # the candidate landed as its own row


def test_eval_prompt_traces_the_harness_run(monkeypatch):
    # /learn's in-process gate streams each harness run to the SAME LangSmith sink the CLI uses (so a prod
    # /learn is observable — "how do we know it ran"), tagged learn-<label> to distinguish it from a CLI
    # eval:* run. Spy on the sink (no key/network needed); the harness itself runs offline via _ConstFake.
    from eval import llm_eval

    calls = []
    monkeypatch.setattr(
        llm_eval, "trace_report", lambda report, **kw: calls.append(kw) or 0
    )
    report = learn._eval_prompt(None, _ConstFake(), label="candidate")
    assert report is not None
    assert len(calls) == 1  # exactly one trace per harness run
    assert calls[0]["run_prefix"] == "learn-candidate"
    assert calls[0]["extra_metadata"] == {"source": "learn", "phase": "candidate"}


def test_run_learn_debounces_an_unchanged_signal_set():
    con = fresh_con()
    _seed_member_with_signal(con)
    learn.run_learn(con, provider=_ConstFake())  # first run does the work
    rows_before = con.execute("SELECT COUNT(*) FROM prompt_versions").fetchone()[0]

    fake2 = _ConstFake()
    again = learn.run_learn(con, provider=fake2)  # identical signals -> cached
    assert again.get("cached") is True
    assert fake2.calls == 0  # zero model calls on a debounced run
    # A debounced run writes NO new row (+0) — the third row-count invariant state (gating -> +1, but a
    # no-op/debounce -> +0), so the daily cap counts only real candidates.
    assert (
        con.execute("SELECT COUNT(*) FROM prompt_versions").fetchone()[0] == rows_before
    )


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


def test_run_learn_noop_names_active_corrections_on_the_other_path():
    """The empty-signals no-op is DIAGNOSTIC. A clinician who applied a range_override (a CORRECTION on
    the deterministic path) and then ran /learn should not read 'no signals' as '/learn picks up
    nothing' — the message names the active correction count and says it feeds the next Scan/Ask, not
    /learn (the symptom the debug report flagged)."""
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
    db.insert_feedback(
        con,
        "M1",
        Feedback(
            kind="range_override",
            target="HbA1c",
            payload={"ref_high": 7.0},
            source="clinician",
        ),
    )
    result = learn.run_learn(con, provider=_ConstFake())
    assert result["status"] == "noop" and result["version"] is None
    assert "1 active correction" in result["reason"]
    assert "not /learn" in result["reason"]


def test_run_learn_noop_explains_why_present_signals_are_inert():
    """The SECOND no-op — the one the new target dropdown steers into: pick 'incorrect', select a
    finding, submit. The control-panel form sends no {question, corrected_answer} payload, so the
    candidate == BASE. The message must name the CAUSE (a payload-less 'incorrect'), not just
    'byte-identical', or it reads as '/learn ignores my feedback' all over again."""
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
    # A signal exactly as the UI form posts it: a finding target, but NO payload.
    db.insert_feedback(
        con,
        "M1",
        Feedback(kind="incorrect", target="obs:x", payload=None, source="clinician"),
    )
    result = learn.run_learn(con, provider=_ConstFake())
    assert result["status"] == "noop"
    assert "incorrect" in result["reason"] and "payload" in result["reason"]


def test_run_learn_noop_names_a_length_screened_incorrect_accurately():
    # RR-A2 reconciled for the judge pivot: assembly is length-only, so the diagnostic's "screened" case is
    # an OVERSIZED complete 'incorrect' (a semantically-bad answer is a Haiku reject at submission, never
    # stored; a direct-insert oversized row is dropped at assembly -> candidate == BASE -> no-op). The
    # message must name the length bound (the row HAS a payload), not "carry no payload".
    con = fresh_con()
    learn.seed_baseline_prompt(
        con
    )  # v0 seeded, so the no-op BASE-branch is reached cleanly
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
    # A complete-but-oversized row inserted directly (bypassing the /feedback bar), as a pre-bar row.
    oversized = "padding " * 500  # > _MAX_CORRECTION_CHARS, both halves present
    db.insert_feedback(
        con, "M1", _incorrect("Tell me about my Fasting glucose", oversized)
    )
    result = learn.run_learn(con, provider=_ConstFake())
    assert result["status"] == "noop"
    assert "length bound" in result["reason"]
    assert "carry no" not in result["reason"]  # NOT misattributed as "no payload"


def test_run_learn_noop_explains_a_lone_helpful_is_advisory_only():
    """A 'helpful' signal toggles no clause no matter what it carries — so even after the form learned to
    send payloads, a lone 'helpful' still lands in the inert branch. The no-op must say so (advisory-only),
    not just 'byte-identical'."""
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
    db.insert_feedback(
        con, "M1", Feedback(kind="helpful", target="obs:x", source="clinician")
    )
    result = learn.run_learn(con, provider=_ConstFake())
    assert result["status"] == "noop"
    assert "advisory-only" in result["reason"]


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


# --------------------------------------------------------------------------------------------------
# Single-flight — the CROSS-PROCESS guard (a bare threading.Lock only serializes threads within one
# worker; db.process_lock extends single-flight across worker processes on the same host — LOAD-BEARING
# under the --workers 2 deploy, or the gate would double-run). flock locks the open file DESCRIPTION,
# so a second fd in THIS process models a second worker faithfully.
# --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(db.fcntl is None, reason="flock is POSIX-only")
def test_single_flight_blocks_a_concurrent_worker(tmp_path):
    con = fresh_con(str(tmp_path / "health.db"))
    lock_path = db._process_lock_path(con, learn.LEARN_LOCK_NAME)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    db.fcntl.flock(
        fd, db.fcntl.LOCK_EX | db.fcntl.LOCK_NB
    )  # stand in for another worker
    try:
        with pytest.raises(learn.LearnBusy):
            with learn._single_flight(con):
                pass  # pragma: no cover — the acquire raises before the body runs
    finally:
        db.fcntl.flock(fd, db.fcntl.LOCK_UN)
        os.close(fd)


def test_single_flight_releases_both_layers_so_a_later_run_reacquires(tmp_path):
    # No leak: after one run's context exits, BOTH the thread lock and the fd are freed, so the next
    # acquisition succeeds (a leaked fd would make this second acquire raise LearnBusy).
    con = fresh_con(str(tmp_path / "health.db"))
    with learn._single_flight(con):
        pass
    with learn._single_flight(con):
        pass
    assert not learn._LOCK.locked()  # thread lock fully released
