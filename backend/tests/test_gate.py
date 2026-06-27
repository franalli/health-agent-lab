"""Phase 4 — the input gate: the deterministic emergency-phrase backstop (pure, no model) and the LLM
classifier's routing, message-floor mapping, one bounded retry, and fail-closed ``couldnt_route``.

Everything runs offline via the scripted ``fake_provider`` (no network). Mirrors §52's "likely to break"
surface: the gate missing crisis/acute (recall), an invalid classification not retried once before
falling back, the ``couldnt_route`` fallback defaulting to ``none`` or dropping below
``clinician_review``, and — the load-bearing one — the emergency-phrase floor not taken in PARALLEL with
the gate (a jailbroken or dead gate must still floor a self-harm / acute message).
"""

import pytest

from health_intelligence import gate, llm
from health_intelligence.config import GATE_MAX_TOKENS, GATE_MODEL
from health_intelligence.gate import GateClassification, emergency_phrase_floor
from health_intelligence.llm import LLMParseError, LLMUnavailable, LLMUsage

# ---- the deterministic emergency-phrase floor (pure: no provider, no network) --------------------


def test_emergency_phrase_floor_fires_on_self_harm_and_acute_regardless_of_casing():
    for msg in (
        "I want to die",
        "I might kill myself",
        "I'm feeling suicidal",
        "I don't want to live",
        "CRUSHING CHEST PAIN since this morning",
        "I can't breathe",
        "I took an overdose",
        "there's severe bleeding",
    ):
        assert emergency_phrase_floor(msg) == "urgent", msg


def test_emergency_phrase_floor_abstains_on_ordinary_health_questions():
    for msg in (
        "what has changed in my ferritin?",
        "is my thyroid okay?",
        "should I worry about my cholesterol?",
        "give me an overview of my results",
    ):
        assert emergency_phrase_floor(msg) == "none", msg


def test_emergency_phrase_floor_survives_smart_apostrophes():
    # iOS/macOS/Word autocorrect ' -> U+2019; the deterministic backstop must still fire (it is a
    # *guarantee*, §564), or a curly-quote self-harm/acute message would silently get no urgent floor
    for msg in (
        "I can’t breathe",
        "I don’t want to live anymore",
        "I don’t want to be alive",
    ):
        assert emergency_phrase_floor(msg) == "urgent", msg


def test_emergency_phrase_floor_word_boundary_avoids_substring_false_positives():
    # "want to die" must NOT fire on "want to diet" (a core nutrition topic for a health assistant) —
    # word-boundary matching, not bare substring containment
    for msg in (
        "I want to diet to lower my cholesterol",
        "I don't want to diet anymore",
        "can you suggest a diet plan?",
    ):
        assert emergency_phrase_floor(msg) == "none", msg


def test_default_provider_without_a_key_degrades_instead_of_crashing(monkeypatch):
    # The anthropic SDK does not raise on a missing key at construction; default_provider() must catch
    # that explicitly and raise LLMUnavailable, so gate.classify degrades to couldnt_route rather than
    # letting a request-time TypeError escape as a 500 (the killswitch invariant).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_DEFAULT_PROVIDER", None)
    with pytest.raises(LLMUnavailable):
        llm.default_provider()
    # and end to end through the gate (provider defaulted, no key) -> fail closed, no crash
    res = gate.classify(
        "what's changed in my results?"
    )  # provider=None -> default_provider
    assert res.route == "couldnt_route" and res.msg_floor == "clinician_review"


# ---- the classifier: route -> message floor ------------------------------------------------------


@pytest.mark.parametrize(
    "route,expected_floor",
    [
        ("none", "none"),
        ("out_of_scope", "none"),
        ("acute_medical", "urgent"),
        ("crisis", "urgent"),
    ],
)
def test_route_maps_to_the_message_floor(fake_provider, route, expected_floor):
    # a benign message contributes no emergency-phrase floor, so msg_floor is the route's floor alone
    res = gate.classify(
        "how are my results looking?",
        provider=fake_provider(GateClassification(route=route)),
    )
    assert res.route == route and res.msg_floor == expected_floor
    assert res.usage is not None and res.usage.model == GATE_MODEL


def test_gate_sees_only_the_raw_message_no_clinical_context(fake_provider):
    prov = fake_provider(GateClassification(route="none"))
    gate.classify("is my thyroid okay?", provider=prov)
    call = prov.calls[0]
    assert (
        call.user == "is my thyroid okay?"
    )  # the raw message verbatim — no profile/labs/notes/analysis
    assert call.model == GATE_MODEL


# ---- two-tier fail-closed validation (architecture §98) ------------------------------------------


def test_one_parse_failure_is_repaired_by_the_bounded_retry(fake_provider):
    # a transient structured-output glitch resolves silently on the single retry — no escalation, no
    # interruption, the recovered route is used
    prov = fake_provider(
        LLMParseError("glitch"), GateClassification(route="out_of_scope")
    )
    res = gate.classify("can you change my metformin dose?", provider=prov)
    assert res.route == "out_of_scope" and len(prov.calls) == 2


def test_two_parse_failures_fail_closed_to_couldnt_route_at_clinician_review(
    fake_provider,
):
    prov = fake_provider(LLMParseError("a"), LLMParseError("b"))
    res = gate.classify("???", provider=prov)
    assert (
        res.route == "couldnt_route"
    )  # never defaults to none (free-composing a glitch)
    assert (
        res.msg_floor == "clinician_review"
    )  # the floor holds — clarification lives in the copy
    assert len(prov.calls) == 2  # exactly one bounded retry, then fall back


def test_provider_down_fails_closed_without_burning_a_retry(fake_provider):
    prov = fake_provider(LLMUnavailable("provider down"))
    res = gate.classify("what's changed?", provider=prov)
    assert res.route == "couldnt_route" and res.msg_floor == "clinician_review"
    assert (
        len(prov.calls) == 1
    )  # LLMUnavailable is degradation, not a parse glitch — no retry


def test_call_structured_preserves_billed_usage_when_retry_goes_unavailable(
    fake_provider,
):
    # parse-fail on attempt 1 (already billed), then provider-down on the bounded retry: call_structured
    # must propagate LLMUnavailable carrying the first attempt's usage, so a degrading caller can still
    # count it (§5 'count every billed attempt'; the bug silently dropped the accumulated total).
    u1 = LLMUsage(model=GATE_MODEL, input_tokens=10, output_tokens=5)
    prov = fake_provider(
        LLMParseError("glitch", usage=u1), LLMUnavailable("429 on the retry")
    )
    with pytest.raises(LLMUnavailable) as ei:
        llm.call_structured(
            prov,
            model=GATE_MODEL,
            system="s",
            user="u",
            schema=GateClassification,
            max_tokens=GATE_MAX_TOKENS,
            tool_name="t",
            tool_description="d",
        )
    assert ei.value.usage is not None
    assert ei.value.usage.total_tokens == u1.total_tokens  # 15 — not dropped
    assert len(prov.calls) == 2  # attempt + one bounded retry


def test_gate_counts_billed_tokens_on_parse_fail_then_unavailable(fake_provider):
    # the same path end to end through the gate: fail-closed at clinician_review, but the gate's usage
    # still reflects the tokens the parse-failed first attempt billed (not dropped on the LLMUnavailable).
    u1 = LLMUsage(model=GATE_MODEL, input_tokens=10, output_tokens=5)
    res = gate.classify(
        "what's changed?",
        provider=fake_provider(
            LLMParseError("glitch", usage=u1), LLMUnavailable("down")
        ),
    )
    assert res.route == "couldnt_route" and res.msg_floor == "clinician_review"
    assert res.usage is not None and res.usage.total_tokens == 15


# ---- the non-overridable emergency floor, taken in PARALLEL with the gate ------------------------


def test_emergency_floor_overrides_a_missed_or_jailbroken_gate(fake_provider):
    # the gate (faked) misclassifies a self-harm message as a normal question; the deterministic floor
    # still forces urgent. Floor-only by design: the (wrong) route is preserved, the FLOOR is not.
    res = gate.classify(
        "things are fine but I want to die",
        provider=fake_provider(GateClassification(route="none")),
    )
    assert res.route == "none"  # the gate's route is untouched (the documented v1 gap)
    assert (
        res.msg_floor == "urgent"
    )  # the phrase check raised it regardless of the gate


def test_emergency_floor_survives_a_dead_gate(fake_provider):
    # gate down -> couldnt_route (clinician_review), but the acute phrase floors it to urgent in parallel:
    # max(clinician_review, urgent) = urgent. The one routing where a miss is unacceptable is a guarantee.
    res = gate.classify(
        "crushing chest pain radiating into my arm",
        provider=fake_provider(LLMUnavailable("down")),
    )
    assert res.route == "couldnt_route"
    assert res.msg_floor == "urgent"
