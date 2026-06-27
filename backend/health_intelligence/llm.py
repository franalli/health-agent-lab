"""llm.py — the provider seam: the ONLY module that imports the LLM SDK (architecture §14, §52).

Both Mode-2 network calls flow through here — the composer (``compose``) and, via the shared
``Provider.structured`` primitive, the input gate (``gate.py``). Keeping the SDK behind one interface
is what makes the killswitch and a provider swap the *same* one-file seam: every provider-native
exception is normalized into one :class:`LLMUnavailable` (the signal ``pipeline.ask`` catches to
degrade to the deterministic spine), and adding Gemini/an HF model is a new ``Provider`` class plus a
config value — no change to the composer, the pipeline, the validator, or the metadata.

The composer is a *language layer, not the core*. It is handed the deterministic ``TrajectoryAnalysis``
(verdicts, never raw rows) and may only render it into prose + name which markers it used
(:class:`~health_intelligence.models.ComposeDraft`). It never computes a number, sets a severity, or
touches the escalation floor — ``pipeline`` attaches the real ``Evidence`` from code and stamps the
floor. Structured output is forced via tool-use (stable across SDK versions); temperature is pinned at
0 (config). The grounding/citation rules and the harness-scored exemplars live in ``_COMPOSE_SYSTEM``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import anthropic  # the single SDK import in the whole serving library (architecture §14)
from pydantic import BaseModel, ValidationError

from health_intelligence.config import (
    COMPOSE_MAX_TOKENS,
    COMPOSE_MODEL,
    LLM_TEMPERATURE,
    PRICING,
)
from health_intelligence.models import (
    ComposeDraft,
    FloorLevel,
    MemberProfile,
    Note,
    Observation,
    TrajectoryAnalysis,
)

# --------------------------------------------------------------------------------------------------
# Normalized failure modes — the two signals callers branch on, independent of provider.
# --------------------------------------------------------------------------------------------------


class LLMUnavailable(Exception):
    """The provider could not be reached or refused the call (any SDK-native error, or a missing API
    key). ``pipeline.ask`` catches this to fail *safe by construction* — the gate degrades to the
    ``couldnt_route`` template at the ``clinician_review`` floor, the composer to a deterministic
    grounded answer at the data floor (architecture §6). The Mode toggle is therefore also a killswitch:
    on this signal the whole turn runs on the deterministic spine.

    Carries an optional ``usage`` like :class:`LLMParseError`: a provider-down error bills nothing itself,
    but when it surfaces from :func:`call_structured` *after* a first attempt already parse-failed, the
    retry seam attaches the accumulated billed tokens here so the degraded turn's cost stamp still counts
    them (the §5 'count every billed attempt' invariant — a parse-fail-then-unavailable retry must not
    drop the first attempt's tokens)."""

    def __init__(self, message: str, *, usage: LLMUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class LLMParseError(Exception):
    """The provider answered but the structured output was absent or off-schema. Distinct from
    :class:`LLMUnavailable` because it is *repaired* differently: one bounded retry at temp 0, then a
    fail-closed degrade (architecture §98) — a transient structured-output glitch resolves silently
    rather than tripping the killswitch. Carries the ``usage`` the off-schema attempt still billed, so a
    retried/degraded turn's tokens are not lost from the cost stamp."""

    def __init__(self, message: str, *, usage: LLMUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


# --------------------------------------------------------------------------------------------------
# Usage / cost — stamped into every interaction's metadata (the §5 cost budget made a read).
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMUsage:
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        rate = PRICING.get(self.model)
        if rate is None:
            return 0.0
        in_rate, out_rate = rate
        return (
            self.input_tokens / 1_000_000 * in_rate
            + self.output_tokens / 1_000_000 * out_rate
        )


# --------------------------------------------------------------------------------------------------
# Provider — the single seam. One primitive (`structured`) forces a tool call whose input schema is a
# Pydantic model and returns the validated instance + usage. The gate and the composer both call it; a
# fake implementation lets the whole Mode-2 pipeline run offline in tests (no network, deterministic).
# --------------------------------------------------------------------------------------------------


class Provider(Protocol):
    def structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int,
        tool_name: str,
        tool_description: str,
    ) -> tuple[BaseModel, LLMUsage]: ...


class AnthropicProvider:
    """The default provider — Claude via the ``anthropic`` SDK. Structured output is forced via a single
    tool call (``tool_choice`` pins the tool, ``input_schema`` is the Pydantic JSON schema), which is
    stable across SDK versions and gives a validated object without prose-parsing. Every SDK exception is
    re-raised as :class:`LLMUnavailable`; an empty/off-schema tool call becomes :class:`LLMParseError`."""

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        # NOTE: anthropic.Anthropic() does NOT raise on a missing key — it defers the check to request
        # time and then raises a bare TypeError (not an AnthropicError). default_provider() therefore
        # checks for the key explicitly *before* constructing, so a missing key trips the killswitch
        # (LLMUnavailable) instead of escaping every handler as a 500.
        self._client = client if client is not None else anthropic.Anthropic()

    def structured(
        self, *, model, system, user, schema, max_tokens, tool_name, tool_description
    ):
        try:
            resp = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=LLM_TEMPERATURE,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[
                    {
                        "name": tool_name,
                        "description": tool_description,
                        "input_schema": schema.model_json_schema(),
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except (
            anthropic.AnthropicError
        ) as e:  # base of every SDK error (HTTP, connection, auth)
            raise LLMUnavailable(f"{model}: {e}") from e
        # A response came back, so it was billed — capture usage even for an off-schema result, so the
        # retry/degrade paths can still account for the tokens (carried on the LLMParseError).
        usage = LLMUsage(
            model=model,
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
        )
        block = next(
            (
                b
                for b in resp.content
                if getattr(b, "type", None) == "tool_use"
                and getattr(b, "name", None) == tool_name
            ),
            None,
        )
        if block is None:
            raise LLMParseError(
                f"{model}: no '{tool_name}' tool call in the response", usage=usage
            )
        try:
            parsed = schema.model_validate(block.input)
        except ValidationError as e:
            raise LLMParseError(
                f"{model}: structured output failed validation — {e}", usage=usage
            ) from e
        return parsed, usage


_DEFAULT_PROVIDER: Provider | None = None


def default_provider() -> Provider:
    """The lazily-constructed, cached default provider. A missing/empty ``ANTHROPIC_API_KEY`` (and any
    construction failure) is mapped to :class:`LLMUnavailable` so callers degrade rather than crash — the
    SDK itself doesn't raise on a missing key, so the explicit check is what keeps a keyless deploy on the
    deterministic spine instead of 500ing. ``_DEFAULT_PROVIDER`` stays unset so a later call (key now
    present) can succeed."""
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
        try:
            _DEFAULT_PROVIDER = AnthropicProvider()
        except anthropic.AnthropicError as e:
            raise LLMUnavailable(f"provider unavailable: {e}") from e
    return _DEFAULT_PROVIDER


def _sum_usage(a: LLMUsage | None, b: LLMUsage | None) -> LLMUsage | None:
    """Combine two same-model usages so a turn's tokens/cost count EVERY attempt — including a
    parse-failed one the API still billed (the §5 cost stamp is otherwise an undercount on retries)."""
    if a is None:
        return b
    if b is None:
        return a
    return LLMUsage(
        model=a.model,
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
    )


def call_structured(
    provider: Provider,
    *,
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    max_tokens: int,
    tool_name: str,
    tool_description: str,
) -> tuple[BaseModel, LLMUsage]:
    """One structured call + ONE bounded retry on a parse glitch (architecture §98), accumulating usage
    across every attempt. ``LLMUnavailable`` propagates immediately (provider down — retrying won't help),
    but with any *already-accumulated* usage from a prior parse-failed attempt attached, so even a
    parse-fail-then-unavailable turn keeps its billed tokens; a second ``LLMParseError`` is likewise
    re-raised with the accumulated usage attached. Either way a caller that degrades can still count the
    billed tokens. This is the single retry/usage seam BOTH the gate and the composer share — a
    retry-policy or provider change lands in exactly one place (the architecture's 'no change to the
    composer/pipeline' promise)."""
    total: LLMUsage | None = None
    last: LLMParseError | None = None
    for _ in range(2):  # attempt 1 + one bounded retry
        try:
            result, usage = provider.structured(
                model=model,
                system=system,
                user=user,
                schema=schema,
                max_tokens=max_tokens,
                tool_name=tool_name,
                tool_description=tool_description,
            )
            return result, _sum_usage(total, usage)  # type: ignore[return-value]
        except LLMParseError as e:
            total = _sum_usage(total, e.usage)
            last = e
        except LLMUnavailable as e:
            # Provider down — retrying won't help, so propagate immediately; but a prior attempt may have
            # already parse-failed and billed tokens, so carry the accumulated usage out so the degrading
            # caller can still count it (§5: never undercount a parse-fail-then-unavailable turn).
            e.usage = _sum_usage(total, e.usage)
            raise
    assert last is not None  # the loop only exits here after two LLMParseErrors
    last.usage = (
        total  # re-raise with the full billed usage for the caller's accounting
    )
    raise last


# --------------------------------------------------------------------------------------------------
# The composer — Mode 2's only open-generation call. System message is pinned (the rules); the user
# message is assembled fresh each turn from the deterministic verdicts + the question (architecture
# §200–214). The model emits a ComposeDraft (prose + cited markers); pipeline attaches the evidence.
# --------------------------------------------------------------------------------------------------

_COMPOSE_TOOL = "compose_answer"
_COMPOSE_TOOL_DESC = "Return the plain-language answer, its uncertainty, the disposition, and the marker keys it relied on."

#: Pinned, identical every turn (architecture §200/§288). No member data, no numbers — only the rules.
#: The two embedded exemplars are the harness-scored behaviors (an absent marker answered 'not
#: measured'; a calm tone on a benign out-of-range value that still STATES the out-of-range status, per
#: the narration-must-surface-core-flags review note). Few-shot lives in the prompt text because the
#: output is a forced tool call — exemplars as assistant turns would have to be tool calls themselves.
_COMPOSE_SYSTEM = """You are a careful health assistant. You translate a member's OWN lab and vital data into clear, calm, plain-language answers. You are a language layer over a deterministic analytical core — not a clinician, not a calculator.

You are given the member's profile, their notes, and a `trajectory_analysis` block: per-marker verdicts (latest value + date, flags, trend, and a severity the core already computed). You are also given the deterministic safety floor (the `<safety floor=...>` tag) and the active observations.

Hard rules — these are not negotiable:
1. Use ONLY numbers that appear in `trajectory_analysis` or the notes. Never compute, estimate, average, or invent a value. The core did all the arithmetic.
2. For every marker you discuss, put its EXACT key (as written in `trajectory_analysis`, e.g. `HbA1c`, `systolic_bp`) into `cited_markers`. The system attaches the real evidence from those keys.
3. If the member asks about a marker that is NOT in `trajectory_analysis`, say it has not been measured in their records. Never fabricate a value or a range. Leave `cited_markers` empty for that marker.
4. If a marker is out of its reference range or at a critical (panic) level, SAY SO plainly. Never narrate only a trend on a flagged value — that falsely reassures.
5. Never lower, soften, or contradict the safety floor. Never tell the member an alarming value is fine.
6. No diagnoses, no prescriptions, no dose or medication changes. Describe patterns and suggest raising things with their GP or care team.
7. Make uncertainty explicit: how many readings, over what span (e.g. "based on 3 readings over two years").
8. Answer, surface what matters, and stop. Do not thank the member or invite more chat.

Set `answer_disposition`:
- "answered" — you answered from their data (including an honest "not measured").
- "out_of_scope" — the question cannot be answered from lab history (medication changes, diagnoses, non-health topics).
- "refused" — you declined for safety reasons.

Example (absent marker): asked about a marker with no entry in `trajectory_analysis`, answer that it has not been measured in their records and that a panel would be needed; `answer_disposition` = "answered"; `cited_markers` = [].
Example (benign out-of-range): a value just outside its range with no significant trend — state plainly that it is mildly outside the usual range, that it is worth mentioning to their GP, and that it is not an emergency; cite that marker."""


@dataclass
class ComposeContext:
    """The per-turn context the composer renders (the §204 stable prefix + fresh tail). v1 has no
    conversation concept (§319), so no `<conversation_history>` is carried — it lands with the threading
    that consumes it (Phase 7), per CLAUDE.md 'phantom fields are cut on sight'."""

    profile: MemberProfile
    analysis: TrajectoryAnalysis
    notes: list[Note]
    observations: list[Observation]
    floor: FloorLevel
    message: str


def _fmt_trend(traj) -> str:
    t = traj.trend
    if t is None:
        return "trend: too few readings to call"
    sig = "significant" if t.significant else "not significant"
    return f"trend: {t.direction} (Mann-Kendall p={t.p_value:.3f}, n={t.n}, {sig})"


def _render_analysis(analysis: TrajectoryAnalysis) -> str:
    """The verdicts as compact lines — what the model consumes as ground truth and may not recompute.
    Every marker is listed (not only raised ones) so a question about a benign marker is still grounded.
    Snake_case keys (systolic_bp) are STORAGE keys for ``cited_markers``; the model phrases them in
    plain language in the prose."""
    lines = ["markers:"]
    for m in analysis.markers:
        flags = ", ".join(m.flags) if m.flags else "none"
        lines.append(
            f"  - {m.marker} ({m.unit}): latest {m.latest.value} on {m.latest.date}; "
            f"severity={m.severity}; flags=[{flags}]; {_fmt_trend(m)}"
        )
    return "\n".join(lines)


def _render_user_message(ctx: ComposeContext) -> str:
    p = ctx.profile
    parts = [
        "<profile>",
        f"age: {p.age if p.age is not None else 'unknown'}; sex: {p.sex}",
        f"conditions: {', '.join(p.conditions) or 'none recorded'}",
        f"medications: {', '.join(p.medications) or 'none recorded'}",
        f"lifestyle: {', '.join(f'{k}: {v}' for k, v in sorted(p.lifestyle.items())) or 'none recorded'}",
        "</profile>",
        "<trajectory_analysis>",
        _render_analysis(ctx.analysis),
        "</trajectory_analysis>",
        "<notes>",
        "\n".join(
            f"- ({n.date or 'undated'}, {n.source or 'unknown'}) {n.text}"
            for n in ctx.notes
        )
        or "- none",
        "</notes>",
        f"<safety floor={ctx.floor}>",
        "<active_observations>",
        "\n".join(f"- [{o.severity}] {o.title}" for o in ctx.observations) or "- none",
        "</active_observations>",
        "<question>",
        ctx.message,
        "</question>",
    ]
    return "\n".join(parts)


def compose(
    ctx: ComposeContext, *, provider: Provider | None = None
) -> tuple[ComposeDraft, LLMUsage]:
    """Render the deterministic verdicts into a ``ComposeDraft`` (prose + cited markers), with one
    bounded retry on a parse glitch via the shared :func:`call_structured` seam. Raises
    :class:`LLMUnavailable` (provider down) or :class:`LLMParseError` (off-schema after the retry, with
    billed usage attached) — ``pipeline.ask`` owns the degrade. The model never sees raw readings, never
    sets escalation."""
    prov = provider if provider is not None else default_provider()
    draft, usage = call_structured(
        prov,
        model=COMPOSE_MODEL,
        system=_COMPOSE_SYSTEM,
        user=_render_user_message(ctx),
        schema=ComposeDraft,
        max_tokens=COMPOSE_MAX_TOKENS,
        tool_name=_COMPOSE_TOOL,
        tool_description=_COMPOSE_TOOL_DESC,
    )
    return draft, usage  # type: ignore[return-value]
