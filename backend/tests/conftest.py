"""Shared fixtures for the Phase-4 (Mode 2) tests — a scripted fake LLM provider so the whole gate/ask
path runs offline, deterministically, with no network and no API key. The provider is the one network
seam (``llm.Provider``), so injecting a fake here is all it takes to exercise compose, the gate's retry
and fail-closed paths, and the LLMUnavailable degradation entirely in-process.
"""

from types import SimpleNamespace

import pytest

from health_intelligence import llm


class FakeProvider:
    """A scripted :class:`llm.Provider`. Each ``structured`` call pops the next scripted item: a Pydantic
    instance to return (paired with a synthetic usage), or an Exception to raise (to simulate
    :class:`~health_intelligence.llm.LLMParseError` / :class:`~health_intelligence.llm.LLMUnavailable`).
    Records every call so a test can assert what the gate or composer received — e.g. that the gate saw
    only the raw message."""

    def __init__(self, *scripted):
        self._queue = list(scripted)
        self.calls: list[SimpleNamespace] = []

    def structured(
        self, *, model, system, user, schema, max_tokens, tool_name, tool_description
    ):
        self.calls.append(
            SimpleNamespace(
                model=model,
                system=system,
                user=user,
                schema=schema,
                tool_name=tool_name,
                max_tokens=max_tokens,
            )
        )
        if not self._queue:
            raise AssertionError(
                "FakeProvider: more structured() calls than scripted results"
            )
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        # Mirror the real provider's contract: it always returns ``schema.model_validate(...)``, i.e. an
        # instance of ``schema``. Asserting it here keeps the fake honest — a test can't script a
        # type-mismatched object the AnthropicProvider could never produce.
        assert isinstance(item, schema), (
            f"FakeProvider scripted a {type(item).__name__}, but this call expects {schema.__name__}"
        )
        return item, llm.LLMUsage(model=model, input_tokens=12, output_tokens=8)


@pytest.fixture
def fake_provider():
    """Return the FakeProvider class; tests script it inline, e.g. ``fake_provider(GateClassification(...), draft)``."""
    return FakeProvider
