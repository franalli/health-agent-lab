"""eval/llm_eval.trace_report — the report-shaped sink entry point shared by the ``make eval`` CLI and
``POST /learn`` (so both instrument IDENTICALLY). The LangSmith network path itself (``trace_run``'s
``RunTree`` posting) is external SaaS and stays untested; these pin the DELEGATION contract: off without a
key, and it threads ``run_prefix`` + builds the per-case triples from the report."""

import types

from eval import llm_eval


def _stub_report():
    """A duck-typed Report: ``trace_report`` reads ``.raw`` (each ``.case`` / ``.responses``), ``.cases``
    (``.mode2``), ``.dataset``, ``.model_version`` — nothing else, so a namespace stub is faithful."""
    case = types.SimpleNamespace(id="A01")
    return types.SimpleNamespace(
        raw=[types.SimpleNamespace(case=case, responses="R")],
        cases=[types.SimpleNamespace(mode2=["S"])],
        dataset="training_data",
        model_version="m",
    )


def test_trace_report_is_a_noop_without_a_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    called = []
    monkeypatch.setattr(llm_eval, "trace_run", lambda *a, **k: called.append(k) or 0)
    assert llm_eval.trace_report(_stub_report()) == 0
    assert (
        called == []
    )  # short-circuits BEFORE delegating (no triples built, no trace_run)


def test_trace_report_delegates_with_prefix_and_builds_triples(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    captured = {}

    def _fake_trace_run(triples, *, dataset, model_version, run_prefix, extra_metadata):
        captured.update(
            triples=triples,
            dataset=dataset,
            model_version=model_version,
            run_prefix=run_prefix,
            extra_metadata=extra_metadata,
        )
        return len(triples)

    monkeypatch.setattr(llm_eval, "trace_run", _fake_trace_run)
    n = llm_eval.trace_report(
        _stub_report(), run_prefix="learn-candidate", extra_metadata={"source": "learn"}
    )
    assert n == 1
    assert (
        captured["run_prefix"] == "learn-candidate"
    )  # distinguishes /learn runs in the shared project
    assert captured["dataset"] == "training_data"
    assert captured["extra_metadata"] == {"source": "learn"}
    # the triple is built from report.raw[i].case / .responses + report.cases[i].mode2
    ((case, responses, scores),) = captured["triples"]
    assert case.id == "A01" and responses == "R" and scores == ["S"]


def test_trace_report_default_prefix_is_eval(monkeypatch):
    # The CLI (eval/__main__) calls trace_report(report) with no prefix -> "eval:*", unchanged behavior.
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    captured = {}
    monkeypatch.setattr(llm_eval, "trace_run", lambda *a, **k: captured.update(k) or 0)
    llm_eval.trace_report(_stub_report())
    assert captured["run_prefix"] == "eval"
    assert captured["extra_metadata"] is None
