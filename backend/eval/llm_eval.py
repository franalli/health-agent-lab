"""eval/llm_eval.py — the offline LangSmith trace sink (architecture §8, Phase 5c).

Additive and cuttable. When `LANGSMITH_API_KEY` is set, the harness streams each eval case to LangSmith
for trace inspection: inputs (the member's question + context), the model output (the Mode-2 answer +
the deterministic escalation), latency/tokens/cost from the response metadata, and the deterministic
scorer verdicts attached as **feedback** — so a reviewer can eye per-case behaviour and run-over-run
drift in one place. The scorers stay the source of truth for pass/fail; LangSmith is the lens, not the
gate, and the local markdown + JSON report stays canonical. Two sinks mirror the canonical report:
`trace_run` (the composer/gate eval cases) and `trace_judge_run` (the Haiku feedback input-judge
battery, each case a run whose `fit_correct` feedback is 1.0 iff the verdict matched its label) — so
nothing in the local report is missing from LangSmith and there is no discrepancy to reconcile.

It is **tracing-only**: it wraps the harness's ALREADY-MADE service calls (this module is the *only*
importer of `langsmith` — the provider stays behind `llm.py`, no LangChain, and the live `/ask` path is
never instrumented in v1). It is **off unless `LANGSMITH_API_KEY` is set**, and used only for the
synthetic eval data here — traces go to a third-party SaaS, so real PHI would need self-hosted LangSmith
or a BAA plus the §6 data controls. Every call is defensive: a LangSmith failure logs a warning and is
swallowed, never breaking the (canonical) local run.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from eval.types import Case, CaseResponses, ScorerResult
from health_intelligence.config import JUDGE_MODEL

if TYPE_CHECKING:
    from eval.judge_eval import JudgeEvalResult

#: Default project name; override with LANGSMITH_PROJECT.
_DEFAULT_PROJECT = "health-intelligence-eval"


def is_enabled() -> bool:
    """The sink is on only when a LangSmith key is present (architecture §8: off by default)."""
    return bool(os.environ.get("LANGSMITH_API_KEY"))


def project_name() -> str:
    return os.environ.get("LANGSMITH_PROJECT", _DEFAULT_PROJECT)


def _open_sink():
    """Construct a synchronous LangSmith ``Client`` + the ``RunTree`` class, or ``None`` if the SDK import
    or client construction fails. Shared by ``trace_run`` and ``trace_judge_run`` so both sinks guard
    construction identically (a misconfigured endpoint is never fatal). ``auto_batch_tracing=False`` posts
    runs SYNCHRONOUSLY, so every error surfaces in the per-case try/except — no background ingest thread
    that would 403 at interpreter exit and crash the process even though the canonical report already
    wrote."""
    try:
        from langsmith import Client
        from langsmith.run_trees import RunTree

        return Client(auto_batch_tracing=False), RunTree
    except Exception as e:  # noqa: BLE001 — SDK import OR client construction; never fatal
        print(
            f"  [langsmith] sink unavailable ({type(e).__name__}: {e}); skipping trace sink."
        )
        return None


def trace_run(
    triples: list[tuple[Case, CaseResponses, list[ScorerResult]]],
    *,
    dataset: str,
    model_version: str,
) -> int:
    """Stream each case to LangSmith as one run with the scorer verdicts as feedback. Returns the number
    of cases traced (0 when the sink is off or unavailable). Never raises — a trace failure must not fail
    the canonical local run."""
    if not is_enabled():
        return 0
    opened = _open_sink()
    if opened is None:
        return 0
    client, RunTree = opened

    project = project_name()
    n = 0
    for case, responses, scores in triples:
        try:
            first = responses.mode2[0] if responses.mode2 else None
            rt = RunTree(
                name=f"eval:{case.id}",
                run_type="chain",
                project_name=project,
                client=client,  # type: ignore[call-arg]  # alias of ls_client; synchronous → no bg thread
                inputs={
                    "question": case.question,
                    "member_id": case.member_id,
                    "category": case.category,
                },
                outputs={
                    "answer": first.answer if first else None,
                    "escalation": first.escalation if first else None,
                    "disposition": first.answer_disposition if first else None,
                    "cited_markers": sorted(
                        {
                            ev.marker
                            for f_ in (first.findings if first else [])
                            for ev in f_.evidence
                        }
                    ),
                },
                extra={
                    "metadata": {
                        "dataset": dataset,
                        "model_version": model_version,
                        "tags": ",".join(case.tags),
                        "latency_ms": first.metadata.latency_ms if first else None,
                        "tokens": first.metadata.tokens if first else None,
                        "cost_usd": first.metadata.cost_usd if first else None,
                        "n_runs": len(responses.mode2),
                    }
                },
            )
            rt.end()
            rt.post()
            for s in scores:
                client.create_feedback(
                    run_id=rt.id,
                    key=s.dimension,
                    score=1.0 if s.passed else 0.0,
                    comment=(
                        s.detail
                        + (f" [never_event={s.never_event}]" if s.never_event else "")
                    ),
                )
            n += 1
        except Exception as e:  # noqa: BLE001 - one bad case must not abort the sink
            print(
                f"  [langsmith] failed to trace {case.id}: {type(e).__name__}: {str(e)[:140]}"
            )
            if n == 0:
                # The FIRST trace failed → a systematic config/auth issue (bad key, or the wrong
                # region/workspace — a 403 is typical of an EU key hitting the default US endpoint), not a
                # transient. Skip the rest rather than repeat the same error per case; the canonical report
                # is already written and unaffected.
                print(
                    "  [langsmith] first trace failed — skipping the sink. Check LANGSMITH_API_KEY, and set "
                    "LANGSMITH_ENDPOINT to your region (e.g. https://eu.api.smith.langchain.com) if non-US."
                )
                break
    if n:
        print(f"  [langsmith] traced {n} cases to project {project!r}.")
    return n


def trace_judge_run(result: JudgeEvalResult | None) -> int:
    """Stream the feedback input-judge battery to LangSmith — the sibling of :func:`trace_run` for the
    Haiku QUALITY gate, so the local report and LangSmith carry the SAME per-case data (no discrepancy to
    reconcile). Each case becomes one ``llm`` run whose ``fit_correct`` feedback is 1.0 iff the judge's
    verdict matched the label. Returns the number of cases traced (0 when the sink is off, unavailable, or
    the run was SKIPPED). Never raises — a trace failure must not fail the canonical local run."""
    if result is None or not is_enabled():
        return 0
    opened = _open_sink()
    if opened is None:
        return 0
    client, RunTree = opened

    project = project_name()
    n = 0
    for i, c in enumerate(result.cases):
        try:
            rt = RunTree(
                name=f"judge:{c.tag}:{i}",
                run_type="llm",
                project_name=project,
                client=client,  # type: ignore[call-arg]  # synchronous → no bg thread
                inputs={
                    "question": c.question,
                    "corrected_answer": c.answer,
                    "tag": c.tag,
                },
                outputs={
                    "fit": c.got_fit,
                    "expected_fit": c.want_fit,
                    "reason": c.reason,
                },
                extra={"metadata": {"kind": "feedback_judge", "model": JUDGE_MODEL}},
            )
            rt.end()
            rt.post()
            client.create_feedback(
                run_id=rt.id,
                key="fit_correct",
                score=1.0 if c.passed else 0.0,
                comment=f"want_fit={c.want_fit} got_fit={c.got_fit}: {c.reason}",
            )
            n += 1
        except Exception as e:  # noqa: BLE001 - one bad case must not abort the sink
            print(
                f"  [langsmith] failed to trace judge:{c.tag}:{i}: {type(e).__name__}: {str(e)[:140]}"
            )
            if n == 0:
                # First trace failed → systematic config/auth (see trace_run); skip the rest.
                print(
                    "  [langsmith] first judge trace failed — skipping the sink. Check LANGSMITH_API_KEY, "
                    "and set LANGSMITH_ENDPOINT to your region if non-US."
                )
                break
    if n:
        print(f"  [langsmith] traced {n} judge cases to project {project!r}.")
    return n
