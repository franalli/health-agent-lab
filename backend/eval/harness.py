"""eval/harness.py — the runner (architecture §8): ``run_eval(cases, client, cfg) -> Report``.

A thin orchestrator over the pure scorers: for each case it drives the live service N times (Mode 2)
and once (Mode 1, byte-identical so a second call only proves it), collects the responses + the
post-run DB escalation snapshot + the analysis marker-value snapshot, and runs every deterministic
scorer. ``score_stats`` grades the pure core on the authored fixtures separately (the second, distinct
label source). The result is a ``Report`` whose JSON is the regression-gate artifact.

The runner itself stays deterministic in structure (the LLM nondeterminism lives inside the service
calls, which is exactly what ``score_consistency`` measures). All judgement lives in the scorers.
"""

from __future__ import annotations

from datetime import UTC, datetime

from eval.client import ServiceClient
from eval.report import CaseReport, RawCase, Report
from eval.scorers import score_mode1, score_mode2, score_stats
from eval.stats_fixtures import STATS_FIXTURES
from eval.types import Case, CaseResponses, EvalConfig
from health_intelligence.config import COMPOSE_MODEL, CONFIG_VERSION
from preprocessing.datasets import dataset_dir


def _collect(case: Case, client: ServiceClient, cfg: EvalConfig) -> CaseResponses:
    """Drive the live service for one case inside its own isolated DB. The N Mode-2 asks run first (they
    write the audit + escalation rows), then the DB snapshot and the Mode-1 overview are read back."""

    def _safe(label, fn, default):
        """A post-ask read failure degrades to a default (and warns) — it must NOT abort the run, the
        same tolerance the per-ask loop already applies. Reverting to no-guard would let one /suggestions
        or /escalations edge case discard every already-scored case."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print(f"  [{case.id}] {label} failed: {type(e).__name__}: {e}")
            return default

    with client.case_session(case.member_id) as s:
        mode2 = []
        for _ in range(cfg.n_runs):
            try:
                mode2.append(s.ask(case.question))
            except (
                Exception
            ) as e:  # one failed run must not abort the case  # noqa: BLE001
                print(f"  [{case.id}] ask failed: {type(e).__name__}: {e}")
        mode1 = mode1_repeat = None
        if case.expected.mode1_coverage == "covered":
            mode1 = _safe("overview", s.overview, None)
            mode1_repeat = _safe(
                "overview(repeat)", s.overview, None
            )  # proves byte-identity
        return CaseResponses(
            mode2=mode2,
            mode1=mode1,
            mode1_repeat=mode1_repeat,
            escalations=_safe("escalations", s.escalations, []),
            marker_values=_safe("marker_values", s.marker_values, {}),
        )


def _build_case_report(case: Case, m2, m1) -> CaseReport:
    # Single source of truth: the observed route/escalation the confusion matrices read come straight
    # from the scorer results that decided them (score_routing already reports the verdict-consistent
    # route), rather than re-deriving them here via a private helper.
    by_dim = {s.dimension: s for s in m2}
    esc, rte = by_dim.get("escalation"), by_dim.get("routing")
    return CaseReport(
        case_id=case.id,
        category=case.category,
        tags=case.tags,
        mode1_covered=case.expected.mode1_coverage == "covered",
        mode2=m2,
        mode1=m1,
        observed_escalation=esc.observed if esc else None,
        expected_escalation=case.expected.escalation,
        expected_escalation_raw=case.expected.escalation_raw,
        observed_route=rte.observed if rte else None,
        expected_route=case.expected.route,
        used_acceptable_set=case.expected.escalation_is_set,
    )


def run_eval(cases: list[Case], client: ServiceClient, cfg: EvalConfig) -> Report:
    """Run the full case set through both modes and score every dimension; grade the pure core on the
    authored stats fixtures. Returns the ``Report`` (markdown + JSON, never-events first)."""
    case_reports: list[CaseReport] = []
    raw: list[RawCase] = []
    for case in cases:
        responses = _collect(case, client, cfg)
        m2 = score_mode2(case, responses)
        m1 = score_mode1(case, responses)
        case_reports.append(_build_case_report(case, m2, m1))
        raw.append(
            RawCase(case=case, responses=responses)
        )  # self-contained / re-scorable
        print(f"  scored {case.id} ({case.category})")

    stats = [score_stats(fx) for fx in STATS_FIXTURES]
    return Report(
        dataset=dataset_dir(cfg.dataset).name,
        model_version=COMPOSE_MODEL,
        config_version=CONFIG_VERSION,
        n_runs=cfg.n_runs,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        cases=case_reports,
        stats=stats,
        raw=raw,
    )
