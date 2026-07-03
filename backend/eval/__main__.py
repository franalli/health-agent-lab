"""eval/__main__.py — ``python -m eval`` (wired to ``make eval``).

Seeds a template DB from the active dataset, runs every case through both modes against the live
service, writes the markdown + JSON report under ``eval/reports/``, and prints a concise summary. Exits
non-zero iff a never-event fired (the blocking safety failures) — the regression-gate semantics: a
healthy v1 run is GREEN with documented dimension *findings* (a labeled under-call, the K⁺ chip gap, an
oblique-crisis routing miss), never a never-event.

Mode 2 calls the real Anthropic API (architecture §8: "not a mock"); ``ANTHROPIC_API_KEY`` must be set
or Mode 2 degrades to the fail-closed spine and the run warns.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from eval import llm_eval
from eval.adapter import load_cases
from eval.client import ServiceClient
from eval.harness import run_eval
from eval.judge_eval import format_summary, run_judge_eval, to_dict, to_markdown_section
from eval.types import EvalConfig
from health_intelligence.learn import LearnUnavailable

_REPORTS = Path(__file__).resolve().parent / "reports"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the evaluation harness (architecture §8)."
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="dataset bundle (default: DATASET env or training_data)",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=3,
        help="Mode-2 repeats per case for the consistency measurement (default 3)",
    )
    ap.add_argument(
        "--out", default=None, help="output directory (default: eval/reports/)"
    )
    ap.add_argument(
        "--judge",
        action="store_true",
        help="enable the judge scorers (Phase 5b — not built yet)",
    )
    args = ap.parse_args(argv)

    # Load backend/.env BEFORE the key check below — otherwise the warning misfires, because api.py's own
    # load_dotenv() runs only when `api` is imported lazily during the first ask (so the key is, in fact,
    # set by run time). Same loader, just pulled earlier so the warning tells the truth.
    load_dotenv()

    if args.judge:
        print(
            "note: --judge (semantic grounding / tone) lands in Phase 5b; running the deterministic scorers only."
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "WARNING: ANTHROPIC_API_KEY is not set — Mode 2 will run DEGRADED (every turn fails closed to\n"
            "         couldnt_route at clinician_review, so routing/escalation are meaningless). Mode 1 and\n"
            "         score_stats stay valid. Set the key for a real Mode-2 run (architecture §8)."
        )

    cfg = EvalConfig(n_runs=args.n, use_judge=False, dataset=args.dataset)
    cases = load_cases(args.dataset)
    print(
        f"running {len(cases)} cases × N={cfg.n_runs} through both modes (Mode 2 = real API) ..."
    )
    with ServiceClient.build(args.dataset) as client:
        report = run_eval(cases, client, cfg)

    # Write the canonical COMPOSER artifact FIRST — before the multi-call real-Haiku judge battery below —
    # so a Ctrl-C or an unexpected (non-LearnUnavailable) error DURING the battery can never discard the
    # whole paid composer run. The judge section is written as PENDING here and REWRITTEN in place once the
    # battery finishes (or is known to have failed); a crash mid-battery therefore leaves a valid composer
    # report whose judge section honestly reads "interrupted", not a false "skipped". (The report must also
    # survive whatever the cuttable LangSmith sink does — architecture §8: the local report stays canonical.)
    out = Path(args.out) if args.out else _REPORTS
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    md_path = out / f"eval-{stamp}.md"
    json_path = out / f"eval-{stamp}.json"
    composer_md = report.to_markdown()
    composer_json = json.loads(report.to_json())

    def _write_artifact(judge_result, *, failed=False, pending=False):
        md_path.write_text(
            composer_md
            + "\n"
            + to_markdown_section(judge_result, failed=failed, pending=pending)
        )
        json_path.write_text(
            json.dumps(
                {
                    **composer_json,
                    "feedback_judge": to_dict(
                        judge_result, failed=failed, pending=pending
                    ),
                },
                indent=2,
            )
        )

    _write_artifact(None, pending=True)  # crash-safe placeholder; rewritten below

    # Feedback input-judge — a Haiku QUALITY gate (distinct from the composer/gate harness above). Only a
    # MISSING KEY is a SKIP (mirrors the degraded-Mode-2 warning above). A LearnUnavailable WITH a key
    # present is a TRANSIENT provider failure mid-battery (run_judge_eval raises on the first failing case) —
    # that must FAIL the run, not silently skip: a blanket skip would let a judge regression (or a bad/
    # expired key) pass CI as a clean skip, the exact blind spot this gate exists to close.
    judge_failed = False
    try:
        judge = run_judge_eval()
    except LearnUnavailable as e:
        judge = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            judge_failed = (
                True  # key present -> a real failure to certify, not a keyless skip
            )
            print(
                f"  feedback judge: FAILED to run despite ANTHROPIC_API_KEY set — {e}"
            )

    # Fold the real judge verdict into the already-written artifact — a keyed failure is recorded as FAILED,
    # NOT as the keyless "skipped" the placeholder would otherwise imply.
    _write_artifact(judge, failed=judge_failed)

    # LangSmith offline trace sink (Phase 5c) — gated on LANGSMITH_API_KEY, a no-op otherwise. It mirrors
    # the canonical local report: the COMPOSER cases via trace_report, then the feedback-judge battery via
    # trace_judge_run, so the two sinks carry the same per-case data (no discrepancy to reconcile). The
    # report is already durable, and both trace_* helpers swallow failures (architecture §8).
    llm_eval.trace_report(report)  # default run_prefix "eval" — the composer/gate cases
    llm_eval.trace_judge_run(judge)

    nes = report.never_events()
    nr = report.no_response_cases()
    red = report.is_red()
    hit, tot = report.safety_recall()
    print()
    print(
        f"=== {'🔴 RED' if red else '🟢 GREEN'} · {len(report.cases)} cases · N={cfg.n_runs} ==="
    )
    for cid, dim, ev in nes:
        print(f"  never-event: {cid} → {ev} ({dim})")
    if nr:
        print(f"  NO RESPONSE (service failed): {', '.join(nr)}")
    print(f"  acute/crisis routing recall: {hit}/{tot}")
    print(
        f"  stats fixtures: {sum(s.passed for s in report.stats)}/{len(report.stats)} passed"
    )
    if judge_failed:
        print(
            "  feedback judge: FAILED (provider error despite ANTHROPIC_API_KEY set) — run NOT certified"
        )
    elif judge is None:
        print("  feedback judge: SKIPPED (no ANTHROPIC_API_KEY)")
    else:
        print("  " + format_summary(judge).replace("\n", "\n  "))
    print(f"report: {md_path}")
    print(f"json:   {json_path}")
    return 1 if (red or judge_failed or (judge is not None and not judge.passed)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
