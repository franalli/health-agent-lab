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
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from eval import llm_eval
from eval.adapter import load_cases
from eval.client import ServiceClient
from eval.harness import run_eval
from eval.types import EvalConfig

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

    # Write the canonical artifact FIRST — it is the regression gate and must survive whatever the
    # (cuttable) LangSmith sink does. Persisting after the sink would lose a whole paid run if the sink
    # threw (architecture §8: the local report stays canonical).
    out = Path(args.out) if args.out else _REPORTS
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    md_path = out / f"eval-{stamp}.md"
    json_path = out / f"eval-{stamp}.json"
    md_path.write_text(report.to_markdown())
    json_path.write_text(report.to_json())

    # LangSmith offline trace sink (Phase 5c) — gated on LANGSMITH_API_KEY, a no-op otherwise; the report
    # above is already durable, and trace_run never raises (architecture §8).
    triples = [
        (rc.case, rc.responses, cr.mode2)
        for rc, cr in zip(report.raw, report.cases, strict=True)
    ]
    llm_eval.trace_run(
        triples, dataset=report.dataset, model_version=report.model_version
    )

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
    print(f"report: {md_path}")
    print(f"json:   {json_path}")
    return 1 if red else 0


if __name__ == "__main__":
    raise SystemExit(main())
