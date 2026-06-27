"""eval/report.py — the per-case results container + the human/machine report (architecture §8).

``Report.to_markdown()`` and ``.to_json()`` over per-case results: **never-events first** (the blocking
safety failures), then per-dimension aggregates, the routing & escalation confusion matrices,
latency/cost percentiles, consistency flip-rates, the acceptable-set audit (Card-4 — the tolerance is
visible), the supplied-vs-added and Mode-1-vs-Mode-2 splits, and the per-case pass/fail table. The header
stamps ``model_version + config_version + data_version`` so two runs are comparable and the JSON is the
regression-gate artifact.

The run is RED iff any never-event fired; ordinary dimension failures (a labeled under-call like E12, the
K⁺ chip-completeness gap on E07) are *expected* and reported, not blocking — that asymmetry is the whole
point of never-events.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from eval.types import Case, CaseResponses, ScorerResult
from health_intelligence.models import FLOOR_ORDER, FloorLevel

_FLOORS = ["none", "clinician_review", "urgent"]
_ROUTES = ["none", "out_of_scope", "acute_medical", "crisis"]


class RawCase(BaseModel):
    """The raw inputs+outputs for one case, persisted in the JSON so the artifact is self-contained and
    re-scorable OFFLINE: a future scorer change can be re-graded against a stored run without re-hitting
    the API (the root-cause fix for "the report predates the scorer"). Carries the full ``Case`` (its
    labels) and the collected ``CaseResponses`` (the N Mode-2 responses, the Mode-1 overview, the DB
    escalation snapshot, the marker-value snapshot)."""

    case: Case
    responses: CaseResponses


class CaseReport(BaseModel):
    """One case's scored result across both modes (the harness fills it; the Report aggregates)."""

    case_id: str
    category: str
    tags: list[str]
    mode1_covered: bool
    mode2: list[ScorerResult] = Field(default_factory=list)
    mode1: list[ScorerResult] = Field(default_factory=list)
    observed_escalation: FloorLevel | None = None
    expected_escalation: list[FloorLevel] = Field(default_factory=list)
    expected_escalation_raw: str = ""
    observed_route: str | None = None
    expected_route: str = "none"
    used_acceptable_set: bool = False

    @property
    def never_events(self) -> list[tuple[str, str]]:
        """(dimension, never_event) for every scorer (both modes) that fired one."""
        return [
            (s.dimension, s.never_event)
            for s in (*self.mode2, *self.mode1)
            if s.never_event is not None
        ]


def _pct(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,1]); 0.0 on empty. No numpy dependency."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * frac


def _rate(results: list[ScorerResult]) -> tuple[int, int]:
    """(passed, total) over a list of scorer results."""
    return sum(1 for r in results if r.passed), len(results)


class Report(BaseModel):
    dataset: str
    model_version: str
    config_version: str
    n_runs: int
    generated_at: str
    cases: list[CaseReport] = Field(default_factory=list)
    stats: list[ScorerResult] = Field(default_factory=list)
    raw: list[RawCase] = Field(
        default_factory=list
    )  # self-contained, re-scorable (not in to_markdown)

    # ---- aggregation ----

    def never_events(self) -> list[tuple[str, str, str]]:
        """(case_id, dimension, never_event) across all cases — the blocking failures, surfaced first."""
        return [(c.case_id, dim, ev) for c in self.cases for dim, ev in c.never_events]

    def no_response_cases(self) -> list[str]:
        """Case ids whose Mode-2 calls ALL failed (zero responses) — a hard run failure: the service
        under test produced nothing, so a GREEN here would be the gate passing on total failure (a
        provider outage or a bad DB path silently masquerading as 'no never-events')."""
        return [rc.case.id for rc in self.raw if not rc.responses.mode2]

    def is_red(self) -> bool:
        return bool(self.never_events()) or bool(self.no_response_cases())

    def _dimension_rate(
        self, dimension: str, *, tag: str | None = None
    ) -> tuple[int, int]:
        rs = [
            s
            for c in self.cases
            if tag is None or tag in c.tags
            for s in c.mode2
            if s.dimension == dimension
        ]
        return _rate(rs)

    def _escalation_matrix(self) -> dict[str, dict[str, int]]:
        """expected min-floor (rows) × observed floor (cols). Upper-triangle = over-escalation (safe)."""
        mat = {e: {o: 0 for o in _FLOORS} for e in _FLOORS}
        for c in self.cases:
            if not c.expected_escalation or c.observed_escalation is None:
                continue
            exp_min = min(c.expected_escalation, key=lambda f: FLOOR_ORDER[f])
            mat[exp_min][c.observed_escalation] += 1
        return mat

    def _routing_matrix(self) -> dict[str, dict[str, int]]:
        mat = {e: {o: 0 for o in _ROUTES} for e in _ROUTES}
        for c in self.cases:
            if c.observed_route is None:
                continue
            mat[c.expected_route][c.observed_route] += 1
        return mat

    def _safety_recall(self) -> tuple[int, int]:
        """Of cases whose expected route is acute_medical/crisis, the fraction routed there (recall)."""
        hit = tot = 0
        for c in self.cases:
            if c.expected_route in ("acute_medical", "crisis"):
                tot += 1
                if c.observed_route == c.expected_route:
                    hit += 1
        return hit, tot

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    # ---- markdown ----

    def to_markdown(self) -> str:
        L: list[str] = []
        red = self.is_red()
        L.append("# Evaluation report")
        L.append("")
        L.append(
            f"- **dataset**: `{self.dataset}`  ·  **model**: `{self.model_version}`  ·  "
            f"**config**: `{self.config_version}`  ·  **N**: {self.n_runs}"
        )
        L.append(f"- **generated**: {self.generated_at}")
        L.append(
            f"- **result**: {'🔴 RED — never-event(s) fired' if red else '🟢 GREEN — no never-events'}"
        )
        L.append("")

        # 1. Never-events — first, always.
        L.append("## Never-events (blocking)")
        nes = self.never_events()
        if not nes:
            L.append("None. ✅")
        else:
            for cid, dim, ev in nes:
                L.append(f"- 🔴 **{cid}** — `{ev}` (scorer: {dim})")
        L.append("")

        # 1b. No-response cases — also blocking (the service produced nothing for these).
        nr = self.no_response_cases()
        if nr:
            L.append("## No-response cases (blocking)")
            L.append(
                f"- 🔴 {len(nr)} case(s) returned ZERO Mode-2 responses (service failed): "
                f"{', '.join(nr)}. The gate is RED — a run where nothing answered must not pass."
            )
            L.append("")

        # 2. Per-dimension pass rates (Mode 2), supplied vs added vs all.
        L.append("## Mode 2 — per-dimension pass rate")
        L.append("")
        L.append("| dimension | supplied | added | all |")
        L.append("|---|---|---|---|")
        for dim in ("escalation", "routing", "grounding", "consistency"):
            sp, spt = self._dimension_rate(dim, tag="supplied")
            ad, adt = self._dimension_rate(dim, tag="added")
            al, alt = self._dimension_rate(dim)
            L.append(f"| {dim} | {sp}/{spt} | {ad}/{adt} | {al}/{alt} |")
        L.append("")

        # 3. Escalation confusion matrix.
        L.append("## Escalation matrix (expected floor ↓ × observed →)")
        L.append("")
        L.append("| exp \\ obs | " + " | ".join(_FLOORS) + " |")
        L.append("|---|" + "|".join("---" for _ in _FLOORS) + "|")
        em = self._escalation_matrix()
        for e in _FLOORS:
            L.append(f"| {e} | " + " | ".join(str(em[e][o]) for o in _FLOORS) + " |")
        L.append("")
        L.append(
            "_Upper triangle = over-escalation (safe, measured, never failed); below diagonal = under-call._"
        )
        L.append("")

        # 4. Routing matrix + safety recall.
        L.append("## Gate routing matrix (expected ↓ × observed →)")
        L.append("")
        L.append("| exp \\ obs | " + " | ".join(_ROUTES) + " |")
        L.append("|---|" + "|".join("---" for _ in _ROUTES) + "|")
        rm = self._routing_matrix()
        for e in _ROUTES:
            L.append(f"| {e} | " + " | ".join(str(rm[e][o]) for o in _ROUTES) + " |")
        hit, tot = self._safety_recall()
        L.append("")
        L.append(
            f"**Recall on acute/crisis**: {hit}/{tot}"
            + (
                " ⚠️ (LLM-only gate; oblique cases expected to miss)"
                if tot and hit < tot
                else ""
            )
        )
        L.append("")

        # 5. Latency / cost percentiles + consistency. Read the per-run metadata straight off the
        # persisted raw responses (one source) rather than a duplicated per-case copy.
        all_lat = [
            float(r.metadata.latency_ms)
            for rc in self.raw
            for r in rc.responses.mode2
            if r.metadata.latency_ms is not None
        ]
        all_cost = [
            r.metadata.cost_usd
            for rc in self.raw
            for r in rc.responses.mode2
            if r.metadata.cost_usd is not None
        ]
        flips = [
            s.metrics.get("citation_flip_rate", 0.0)
            for c in self.cases
            for s in c.mode2
            if s.dimension == "consistency"
        ]
        L.append("## Latency / cost / consistency (Mode 2)")
        L.append("")
        L.append(
            f"- latency p50 **{_pct(all_lat, 0.5):.0f} ms** · p95 **{_pct(all_lat, 0.95):.0f} ms**"
        )
        L.append(
            f"- cost mean **${(sum(all_cost) / len(all_cost) if all_cost else 0):.5f}** · p95 **${_pct(all_cost, 0.95):.5f}** per interaction"
        )
        L.append(
            f"- mean citation flip-rate **{(sum(flips) / len(flips) if flips else 0):.2f}** (escalation/floor is byte-stable by construction)"
        )
        L.append("")

        # 6. Mode 1 — coverage + quality.
        covered = [c for c in self.cases if c.mode1_covered]
        m1_esc = _rate(
            [s for c in covered for s in c.mode1 if s.dimension == "mode1_escalation"]
        )
        m1_gnd = _rate(
            [s for c in covered for s in c.mode1 if s.dimension == "mode1_grounding"]
        )
        m1_con = _rate(
            [s for c in covered for s in c.mode1 if s.dimension == "mode1_consistency"]
        )
        L.append("## Mode 1 — coverage + quality (≈0 cost, byte-identical)")
        L.append("")
        L.append(
            f"- **coverage**: {len(covered)}/{len(self.cases)} cases handled by the preset surface"
        )
        L.append(
            f"- escalation {m1_esc[0]}/{m1_esc[1]} · grounding {m1_gnd[0]}/{m1_gnd[1]} · byte-identical {m1_con[0]}/{m1_con[1]} (on covered)"
        )
        L.append("")

        # 7. Acceptable-set audit (Card-4).
        audit = [c for c in self.cases if c.used_acceptable_set]
        L.append("## Acceptable-set audit")
        L.append("")
        if not audit:
            L.append("No case relied on an acceptable-set.")
        else:
            L.append("| case | acceptable-set | raw label | why |")
            L.append("|---|---|---|---|")
            for c in audit:
                L.append(
                    f"| {c.case_id} | {{{', '.join(c.expected_escalation)}}} | "
                    f"{c.expected_escalation_raw or '—'} | ambiguous label → over-escalation is the safe direction |"
                )
        L.append("")

        # 8. score_stats (separate label source).
        L.append("## score_stats — pure core vs authored fixtures")
        L.append("")
        for s in self.stats:
            L.append(f"- {'✅' if s.passed else '❌'} {s.detail}")
        L.append("")

        # 9. Per-case detail.
        L.append("## Per-case results")
        L.append("")
        L.append("| case | cat | tags | esc | route | grounding | never-event |")
        L.append("|---|---|---|---|---|---|---|")

        def _mark(cr: CaseReport, dim: str) -> str:
            s = next((x for x in cr.mode2 if x.dimension == dim), None)
            return "—" if s is None else ("✅" if s.passed else "❌")

        for c in self.cases:
            nes = ", ".join(ev for _, ev in c.never_events) or "—"
            L.append(
                f"| {c.case_id} | {c.category} | {','.join(c.tags)} | {_mark(c, 'escalation')} | "
                f"{_mark(c, 'routing')} | {_mark(c, 'grounding')} | {nes} |"
            )
        L.append("")
        return "\n".join(L)
