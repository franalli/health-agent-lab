"""eval/client.py — the live-service client with per-case DB isolation (architecture §8).

The harness "calls the real service API — not a mock" (§8). ``ServiceClient`` drives the actual FastAPI
app through ``fastapi.testclient.TestClient`` — the real ASGI stack: the same routes, the same per-request
DB connection seam, and the same default LLM provider the deploy uses. (A uvicorn subprocess + httpx would
be truer to "a socket" but adds process management for no fidelity the in-process ASGI app lacks.)

**Per-case DB isolation is mandatory.** The chat-escalation dedup key is ``chat:{member_id}:{today}`` and
the same member recurs across cases (C07 in E07 *and* E17, C02 in E02 *and* E16). On a shared DB the
second case's escalation row would be a silent UNIQUE no-op, so the escalation scorer's DB check would
read a false negative. So each case runs against a FRESH copy of a seeded template DB: seed the dataset
once into a template file, then ``shutil.copy`` it per case (cheap) and point the app at the copy. The N
consistency repeats run inside that one per-case DB — which also makes the idempotency proof clean (after
N asks: exactly one chat escalation row, not N).

The app reads its DB path from the module global ``api._DB_PATH`` (set at import — api.py documents it as
"the test seam"), so the session patches that global before entering the ``TestClient`` context (whose
lifespan re-runs ``init_db``). ``marker_values`` reads the member's latest value per marker straight off
the per-case DB (in-process ``analyze`` — ground truth for the scorers, not part of the measured service
call), which the number-tracer needs to spot a discussed-but-uncited value.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from health_intelligence import db
from health_intelligence.analysis import analyze
from health_intelligence.config import ANALYSIS_CONFIG
from health_intelligence.models import (
    Escalation,
    HealthIntelligenceResponse,
    SuggestedPrompt,
)
from preprocessing.ingest import ingest_dataset

#: The deterministic "what's changed" overview anchor (templates.suggest_prompts) — the Mode-1 response
#: the harness scores as Mode 1's analog of "what does my data show". Matched on its exact prompt text.
OVERVIEW_PROMPT = "What's changed since my last results?"


def build_seeded_template(workdir: Path, dataset: str | None = None) -> Path:
    """Seed one template DB (schema + the whole dataset bundle) that per-case copies are made from.
    Reuses the ingestion firewall (``ingest_dataset``) — the same path ``make seed`` runs."""
    template = workdir / "template.db"
    con = db.connect(str(template))
    try:
        db.init_db(con)
        ingest_dataset(con, dataset)
    finally:
        con.close()
    return template


def ground_truth_markers(con, member_id: str) -> dict[str, float]:
    """The member's latest value per canonical marker (in-process ``analyze`` — the number-tracer's
    ground truth, not a measured service call); empty if the member is absent. The ONE definition both
    eval clients share (the HTTP ``ServiceClient`` and the in-process ``InProcessClient``), so the
    ground-truth read can't drift between ``make eval`` and the ``/learn`` gate."""
    try:
        member, results, ranges, age, data_version = db.load_for_analysis(
            con, member_id
        )
    except KeyError:
        return {}
    out = analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version
    )
    return {m.marker: m.latest.value for m in out.markers}


def overview_response(
    prompts: list[SuggestedPrompt],
) -> HealthIntelligenceResponse | None:
    """The Mode-1 "what's changed" overview from a list of suggested prompts, or ``None`` if absent — the
    one filter both clients use to pick Mode 1's scored anchor (matched on :data:`OVERVIEW_PROMPT`)."""
    return next((sp.response for sp in prompts if sp.prompt == OVERVIEW_PROMPT), None)


class _CaseSession:
    """A live session bound to one case's isolated DB. ``ask``/``suggestions``/``scan`` go over HTTP
    (the measured service calls); ``escalations``/``observations`` read back DB state; ``marker_values``
    is an in-process ground-truth read for the scorers."""

    def __init__(self, http, member_id: str, case_db: Path):
        self._http = http
        self.member_id = member_id
        self._case_db = case_db

    def ask(self, message: str) -> HealthIntelligenceResponse:
        r = self._http.post(f"/members/{self.member_id}/ask", json={"message": message})
        r.raise_for_status()
        return HealthIntelligenceResponse.model_validate(r.json())

    def suggestions(
        self, focus: str | None = None, asked: tuple[str, ...] = ()
    ) -> list[SuggestedPrompt]:
        params: dict = {}
        if focus is not None:
            params["focus"] = focus
        if asked:
            params["asked"] = list(asked)
        r = self._http.get(f"/members/{self.member_id}/suggestions", params=params)
        r.raise_for_status()
        return [SuggestedPrompt.model_validate(x) for x in r.json()]

    def overview(self) -> HealthIntelligenceResponse | None:
        """The Mode-1 "what's changed" overview response, or ``None`` if absent (e.g. an unknown member)."""
        return overview_response(self.suggestions())

    def escalations(self) -> list[Escalation]:
        r = self._http.get(f"/members/{self.member_id}/escalations")
        r.raise_for_status()
        return [Escalation.model_validate(x) for x in r.json()]

    def marker_values(self) -> dict[str, float]:
        """The member's latest value per canonical marker, read straight off the per-case DB (the shared
        :func:`ground_truth_markers` — ground truth for the number-tracer, not a measured service call)."""
        con = db.connect(str(self._case_db))
        try:
            return ground_truth_markers(con, self.member_id)
        finally:
            con.close()


class ServiceClient:
    """Owns the seeded template DB + a scratch dir for per-case copies. One ``case_session`` per case."""

    def __init__(self, template_db: Path, workdir: Path):
        self._template = template_db
        self._workdir = workdir
        self._n = 0

    @classmethod
    @contextmanager
    def build(cls, dataset: str | None = None) -> Generator[ServiceClient]:
        """Seed a template DB in a scratch dir and yield a client; clean both up on exit."""
        tmp = Path(tempfile.mkdtemp(prefix="eval-"))
        try:
            template = build_seeded_template(tmp, dataset)
            yield cls(template, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @contextmanager
    def case_session(self, member_id: str) -> Generator[_CaseSession]:
        """A fresh, isolated DB (a copy of the template) bound to the app for the duration of one case.
        Patches ``api._DB_PATH`` (the documented test seam) and runs the app's lifespan via ``TestClient``."""
        from fastapi.testclient import TestClient

        import api  # top-level module at backend/api.py; resolved when run from backend/ (make eval)

        self._n += 1
        case_db = self._workdir / f"case-{self._n}.db"
        shutil.copy(self._template, case_db)
        prev = api._DB_PATH
        api._DB_PATH = str(case_db)
        try:
            with TestClient(api.app) as http:
                yield _CaseSession(http, member_id, case_db)
        finally:
            api._DB_PATH = prev
            case_db.unlink(missing_ok=True)
