"""eval/inprocess.py — an in-process client for the ``/learn`` gate (Phase 7).

Same duck-typed interface as :class:`eval.client.ServiceClient` (``case_session`` -> a session with
``ask`` / ``overview`` / ``escalations`` / ``marker_values``), so ``harness.run_eval`` drives it
UNCHANGED — but it calls ``pipeline.ask`` / ``pipeline.suggestions`` / ``db`` reads DIRECTLY against a
per-case copy-DB connection, with **no FastAPI ``TestClient`` and no ``api._DB_PATH`` global patch**.

Why not reuse ``ServiceClient`` for ``/learn``: ``ServiceClient.case_session`` runs ``TestClient(api.app)``
and patches the process-global ``api._DB_PATH`` for the whole multi-minute eval. That is fine for
``make eval`` (its own process), but ``/learn`` runs INSIDE the live API process — nesting a TestClient
inside a live request and hijacking the global DB path would break every concurrent ``/ask`` during the
eval. Driving the real ``pipeline`` + the pure scorers directly is faithful to §8's "the same scorers
gate the candidate" and differs from ``make eval`` only in transport (routes are thin adapters, so
nothing load-bearing is bypassed).

**Seed the prompt-under-test.** ``build_seeded_template`` ingests the dataset but writes NO
``prompt_versions`` row, so the version-aware composer (``pipeline`` -> ``db.get_active_prompt``) would
fall back to ``BASE_COMPOSE_SYSTEM`` and the eval would silently measure the BASE prompt (the gate would
look alive but gate nothing). So ``build(seed_prompt=...)`` inserts the candidate as the active promoted
row in the template BEFORE the per-case copies are made — that seeding is the mechanism that makes the
gate gate. ``seed_prompt=None`` (the baseline run) leaves the template empty -> BASE, i.e. the live
default prompt.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from eval.client import (
    build_seeded_template,
    ground_truth_markers,
    overview_response,
)
from health_intelligence import db, llm, pipeline
from health_intelligence.models import Escalation, HealthIntelligenceResponse

#: Template-local version stamped on the seeded prompt-under-test. Never persisted to the live DB — it
#: lives only inside the throwaway eval template, so its exact value is immaterial (it just has to be the
#: single promoted row so ``get_active_prompt`` returns the candidate).
_SEED_VERSION = 1


class _InProcessSession:
    """One case's session, bound to a copy-DB connection — the ``_CaseSession`` analogue that calls the
    pipeline directly instead of over HTTP."""

    def __init__(self, con, member_id: str, provider: llm.Provider | None):
        self._con = con
        self.member_id = member_id
        self._provider = provider

    def ask(self, message: str) -> HealthIntelligenceResponse:
        return pipeline.ask(self._con, self.member_id, message, provider=self._provider)

    def overview(self) -> HealthIntelligenceResponse | None:
        return overview_response(pipeline.suggestions(self._con, self.member_id))

    def escalations(self) -> list[Escalation]:
        return db.get_escalations(self._con, self.member_id)

    def marker_values(self) -> dict[str, float]:
        return ground_truth_markers(self._con, self.member_id)


class InProcessClient:
    """Owns a seeded template DB (with the prompt-under-test, if any) + a scratch dir; one isolated
    copy-DB ``case_session`` per case (mandatory: the chat dedup key is day-scoped, so a shared DB would
    silently no-op the second same-member case — the same reason ``ServiceClient`` isolates per case)."""

    def __init__(self, template: Path, workdir: Path, provider: llm.Provider | None):
        self._template = template
        self._workdir = workdir
        self._provider = provider
        self._n = 0

    @classmethod
    @contextmanager
    def build(
        cls,
        dataset: str | None = None,
        *,
        seed_prompt: str | None = None,
        provider: llm.Provider | None = None,
    ) -> Generator[InProcessClient]:
        """Seed a template from ``dataset`` (default: the active ``DATASET``); when ``seed_prompt`` is
        given, insert it as the active promoted prompt so the composer-under-test runs it. ``provider``
        threads to ``pipeline.ask`` (None -> the real default provider; tests inject a fake)."""
        tmp = Path(tempfile.mkdtemp(prefix="learn-"))
        try:
            template = build_seeded_template(tmp, dataset)
            if seed_prompt is not None:
                con = db.connect(str(template))
                try:
                    db.insert_prompt_version(
                        con,
                        version=_SEED_VERSION,
                        prompt_text=seed_prompt,
                        status="promoted",
                    )
                finally:
                    con.close()
            yield cls(template, tmp, provider)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @contextmanager
    def case_session(self, member_id: str) -> Generator[_InProcessSession]:
        self._n += 1
        case_db = self._workdir / f"case-{self._n}.db"
        shutil.copy(self._template, case_db)
        con = db.connect(str(case_db))
        try:
            yield _InProcessSession(con, member_id, self._provider)
        finally:
            con.close()
            case_db.unlink(missing_ok=True)
