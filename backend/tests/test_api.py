"""The Phase-6 member-CRUD routes over the live FastAPI app (``TestClient``) — ``GET/POST/DELETE
/members``. These are the routes the consumer surface adds: the member-picker (``GET``), Upload
bundle (``POST``), and Clear member (``DELETE``).

The app captures ``HEALTH_DB_PATH`` at import (api.py reads it module-level), so we point it at a temp
FILE *before* importing ``api`` — ``:memory:`` is unsupported (one connection per request -> a fresh
empty in-memory DB each call; api.py fails fast on it). Each test gets a fresh schema seeded from the
training dataset on that file.
"""

import os
import pathlib
import tempfile

import pytest

# Point the app at a temp DB BEFORE importing api (it reads HEALTH_DB_PATH at import time; load_dotenv
# runs with override=False, so this value wins).
_DB_FILE = str(pathlib.Path(tempfile.mkdtemp()) / "test_api_health.db")
os.environ["HEALTH_DB_PATH"] = _DB_FILE

from builders import bundle_dict as _bundle  # noqa: E402
from builders import fresh_con  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from health_intelligence import db, learn  # noqa: E402
from preprocessing.ingest import ingest_dataset  # noqa: E402

# Every member-scoped child table — the cascade must leave none of these behind on a delete.
_CHILD_TABLES = [
    "escalations",
    "observations",
    "interactions",
    "lab_results",
    "notes",
    "feedback",
]


def _reset_and_seed() -> None:
    """Fresh schema + the 15-member training dataset on the file the app reads. reference_ranges are
    global (upserted by range_id), so they need no wipe."""
    con = fresh_con(_DB_FILE)
    for t in [*_CHILD_TABLES, "members"]:
        con.execute(f"DELETE FROM {t}")
    con.commit()
    ingest_dataset(con)
    con.close()


@pytest.fixture
def client():
    _reset_and_seed()
    with TestClient(api.app) as c:
        yield c


# ---- GET /members ---------------------------------------------------------------------------------


def test_get_members_projection(client):
    r = client.get("/members")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 15  # the seeded training dataset
    for m in body:  # exactly the picker projection — nothing heavier leaks
        assert set(m.keys()) == {"member_id", "age", "sex"}
    ids = [m["member_id"] for m in body]
    assert ids == sorted(ids)  # ordered by id
    c01 = next(m for m in body if m["member_id"] == "C01")
    assert c01["age"] == 46 and c01["sex"] == "male"


# ---- POST /members --------------------------------------------------------------------------------


def test_post_member_creates_and_appears_in_picker(client):
    assert "T99" not in [m["member_id"] for m in client.get("/members").json()]
    r = client.post("/members", json=_bundle("T99", age=40, sex="female"))
    assert r.status_code == 200
    assert r.json()["member_id"] == "T99"
    t99 = next(m for m in client.get("/members").json() if m["member_id"] == "T99")
    assert t99["age"] == 40 and t99["sex"] == "female"


def test_post_member_upserts_not_duplicates(client):
    client.post("/members", json=_bundle("T99", age=40, sex="female"))
    client.post(
        "/members", json=_bundle("T99", age=41, sex="female")
    )  # re-POST, changed age
    rows = [m for m in client.get("/members").json() if m["member_id"] == "T99"]
    assert len(rows) == 1 and rows[0]["age"] == 41  # upsert, not a second row


def test_post_member_malformed_returns_422(client):
    bad = _bundle("T98")
    del bad["profile"][
        "sex"
    ]  # required field -> Pydantic rejects (the upload is the format check)
    assert client.post("/members", json=bad).status_code == 422


def test_post_member_id_mismatch_returns_422(client):
    bad = _bundle("T97")
    bad["profile"]["member_id"] = "OTHER"  # bundle root and profile disagree
    assert client.post("/members", json=bad).status_code == 422


def test_post_member_semantic_error_returns_422_not_500(client):
    # Shape-valid (passes Pydantic) but firewall-rejected: an unparseable reference_range raises a
    # ValueError deep in ingest. The route must turn that into a clear 422, not an opaque 500 — the
    # upload is the format check (ui-ux §2), and surfacing the cause is the holdout path's whole point.
    bad = _bundle("T96")
    bad["panels"][0]["results"][0]["reference_range"] = "not-a-range"
    r = client.post("/members", json=bad)
    assert r.status_code == 422  # not 500
    assert "T96" not in [m["member_id"] for m in client.get("/members").json()]


# ---- DELETE /members/{id} -------------------------------------------------------------------------


def test_delete_member_removes_and_cascades(client):
    # C07 is the panic member; scanning it writes interactions + observations + an escalation, so a
    # delete that leaves no orphans proves the cascade reaches the audit tables.
    client.post("/members/C07/scan")
    con = db.connect(_DB_FILE)
    try:
        assert (
            con.execute(
                "SELECT COUNT(*) FROM observations WHERE member_id='C07'"
            ).fetchone()[0]
            > 0
        )
        assert (
            con.execute(
                "SELECT COUNT(*) FROM escalations WHERE member_id='C07'"
            ).fetchone()[0]
            > 0
        )
    finally:
        con.close()

    r = client.delete("/members/C07")
    assert r.status_code == 200 and r.json() == {"deleted": True}
    assert "C07" not in [m["member_id"] for m in client.get("/members").json()]

    con = db.connect(_DB_FILE)
    try:
        for t in _CHILD_TABLES:
            n = con.execute(
                f"SELECT COUNT(*) FROM {t} WHERE member_id='C07'"
            ).fetchone()[0]
            assert n == 0, f"orphan rows left in {t}"
    finally:
        con.close()


def test_delete_absent_member_404(client):
    assert client.delete("/members/NOPE").status_code == 404


def test_delete_twice_second_is_404(client):
    assert client.delete("/members/C01").status_code == 200
    assert (
        client.delete("/members/C01").status_code == 404
    )  # not present the second time


# ---- POST /members/{id}/feedback (Phase 7) --------------------------------------------------------


def test_post_feedback_records_an_override(client):
    r = client.post(
        "/members/C01/feedback",
        json={
            "kind": "range_override",
            "target": "LDL cholesterol",
            "payload": {"ref_high": 250.0},
            "source": "clinician",
        },
    )
    assert r.status_code == 200 and r.json()["feedback_id"]


def test_post_feedback_unknown_member_404(client):
    r = client.post(
        "/members/NOPE/feedback",
        json={"kind": "preference", "payload": {"text": "brief"}, "source": "member"},
    )
    assert r.status_code == 404


# ---- GET /members/{id}/trajectory (Phase 7) -------------------------------------------------------


def test_get_trajectory_returns_series(client):
    r = client.get("/members/C01/trajectory")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and body
    keys = {
        "marker",
        "unit",
        "readings",
        "trend",
        "flags",
        "severity",
        "reference_range",
        "theil_sen",
    }
    assert keys <= set(body[0].keys())


def test_get_trajectory_marker_filter(client):
    r = client.get("/members/C01/trajectory", params={"marker": "HbA1c"})
    assert r.status_code == 200
    assert [t["marker"] for t in r.json()] == ["HbA1c"]


def test_get_trajectory_unknown_member_404(client):
    assert client.get("/members/NOPE/trajectory").status_code == 404


# ---- POST /reset (Phase 7) ------------------------------------------------------------------------


def test_reset_deactivates_feedback(client):
    client.post(
        "/members/C01/feedback",
        json={
            "kind": "suppress_marker",
            "target": "LDL cholesterol",
            "source": "clinician",
        },
    )
    r = client.post("/reset")
    assert r.status_code == 200 and r.json()["learning_reset"] is True
    con = db.connect(_DB_FILE)
    try:
        active = con.execute("SELECT COUNT(*) FROM feedback WHERE active=1").fetchone()[
            0
        ]
    finally:
        con.close()
    assert active == 0


# ---- POST /learn (Phase 7 — route wiring only; the gate logic is covered in test_learn.py) ---------


def test_learn_route_returns_run_result(client, monkeypatch):
    monkeypatch.setattr(
        learn,
        "run_learn",
        lambda con: {"status": "noop", "version": None, "reason": "x", "report": None},
    )
    r = client.post("/learn")
    assert r.status_code == 200 and r.json()["status"] == "noop"


def test_learn_route_maps_busy_to_409(client, monkeypatch):
    def _busy(con):
        raise learn.LearnBusy("busy")

    monkeypatch.setattr(learn, "run_learn", _busy)
    assert client.post("/learn").status_code == 409


def test_learn_route_maps_unavailable_to_503(client, monkeypatch):
    def _down(con):
        raise learn.LearnUnavailable("no key")

    monkeypatch.setattr(learn, "run_learn", _down)
    assert client.post("/learn").status_code == 503


# ---- POST /admin/reseed (Phase 7) -----------------------------------------------------------------


def test_reseed_restores_initial_members_and_drops_holdouts(client):
    client.delete("/members/C01")  # remove a seeded member
    client.post("/members", json=_bundle("T99"))  # add an uploaded holdout
    r = client.post("/admin/reseed")
    assert r.status_code == 200 and r.json()["reseeded"] is True
    ids = [m["member_id"] for m in client.get("/members").json()]
    assert "C01" in ids  # the deleted seeded member is back
    assert "T99" not in ids  # the uploaded holdout is dropped
    assert len(ids) == 15  # back to the initial state
