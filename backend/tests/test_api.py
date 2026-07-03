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
import shutil
import tempfile

import pytest

# Point the app at a temp DB BEFORE importing api (it reads HEALTH_DB_PATH at import time; load_dotenv
# runs with override=False, so this value wins).
_DB_FILE = str(pathlib.Path(tempfile.mkdtemp()) / "test_api_health.db")
os.environ["HEALTH_DB_PATH"] = _DB_FILE

from builders import bundle_dict as _bundle  # noqa: E402
from builders import fresh_con, make_panel, make_result  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import preprocessing.datasets as datasets  # noqa: E402
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
    global (upserted by range_id), so they need no wipe. ``prompt_versions`` IS wiped so each test is a
    true cold start (an empty learning store) — the app's lifespan then re-seeds v0, exactly like a fresh
    deploy, which is what ``test_cold_start_seeds_the_v0_baseline_prompt`` pins."""
    con = fresh_con(_DB_FILE)
    for t in [*_CHILD_TABLES, "members", "prompt_versions"]:
        con.execute(f"DELETE FROM {t}")
    con.commit()
    ingest_dataset(con)
    con.close()


def _isolate_data_root(tmp_path, monkeypatch):
    """Point ``HEALTH_DATA_ROOT`` at an isolated temp dir seeded with a copy of the shipped
    ``training_data``, so NO client-based test mutates the repo's ``backend/data/``. This matters
    specifically because ``POST /admin/reseed`` now removes uploaded dataset FOLDERS from ``data_root()``
    (``datasets.remove_uploaded_datasets``) — against the real root a reseed test would delete a developer's
    local uploads. Returns the temp root (``.data_root`` for tests that assert on the created folder)."""
    root = tmp_path / "data"
    root.mkdir()
    shutil.copytree(_REAL_TRAINING, root / "training_data")
    monkeypatch.setenv("HEALTH_DATA_ROOT", str(root))
    return root


@pytest.fixture
def client(tmp_path, monkeypatch):
    _isolate_data_root(tmp_path, monkeypatch)
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


# ---- GET /escalations (global triage queue) -------------------------------------------------------


def test_global_escalations_queue_spans_members_worst_first(client):
    """The cross-member worklist: scanning C07 (panic K+ -> urgent) and C01 (an adverse trend ->
    clinician_review) must both surface in ONE global read, the urgent ranked ahead. This is the
    property a per-member route structurally can't deliver (you'd have to know C07 to look)."""
    client.post("/members/C07/scan")  # -> urgent
    client.post("/members/C01/scan")  # -> clinician_review

    r = client.get("/escalations")
    assert r.status_code == 200
    queue = r.json()
    members = {e["member_id"] for e in queue}
    assert {"C01", "C07"} <= members  # genuinely cross-member (the whole point)

    # worst-first: every urgent precedes every clinician_review.
    rank = {"urgent": 0, "clinician_review": 1}
    ranks = [rank[e["level"]] for e in queue]
    assert ranks == sorted(ranks)
    assert queue[0]["level"] == "urgent" and queue[0]["member_id"] == "C07"
    # every item carries its severity on the wire (derived from level at read), and the queue is
    # sorted highest-severity first on that same axis.
    sev_rank = {"urgent": 0, "attention": 1}
    sevs = [sev_rank[e["severity"]] for e in queue]
    assert sevs == sorted(sevs)
    assert queue[0]["severity"] == "urgent"


def test_global_escalations_queue_empty_before_any_scan(client):
    """No scan run yet -> no escalations anywhere -> an empty queue (a clean 200, never a 404)."""
    r = client.get("/escalations")
    assert r.status_code == 200 and r.json() == []


def test_scan_route_returns_envelope_with_new_observation_count(client):
    """POST /scan serves the ScanResult envelope: the observations projection plus how many rows this
    run newly persisted — the count the operator panel's scan-all readout sums. First scan: all new;
    idempotent re-scan: 0 (rows refresh in place)."""
    r = client.post("/members/C07/scan")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"observations", "new_observations"}
    assert body["new_observations"] == len(body["observations"]) > 0
    assert client.post("/members/C07/scan").json()["new_observations"] == 0


def test_per_member_escalations_is_the_drill_in_not_the_queue(client):
    """The per-member route is the drill-in: it returns ONLY that member's escalations, so it can't
    serve as the queue. C07 scanned (urgent), C01 not -> C07's drill-in has its row, C01's is empty,
    and neither sees the other's."""
    client.post("/members/C07/scan")
    c07 = client.get("/members/C07/escalations").json()
    c01 = client.get("/members/C01/escalations").json()
    assert c07 and all(e["member_id"] == "C07" for e in c07)
    # the drill-in carries the same derived severity field as the queue, consistent with each level
    assert all(
        e["severity"] == ("urgent" if e["level"] == "urgent" else "attention")
        for e in c07
    )
    assert c01 == []  # unscanned -> nothing; never C07's rows
    assert (
        client.get("/members/NOPE/escalations").status_code == 404
    )  # drill-in still 404s


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


# ---- POST /members/upload (full bundle persisted as a new dataset folder; additive; auto-scan) ------

#: The shipped bundle, resolved from the DEFAULT root (not data_root()) so it's found whatever the
#: HEALTH_DATA_ROOT override is — the source we copy into each test's isolated root.
_REAL_TRAINING = datasets._DEFAULT_DATA_ROOT / "training_data"


@pytest.fixture
def upload_client(tmp_path, monkeypatch):
    """A ``client`` that also exposes ``.data_root`` for tests that assert on the created dataset folder —
    an upload CREATES its folder in the isolated temp root (see :func:`_isolate_data_root`), never the repo,
    and teardown is automatic."""
    root = _isolate_data_root(tmp_path, monkeypatch)
    _reset_and_seed()  # data_root() now resolves to `root`; seeds the 15 from the copied training_data
    with TestClient(api.app) as c:
        c.data_root = root  # let tests assert on the created folder
        yield c


def _valid_eval_bytes(ids, member_id):
    """A valid JSONL eval file (one :class:`EvalCaseInput`-shaped object per line) — the ``.jsonl`` role."""
    import json

    return "\n".join(json.dumps(r) for r in _eval_records(ids, member_id)).encode()


def _valid_csv_bytes(member_ids):
    """A valid lab-panels CSV (exact header + one HbA1c row per member) — the ``.csv`` role."""
    header = "member_id,panel_id,collected_date,analyte,value,unit,reference_range"
    rows = [f"{mid},{mid}-P1,2024-01-15,HbA1c,5.3,%,<5.7" for mid in member_ids]
    return ("\n".join([header, *rows]) + "\n").encode()


def _dataset_zip(member_ids, *, top="bundle", extra=None, mutate=None, required=True):
    """Build a COMPLETE dataset .zip — ``members.json`` + ``eval_set.jsonl`` + ``lab_panels.csv`` (the
    three files the upload now requires by extension) under a single wrapping folder ``top`` (``None`` ->
    at the zip root), plus optional ``extra`` ``{relpath: str|bytes}``. Pass ``required=False`` to write
    ONLY ``members.json`` (for the missing-required-file rejection tests). ``mutate`` edits each member
    dict before serialization."""
    import io
    import json
    import zipfile

    members = []
    for mid in member_ids:
        b = _bundle(mid)
        if mutate:
            mutate(b)
        members.append(b)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        prefix = f"{top}/" if top else ""
        zf.writestr(prefix + "members.json", json.dumps(members))
        if required:
            zf.writestr(
                prefix + "eval_set.jsonl", _valid_eval_bytes(["E1"], member_ids[0])
            )
            zf.writestr(prefix + "lab_panels.csv", _valid_csv_bytes(member_ids))
        for rel, content in (extra or {}).items():
            zf.writestr(prefix + rel, content)
    return buf.getvalue()


def test_upload_zip_persists_folder_and_ingests_additively(upload_client):
    # The whole requirement: keep EVERYTHING (members.json + csv + jsonl) as a new dataset folder, AND
    # add the members on top of the existing 15 (no reseed).
    data = _dataset_zip(["U01", "U02", "U03"], top="holdout")
    r = upload_client.post(
        "/members/upload", files={"file": ("holdout.zip", data, "application/zip")}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"] == "holdout"
    assert body["members"] == 3 and body["member_ids"] == ["U01", "U02", "U03"]
    assert set(body["files"]) == {"members.json", "lab_panels.csv", "eval_set.jsonl"}

    picker = {m["member_id"] for m in upload_client.get("/members").json()}
    assert {"U01", "U02", "U03"} <= picker  # new members added
    assert "C01" in picker  # ADDITIVE: the seeded members are untouched (no reseed)

    folder = upload_client.data_root / "holdout"
    assert (folder / "members.json").is_file()
    assert (
        folder / "lab_panels.csv"
    ).is_file()  # the WHOLE bundle persisted, not just members.json
    assert (folder / "eval_set.jsonl").is_file()


def test_upload_auto_scans_new_members(upload_client):
    # A flagged marker (HbA1c 9.0 vs <5.7) must surface as an Observation immediately after upload —
    # proving the auto-scan ran and PERSISTED, so the member's Observations match its live Trajectory.
    import io
    import json
    import zipfile

    member = _bundle(
        "S01",
        panels=[
            make_panel("S01-P1", "2024-01-15", [make_result("HbA1c", 9.0, "%", "<5.7")])
        ],
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("members.json", json.dumps([member]))
        zf.writestr("eval_set.jsonl", _valid_eval_bytes(["E1"], "S01"))
        zf.writestr("lab_panels.csv", _valid_csv_bytes(["S01"]))
    r = upload_client.post(
        "/members/upload",
        files={"file": ("scanme.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["scanned"] == 1
    obs = upload_client.get("/members/S01/observations").json()
    assert len(obs) >= 1  # auto-scan produced + stored the out-of-range observation
    # the sweep's newly-persisted total rides the upload response (a fresh member: all rows are new)
    assert r.json()["new_observations"] == len(obs)
    hba1c = next(o for o in obs if o["title"].startswith("HbA1c"))
    # the member-facing field rides the wire, states the numeric range, and leaks no clinician stats
    assert "above the normal range" in hba1c["member_explanation"]
    assert "under 5.7" in hba1c["member_explanation"]
    assert "Mann-Kendall" not in hba1c["member_explanation"]


def test_upload_bare_json_rejected_must_be_zip(upload_client):
    # A bare (non-zip) .json can't carry the required .jsonl + .csv, so it is rejected with a clear,
    # structured reason — NOT ingested as a one-file dataset (the pre-validation contract accepted it).
    import io
    import json

    data = io.BytesIO(json.dumps([_bundle("U10"), _bundle("U11")]).encode())
    r = upload_client.post(
        "/members/upload", files={"file": ("myset.json", data, "application/json")}
    )
    assert r.status_code == 422
    failures = r.json()["detail"]["failures"]
    assert any("must be a .zip" in f["detail"] for f in failures)
    assert not (upload_client.data_root / "myset").exists()  # no side effect
    assert "U10" not in {m["member_id"] for m in upload_client.get("/members").json()}


def test_upload_explicit_name_field_wins(upload_client):
    data = _dataset_zip(["U40"], top="ignored_inner_name")
    r = upload_client.post(
        "/members/upload",
        files={"file": ("alsoignored.zip", data, "application/zip")},
        data={"name": "chosen"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["dataset"] == "chosen"
    assert (upload_client.data_root / "chosen").is_dir()


def test_upload_collision_returns_409(upload_client):
    data = _dataset_zip(["U50"])
    first = upload_client.post(
        "/members/upload",
        files={"file": ("dup.zip", data, "application/zip")},
        data={"name": "dup"},
    )
    assert first.status_code == 200, first.text
    second = upload_client.post(
        "/members/upload",
        files={"file": ("dup.zip", _dataset_zip(["U51"]), "application/zip")},
        data={"name": "dup"},
    )
    assert second.status_code == 409  # uploads never overwrite an existing dataset


def test_upload_training_data_name_collision_protects_shipped(upload_client):
    # Uploading a zip whose stem is `training_data` must not clobber the shipped bundle.
    r = upload_client.post(
        "/members/upload",
        files={"file": ("training_data.zip", _dataset_zip(["U60"]), "application/zip")},
    )
    assert r.status_code == 409


def test_upload_rejects_zip_slip_with_no_side_effects(upload_client):
    # A path-traversal entry must be rejected BEFORE any DB row or escaped file is written.
    data = _dataset_zip(["EVIL"], top="bundle", extra={"../escape.txt": "pwned"})
    r = upload_client.post(
        "/members/upload", files={"file": ("evil.zip", data, "application/zip")}
    )
    assert r.status_code == 422
    assert "EVIL" not in {m["member_id"] for m in upload_client.get("/members").json()}
    assert not (upload_client.data_root / "evil").exists()  # no folder created
    assert not (
        upload_client.data_root.parent / "escape.txt"
    ).exists()  # nothing escaped


@pytest.mark.parametrize("bad_name", ["../evil", "a/b", "..", ".hidden"])
def test_upload_rejects_unsafe_dataset_name(upload_client, bad_name):
    data = _dataset_zip(["U70"])
    r = upload_client.post(
        "/members/upload",
        files={"file": ("x.zip", data, "application/zip")},
        data={"name": bad_name},
    )
    assert r.status_code == 422


def test_upload_empty_derived_name_rejected(upload_client):
    # A filename with no usable stem (".json") and no explicit name -> sanitizes to "" -> 422.
    import io

    data = io.BytesIO(b"[]")
    r = upload_client.post(
        "/members/upload", files={"file": (".json", data, "application/json")}
    )
    assert r.status_code == 422


def test_upload_shape_malformed_422_no_partial_set(upload_client):
    # A SHAPE error (extra='forbid') on the FIRST member is caught by the first-record format gate —
    # before ANY DB write and before the folder is created. `mutate` corrupts every member, so the first
    # is bad and the gate fires (a later-only bad row would instead be skipped at ingest, not rejected).
    # Neither the sibling member nor the folder lands, and the structured `failures` list rides the wire.
    data = _dataset_zip(["U20", "U21"], mutate=lambda b: b["profile"].update(oops="x"))
    r = upload_client.post(
        "/members/upload", files={"file": ("bad.zip", data, "application/zip")}
    )
    assert r.status_code == 422
    assert r.json()["detail"][
        "failures"
    ]  # structured file+row failures, not a flat string
    picker = {m["member_id"] for m in upload_client.get("/members").json()}
    assert not ({"U20", "U21"} & picker)
    assert not (upload_client.data_root / "bad").exists()  # no folder on a shape error


def test_upload_late_semantic_error_skips_row_not_upload(upload_client):
    # The new skip semantics: the first member is good, a LATER member carries a bad reference_range. The
    # first-record format gate passes it (MemberBundle keeps the range as a raw string, parsed only at
    # ingest — the documented residual), so the semantic firewall parse fails per-member at ingest. That
    # bad row is now SKIPPED, not rejected: the upload SUCCEEDS (200), the good member ingests + scans, the
    # bad one lands in `skipped` with its member_id + a range-parse detail, the folder IS created, and
    # `member_ids` excludes the skipped row.
    import io
    import json
    import zipfile

    good, bad = _bundle("U22"), _bundle("U23")
    bad["panels"][0]["results"][0]["reference_range"] = "not-a-range"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("members.json", json.dumps([good, bad]))
        zf.writestr("eval_set.jsonl", _valid_eval_bytes(["E1"], "U22"))
        zf.writestr("lab_panels.csv", _valid_csv_bytes(["U22", "U23"]))
    r = upload_client.post(
        "/members/upload",
        files={"file": ("sem.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["member_ids"] == ["U22"]  # skipped row excluded from the ingested set
    skipped = body["skipped"]
    assert [s["member_id"] for s in skipped] == ["U23"]
    assert (
        skipped[0]["file"] == "members.json" and "not-a-range" in skipped[0]["detail"]
    )
    picker = {m["member_id"] for m in upload_client.get("/members").json()}
    assert "U22" in picker and "U23" not in picker  # good member in, skipped member out
    assert (upload_client.data_root / "sem").is_dir()  # folder still created on success


def test_upload_non_dict_later_member_row_skipped_not_fatal(upload_client):
    # A NON-DICT later row (a bare string, a null) must be SKIPPED at ingest like a shape-invalid row, NOT
    # 422 the whole upload — the first-record gate checks only records[0]'s shape. This is the contract "a
    # buggy later member row is skipped, not fatal", and it makes a non-dict late row and a
    # dict-with-bad-fields late row behave the SAME (both skipped), not opposite outcomes.
    import io
    import json
    import zipfile

    members = [_bundle("G1"), "junk-not-an-object", None, _bundle("G2")]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("members.json", json.dumps(members))
        zf.writestr("eval_set.jsonl", _valid_eval_bytes(["E1"], "G1"))
        zf.writestr("lab_panels.csv", _valid_csv_bytes(["G1", "G2"]))
    r = upload_client.post(
        "/members/upload",
        files={"file": ("mixed.zip", buf.getvalue(), "application/zip")},
        data={"name": "mixed"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["members"] == 2 and set(body["member_ids"]) == {"G1", "G2"}
    assert len(body["skipped"]) == 2  # the string + the null skipped, not fatal
    picker = {m["member_id"] for m in upload_client.get("/members").json()}
    assert {"G1", "G2"} <= picker
    assert (upload_client.data_root / "mixed").is_dir()  # good rows persisted


def test_upload_all_member_rows_bad_is_422_and_rolls_back_folder(upload_client):
    # Every member row is shape-valid (passes the first-record gate) but semantically bad (an unparseable
    # reference_range), so ingest skips ALL of them -> members==0. An upload that ingests NOBODY is a FAILED
    # upload, not a 200 with an empty dataset: it 422s with the per-row detail AND the reserved folder is
    # rolled back (no orphan dataset that only 409s on retry).
    def _break(b):
        b["panels"][0]["results"][0]["reference_range"] = "not-a-range"

    data = _dataset_zip(["B1", "B2"], top="allbad", mutate=_break)
    r = upload_client.post(
        "/members/upload",
        files={"file": ("allbad.zip", data, "application/zip")},
        data={"name": "allbad"},
    )
    assert r.status_code == 422, r.text
    failures = r.json()["detail"]["failures"]
    assert failures and all(f["file"] == "members.json" for f in failures)
    assert any("not-a-range" in f["detail"] for f in failures)  # per-row cause surfaced
    assert not (
        upload_client.data_root / "allbad"
    ).exists()  # folder rolled back, no orphan
    picker = {m["member_id"] for m in upload_client.get("/members").json()}
    assert not ({"B1", "B2"} & picker)  # nothing ingested


def test_upload_not_a_zip_returns_422(upload_client):
    r = upload_client.post(
        "/members/upload",
        files={"file": ("x.zip", b"definitely not a zip", "application/zip")},
    )
    assert r.status_code == 422


def test_upload_resolves_roles_by_extension_not_name(upload_client):
    # Roles are bucketed by EXTENSION, not base name: a zip whose files carry arbitrary stems but the
    # three required extensions (.json members, .jsonl eval, .csv panels) ingests, and each file is
    # renamed to its CANONICAL on-disk name so every downstream reader stays name-based.
    data = _zip_of(
        {
            "roster.json": _members_bytes(["AX1", "AX2"]),
            "cases.jsonl": _valid_eval_bytes(["E1"], "AX1"),
            "panel.csv": _valid_csv_bytes(["AX1", "AX2"]),
        }
    )
    r = upload_client.post(
        "/members/upload",
        files={"file": ("holdoutA.zip", data, "application/zip")},
        data={"name": "byext"},
    )
    assert r.status_code == 200, r.text
    assert set(r.json()["member_ids"]) == {"AX1", "AX2"}
    folder = upload_client.data_root / "byext"
    assert (
        folder / "members.json"
    ).is_file()  # renamed to canonical, not the upload's name
    assert (folder / "lab_panels.csv").is_file()
    assert (folder / "eval_set.jsonl").is_file()
    assert not (folder / "roster.json").exists()  # the arbitrary stem is not kept


def test_upload_bom_members_and_eval(upload_client):
    # #5: a UTF-8 BOM on the members AND eval files must not break the format gate or the eval load. The
    # bundle is a full zip: BOM'd .json members, BOM'd .jsonl eval, and a valid .csv (BOM tolerance must
    # survive the gate AND the later load_supplied_cases read).
    from eval.adapter import load_supplied_cases

    bom = b"\xef\xbb\xbf"
    data = _zip_of(
        {
            "m.json": bom + _members_bytes(["BX1"]),  # BOM'd members
            "e.jsonl": bom + _valid_eval_bytes(["EB1"], "BX1"),  # BOM'd eval
            "p.csv": _valid_csv_bytes(["BX1"]),
        }
    )
    r = upload_client.post(
        "/members/upload",
        files={"file": ("bom.zip", data, "application/zip")},
        data={"name": "bomset"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["member_ids"] == ["BX1"]
    assert [c.id for c in load_supplied_cases("bomset")] == [
        "EB1"
    ]  # BOM eval still loads


def test_upload_dot_entry_rejected_no_side_effects(upload_client):
    # #6: a zip entry named "." slips a naive traversal guard; it must be rejected (422), not crash (500),
    # and ingest nothing.
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("members.json", _members_bytes(["DOT1"]))
        zf.writestr(".", b"x")  # entry that normalizes to the directory itself
    r = upload_client.post(
        "/members/upload",
        files={"file": ("dot.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 422
    assert "DOT1" not in {m["member_id"] for m in upload_client.get("/members").json()}
    assert not (upload_client.data_root / "dot").exists()


@pytest.mark.parametrize("compression", ["stored", "deflate", "bzip2", "lzma"])
def test_upload_corrupt_zip_entry_is_422_not_500(upload_client, compression):
    # #3: a zip that OPENS but whose entry can't be READ must be a clean 422, not a 500 — across EVERY
    # compression method, since a corrupt stream raises a method-specific error: STORED bad-CRC ->
    # BadZipFile, DEFLATE (the `zip -r`/macOS default) -> zlib.error, BZIP2 -> OSError, LZMA -> LZMAError.
    # A STORED/DEFLATE-only test misses bzip2's OSError (which the route's OSError->500 would mis-handle).
    import io
    import zipfile

    members = _members_bytes(["CR1"])
    mode = {
        "stored": zipfile.ZIP_STORED,
        "deflate": zipfile.ZIP_DEFLATED,
        "bzip2": zipfile.ZIP_BZIP2,
        "lzma": zipfile.ZIP_LZMA,
    }[compression]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", mode) as zf:
        zf.writestr("members.json", members)
        zf.writestr(
            "filler.json", members
        )  # 2nd entry so corruption lands mid-archive, not at EOF
    raw = bytearray(buf.getvalue())
    # Corrupt the FIRST entry's payload so zf.read() raises at READ time, not at open.
    if compression == "stored":
        idx = raw.find(members)
        raw[idx : idx + 4] = b"XXXX"  # CRC mismatch -> BadZipFile
    else:
        # flip bytes just past the first local-file-header content start (inside the compressed stream)
        idx = raw.find(b"members.json") + len(b"members.json") + 6
        for i in range(idx, idx + 8):
            raw[i] ^= (
                0xFF  # corrupt the compressed stream -> zlib.error / OSError / LZMAError on inflate
            )
    r = upload_client.post(
        "/members/upload",
        files={"file": ("corrupt.zip", bytes(raw), "application/zip")},
    )
    assert r.status_code == 422  # not 500


def test_upload_non_utf8_members_is_422_not_raw_error(upload_client):
    # #X1: a non-UTF-8 (no-BOM) .json members file is not valid JSON (JSON must be Unicode); read_records
    # must surface a clean whole-file ValueError (row=None) -> 422, never a raw UnicodeDecodeError/500. The
    # rest of the bundle is valid, so the failure is unambiguously the members read.
    # cp1252 bytes that are not valid UTF-8 (0x96 = en-dash in cp1252, an invalid UTF-8 start byte)
    data = _zip_of(
        {
            "m.json": b'[{"member_id": "\x96bad"}]',
            "e.jsonl": _valid_eval_bytes(["E1"], "X1"),
            "p.csv": _valid_csv_bytes(["X1"]),
        }
    )
    r = upload_client.post(
        "/members/upload", files={"file": ("nonutf8.zip", data, "application/zip")}
    )
    assert r.status_code == 422  # clean validation error, never a 500/raw traceback
    failures = r.json()["detail"]["failures"]
    assert any(f["file"] == "members.json" and f["row"] is None for f in failures)
    assert not (
        upload_client.data_root / "nonutf8"
    ).exists()  # rejected before any folder


def test_read_records_wraps_non_utf8_as_plain_valueerror():
    # #X1 unit: read_records must honor its "Raises ValueError" contract on non-UTF-8 bytes by wrapping
    # them as a PLAIN ValueError — not letting a raw UnicodeDecodeError escape (which, though a ValueError
    # subclass, is the un-wrapped error the direct-read paths (ingest_dataset / load_supplied_cases) would
    # surface as a confusing traceback). Asserting `not UnicodeDecodeError` is what distinguishes the fix.
    from preprocessing.ingest import read_records

    with pytest.raises(ValueError) as ei:
        read_records(b'{"x": "\x96"}')  # 0x96 is invalid UTF-8
    assert not isinstance(ei.value, UnicodeDecodeError)
    assert "not valid JSON" in str(ei.value)


def test_upload_duplicate_member_id_counts_distinct(upload_client):
    # #10: a duplicate member_id in members.json must not inflate the members/scanned counts (upsert -> one).
    data = _zip_of(
        {
            "members.json": _members_bytes(["DUP", "DUP"]),
            "eval_set.jsonl": _valid_eval_bytes(["E1"], "DUP"),
            "lab_panels.csv": _valid_csv_bytes(["DUP"]),
        }
    )
    r = upload_client.post(
        "/members/upload", files={"file": ("dup.zip", data, "application/zip")}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (
        body["members"] == 1 and body["member_ids"] == ["DUP"] and body["scanned"] == 1
    )


def test_upload_dotted_path_member_file_kept(upload_client):
    # #11: a legitimately dot-named bundle path is NOT discarded as junk (only __MACOSX/.DS_Store/._* are).
    # The members .json sits under a `.v2/` folder; the eval/csv roles sit at the root (so nothing is
    # stripped as a common top folder). Role resolution is by extension, so the dotted path still ingests.
    data = _zip_of(
        {
            ".v2/roster.json": _members_bytes(["DV1"]),
            "eval_set.jsonl": _valid_eval_bytes(["E1"], "DV1"),
            "lab_panels.csv": _valid_csv_bytes(["DV1"]),
        }
    )
    r = upload_client.post(
        "/members/upload", files={"file": ("dotted.zip", data, "application/zip")}
    )
    assert r.status_code == 200, r.text
    assert r.json()["member_ids"] == ["DV1"]


def test_upload_name_collides_with_existing_file_is_409_not_500(upload_client):
    # #2: a name that collides with an existing FILE (not just a dir) under the data root must be a clean
    # 409 with NO members ingested — create_dataset_dir's mkdir is the single, drift-free collision check.
    (upload_client.data_root / "afile").write_text(
        "x"
    )  # a plain file, not a dataset dir
    r = upload_client.post(
        "/members/upload",
        files={"file": ("x.zip", _dataset_zip(["CF1"]), "application/zip")},
        data={"name": "afile"},
    )
    assert r.status_code == 409
    assert "CF1" not in {m["member_id"] for m in upload_client.get("/members").json()}


# ---- content-based discovery (hold-out ships identical schemas under possibly-different names) ------


def _zip_of(files):
    """A zip from an explicit ``{relpath: bytes|str}`` map — for arbitrary-filename / content-routing
    tests (vs ``_dataset_zip`` which always names the members file ``members.json``)."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel, content in files.items():
            zf.writestr(rel, content)
    return buf.getvalue()


def _members_bytes(member_ids):
    import json

    return json.dumps([_bundle(m) for m in member_ids]).encode()


def _eval_records(ids, member_id="AX1"):
    return [
        {
            "id": i,
            "member_id": member_id,
            "category": "grounded_qa",
            "input": "what changed?",
            "expected_behavior": "describe trends",
            "must_include": [],
            "must_not": [],
            "escalation_expected": "none",
        }
        for i in ids
    ]


def test_upload_eval_loads_through_harness_array_or_jsonl(upload_client):
    # Criterion 4, END-TO-END: a valid JSONL eval file (one object per line) uploads and loads via the
    # SAME harness adapter with identical case ids — upload WRITES the canonical eval_set.jsonl, the
    # harness READS it.
    from eval.adapter import load_supplied_cases

    data = _zip_of(
        {
            "m.json": _members_bytes(["AX1"]),
            "ev.jsonl": _valid_eval_bytes(["E0", "E1", "E2"], "AX1"),
            "p.csv": _valid_csv_bytes(["AX1"]),
        }
    )
    upload_client.post(
        "/members/upload",
        files={"file": ("jl.zip", data, "application/zip")},
        data={"name": "evjsonl"},
    )
    assert [c.id for c in load_supplied_cases("evjsonl")] == ["E0", "E1", "E2"]


def test_upload_eval_first_row_json_array_rejected(upload_client):
    # Negative: the .jsonl eval role is gated line-by-line, so a first line that is a JSON ARRAY (not one
    # EvalCaseInput object) fails the first-row format gate -> 422, before any side effect.
    import json

    array_line = json.dumps(
        _eval_records(["E1"], "AX1")
    ).encode()  # a whole array on line 1
    data = _zip_of(
        {
            "m.json": _members_bytes(["AX1"]),
            "ev.jsonl": array_line,
            "p.csv": _valid_csv_bytes(["AX1"]),
        }
    )
    r = upload_client.post(
        "/members/upload",
        files={"file": ("arr.zip", data, "application/zip")},
        data={"name": "evarr"},
    )
    assert r.status_code == 422
    failures = r.json()["detail"]["failures"]
    assert any(f["file"] == "eval_set.jsonl" for f in failures)
    assert not (upload_client.data_root / "evarr").exists()


_GOOD_CSV_HEADER = (
    "member_id,panel_id,collected_date,analyte,value,unit,reference_range"
)


@pytest.mark.parametrize(
    "csv_text, expect_detail",
    [
        # wrong header (missing the reference_range column) — the header check fires before any data row
        (
            "member_id,panel_id,collected_date,analyte,value,unit\nX1,X1-P1,2024-01-15,HbA1c,5.3,%\n",
            "header must be exactly",
        ),
        # non-numeric value
        (
            _GOOD_CSV_HEADER + "\nX1,X1-P1,2024-01-15,HbA1c,notanum,%,<5.7\n",
            "not numeric",
        ),
        # non-ISO collected_date
        (
            _GOOD_CSV_HEADER + "\nX1,X1-P1,15-01-2024,HbA1c,5.3,%,<5.7\n",
            "not an ISO date",
        ),
        # wrong column count in the first data row (6 cells under the 7-column header)
        (
            _GOOD_CSV_HEADER + "\nX1,X1-P1,2024-01-15,HbA1c,5.3,%\n",
            "expected 7 columns",
        ),
        # unparseable reference_range (parsed by the SAME parse_reference_range the ingest uses)
        (
            _GOOD_CSV_HEADER + "\nX1,X1-P1,2024-01-15,HbA1c,5.3,%,not-a-range\n",
            "not a recognized range",
        ),
        # empty required field (member_id)
        (
            _GOOD_CSV_HEADER + "\n,X1-P1,2024-01-15,HbA1c,5.3,%,<5.7\n",
            "must not be empty",
        ),
    ],
)
def test_upload_bad_csv_first_row_rejected_422(upload_client, csv_text, expect_detail):
    # The lab-panels CSV first-row format gate — previously ZERO coverage while the members + eval gates
    # were both tested. A malformed header or first data row must 422 with a lab_panels.csv failure carrying
    # the specific cause, before any side effect (not a 500, not a silently-accepted bad CSV).
    data = _zip_of(
        {
            "m.json": _members_bytes(["X1"]),
            "ev.jsonl": _valid_eval_bytes(["E1"], "X1"),
            "p.csv": csv_text.encode(),
        }
    )
    r = upload_client.post(
        "/members/upload",
        files={"file": ("badcsv.zip", data, "application/zip")},
        data={"name": "badcsv"},
    )
    assert r.status_code == 422, r.text
    failures = r.json()["detail"]["failures"]
    assert any(
        f["file"] == "lab_panels.csv" and expect_detail in f["detail"] for f in failures
    )
    assert not (upload_client.data_root / "badcsv").exists()


def test_upload_opaque_member_id_roundtrips_through_scan(upload_client):
    # Different ID space: an opaque id (H-001, a UUID) must ingest, auto-scan, and read back — no format
    # assumption anywhere. The flagged HbA1c proves the full ingest -> auto-scan -> observations path.
    import io
    import json
    import zipfile

    uid, uuid = "H-001", "550e8400-e29b-41d4-a716-446655440000"
    flagged = _bundle(
        uid,
        panels=[
            make_panel(
                f"{uid}-P1", "2024-01-15", [make_result("HbA1c", 9.4, "%", "<5.7")]
            )
        ],
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("members.json", json.dumps([flagged, _bundle(uuid)]))
        zf.writestr("eval_set.jsonl", _valid_eval_bytes(["E1"], uid))
        zf.writestr("lab_panels.csv", _valid_csv_bytes([uid, uuid]))
    r = upload_client.post(
        "/members/upload",
        files={"file": ("opaque.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 200, r.text
    assert set(r.json()["member_ids"]) == {uid, uuid} and r.json()["scanned"] == 2
    picker = {m["member_id"] for m in upload_client.get("/members").json()}
    assert {uid, uuid} <= picker
    assert (
        len(upload_client.get(f"/members/{uid}/observations").json()) >= 1
    )  # auto-scanned


def test_upload_no_members_file_422_no_side_effects(upload_client):
    # Under the by-extension rules a `.json` file is bucketed as the MEMBERS role regardless of content, so
    # an eval-shaped `cases.json` fills the members slot — but the required `.jsonl` role is missing, and
    # that structural role check fires first. Either the missing role or the members first-row gate -> 422,
    # before any ingest or folder.
    import json

    data = _zip_of(
        {
            "panel.csv": _valid_csv_bytes(["AX1"]),
            "cases.json": json.dumps(
                _eval_records(["E1"])
            ).encode(),  # eval-shaped, but the lone .json -> members role
        }
    )
    r = upload_client.post(
        "/members/upload", files={"file": ("nomembers.zip", data, "application/zip")}
    )
    assert r.status_code == 422
    assert not (upload_client.data_root / "nomembers").exists()


def test_upload_two_members_files_422_no_partial(upload_client):
    # Two .json files -> the members role is ambiguous by extension (expected exactly one) -> reject with no
    # partial load. The OTHER two roles ARE present, so the duplicate-.json check is what fires in ISOLATION
    # (not masked by a missing-role check firing first) — and the failure names the members role explicitly.
    data = _zip_of(
        {
            "a.json": _members_bytes(["P1"]),
            "b.json": _members_bytes(["P2"]),
            "ev.jsonl": _valid_eval_bytes(["E1"], "P1"),
            "p.csv": _valid_csv_bytes(["P1"]),
        }
    )
    r = upload_client.post(
        "/members/upload", files={"file": ("ambig.zip", data, "application/zip")}
    )
    assert r.status_code == 422
    failures = r.json()["detail"]["failures"]
    assert any(
        f["file"] == "members.json" and "exactly one" in f["detail"] for f in failures
    )
    picker = {m["member_id"] for m in upload_client.get("/members").json()}
    assert not ({"P1", "P2"} & picker)  # nothing ingested
    assert not (upload_client.data_root / "ambig").exists()


def test_upload_duplicate_csv_role_rejected(upload_client):
    # The "exactly one .csv" contract IN ISOLATION (all three roles present, but TWO .csv) — the panels role
    # is ambiguous by extension -> 422, no side effect. Previously untested (the two-.json test above was
    # confounded by also-missing roles, and neither a duplicate .csv nor .jsonl had any coverage).
    data = _zip_of(
        {
            "m.json": _members_bytes(["D1"]),
            "ev.jsonl": _valid_eval_bytes(["E1"], "D1"),
            "a.csv": _valid_csv_bytes(["D1"]),
            "b.csv": _valid_csv_bytes(["D1"]),
        }
    )
    r = upload_client.post(
        "/members/upload",
        files={"file": ("dupcsv.zip", data, "application/zip")},
        data={"name": "dupcsv"},
    )
    assert r.status_code == 422
    failures = r.json()["detail"]["failures"]
    assert any(
        f["file"] == "lab_panels.csv" and "exactly one" in f["detail"] for f in failures
    )
    assert not (upload_client.data_root / "dupcsv").exists()


def test_upload_duplicate_jsonl_role_rejected(upload_client):
    # Same "exactly one" contract for the eval role: two .jsonl -> ambiguous -> 422, no side effect.
    data = _zip_of(
        {
            "m.json": _members_bytes(["D1"]),
            "a.jsonl": _valid_eval_bytes(["E1"], "D1"),
            "b.jsonl": _valid_eval_bytes(["E2"], "D1"),
            "p.csv": _valid_csv_bytes(["D1"]),
        }
    )
    r = upload_client.post(
        "/members/upload",
        files={"file": ("dupjsonl.zip", data, "application/zip")},
        data={"name": "dupjsonl"},
    )
    assert r.status_code == 422
    failures = r.json()["detail"]["failures"]
    assert any(
        f["file"] == "eval_set.jsonl" and "exactly one" in f["detail"] for f in failures
    )
    assert not (upload_client.data_root / "dupjsonl").exists()


@pytest.mark.parametrize("good", ["holdout", "data-2024", "set.v1", "A_b-9"])
def test_sanitize_dataset_name_accepts_safe(good):
    assert datasets.sanitize_dataset_name(good) == good


@pytest.mark.parametrize(
    "bad", ["../x", "a/b", "a\\b", "..", ".", "", "   ", ".hidden", "x" * 65]
)
def test_sanitize_dataset_name_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        datasets.sanitize_dataset_name(bad)


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
    # A correction feeds the deterministic core, NOT /learn — the response says so authoritatively.
    assert r.json()["channel"] == "core"


def test_post_feedback_response_advertises_learn_only_for_learn_channel(client):
    # The fix for the misleading "Feeds the next Run learn gate" copy: an advisory signal
    # (escalation_accept) must report channel="advisory" (not "learn") so the UI won't advertise Run learn.
    # (No Haiku judge on this path — validate_feedback only screens 'incorrect' — so it's keyless-safe.)
    r = client.post(
        "/members/C01/feedback",
        json={"kind": "escalation_accept", "target": "f:x", "source": "clinician"},
    )
    assert r.status_code == 200
    assert r.json()["channel"] == "advisory"
    assert "does not feed /learn" in r.json()["effect"]


def test_post_feedback_unknown_member_404(client):
    r = client.post(
        "/members/NOPE/feedback",
        json={"kind": "preference", "payload": {"text": "brief"}, "source": "member"},
    )
    assert r.status_code == 404


def _incorrect_feedback(answer):
    return {
        "kind": "incorrect",
        "target": "obs:x",
        "payload": {
            "question": "Tell me about my Fasting glucose",
            "corrected_answer": answer,
        },
        "source": "clinician",
    }


def test_post_feedback_empty_corrected_answer_422_deterministically(client):
    # The DETERMINISTIC bound runs before the judge, so an empty corrected answer 422s with no model call
    # (no API key needed) — the cheap floor beneath the Haiku input-judge.
    r = client.post("/members/C01/feedback", json=_incorrect_feedback("   "))
    assert r.status_code == 422 and "empty" in r.json()["detail"]


def test_post_feedback_maps_judge_unavailable_to_503_fail_closed(client, monkeypatch):
    # Route wiring: when the Haiku judge can't run, validate_feedback raises LearnUnavailable and the route
    # maps it to 503 — FAIL-CLOSED (retry, never store unjudged), the safe direction since the bar is the
    # only off-eval-junk catch. (The real fail-closed path lives in test_learn.py; this pins the mapping.)
    def _down(fb):
        raise learn.LearnUnavailable(
            "the feedback review service is unavailable; try again"
        )

    monkeypatch.setattr(learn, "validate_feedback", _down)
    r = client.post(
        "/members/C01/feedback", json=_incorrect_feedback("a well-formed answer here")
    )
    # 503 -> the route raised before db.insert_feedback, so nothing was stored.
    assert r.status_code == 503


def test_post_feedback_maps_a_judge_rejection_to_422(client, monkeypatch):
    # Route wiring: when the judge deems a corrected answer unfit, validate_feedback returns its reason and
    # the route maps it to a 422 with that reason (the judge itself is exercised in test_learn.py).
    monkeypatch.setattr(
        learn, "validate_feedback", lambda fb: "incoherent — not an answer"
    )
    r = client.post("/members/C01/feedback", json=_incorrect_feedback("makes no sense"))
    assert r.status_code == 422 and r.json()["detail"] == "incoherent — not an answer"


def test_post_feedback_non_incorrect_correction_stores_without_a_judge(client):
    # A range_override is not an exemplar, so validate_feedback never calls the judge — it 200s even with
    # no LLM configured (only 'incorrect' corrections are judged).
    r = client.post(
        "/members/C01/feedback",
        json={
            "kind": "suppress_marker",
            "target": "LDL cholesterol",
            "source": "clinician",
        },
    )
    assert r.status_code == 200 and r.json()["feedback_id"]


def test_post_feedback_core_correction_auto_scans_the_member(client):
    # ingest_dataset alone never scans (the CLI's auto-scan lives in its main()), so this fixture starts
    # with NO persisted observations — what populates them below is the feedback route's own auto-scan.
    assert client.get("/members/C01/observations").json() == []
    r = client.post(
        "/members/C01/feedback",
        json={
            "kind": "range_override",
            "target": "LDL cholesterol",
            "payload": {"ref_high": 250.0},
            "source": "clinician",
        },
    )
    assert r.status_code == 200 and r.json()["rescanned"] is True
    titles = [o["title"] for o in client.get("/members/C01/observations").json()]
    assert titles  # the auto-scan persisted C01's findings — no manual /scan call anywhere in this test
    # ...and the scan consumed the JUST-committed override (insert-then-scan ordering): un-overridden,
    # C01's LDL scans as "LDL cholesterol above range"; the wide re-bound clears that flag, so the
    # auto-scan raised no LDL finding.
    assert not any(t.startswith("LDL cholesterol") for t in titles)


def test_post_feedback_auto_scan_equals_a_manual_rescan(client):
    # The auto-scan is the SAME idempotent scan the Scan button runs: a manual re-scan right after a core
    # correction must change nothing — the feedback route already left the projection current. (Also pins
    # suppress-through-auto-scan: the suppressed marker's observation is pruned without a manual scan.)
    client.post(
        "/members/C01/feedback",
        json={"kind": "suppress_marker", "target": "HbA1c", "source": "clinician"},
    )
    after_feedback = client.get("/members/C01/observations").json()
    assert after_feedback  # C01 still raises its other findings
    assert not any(o["title"].startswith("HbA1c") for o in after_feedback)
    client.post("/members/C01/scan")
    after_manual = client.get("/members/C01/observations").json()
    assert after_manual == after_feedback


def test_post_feedback_non_core_kinds_skip_the_auto_scan(client, monkeypatch):
    # Only a core correction changes analyze()'s inputs; a preference (composer channel) and the signal
    # kinds (learn/advisory) must not pay for — or imply — a rescan. Spy on the seam the route calls.
    calls = []
    monkeypatch.setattr(
        api.pipeline,
        "scan_members",
        lambda con, ids: calls.append(list(ids)) or api.pipeline.ScanSweep(1, 0),
    )
    for body in (
        {
            "kind": "preference",
            "payload": {"text": "keep it brief"},
            "source": "member",
        },
        {"kind": "helpful", "target": "obs:x", "source": "clinician"},
        {"kind": "escalation_accept", "target": "esc:x", "source": "clinician"},
        {"kind": "escalation_reject", "target": "esc:x", "source": "clinician"},
    ):
        r = client.post("/members/C01/feedback", json=body)
        assert r.status_code == 200 and r.json()["rescanned"] is False
    assert calls == []


def test_post_feedback_escalation_reject_reason_descriptor(client):
    # The rejection descriptor rides in the payload: a filled reason stores (200); a present-but-blank one
    # 422s DETERMINISTICALLY (no judge, no key needed — the reason is audit text, never prompt text). The
    # form requires the reason; the API bounds it when present.
    r = client.post(
        "/members/C01/feedback",
        json={
            "kind": "escalation_reject",
            "target": "esc:x",
            "payload": {"reason": "expected under the member's current treatment plan"},
            "source": "clinician",
        },
    )
    assert r.status_code == 200 and r.json()["feedback_id"]
    r = client.post(
        "/members/C01/feedback",
        json={
            "kind": "escalation_reject",
            "target": "esc:x",
            "payload": {"reason": "   "},
            "source": "clinician",
        },
    )
    assert r.status_code == 422 and "reason is empty" in r.json()["detail"]


def test_escalation_accept_acknowledges_on_both_reads(client):
    # An accept flips the targeted escalation to 'acknowledged' on BOTH the global queue and the member
    # drill-in (one shared read path) — still LISTED, never hidden; the response reports it; a re-scan
    # cannot undo it (nothing stored changed — the overlay derives from the active accept row); /reset
    # (deactivating that row) reverts it to 'open' symmetrically.
    client.post("/members/C01/scan")
    esc = next(e for e in client.get("/escalations").json() if e["member_id"] == "C01")
    assert esc["status"] == "open"
    r = client.post(
        "/members/C01/feedback",
        json={
            "kind": "escalation_accept",
            "target": esc["observation_id"],
            "source": "clinician",
        },
    )
    assert r.status_code == 200 and r.json()["acknowledged"] is True

    def queue_status():
        return next(
            e["status"]
            for e in client.get("/escalations").json()
            if e["escalation_id"] == esc["escalation_id"]
        )

    assert queue_status() == "acknowledged"  # on the queue, visibly triaged
    drill = next(
        e
        for e in client.get("/members/C01/escalations").json()
        if e["escalation_id"] == esc["escalation_id"]
    )
    assert drill["status"] == "acknowledged"
    client.post("/members/C01/scan")  # reconcile re-runs; the overlay survives it
    assert queue_status() == "acknowledged"
    client.post("/reset")  # deactivates the accept row -> symmetric revert
    assert queue_status() == "open"


def test_escalation_accept_miss_reports_false_and_severity_dominates_ordering(client):
    # A target matching no live escalation acknowledges nothing (reported False; the row is still stored
    # for audit). And on the queue, severity DOMINATES the acknowledged ordering: an acknowledged urgent
    # is never buried below an open clinician_review — acknowledged only sorts last WITHIN its tier.
    client.post("/members/C07/scan")  # -> urgent
    client.post("/members/C01/scan")  # -> clinician_review
    r = client.post(
        "/members/C07/feedback",
        json={"kind": "escalation_accept", "target": "esc:nope", "source": "clinician"},
    )
    assert r.status_code == 200 and r.json()["acknowledged"] is False
    urgent = next(
        e for e in client.get("/escalations").json() if e["level"] == "urgent"
    )
    r = client.post(
        "/members/C07/feedback",
        json={
            "kind": "escalation_accept",
            "target": urgent[
                "escalation_id"
            ],  # accept by escalation_id (the chat-escalation path)
            "source": "clinician",
        },
    )
    assert r.json()["acknowledged"] is True
    queue = client.get("/escalations").json()
    keys = [
        (
            {"urgent": 0, "clinician_review": 1}[e["level"]],
            e["status"] == "acknowledged",
        )
        for e in queue
    ]
    assert keys == sorted(keys)  # severity first, acknowledged-last within a tier
    idx_acked = next(
        i for i, e in enumerate(queue) if e["escalation_id"] == urgent["escalation_id"]
    )
    assert queue[idx_acked]["status"] == "acknowledged"
    first_cr = next(i for i, e in enumerate(queue) if e["level"] == "clinician_review")
    assert (
        idx_acked < first_cr
    )  # the acknowledged urgent still outranks every clinician_review


def test_escalation_accept_from_member_source_is_inert_on_the_queue(client):
    # The acknowledged overlay is SOURCE-FILTERED like resolve_overrides ("the member experience cannot
    # lobby the safety logic"): a member-sourced accept is stored for audit but must not mark a clinician
    # triage item as already-reviewed or demote it within its tier — only clinician/system accepts do
    # (the docs promise the overlay derives from a CLINICIAN's accept).
    client.post("/members/C01/scan")
    esc = next(e for e in client.get("/escalations").json() if e["member_id"] == "C01")
    r = client.post(
        "/members/C01/feedback",
        json={
            "kind": "escalation_accept",
            "target": esc["observation_id"],
            "source": "member",
        },
    )
    assert r.status_code == 200 and r.json()["acknowledged"] is False
    assert (
        next(
            e["status"]
            for e in client.get("/escalations").json()
            if e["escalation_id"] == esc["escalation_id"]
        )
        == "open"
    )  # still visibly un-triaged on the queue


def test_supersede_wins_over_acknowledged(client):
    # An acknowledged clinician_review whose finding an override then CLEARS is superseded off the active
    # queue — the stored data-cleared fact beats the overlay (the drill-in shows 'superseded', never
    # 'acknowledged'): dropping a cleared finding off the worklist is the stronger, safe direction.
    client.post("/members/C01/scan")
    esc = next(e for e in client.get("/escalations").json() if e["member_id"] == "C01")
    obs = next(
        o
        for o in client.get("/members/C01/observations").json()
        if o["observation_id"] == esc["observation_id"]
    )
    marker = next(
        t["marker"]
        for t in client.get("/members/C01/trajectory").json()
        if obs["title"].startswith(t["marker"])
    )
    client.post(
        "/members/C01/feedback",
        json={
            "kind": "escalation_accept",
            "target": esc["observation_id"],
            "source": "clinician",
        },
    )
    # Clear the finding: suppress the marker (core channel -> auto-scan -> reconcile supersedes).
    client.post(
        "/members/C01/feedback",
        json={"kind": "suppress_marker", "target": marker, "source": "clinician"},
    )
    drill = next(
        e
        for e in client.get("/members/C01/escalations").json()
        if e["escalation_id"] == esc["escalation_id"]
    )
    assert drill["status"] == "superseded"
    assert esc["escalation_id"] not in {
        e["escalation_id"] for e in client.get("/escalations").json()
    }


def test_get_feedback_returns_the_audit_trail_newest_first(client):
    # The read-only audit projection over the feedback table: every submitted row comes back newest-first
    # with its payload (an escalation_reject's reason round-trips) and storage fields; /reset flips
    # `active` without dropping rows — the trail survives, flagged.
    assert (
        client.get("/members/NOPE/feedback").status_code == 404
    )  # typo'd id != empty trail
    assert client.get("/members/C01/feedback").json() == []
    client.post(
        "/members/C01/feedback",
        json={"kind": "suppress_marker", "target": "HbA1c", "source": "clinician"},
    )
    client.post(
        "/members/C01/feedback",
        json={
            "kind": "escalation_reject",
            "target": "esc:x",
            "payload": {"reason": "expected under treatment"},
            "source": "clinician",
        },
    )
    rows = client.get("/members/C01/feedback").json()
    assert [r["kind"] for r in rows] == ["escalation_reject", "suppress_marker"]
    assert rows[0]["payload"] == {"reason": "expected under treatment"}
    assert all(r["active"] and r["feedback_id"] and r["created_at"] for r in rows)
    client.post("/reset")
    rows = client.get("/members/C01/feedback").json()
    assert len(rows) == 2 and not any(
        r["active"] for r in rows
    )  # deactivated, never deleted


def test_post_feedback_auto_scan_failure_never_fails_the_correction(
    client, monkeypatch
):
    # Best-effort, same contract as every auto-scan path: the correction commits BEFORE the scan runs, so
    # a scan blow-up must surface as rescanned=False on a 200 — never a 500 (the client would read failure
    # for a recorded override) and never a lost row. scan_members itself absorbs per-member failures; this
    # raises from the seam directly to pin the route's own envelope against a systemic crash.
    def _boom(con, ids):
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(api.pipeline, "scan_members", _boom)
    r = client.post(
        "/members/C01/feedback",
        json={"kind": "suppress_marker", "target": "HbA1c", "source": "clinician"},
    )
    assert r.status_code == 200 and r.json()["feedback_id"]
    assert r.json()["rescanned"] is False
    con = db.connect(_DB_FILE)
    try:
        stored = con.execute(
            "SELECT COUNT(*) FROM feedback WHERE kind='suppress_marker' AND active=1"
        ).fetchone()[0]
    finally:
        con.close()
    assert stored == 1


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


@pytest.mark.skipif(db.fcntl is None, reason="flock is POSIX-only")
def test_reset_and_reseed_409_while_learn_holds_the_learning_state_lock(client):
    """The multi-worker exclusion (--workers 2): while a /learn run holds the cross-process
    learning-state lock (here: a raw flock standing in for the OTHER worker's run), /reset and
    /admin/reseed must 409 rather than truncate/flip the feedback + prompt_versions tables the run is
    gating against — and both must succeed again the moment the lock is released."""
    con = db.connect(_DB_FILE)
    lock_path = db._process_lock_path(con, learn.LEARN_LOCK_NAME)
    con.close()
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    db.fcntl.flock(fd, db.fcntl.LOCK_EX | db.fcntl.LOCK_NB)  # the other worker's /learn
    try:
        assert client.post("/reset").status_code == 409
        assert client.post("/admin/reseed").status_code == 409
    finally:
        db.fcntl.flock(fd, db.fcntl.LOCK_UN)
        os.close(fd)
    assert client.post("/reset").status_code == 200  # released -> normal service
    assert client.post("/admin/reseed").status_code == 200


@pytest.mark.skipif(db.fcntl is None, reason="flock is POSIX-only")
def test_ingest_routes_and_reseed_409_while_the_dataset_lock_is_held(client):
    """The dataset/bulk-ingest exclusion (--workers 2): while another worker holds the shared 'data'
    lock (a reseed mid-wipe, or another upload mid-ingest), POST /members and /admin/reseed must 409
    rather than interleave per-member commits with a truncate + dataset-folder rmtree — and succeed
    again the moment the lock is released. (POST /members/upload takes the same lock via the same
    helper.)"""
    con = db.connect(_DB_FILE)
    lock_path = db._process_lock_path(con, api._DATA_LOCK_NAME)
    con.close()
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    db.fcntl.flock(fd, db.fcntl.LOCK_EX | db.fcntl.LOCK_NB)  # the other worker's ingest
    try:
        assert client.post("/members", json=_bundle("Z99")).status_code == 409
        assert client.post("/admin/reseed").status_code == 409
    finally:
        db.fcntl.flock(fd, db.fcntl.LOCK_UN)
        os.close(fd)
    assert client.post("/members", json=_bundle("Z99")).status_code == 200
    assert client.post("/admin/reseed").status_code == 200  # released -> normal service


def test_seed_if_empty_is_atomic_on_a_mid_seed_failure(tmp_path, monkeypatch):
    """The startup seed is ONE transaction: any mid-seed failure leaves the DB EMPTY (rollback), never a
    committed partial prefix — with per-member commits, the emptiness guard would read a stranded prefix
    as 'seeded' forever, and under --workers 2 the sibling lifespan would boot clean over it in the same
    cycle, masking the crash. The single-transaction shape also covers process death (SIGKILL/OOM): an
    uncommitted transaction dies with the process."""
    import preprocessing.ingest as ingest_mod

    con = fresh_con(str(tmp_path / "seed.db"))
    real = ingest_mod.ingest_bundle
    calls = {"n": 0}

    def failing(con_, bundle, commit=True):
        calls["n"] += 1
        if calls["n"] > 3:  # fail mid-loop, after some members are already written
            raise RuntimeError("boom mid-seed")
        return real(con_, bundle, commit=commit)

    monkeypatch.setattr(ingest_mod, "ingest_bundle", failing)
    with pytest.raises(RuntimeError):
        ingest_mod.seed_if_empty(con)
    assert db.list_members(con) == []  # rolled back to EMPTY — the next boot retries
    monkeypatch.undo()
    out = ingest_mod.seed_if_empty(con)  # the retry seeds cleanly
    assert out["seeded"] is True and db.list_members(con)
    con.close()


# ---- GET /prompts (the operator's Prompt-history readout) ------------------------------------------


def test_prompt_history_route_serves_version_trail(client):
    """GET /prompts — the Learning group's Prompt-history button: the composer-prompt version trail,
    newest first. On a fresh instance the lifespan-seeded v0 baseline is present (promoted, active,
    full prompt text, no eval summary — the report attaches lazily on the first /learn), and exactly
    ONE row is flagged active (the version the composer serves). A candidate row a later /learn wrote
    joins the trail with its gate verdict; a rejected one never steals `active`."""
    r = client.get("/prompts")
    assert r.status_code == 200
    hist = r.json()
    assert hist  # v0 is seeded at startup — the history is never empty
    versions = [p["version"] for p in hist]
    assert versions == sorted(versions, reverse=True)  # newest first
    v0 = next(p for p in hist if p["version"] == 0)
    assert v0["status"] == "promoted" and v0["active"] is True
    assert v0["prompt_text"]  # the full text is the inspectable artifact
    assert v0["eval_summary"] is None  # v0 seeds report-less
    assert sum(p["active"] for p in hist) == 1
    # a gated /learn candidate joins the trail; REJECTED -> recorded but never active
    con = db.connect(_DB_FILE)
    try:
        db.insert_prompt_version(
            con, version=99, prompt_text="candidate", status="rejected"
        )
    finally:
        con.close()
    hist = client.get("/prompts").json()
    assert hist[0]["version"] == 99 and hist[0]["status"] == "rejected"
    assert hist[0]["active"] is False
    assert next(p for p in hist if p["version"] == 0)["active"] is True


# ---- POST /admin/reseed (Phase 7) -----------------------------------------------------------------


def test_reseed_restores_initial_members_and_drops_holdouts(client):
    client.delete("/members/C01")  # remove a seeded member
    client.post("/members", json=_bundle("T99"))  # add an uploaded holdout
    r = client.post("/admin/reseed")
    assert r.status_code == 200 and r.json()["reseeded"] is True
    assert r.json()["scanned"] == 15  # every ingest path auto-scans what it loaded
    assert (
        r.json()["new_observations"] > 0
    )  # the truncate emptied observations, so the reseed sweep re-creates them all as new
    ids = [m["member_id"] for m in client.get("/members").json()]
    assert "C01" in ids  # the deleted seeded member is back
    assert "T99" not in ids  # the uploaded holdout is dropped
    assert len(ids) == 15  # back to the initial state
    # Observations are consistent with the restored data WITHOUT a manual scan: C07 (the panic member)
    # has findings and its urgent escalation straight off the factory reset.
    n_auto = len(client.get("/members/C07/observations").json())
    assert n_auto >= 1
    queue = client.get("/escalations").json()
    assert any(e["member_id"] == "C07" and e["level"] == "urgent" for e in queue)
    # Idempotent under the manual button: a reviewer clicking Scan after the auto-scan re-runs the same
    # scan (observations overwrite on their dedup key, escalations fire once) — no duplicate findings.
    client.post("/members/C07/scan")
    assert len(client.get("/members/C07/observations").json()) == n_auto
    urgent = [
        e
        for e in client.get("/escalations").json()
        if e["member_id"] == "C07" and e["level"] == "urgent"
    ]
    assert len(urgent) == len(
        [e for e in queue if e["member_id"] == "C07" and e["level"] == "urgent"]
    )
    # The v0 baseline prompt is materialized post-reseed (not left to a lazy /learn) — promoted, BASE
    # text, no report yet — so the prompt store is never empty after a factory reset.
    con = db.connect(_DB_FILE)
    try:
        rows = con.execute(
            "SELECT version, status, eval_report_json FROM prompt_versions"
        ).fetchall()
    finally:
        con.close()
    assert [(r["version"], r["status"]) for r in rows] == [(0, "promoted")]
    assert rows[0]["eval_report_json"] is None


def test_reseed_removes_uploaded_dataset_folders_keeping_training_data(upload_client):
    # The bug: a DB-only reseed dropped an uploaded holdout's MEMBERS but left its <root>/<name>/ FOLDER
    # stranded on disk — still discoverable and re-selectable via DATASET=<name>. A factory reset must
    # return the datasets root to just the shipped bundle.
    data = _dataset_zip(["U01", "U02"], top="holdout")
    up = upload_client.post(
        "/members/upload", files={"file": ("holdout.zip", data, "application/zip")}
    )
    assert up.status_code == 200, up.text
    holdout = upload_client.data_root / "holdout"
    assert holdout.is_dir()  # the upload created the folder

    r = upload_client.post("/admin/reseed")
    assert r.status_code == 200, r.text
    assert r.json()["datasets_removed"] == ["holdout"]  # reported on the response
    assert not holdout.exists()  # folder gone from disk
    assert (upload_client.data_root / "training_data").is_dir()  # shipped bundle KEPT
    ids = [m["member_id"] for m in upload_client.get("/members").json()]
    assert (
        "C01" in ids and "U01" not in ids
    )  # DB back to the 15 shipped; uploaded members dropped


def test_remove_uploaded_datasets_keeps_training_data_and_files(tmp_path, monkeypatch):
    # Unit-level: only non-default SUB-DIRECTORIES go; the shipped bundle and any FILE (the DB, a README,
    # a stray zip) are untouched.
    monkeypatch.setenv("HEALTH_DATA_ROOT", str(tmp_path))
    (tmp_path / "training_data").mkdir()
    (tmp_path / "holdout_a").mkdir()
    (tmp_path / "holdout_b").mkdir()
    (tmp_path / "health.db").write_text("db")
    (tmp_path / "README.md").write_text("readme")
    (tmp_path / "upload_test.zip").write_text("zip")

    assert datasets.remove_uploaded_datasets() == ["holdout_a", "holdout_b"]  # sorted
    assert (tmp_path / "training_data").is_dir()  # shipped bundle kept
    assert (
        not (tmp_path / "holdout_a").exists() and not (tmp_path / "holdout_b").exists()
    )
    assert (tmp_path / "health.db").is_file()  # files never touched
    assert (tmp_path / "README.md").is_file()
    assert (tmp_path / "upload_test.zip").is_file()


def test_remove_uploaded_datasets_skips_symlinks(tmp_path, monkeypatch):
    # rmtree must never follow a symlink out of the root and delete external content.
    monkeypatch.setenv("HEALTH_DATA_ROOT", str(tmp_path))
    (tmp_path / "training_data").mkdir()
    external = tmp_path.parent / "external_precious"
    external.mkdir()
    (external / "keep.txt").write_text("precious")
    (tmp_path / "linked").symlink_to(external, target_is_directory=True)

    assert "linked" not in datasets.remove_uploaded_datasets()  # symlink skipped
    assert (external / "keep.txt").is_file()  # external content never followed/deleted


def test_remove_uploaded_datasets_swallows_a_listing_error(tmp_path, monkeypatch):
    # Best-effort AND TOTAL: a raising root.iterdir() (unreadable root / TOCTOU) is logged+swallowed, not
    # propagated — else it 500s a reseed that already committed. The original guarded only rmtree.
    monkeypatch.setenv("HEALTH_DATA_ROOT", str(tmp_path))
    (tmp_path / "training_data").mkdir()

    def boom_iterdir(self):
        raise PermissionError("listing denied")

    monkeypatch.setattr(pathlib.Path, "iterdir", boom_iterdir)
    assert datasets.remove_uploaded_datasets() == []  # swallowed, not raised


def test_remove_uploaded_datasets_swallows_a_per_child_stat_error(
    tmp_path, monkeypatch
):
    # Best-effort: an OSError from a per-child stat (e.g. EACCES, which pathlib re-raises) is skipped, not
    # raised. The original stat calls sat OUTSIDE the try that wrapped only rmtree, so this would have 500'd.
    monkeypatch.setenv("HEALTH_DATA_ROOT", str(tmp_path))
    (tmp_path / "training_data").mkdir()
    (tmp_path / "boom").mkdir()
    real_is_dir = pathlib.Path.is_dir

    def flaky_is_dir(self):
        if self.name == "boom":
            raise PermissionError("stat denied")
        return real_is_dir(self)

    monkeypatch.setattr(pathlib.Path, "is_dir", flaky_is_dir)
    assert (
        datasets.remove_uploaded_datasets() == []
    )  # 'boom' skipped on its stat error, not raised
    assert (tmp_path / "boom").exists()  # never removed


def test_cold_start_seeds_the_v0_baseline_prompt(client):
    # The lifespan (cold start) MUST materialize v0. With the tightened _baseline_report (/learn no longer
    # mints the baseline), this startup seed — api.py's learn.seed_baseline_prompt — is the ONLY guarantee
    # the prompt store isn't empty on a fresh deploy. _reset_and_seed wipes prompt_versions (a true cold
    # start); the TestClient context runs the lifespan, which seeds v0. Comment out that seed call and this
    # test fails — it guards the production path the error-not-mint design now depends on.
    con = db.connect(_DB_FILE)
    try:
        rows = con.execute("SELECT version, status FROM prompt_versions").fetchall()
    finally:
        con.close()
    assert [(r["version"], r["status"]) for r in rows] == [(0, "promoted")]


def test_cold_start_seeds_and_auto_scans_the_members():
    # A TRUE cold start (no members — a fresh ephemeral deploy / spun-down free tier): the lifespan
    # seeds the 15 AND auto-scans them, so a reviewer's first page load shows findings without knowing
    # to click Scan. The standard `client` fixture pre-seeds (its lifespan seed no-ops), so this test
    # wipes members itself and lets the lifespan do BOTH.
    con = fresh_con(_DB_FILE)
    for t in [*_CHILD_TABLES, "members", "prompt_versions"]:
        con.execute(f"DELETE FROM {t}")
    con.commit()
    con.close()
    with TestClient(api.app) as c:
        assert len(c.get("/members").json()) == 15  # the lifespan seeded
        # ...and scanned: C07's panic K+ is an observation + an urgent escalation with NO manual scan.
        assert len(c.get("/members/C07/observations").json()) >= 1
        queue = c.get("/escalations").json()
        assert any(e["member_id"] == "C07" and e["level"] == "urgent" for e in queue)
