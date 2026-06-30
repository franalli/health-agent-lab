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


# ---- POST /members/upload (full bundle persisted as a new dataset folder; additive; auto-scan) ------

#: The shipped bundle, resolved from the DEFAULT root (not data_root()) so it's found whatever the
#: HEALTH_DATA_ROOT override is — the source we copy into each test's isolated root.
_REAL_TRAINING = datasets._DEFAULT_DATA_ROOT / "training_data"


@pytest.fixture
def upload_client(tmp_path, monkeypatch):
    """A client whose datasets root is an isolated temp dir (``HEALTH_DATA_ROOT``) seeded with a copy of
    the shipped ``training_data`` — so an upload CREATES its dataset folder in the temp dir, never the
    repo, and teardown is automatic. ``monkeypatch.setenv`` is per-test (no process-wide leak into the
    other test modules that read the real root)."""
    root = tmp_path / "data"
    root.mkdir()
    shutil.copytree(_REAL_TRAINING, root / "training_data")
    monkeypatch.setenv("HEALTH_DATA_ROOT", str(root))
    _reset_and_seed()  # data_root() now resolves to `root`; seeds the 15 from the copied training_data
    with TestClient(api.app) as c:
        c.data_root = root  # let tests assert on the created folder
        yield c


def _dataset_zip(member_ids, *, top="bundle", extra=None, mutate=None):
    """Build a dataset .zip: ``members.json`` (+ optional ``extra`` ``{relpath: str|bytes}``) under a
    single wrapping folder ``top`` (``None`` -> at the zip root)."""
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
        for rel, content in (extra or {}).items():
            zf.writestr(prefix + rel, content)
    return buf.getvalue()


def test_upload_zip_persists_folder_and_ingests_additively(upload_client):
    # The whole requirement: keep EVERYTHING (members.json + csv + jsonl) as a new dataset folder, AND
    # add the members on top of the existing 15 (no reseed).
    data = _dataset_zip(
        ["U01", "U02", "U03"],
        top="holdout",
        extra={"lab_panels.csv": "a,b\n1,2\n", "eval_set.jsonl": '{"input":"hi"}\n'},
    )
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

    member = _bundle(
        "S01",
        panels=[
            make_panel("S01-P1", "2024-01-15", [make_result("HbA1c", 9.0, "%", "<5.7")])
        ],
    )
    data = io.BytesIO(json.dumps([member]).encode())
    r = upload_client.post(
        "/members/upload", files={"file": ("scanme.json", data, "application/json")}
    )
    assert r.status_code == 200, r.text
    assert r.json()["scanned"] == 1
    obs = upload_client.get("/members/S01/observations").json()
    assert len(obs) >= 1  # auto-scan produced + stored the out-of-range observation
    hba1c = next(o for o in obs if o["title"].startswith("HbA1c"))
    # the member-facing field rides the wire, states the numeric range, and leaks no clinician stats
    assert "above the normal range" in hba1c["member_explanation"]
    assert "under 5.7" in hba1c["member_explanation"]
    assert "Mann-Kendall" not in hba1c["member_explanation"]


def test_upload_bare_json_array_creates_dataset(upload_client):
    import io
    import json

    data = io.BytesIO(json.dumps([_bundle("U10"), _bundle("U11")]).encode())
    r = upload_client.post(
        "/members/upload", files={"file": ("myset.json", data, "application/json")}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"] == "myset" and body["member_ids"] == ["U10", "U11"]
    assert (upload_client.data_root / "myset" / "members.json").is_file()


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
    # A SHAPE error (extra='forbid') is caught in the up-front Pydantic pass — before ANY DB write and
    # before the folder is created. Neither the sibling member nor the folder lands.
    data = _dataset_zip(["U20", "U21"], mutate=lambda b: b["profile"].update(oops="x"))
    r = upload_client.post(
        "/members/upload", files={"file": ("bad.zip", data, "application/zip")}
    )
    assert r.status_code == 422
    picker = {m["member_id"] for m in upload_client.get("/members").json()}
    assert not ({"U20", "U21"} & picker)
    assert not (upload_client.data_root / "bad").exists()  # no folder on a shape error


def test_upload_late_semantic_error_422_partial_db_no_folder(upload_client):
    # The documented boundary: the SEMANTIC firewall parse runs per-member inside the committing write
    # loop, so a parse failure in a LATER bundle raises after the earlier valid member is written — a
    # partial DB set persists by design. But the folder is created only AFTER ingest, so it is NOT left
    # behind. If this flips, update ingest_members' + the route's docstrings.
    import io
    import json
    import zipfile

    good, bad = _bundle("U22"), _bundle("U23")
    bad["panels"][0]["results"][0]["reference_range"] = "not-a-range"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("members.json", json.dumps([good, bad]))
    r = upload_client.post(
        "/members/upload",
        files={"file": ("sem.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 422
    picker = {m["member_id"] for m in upload_client.get("/members").json()}
    assert "U22" in picker and "U23" not in picker  # partial DB set, by design
    assert not (
        upload_client.data_root / "sem"
    ).exists()  # folder created only after a clean ingest


def test_upload_not_a_zip_returns_422(upload_client):
    r = upload_client.post(
        "/members/upload",
        files={"file": ("x.zip", b"definitely not a zip", "application/zip")},
    )
    assert r.status_code == 422


def test_upload_single_object_accepted_as_one_member_dataset(upload_client):
    # A bare member-bundle OBJECT (not an array) is members-shaped, so the content classifier accepts it
    # as a one-member dataset — robustly, whether compact or pretty-printed (read_records handles `{`).
    import io
    import json

    data = io.BytesIO(
        json.dumps(_bundle("U30"), indent=2).encode()
    )  # pretty-printed on purpose
    r = upload_client.post(
        "/members/upload", files={"file": ("one.json", data, "application/json")}
    )
    assert r.status_code == 200, r.text
    assert r.json()["member_ids"] == ["U30"]


def test_upload_members_non_json_extension_resolves_by_content(upload_client):
    # #8: discovery is by CONTENT, not extension — a members file under a non-.json name still resolves.
    data = _zip_of({"roster.dat": _members_bytes(["AX1"]), "panel.csv": b"a,b\n1,2\n"})
    r = upload_client.post(
        "/members/upload", files={"file": ("odd.zip", data, "application/zip")}
    )
    assert r.status_code == 200, r.text
    assert r.json()["member_ids"] == ["AX1"]


def test_upload_bom_members_and_eval(upload_client):
    # #5: a UTF-8 BOM on the members AND eval files must not break classification or the eval load.
    import json

    from eval.adapter import load_supplied_cases

    bom = b"\xef\xbb\xbf"
    data = _zip_of(
        {
            "m.json": bom + _members_bytes(["BX1"]),  # BOM'd array members
            "e.json": bom
            + json.dumps(_eval_records(["EB1"])).encode(),  # BOM'd array eval
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


def test_upload_jsonl_members_accepted(upload_client):
    # #7: a JSONL-form members file (one bundle per line) is accepted, not rejected as "not valid JSON".
    import json

    jsonl = "\n".join(json.dumps(_bundle(m)) for m in ["JL1", "JL2"]).encode()
    data = _zip_of({"members.jsonl": jsonl})
    r = upload_client.post(
        "/members/upload", files={"file": ("jl.zip", data, "application/zip")}
    )
    assert r.status_code == 200, r.text
    assert set(r.json()["member_ids"]) == {"JL1", "JL2"}


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
    # #X1: a non-UTF-8 (no-BOM) members file is not valid JSON (JSON must be Unicode); read_records must
    # surface a clean ValueError -> 422, not let a raw UnicodeDecodeError escape.
    import io

    # cp1252 bytes that are not valid UTF-8 (0x96 = en-dash in cp1252, an invalid UTF-8 start byte)
    data = io.BytesIO(b'[{"member_id": "\x96bad"}]')
    r = upload_client.post(
        "/members/upload", files={"file": ("m.json", data, "application/json")}
    )
    assert r.status_code == 422  # clean validation error, never a 500/raw traceback


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


def test_uploaded_jsonl_members_dataset_is_reingestable(upload_client):
    # #2-followup: a JSONL/non-array members file persists verbatim; re-reading the dataset (reseed / make
    # eval -> ingest_dataset) must succeed via the shared read_records, not crash on the non-array shape.
    import json

    from preprocessing.ingest import ingest_dataset

    jsonl = "\n".join(json.dumps(_bundle(m)) for m in ["RT1", "RT2"]).encode()
    r = upload_client.post(
        "/members/upload",
        files={
            "file": ("jl.zip", _zip_of({"members.jsonl": jsonl}), "application/zip")
        },
        data={"name": "rtset"},
    )
    assert r.status_code == 200, r.text
    # re-read the persisted dataset the way reseed / `DATASET=rtset make eval` would
    con = db.connect(_DB_FILE)
    try:
        summary = ingest_dataset(con, "rtset")
    finally:
        con.close()
    assert summary["members"] == 2  # round-trips, no "must be a JSON array" crash


def test_upload_extra_cannot_clobber_canonical_file(upload_client):
    # #3-followup: an unclassified extra named like a canonical file must NOT overwrite it on disk.
    data = _zip_of(
        {
            "roster.json": _members_bytes(
                ["CL1"]
            ),  # the real members -> canonical members.json
            "members.json": b"junk, not json",  # non-JSON extra that shares the canonical name
        }
    )
    r = upload_client.post(
        "/members/upload",
        files={"file": ("clob.zip", data, "application/zip")},
        data={"name": "clobset"},
    )
    assert r.status_code == 200, r.text
    on_disk = (upload_client.data_root / "clobset" / "members.json").read_bytes()
    assert (
        b"CL1" in on_disk and b"junk" not in on_disk
    )  # canonical survived, extra did not clobber


def test_upload_members_only_then_make_eval_is_empty_not_crash(upload_client):
    # #4: a members-only upload creates a dataset with no eval_set.jsonl; load_supplied_cases must return
    # [] (a members-only dataset has no supplied cases), not crash with FileNotFoundError.
    import io

    from eval.adapter import load_supplied_cases

    data = io.BytesIO(_members_bytes(["MO1"]))
    r = upload_client.post(
        "/members/upload",
        files={"file": ("m.json", data, "application/json")},
        data={"name": "memonly"},
    )
    assert r.status_code == 200, r.text
    assert not (upload_client.data_root / "memonly" / "eval_set.jsonl").exists()
    assert load_supplied_cases("memonly") == []


def test_upload_duplicate_member_id_counts_distinct(upload_client):
    # #10: a duplicate member_id in the array must not inflate the members/scanned counts (upsert -> one).
    import io
    import json

    data = io.BytesIO(json.dumps([_bundle("DUP"), _bundle("DUP")]).encode())
    r = upload_client.post(
        "/members/upload", files={"file": ("dup.json", data, "application/json")}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (
        body["members"] == 1 and body["member_ids"] == ["DUP"] and body["scanned"] == 1
    )


def test_upload_dotted_path_member_file_kept(upload_client):
    # #11: a legitimately dot-named bundle file is NOT discarded as junk (only __MACOSX/.DS_Store/._* are).
    data = _zip_of({".v2/roster.json": _members_bytes(["DV1"])})
    r = upload_client.post(
        "/members/upload", files={"file": ("dotted.zip", data, "application/zip")}
    )
    assert r.status_code == 200, r.text
    assert r.json()["member_ids"] == ["DV1"]


def test_upload_name_collides_with_existing_file_is_409_not_500(upload_client):
    # #2: a name that collides with an existing FILE (not just a dir) under the data root must be a clean
    # 409 with NO members ingested — create_dataset_dir's mkdir is the single, drift-free collision check.
    import io

    (upload_client.data_root / "afile").write_text(
        "x"
    )  # a plain file, not a dataset dir
    data = io.BytesIO(_members_bytes(["CF1"]))
    r = upload_client.post(
        "/members/upload",
        files={"file": ("x.json", data, "application/json")},
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


def test_upload_arbitrary_filenames_resolve_by_content(upload_client):
    # The hold-out ships the same schemas under DIFFERENT names — discovery must be by content, not name.
    import json

    data = _zip_of(
        {
            "roster.json": _members_bytes(
                ["AX1", "AX2"]
            ),  # members — by profile/panels shape
            "panel.csv": b"analyte,value\nHbA1c,5.4\n",  # panels — the lone CSV
            "cases.json": json.dumps(
                _eval_records(["E1"])
            ).encode(),  # eval — the other JSON
        }
    )
    r = upload_client.post(
        "/members/upload", files={"file": ("holdoutA.zip", data, "application/zip")}
    )
    assert r.status_code == 200, r.text
    assert set(r.json()["member_ids"]) == {"AX1", "AX2"}
    # files normalized to CANONICAL names on disk, regardless of what the upload called them
    folder = upload_client.data_root / "holdoutA"
    assert (folder / "members.json").is_file()
    assert (folder / "lab_panels.csv").is_file()
    assert (folder / "eval_set.jsonl").is_file()
    assert not (folder / "roster.json").exists()  # renamed to canonical, not duplicated


def test_upload_eval_loads_through_harness_array_or_jsonl(upload_client):
    # Criterion 4, END-TO-END: the eval file as a JSON ARRAY (its name changed from .jsonl) must load via
    # the SAME harness adapter as JSONL, with identical case ids — upload WRITES, the harness READS.
    import json

    from eval.adapter import load_supplied_cases

    cases = _eval_records(["E0", "E1", "E2"])
    upload_client.post(
        "/members/upload",
        files={
            "file": (
                "arr.zip",
                _zip_of(
                    {
                        "m.json": _members_bytes(["AX1"]),
                        "ev.json": json.dumps(cases).encode(),
                    }
                ),
                "application/zip",
            )
        },
        data={"name": "evarr"},
    )
    jsonl = "\n".join(json.dumps(c) for c in cases).encode()
    upload_client.post(
        "/members/upload",
        files={
            "file": (
                "jl.zip",
                _zip_of({"m.json": _members_bytes(["AX9"]), "ev.jsonl": jsonl}),
                "application/zip",
            )
        },
        data={"name": "evjsonl"},
    )
    assert [c.id for c in load_supplied_cases("evarr")] == ["E0", "E1", "E2"]
    assert [c.id for c in load_supplied_cases("evjsonl")] == [
        "E0",
        "E1",
        "E2",
    ]  # same, both forms


def test_upload_opaque_member_id_roundtrips_through_scan(upload_client):
    # Different ID space: an opaque id (H-001, a UUID) must ingest, auto-scan, and read back — no format
    # assumption anywhere. The flagged HbA1c proves the full ingest -> auto-scan -> observations path.
    import io
    import json

    uid, uuid = "H-001", "550e8400-e29b-41d4-a716-446655440000"
    flagged = _bundle(
        uid,
        panels=[
            make_panel(
                f"{uid}-P1", "2024-01-15", [make_result("HbA1c", 9.4, "%", "<5.7")]
            )
        ],
    )
    data = io.BytesIO(json.dumps([flagged, _bundle(uuid)]).encode())
    r = upload_client.post(
        "/members/upload", files={"file": ("opaque.json", data, "application/json")}
    )
    assert r.status_code == 200, r.text
    assert set(r.json()["member_ids"]) == {uid, uuid} and r.json()["scanned"] == 2
    picker = {m["member_id"] for m in upload_client.get("/members").json()}
    assert {uid, uuid} <= picker
    assert (
        len(upload_client.get(f"/members/{uid}/observations").json()) >= 1
    )  # auto-scanned


def test_upload_no_members_file_422_no_side_effects(upload_client):
    # A bundle with no members-shaped JSON can't resolve -> 422, before any ingest or folder.
    import json

    data = _zip_of(
        {
            "panel.csv": b"a,b\n1,2\n",
            "cases.json": json.dumps(
                _eval_records(["E1"])
            ).encode(),  # eval-shaped, not members
        }
    )
    r = upload_client.post(
        "/members/upload", files={"file": ("nomembers.zip", data, "application/zip")}
    )
    assert r.status_code == 422
    assert not (upload_client.data_root / "nomembers").exists()


def test_upload_two_members_files_422_no_partial(upload_client):
    # Two members-shaped JSON is ambiguous -> reject with no partial load.
    data = _zip_of({"a.json": _members_bytes(["P1"]), "b.json": _members_bytes(["P2"])})
    r = upload_client.post(
        "/members/upload", files={"file": ("ambig.zip", data, "application/zip")}
    )
    assert r.status_code == 422
    picker = {m["member_id"] for m in upload_client.get("/members").json()}
    assert not ({"P1", "P2"} & picker)  # nothing ingested
    assert not (upload_client.data_root / "ambig").exists()


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
