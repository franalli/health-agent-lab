"""The firewall's pure contract — every reference-range shape parses to the correct bounds.

``parse_reference_range`` is the one transform the architecture mandates a unit test for (Phase 2):
the five shapes the supplied data prints must each reduce to the right ``(sex, ref_low, ref_high)``
rows, with the Vitamin-D multi-band — the flagged #1 parse risk — pinned explicitly. Pure functions,
no SQLite.
"""

import json

from preprocessing.ingest import parse_reference_range, read_records


def test_parse_reference_range_all_five_shapes():
    # upper-bound only (<high): HbA1c, CRP, LDL, Total chol, Triglycerides
    assert parse_reference_range("<5.7", "HbA1c") == [("any", None, 5.7)]
    # bounded low-high: Fasting glucose, TSH, AST, ALT, Potassium
    assert parse_reference_range("70-99", "Fasting glucose") == [("any", 70.0, 99.0)]
    # lower-bound only (>=low): eGFR
    assert parse_reference_range(">=90", "eGFR") == [("any", 90.0, None)]
    # sex-split wrapping a bounded form: Creatinine (also Hemoglobin, Ferritin)
    assert parse_reference_range(
        "0.74-1.35 (male) / 0.59-1.04 (female)", "Creatinine"
    ) == [
        ("male", 0.74, 1.35),
        ("female", 0.59, 1.04),
    ]
    # sex-split wrapping a one-sided '>low' form: HDL
    assert parse_reference_range(">40 (male) / >50 (female)", "HDL cholesterol") == [
        ("male", 40.0, None),
        ("female", 50.0, None),
    ]


def test_vitamin_d_multiband_reduces_to_deficiency_floor():
    # The graded string reduces to the deficiency floor (ref_low=20); the suff/insuff/defic nuance is
    # owned by config.graded_bands (band-crossing), so it is not double-counted as a range flag.
    assert parse_reference_range(
        ">=30 sufficient; 20-29 insufficient; <20 deficient", "Vitamin D (25-OH)"
    ) == [("any", 20.0, None)]


def test_read_records_reads_array_object_and_jsonl_shapes():
    # read_records is the shared reader every persisted members.json is re-read through (reseed /
    # `DATASET=x make eval`). An UPLOAD can persist a members file as a JSON array, a single object, OR
    # JSONL (all shapes it accepts), so all three must round-trip to a list — a regression to the
    # single-object wrapping (`obj if isinstance(obj, list) else [obj]`) or the JSONL branch would crash a
    # reseed of such a dataset in production, uncovered since the upload-level tests for it were removed.
    assert read_records(json.dumps([{"a": 1}, {"b": 2}]).encode()) == [
        {"a": 1},
        {"b": 2},
    ]
    assert read_records(json.dumps({"member_id": "Z1"}).encode()) == [
        {"member_id": "Z1"}
    ]  # a single object -> a one-record list
    assert read_records(b'{"a": 1}\n{"b": 2}\n') == [
        {"a": 1},
        {"b": 2},
    ]  # JSONL -> list
    assert read_records(b'\xef\xbb\xbf[{"a": 1}]') == [{"a": 1}]  # BOM-tolerant


def test_load_supplied_cases_missing_eval_file_degrades_to_empty(tmp_path, monkeypatch):
    # A dataset folder with NO eval_set.jsonl must degrade to [] (not FileNotFoundError). `/learn`'s gate
    # runs load_cases on the ACTIVE dataset, so an eval-less / hand-assembled dataset must not crash it —
    # coverage lost when the members-only-upload test was removed (an upload now always writes an eval file,
    # but a dataset assembled another way may not).
    from eval.adapter import load_supplied_cases

    monkeypatch.setenv("HEALTH_DATA_ROOT", str(tmp_path))
    (
        tmp_path / "noeval"
    ).mkdir()  # a dataset folder that exists but carries no eval_set.jsonl
    assert load_supplied_cases("noeval") == []
