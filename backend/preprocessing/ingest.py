"""The one normalization firewall — member-bundle JSON -> validated domain objects -> SQLite.

Everything raw enters through here exactly once; nothing downstream re-parses input (architecture
§14). The firewall's jobs, all pure up to the final hand-off to ``db.replace_member``:

  * flatten each member's ``panels`` into dated ``LabResult``s that keep their ``panel_id``;
  * fold the three vitals (BP, BMI) in as markers, with units the data doesn't print (from config);
  * parse the reference-range string each result prints — five shapes (``low-high``, ``<high``,
    ``>low``/``>=low``, sex-split, and the Vitamin-D multi-band) — into ``(sex, ref_low, ref_high)``;
  * transcribe the curated panic thresholds (and the vital normal bounds) from ``config`` onto the
    matching ``reference_ranges`` rows, so the deterministic safety floor has values to read;
  * carry notes with their ``source``.

``analyte`` is already canonical (config's marker keys match the data's analyte strings verbatim),
and units are consistent per marker, so there is deliberately no alias resolution and no unit
conversion. ``db.py`` owns the actual SQL write (the single store seam); this module never touches
SQLite directly — it normalizes and delegates.

Runnable as the loader CLI:  ``python -m preprocessing.ingest [--dataset NAME] [--db PATH] [--verify]``
(the live ``POST /members`` equivalent lands in Phase 6).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import logging
import lzma
import os
import pathlib
import re
import shutil
import zipfile
import zlib

from pydantic import ValidationError

from health_intelligence import db
from health_intelligence.config import CONFIG_VERSION, MARKERS
from health_intelligence.models import (
    EvalCaseInput,
    LabResult,
    MemberBundle,
    RangeSex,
    ReferenceRange,
)
from preprocessing.datasets import (
    create_dataset_dir,
    derive_dataset_name,
    members_path,
)

logger = logging.getLogger(__name__)

#: The three vitals every panel carries; folded into markers with config-supplied units (the data
#: prints no vital units or ranges, so config is the source for both).
VITALS: tuple[str, ...] = ("systolic_bp", "diastolic_bp", "bmi")


# --------------------------------------------------------------------------------------------------
# Reference-range parsing — the firewall's pure, unit-tested contract. Returns one row per applicable
# sex: a sex-agnostic form yields a single ('any', ...) row; a sex-split form yields ('male', ...)
# and ('female', ...). One-sided forms leave the absent bound None.
# --------------------------------------------------------------------------------------------------


def parse_reference_range(
    raw: str, marker: str
) -> list[tuple[RangeSex, float | None, float | None]]:
    """Parse one printed range string into ``[(sex, ref_low, ref_high)]``.

    Precedence is deliberate — (1) Vitamin-D multi-band, (2) sex-split, (3) scalar — because the
    multi-band string contains ``<20`` / ``20-29`` substrings the scalar patterns would otherwise
    mis-match, and the sex-split wrapper must be peeled before its halves reach the scalar parser.
    """
    del (
        marker
    )  # reserved (architecture's documented signature; dispatch is structural on `raw`)
    s = raw.strip()
    # (1) Vitamin-D multi-band: graded, semicolon-separated (e.g. ">=30 sufficient; 20-29 ...; <20 ...").
    if ";" in s:
        return _parse_graded_deficiency_floor(s)
    # (2) sex-split: "<inner> (male) / <inner> (female)" — peel the wrapper, parse each half as scalar.
    if "(male)" in s and "(female)" in s:
        m = re.match(r"^\s*(.*?)\(male\)\s*/\s*(.*?)\(female\)\s*$", s)
        if m:
            male_low, male_high = _parse_scalar(m.group(1))
            female_low, female_high = _parse_scalar(m.group(2))
            return [("male", male_low, male_high), ("female", female_low, female_high)]
    # (3) scalar -> a single sex-agnostic row.
    low, high = _parse_scalar(s)
    return [("any", low, high)]


def _parse_graded_deficiency_floor(
    s: str,
) -> list[tuple[RangeSex, float | None, float | None]]:
    """Reduce the Vitamin-D multi-band to one reference interval: the deficiency floor. ``ref_low`` is
    the threshold from the ``<NN deficient`` clause (below it is unambiguously abnormal -> below_range);
    the sufficient/insufficient/deficient nuance is owned by ``config.graded_bands`` (band-crossing),
    so it is not double-counted here. ``ref_high`` is None (no upper reference for Vitamin D)."""
    m = re.search(r"<\s*([0-9.]+)\s*deficient", s)
    if m:
        return [("any", float(m.group(1)), None)]
    raise ValueError(f"unrecognized graded range string: {s!r}")


def _parse_scalar(s: str) -> tuple[float | None, float | None]:
    """Parse a single scalar form into ``(ref_low, ref_high)``. ``<=``/``>=`` are matched before
    ``<``/``>`` so the longer operator wins; ``fullmatch`` rejects anything unexpected loudly."""
    s = s.strip()
    if m := re.fullmatch(r"<=\s*([0-9.]+)", s):
        return (None, float(m.group(1)))
    if m := re.fullmatch(r"<\s*([0-9.]+)", s):
        return (None, float(m.group(1)))
    if m := re.fullmatch(r">=\s*([0-9.]+)", s):
        return (float(m.group(1)), None)
    if m := re.fullmatch(r">\s*([0-9.]+)", s):
        return (float(m.group(1)), None)
    if m := re.fullmatch(r"([0-9.]+)\s*-\s*([0-9.]+)", s):
        return (float(m.group(1)), float(m.group(2)))
    raise ValueError(f"unrecognized reference range: {s!r}")


# --------------------------------------------------------------------------------------------------
# Per-bundle ingest — normalize, then hand domain objects to db.replace_member (the SQL seam).
# --------------------------------------------------------------------------------------------------


def ingest_bundle(con, bundle: MemberBundle, *, commit: bool = True) -> dict:
    """Normalize one member and persist it (replacing any prior version). Returns summary counts.
    ``commit=False`` defers the write's commit to the caller's transaction (the atomic reseed)."""
    results: list[LabResult] = []
    # marker -> (printed reference_range string, data unit); constant per marker, so parse it once.
    lab_range_src: dict[str, tuple[str, str]] = {}

    for panel in bundle.panels:
        for res in panel.results:
            results.append(
                LabResult(
                    marker=res.analyte,  # analyte -> marker is identity (canonical already)
                    value=res.value,
                    unit=res.unit,
                    panel_id=panel.panel_id,
                    panel_date=panel.collected_date,
                )
            )
            lab_range_src.setdefault(res.analyte, (res.reference_range, res.unit))
        for vk in VITALS:
            results.append(
                LabResult(
                    marker=vk,
                    value=getattr(panel.vitals, vk),
                    unit=_require_unit(
                        vk
                    ),  # vitals' unit comes from config (data prints none)
                    panel_id=panel.panel_id,
                    panel_date=panel.collected_date,
                )
            )

    _assert_unique_markers_per_panel(results)
    ranges = _build_ranges(lab_range_src)
    db.replace_member(
        con,
        profile=bundle.profile,
        results=results,
        ranges=ranges,
        notes=bundle.notes,
        commit=commit,
    )
    return {
        "member_id": bundle.member_id,
        "results": len(results),
        "ranges": len(ranges),
    }


def _assert_unique_markers_per_panel(results: list[LabResult]) -> None:
    """Firewall guard: labs and vitals share one marker namespace, and result_id is
    ``{member}:{panel_id}:{marker}``. A lab analyte named like a vital (``bmi``/``systolic_bp``/
    ``diastolic_bp``), or a marker repeated within a panel, would collide on that PK and surface as an
    opaque UNIQUE-constraint error from db.replace_member. Reject it here with a clear message."""
    seen: set[tuple[str, str]] = set()
    for r in results:
        key = (r.panel_id, r.marker)
        if key in seen:
            raise ValueError(
                f"duplicate marker {r.marker!r} in panel {r.panel_id!r} "
                f"(a lab analyte may collide with a vital name)"
            )
        seen.add(key)


def _build_ranges(lab_range_src: dict[str, tuple[str, str]]) -> list[ReferenceRange]:
    """Build the reference_ranges rows: parse each lab's printed string and transcribe its curated
    panic thresholds from config; synthesize each vital's row entirely from config (bounds, unit,
    panic). Panic is transcribed onto EVERY row — the safety floor reads it from here."""
    ranges: list[ReferenceRange] = []

    for marker, (raw, unit) in lab_range_src.items():
        mcfg = MARKERS.get(marker)
        panic_low = mcfg.panic_low if mcfg else None
        panic_high = mcfg.panic_high if mcfg else None
        parsed = parse_reference_range(raw, marker)
        for sex, ref_low, ref_high in parsed:
            ranges.append(
                ReferenceRange(
                    marker=marker,
                    sex=sex,
                    unit=unit,
                    ref_low=ref_low,
                    ref_high=ref_high,
                    panic_low=panic_low,
                    panic_high=panic_high,
                    config_version=CONFIG_VERSION,
                )
            )
        # A sex-split marker has only male/female rows. An 'other'/'unknown'-sex member would fall
        # through _range_for (sex row -> 'any') to a non-existent 'any' row and get a `no_reference`
        # flag *before* the panic check — so a sex-independent panic (e.g. Hemoglobin 7.0) would
        # silently never fire for them. If this marker carries a panic, also emit a panic-only 'any'
        # row (ref bounds None — we can't pick a sex's normal range) so the safety floor still fires.
        # "Rather over-escalate than miss" (architecture §6). Only markers WITH a panic get the row,
        # so non-panic sex-split markers keep their honest `no_reference` for an unknown-sex member.
        if all(sex != "any" for sex, _, _ in parsed) and (
            panic_low is not None or panic_high is not None
        ):
            ranges.append(
                ReferenceRange(
                    marker=marker,
                    sex="any",
                    unit=unit,
                    ref_low=None,
                    ref_high=None,
                    panic_low=panic_low,
                    panic_high=panic_high,
                    config_version=CONFIG_VERSION,
                )
            )

    for vk in VITALS:
        mcfg = MARKERS[vk]
        ranges.append(
            ReferenceRange(
                marker=vk,
                sex="any",
                unit=_require_unit(vk),
                ref_low=mcfg.ref_low,
                ref_high=mcfg.ref_high,
                panic_low=mcfg.panic_low,
                panic_high=mcfg.panic_high,
                config_version=CONFIG_VERSION,
            )
        )

    return ranges


def _require_unit(marker: str) -> str:
    """A vital's unit must be config-supplied (the data prints none); fail loudly if it isn't, rather
    than write an empty unit that would violate the NOT NULL column."""
    unit = MARKERS[marker].unit
    if unit is None:
        raise ValueError(f"vital {marker!r} has no config-supplied unit")
    return unit


# --------------------------------------------------------------------------------------------------
# Whole-dataset ingest + CLI
# --------------------------------------------------------------------------------------------------


def ingest_dataset(con, dataset: str | None = None, *, commit: bool = True) -> dict:
    """Load ``members.json`` for the active dataset, ingest each bundle SKIPPING any buggy row (via
    ``ingest_members``), and return summary counts (including a ``skipped`` list). An entry that fails its
    Pydantic shape or the semantic firewall parse is recorded and skipped, not raised — one malformed
    bundle never aborts the batch. An empty file is still a loud error (below), since a seed dataset must
    have members.

    Reads via the shared ``read_records`` (not a bare ``json.loads``), so a dataset whose ``members.json``
    was persisted by ``POST /members/upload`` in a non-array shape (a single object, JSONL, or BOM-prefixed
    — all of which the upload accepts) re-ingests correctly on reseed / ``DATASET=<name> make eval``. The
    write side and this re-read side share one reader, so the persisted bundle is always round-trippable."""
    records = read_records(members_path(dataset).read_bytes())
    if not records:
        # An empty/whitespace members.json (e.g. a truncated write on a durable disk) parses to [] —
        # fail LOUDLY here rather than silently seeding zero members (a seed dataset always has members).
        raise ValueError(
            f"dataset {dataset!r} members.json is empty — nothing to ingest"
        )
    result = ingest_members(con, records, commit=commit)
    # ``ingest_members`` is lenient (a buggy row is skipped, not fatal) so one bad bundle never aborts a
    # batch — but on the SEED path (this function backs seed_if_empty / reseed / the CLI) a skipped member
    # is an ANOMALY the old fail-loud path used to surface. Log it LOUDLY so a partial seed can't read as a
    # clean success (a fresh instance would otherwise boot missing that member silently, and never re-heal
    # since the DB is no longer empty). Availability is kept (the good members still seed); the noise is the
    # fix — a trusted seed dataset is expected to be clean, so any skip warrants operator attention.
    if result["skipped"]:
        logger.warning(
            "ingest_dataset(%r): %d member row(s) skipped as malformed — seeded %d of %d: %s",
            dataset,
            len(result["skipped"]),
            result["members"],
            len(records),
            result["skipped"],
        )
    return result


def _short_validation_error(e: Exception) -> str:
    """A compact one-line reason for a skipped member row — the first field:msg of a Pydantic error, or
    the plain message of a bare ``ValueError`` (e.g. the member_id-mismatch validator)."""
    if isinstance(e, ValidationError):
        errs = e.errors()
        if errs:
            loc = ".".join(str(p) for p in errs[0].get("loc", ()))
            return (
                f"{loc}: {errs[0].get('msg', 'invalid')}"
                if loc
                else errs[0].get("msg", "invalid")
            )
    return str(e)


def ingest_members(con, raw: object, *, commit: bool = True) -> dict:
    """Write a ``members.json``-shaped array of bundles, SKIPPING any buggy row. The shared core of
    ``ingest_dataset`` (the seed/startup path) and the ``/members/upload`` zip route — both feed it the
    parsed array. Returns ``{members, results, ranges, member_ids, skipped}``; ``member_ids`` lets the
    uploader focus the picker on what just landed, and ``skipped`` lists the rows that didn't ingest —
    each ``{file, row, member_id, detail}`` — so the caller can surface them without failing the upload.

    Lenient by design: a row that fails its *Pydantic shape* OR the firewall's *semantic* parse
    (reference-range shapes, marker-uniqueness) is recorded in ``skipped`` and the loop moves on, so one
    malformed bundle never aborts a good batch (the upload's first-record format gate has already
    confirmed the file is the right kind). ``ingest_bundle`` upserts, so the surviving rows land
    additively; a skip leaves no partial write for that row (its parse fails before the row's own
    ``replace_member`` write)."""
    if not isinstance(raw, list):
        raise ValueError(
            "members.json must be a JSON array of member bundles "
            f"(got {type(raw).__name__}); a single bundle goes to POST /members"
        )
    skipped: list[dict] = []
    # Validate each entry's shape, skipping (not raising on) a bad one. De-dup by member_id (LAST
    # occurrence wins, matching upsert order) so a duplicate id doesn't inflate the counts.
    valid: dict[str, tuple[int, MemberBundle]] = {}
    for i, entry in enumerate(raw):
        try:
            bundle = MemberBundle.model_validate(entry)
        except (ValidationError, ValueError) as e:
            mid = entry.get("member_id") if isinstance(entry, dict) else None
            skipped.append(
                {
                    "file": CANONICAL_MEMBERS,
                    "row": i
                    + 1,  # 1-based record number, consistent with the gate + eval/CSV file lines
                    "member_id": mid,
                    "detail": _short_validation_error(e),
                }
            )
            continue
        valid[bundle.member_id] = (i, bundle)
    member_ids = []
    total_results = 0
    for row_i, bundle in valid.values():
        try:
            summary = ingest_bundle(
                con, bundle, commit=commit
            )  # semantic parse (ranges) can still raise here
        except ValueError as e:
            skipped.append(
                {
                    "file": CANONICAL_MEMBERS,
                    "row": row_i + 1,  # 1-based record number (see the gate above)
                    "member_id": bundle.member_id,
                    "detail": str(e),
                }
            )
            continue
        member_ids.append(bundle.member_id)
        total_results += summary["results"]
    n_ranges = len(
        db.get_ranges(con)
    )  # via db.py (the one SQLite seam), scoped to the active config
    return {
        "members": len(member_ids),
        "results": total_results,
        "ranges": n_ranges,
        "member_ids": member_ids,
        "skipped": skipped,
    }


# --------------------------------------------------------------------------------------------------
# Uploaded-bundle handling — the firewall for a runtime hold-out. The hold-out ships the SAME three
# kinds of file as `training_data` and — by contract — the SAME three extensions (`.json` members,
# `.jsonl` eval set, `.csv` lab panels), even though the base filenames may differ. So the upload
# REQUIRES exactly one of each extension and FORMAT-GATES THE FIRST RECORD of each file against its
# expected shape up front (before any side effect) — a first-record failure or a missing/duplicated role
# 422s here; later rows are NOT gated (a buggy later member row is skipped at ingest, and later eval/CSV
# rows ride through unchecked). Only then does it ingest the members and write each file to the new dataset
# folder under its CANONICAL name — so every downstream reader (the seed loader, `members_path`, the
# eval adapter) stays name-based and UNCHANGED; this upload boundary is the single place a name is
# normalized. A first-record/role failure is reported at file+row granularity, never a partial load.
# --------------------------------------------------------------------------------------------------

#: The on-disk names every dataset folder uses, whatever the upload called its files.
CANONICAL_MEMBERS = "members.json"
CANONICAL_PANELS = "lab_panels.csv"
CANONICAL_EVAL = "eval_set.jsonl"

#: The exact header the lab-panels CSV must carry, in order (the shape of `training_data/lab_panels.csv`).
_CSV_COLUMNS: tuple[str, ...] = (
    "member_id",
    "panel_id",
    "collected_date",
    "analyte",
    "value",
    "unit",
    "reference_range",
)


class BundleValidationError(ValueError):
    """An uploaded bundle failed validation. Carries ``failures`` — a list of
    ``{file, row, field, detail}`` dicts so the route can surface each problem at the exact file and
    row it occurs (``row`` is ``None`` for a whole-file/whole-bundle problem, e.g. a missing file).

    A ``ValueError`` subclass so any caller catching ``ValueError`` still treats it as bad input, but
    the route matches it FIRST to serialize the structured ``failures`` rather than a flat string."""

    def __init__(self, failures: list[dict]):
        self.failures = failures
        n = len(failures)
        super().__init__(f"bundle validation failed with {n} issue(s)")


def read_records(data: bytes) -> list:
    """Parse a JSON file into a list of records, accepting any of the three shapes a hold-out might ship:
    a JSON **array**, a single JSON **object** (-> a one-record list), or line-delimited **JSONL**.
    Tolerates a leading UTF-8 BOM (common in Windows/Excel exports). The SINGLE reader shared by the
    upload classifier, the members parse, and the eval adapter — one place, so the forms can never drift
    apart. Raises ``ValueError`` on bytes that are none of the three (e.g. a CSV)."""
    if (
        data[:3] == b"\xef\xbb\xbf"
    ):  # strip a UTF-8 BOM so the parse below sees real content
        data = data[3:]
    head = data.lstrip()
    if not head:
        return []
    # Try a WHOLE-document parse first — that cleanly covers a JSON array AND a single JSON object
    # (-> a one-record list), each robust to pretty-printing. Only if that fails (a JSONL file is many
    # documents, so it raises "Extra data") fall back to line-by-line JSONL. A CSV fails both -> ValueError.
    # Catch UnicodeDecodeError alongside JSONDecodeError: json.loads on bytes decodes as UTF-8/16/32, so
    # a non-UTF file (latin-1/cp1252 with no BOM — not valid JSON, which must be Unicode) raises
    # UnicodeDecodeError (a ValueError but NOT a JSONDecodeError); without this it would escape both
    # handlers and break the documented "Raises ValueError" contract (a raw traceback on the direct-read
    # paths — ingest_dataset / load_supplied_cases).
    _ParseError = (json.JSONDecodeError, UnicodeDecodeError)
    whole_doc_err: Exception | None = None
    try:
        obj = json.loads(data)
        return obj if isinstance(obj, list) else [obj]
    except _ParseError as e:
        whole_doc_err = (
            e  # save it — the `as` name is unbound once the except block exits
        )
    try:
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    except _ParseError as jsonl_err:
        # When the content was clearly meant as ONE document (starts with [ or {), the whole-document
        # error is the real one — surfacing the per-line JSONL error would point at a structural line
        # (e.g. the bare "[") and mislead. Otherwise report the JSONL error.
        err = whole_doc_err if head[:1] in (b"[", b"{") else jsonl_err
        raise ValueError(f"not valid JSON or JSONL: {err}") from err


def _open_zip(data: bytes) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ValueError(f"not a valid zip file: {e}") from e


def _is_junk(name: str) -> bool:
    """Archive cruft a zip of a folder carries that is not part of the bundle: the macOS ``__MACOSX/``
    resource-fork tree, ``.DS_Store``, and AppleDouble ``._*`` sidecars. Deliberately NARROW — only these
    known artifacts are dropped, so a legitimately dot-named bundle file (``.members.json``, a file under
    a ``.v2/`` folder) is NOT silently discarded, and a ``.``/``..`` traversal segment still reaches the
    path guard rather than being swallowed."""
    base = name.rsplit("/", 1)[-1]
    return name.startswith("__MACOSX/") or base == ".DS_Store" or base.startswith("._")


def _common_top_folder(names: list[str]) -> str:
    """The single wrapping folder to strip, or ``""``. ``zip -r training_data`` yields every entry under
    ``training_data/``; we materialize the bundle's *contents* at the dataset-folder root, so that one
    shared top segment is peeled. Returns ``""`` when entries don't all share one top folder (e.g. the
    files already sit at the zip root), so nothing is stripped in that case."""
    tops = {n.split("/", 1)[0] for n in names if "/" in n}
    roots = {n for n in names if "/" not in n}
    if len(tops) == 1 and not roots:
        return next(iter(tops)) + "/"
    return ""


def _zip_entries(zf: zipfile.ZipFile) -> list[tuple[str, bytes]]:
    """Every bundle file as ``(relative-path, bytes)`` — wrapping folder stripped, junk dropped.

    SECURITY (zip-slip): rejects a path-traversal entry LEXICALLY here, before the bytes are used for any
    classify/ingest/write decision — a malicious entry name (``../escape``, an absolute path, or one that
    normalizes to ``.``/empty, i.e. the directory itself) raises ``ValueError`` so the upload has zero
    side effects. The resolve-based check in :func:`_write_dataset_folder` re-guards the actual writes
    (defense in depth). A per-entry read error (encrypted entry, unsupported compression, bad CRC) is also
    mapped to ``ValueError`` so a damaged/odd zip stays a clean 422, never an opaque 500."""
    files = [i for i in zf.infolist() if not i.is_dir() and not _is_junk(i.filename)]
    prefix = _common_top_folder([i.filename for i in files])
    out: list[tuple[str, bytes]] = []
    for info in files:
        rel = info.filename[len(prefix) :] if prefix else info.filename
        norm = os.path.normpath(rel)
        if (
            not rel
            or norm in ("", ".")
            or os.path.isabs(norm)
            or norm == ".."
            or norm.startswith(".." + os.sep)
        ):
            raise ValueError(f"unsafe or empty path in zip: {info.filename!r}")
        try:
            payload = zf.read(info)
        except (
            RuntimeError,  # encrypted entry (password required)
            NotImplementedError,  # unsupported compression method
            zipfile.BadZipFile,  # bad CRC on a STORED entry / structurally bad
            zlib.error,  # corrupt DEFLATE stream — the DEFAULT compression for `zip -r`/macOS Compress
            lzma.LZMAError,  # corrupt ZIP_LZMA stream
            EOFError,  # truncated compressed stream
            OSError,  # corrupt ZIP_BZIP2 stream raises a bare OSError (zf reads in-memory, so no real I/O)
        ) as e:
            # a damaged/odd entry is bad INPUT, not a server error — keep the "bad zip -> 422, never 500"
            # contract across ALL compression methods (DEFLATE/BZIP2/LZMA/STORED), not just the default.
            raise ValueError(f"could not read zip entry {info.filename!r}: {e}") from e
        out.append(
            (norm, payload)
        )  # store the NORMALIZED rel so downstream paths are clean (no './')
    return out


def _pydantic_failures(exc: ValidationError, file: str, row: int) -> list[dict]:
    """Flatten a Pydantic ``ValidationError`` into one ``{file, row, field, detail}`` per error, so a
    record with several bad fields surfaces each one (``field`` is the dotted ``loc`` path, ``""`` for a
    whole-record/root error). Same shape the file-level and CSV checks emit — one uniform failure list."""
    out: list[dict] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        out.append(
            {
                "file": file,
                "row": row,
                "field": loc,
                "detail": err.get("msg", "invalid"),
            }
        )
    return out


def _validate_members(data: bytes) -> tuple[list, list[dict]]:
    """Format-gate the members ``.json`` file on its FIRST record against :class:`MemberBundle` — the
    SAME model the ingest uses, so the gate can never drift from what actually ingests. Returns
    ``(records, failures)``: ``records`` is the parsed array (handed forward so ingest doesn't re-parse),
    ``failures`` is empty when the first record is well-formed, else one entry per bad field on that
    record (``row`` 0). A file that isn't a JSON array, is empty, or whose FIRST record isn't an object is
    a whole-file failure (``row=None``). A LATER non-object (or otherwise malformed) row is NOT gated here
    — it is skipped at ingest time, matching the documented "a buggy later member row is skipped, not
    fatal" contract (and the same treatment a dict-with-bad-fields later row already gets)."""
    try:
        records = read_records(data)  # BOM-tolerant; JSON array or JSONL -> list
    except ValueError as e:
        return [], [
            {"file": CANONICAL_MEMBERS, "row": None, "field": "", "detail": str(e)}
        ]
    if not isinstance(records, list):
        return [], [
            {
                "file": CANONICAL_MEMBERS,
                "row": None,
                "field": "",
                "detail": "must be a JSON array of member-bundle objects",
            }
        ]
    if not records:
        return records, [
            {
                "file": CANONICAL_MEMBERS,
                "row": None,
                "field": "",
                "detail": "file has no member records",
            }
        ]
    if not isinstance(records[0], dict):
        # Gate the FIRST record only (like the eval/CSV gates): a non-object first record means the file is
        # the wrong KIND. A later non-object row is deliberately NOT fatal here — ingest_members skips it
        # (its `isinstance(entry, dict)` guard + MemberBundle.model_validate raising), so a non-dict late row
        # and a dict-with-bad-fields late row are treated the SAME (both skipped), not opposite outcomes.
        return records, [
            {
                "file": CANONICAL_MEMBERS,
                "row": None,
                "field": "",
                "detail": "must be a JSON array of member-bundle objects",
            }
        ]
    try:
        MemberBundle.model_validate(records[0])  # gate on the FIRST record only
    except ValidationError as e:
        # Row numbers are 1-BASED (record 1 = first member), matching the 1-based file-line numbering the
        # eval (`start=1`) and CSV (header = row 1) gates emit — so a mixed 422 doesn't put "members.json
        # row 0" beside "eval_set.jsonl row 1" for what is the first record of each file.
        return records, _pydantic_failures(e, CANONICAL_MEMBERS, 1)
    except (
        ValueError
    ) as e:  # model_validator (member_id mismatch) raises a bare ValueError
        return records, [
            {"file": CANONICAL_MEMBERS, "row": 1, "field": "", "detail": str(e)}
        ]
    return records, []


def _validate_eval(data: bytes) -> list[dict]:
    """Format-gate the eval ``.jsonl`` file on its FIRST record against :class:`EvalCaseInput`. Returns
    ``[]`` when the first non-blank line is well-formed, else the failure(s) for that line — ``row`` is
    the true 1-based FILE line number the operator can open to. BOM-tolerant on the first line; an empty
    file is a whole-file failure (``row=None``)."""
    if data[:3] == b"\xef\xbb\xbf":
        data = data[3:]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        return [
            {
                "file": CANONICAL_EVAL,
                "row": None,
                "field": "",
                "detail": f"not valid UTF-8: {e}",
            }
        ]
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            return [
                {
                    "file": CANONICAL_EVAL,
                    "row": lineno,
                    "field": "",
                    "detail": f"not valid JSON: {e}",
                }
            ]
        try:
            EvalCaseInput.model_validate(rec)
        except ValidationError as e:
            return _pydantic_failures(e, CANONICAL_EVAL, lineno)
        return []  # first non-blank line gated; the rest ride opaquely
    return [
        {"file": CANONICAL_EVAL, "row": None, "field": "", "detail": "file is empty"}
    ]


def _validate_panels_csv(data: bytes) -> list[dict]:
    """Format-gate the lab-panels ``.csv`` file against the shape of ``training_data/lab_panels.csv``:
    the exact header :data:`_CSV_COLUMNS`, then its FIRST data row — ``value`` numeric, ``reference_range``
    parseable by the SAME :func:`parse_reference_range` the ingest uses (so 'valid range' means exactly
    'a range the system can parse'), a well-formed ``collected_date`` (``YYYY-MM-DD``), and non-empty
    ``member_id``/``panel_id``/``analyte``/``unit``. ``row`` is the 1-based FILE line (header is line 1;
    the first data row is line 2), and ``field`` names the offending column. Later rows are not gated."""
    if data[:3] == b"\xef\xbb\xbf":
        data = data[3:]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        return [
            {
                "file": CANONICAL_PANELS,
                "row": None,
                "field": "",
                "detail": f"not valid UTF-8: {e}",
            }
        ]
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return [
            {
                "file": CANONICAL_PANELS,
                "row": None,
                "field": "",
                "detail": "file is empty",
            }
        ]
    if [c.strip() for c in header] != list(_CSV_COLUMNS):
        return [
            {
                "file": CANONICAL_PANELS,
                "row": 1,
                "field": "",
                "detail": f"header must be exactly {','.join(_CSV_COLUMNS)} (got {','.join(header)})",
            }
        ]
    ncols = len(_CSV_COLUMNS)
    for lineno, cells in enumerate(reader, start=2):
        if not any(c.strip() for c in cells):
            continue  # skip blank lines to reach the first real data row
        return _check_csv_row(cells, lineno, ncols)  # gate on the FIRST data row only
    return [
        {
            "file": CANONICAL_PANELS,
            "row": None,
            "field": "",
            "detail": "file has a header but no data rows",
        }
    ]


def _check_csv_row(cells: list[str], lineno: int, ncols: int) -> list[dict]:
    """The per-row CSV format checks, factored out so the gate (first row) reads them once."""
    if len(cells) != ncols:
        return [
            {
                "file": CANONICAL_PANELS,
                "row": lineno,
                "field": "",
                "detail": f"expected {ncols} columns, got {len(cells)}",
            }
        ]
    row = dict(
        zip(_CSV_COLUMNS, cells, strict=True)
    )  # lengths verified equal just above
    failures: list[dict] = []
    for col in ("member_id", "panel_id", "analyte", "unit"):
        if not row[col].strip():
            failures.append(
                {
                    "file": CANONICAL_PANELS,
                    "row": lineno,
                    "field": col,
                    "detail": f"{col} must not be empty",
                }
            )
    try:
        float(row["value"])
    except ValueError:
        failures.append(
            {
                "file": CANONICAL_PANELS,
                "row": lineno,
                "field": "value",
                "detail": f"value {row['value']!r} is not numeric",
            }
        )
    try:
        datetime.date.fromisoformat(row["collected_date"].strip())
    except ValueError:
        failures.append(
            {
                "file": CANONICAL_PANELS,
                "row": lineno,
                "field": "collected_date",
                "detail": f"collected_date {row['collected_date']!r} is not an ISO date (YYYY-MM-DD)",
            }
        )
    try:
        parse_reference_range(row["reference_range"], row["analyte"])
    except ValueError:
        failures.append(
            {
                "file": CANONICAL_PANELS,
                "row": lineno,
                "field": "reference_range",
                "detail": f"reference_range {row['reference_range']!r} is not a recognized range",
            }
        )
    return failures


#: How the three required roles are bucketed — by file EXTENSION (the hold-out ships stable extensions,
#: only the base names differ), each mapped to its canonical on-disk name.
_ROLE_BY_EXT: dict[str, str] = {
    ".json": CANONICAL_MEMBERS,
    ".jsonl": CANONICAL_EVAL,
    ".csv": CANONICAL_PANELS,
}


def validate_bundle(
    entries: list[tuple[str, bytes]],
) -> tuple[dict[str, bytes], list, list[tuple[str, bytes]]]:
    """Validate an uploaded bundle and resolve its roles by EXTENSION. Returns
    ``(canonical, members_records, extras)`` on success:

      * ``canonical`` maps each CANONICAL on-disk name -> uploaded bytes: ``members.json`` (from the
        ``.json``), ``eval_set.jsonl`` (from the ``.jsonl``), ``lab_panels.csv`` (from the ``.csv``);
      * ``members_records`` is the members file ALREADY PARSED (so the caller ingests without re-parsing);
      * ``extras`` are any other files (e.g. a ``README.md``), kept verbatim.

    REQUIRES exactly one ``.json``, one ``.jsonl``, and one ``.csv`` — a missing OR duplicated extension
    is a failure. Then the FIRST record of each file is format-gated against its expected shape
    (:class:`MemberBundle`, :class:`EvalCaseInput`, and the CSV schema) — a cheap "is this the right kind
    of file" check, not a whole-file scan; a buggy LATER member row is skipped at ingest time, not
    rejected here. Problems across all three files are collected, and on ANY failure this raises
    :class:`BundleValidationError` carrying them at file+row granularity — with NO side effect (the caller
    has not yet created the folder or written a DB row)."""
    buckets: dict[str, list[tuple[str, bytes]]] = {ext: [] for ext in _ROLE_BY_EXT}
    extras: list[tuple[str, bytes]] = []
    for name, payload in entries:
        ext = pathlib.PurePosixPath(name.lower()).suffix
        if ext in buckets:
            buckets[ext].append((name, payload))
        else:
            extras.append((name, payload))

    failures: list[dict] = []
    for ext, canon in _ROLE_BY_EXT.items():
        n = len(buckets[ext])
        if n == 0:
            failures.append(
                {
                    "file": canon,
                    "row": None,
                    "field": "",
                    "detail": f"bundle is missing its {ext} file",
                }
            )
        elif n > 1:
            names = [nm for nm, _ in buckets[ext]]
            failures.append(
                {
                    "file": canon,
                    "row": None,
                    "field": "",
                    "detail": f"expected exactly one {ext} file, found {n}: {names}",
                }
            )

    # Missing/duplicate roles make row validation ambiguous — report the structural failure and stop.
    if failures:
        raise BundleValidationError(failures)

    members_bytes = buckets[".json"][0][1]
    eval_bytes = buckets[".jsonl"][0][1]
    csv_bytes = buckets[".csv"][0][1]

    members_records, member_failures = _validate_members(members_bytes)
    failures.extend(member_failures)
    failures.extend(_validate_eval(eval_bytes))
    failures.extend(_validate_panels_csv(csv_bytes))
    if failures:
        raise BundleValidationError(failures)

    canonical = {
        CANONICAL_MEMBERS: members_bytes,
        CANONICAL_EVAL: eval_bytes,
        CANONICAL_PANELS: csv_bytes,
    }
    return canonical, members_records, extras


def _write_dataset_folder(
    dest: pathlib.Path,
    canonical: dict[str, bytes],
    extras: list[tuple[str, bytes]],
) -> list[str]:
    """Write the classified bundle into ``dest`` (a freshly-created, empty dataset folder): the three
    roles under their CANONICAL names, then any extras verbatim. Canonical names are safe by
    construction; the extras' relative paths are re-checked against ``dest`` (zip-slip defense in depth),
    and an extra is SKIPPED if its resolved path collides with an already-written file — so a stray entry
    named like a canonical file (``members.json`` etc.) can never overwrite the authoritative one (the
    canonical writes win). Returns the relative paths written, sorted."""
    written: list[str] = []
    dest_resolved = dest.resolve()
    written_targets: set[pathlib.Path] = set()
    for cname, payload in canonical.items():
        (dest / cname).write_bytes(payload)
        written.append(cname)
        written_targets.add((dest / cname).resolve())
    for rel, payload in extras:
        target = (dest / rel).resolve()
        if target == dest_resolved or dest_resolved not in target.parents:
            raise ValueError(f"unsafe path (path traversal): {rel!r}")
        if target in written_targets:
            continue  # never let an extra overwrite a canonical file (or a prior extra) — first write wins
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        written.append(rel)
        written_targets.add(target)
    return sorted(written)


def ingest_uploaded_dataset(
    con, *, data: bytes, filename: str | None, name: str | None
) -> dict:
    """The ``POST /members/upload`` firewall: format-gate the FIRST record of each file in an uploaded
    hold-out bundle, ingest its members ADDITIVELY, and persist the whole bundle as a new dataset folder
    under the datasets root.

    Accepts ONLY a ``.zip`` of a ``training_data``-shaped folder that carries exactly three files by
    EXTENSION — one ``.json`` (members), one ``.jsonl`` (eval set), one ``.csv`` (lab panels), under ANY
    base names — plus optional extras (e.g. a ``README.md``). Order is deliberate so a failure leaves the
    cleanest state (the spec's "no partial load"):

      1. Read the bundle (zip -> entries, with zip-slip + per-entry read errors mapped to a clean
         ``ValueError`` here) and ``validate_bundle`` it: require the three roles by extension AND
         FORMAT-GATE THE FIRST RECORD of each against its expected shape (a whole-file scan is NOT run — see
         ``validate_bundle``). A bad zip, a missing/duplicated role, or a first-record format failure raises
         HERE (``BundleValidationError`` carrying file+row detail), before ANY side effect — so it ingests
         nothing and writes no folder. Later rows are NOT gated here: a buggy later MEMBER row is skipped at
         ingest (step 3, returned in ``skipped``), and later eval/CSV rows ride through unchecked (a bad one
         surfaces only when that file is later consumed, e.g. ``DATASET=<name> make eval``).
      2. ``create_dataset_dir`` — ``mkdir(exist_ok=False)`` atomically RESERVES the folder name BEFORE the
         DB write. This is the single collision check (it matches what it creates — no ``is_dir`` vs
         ``mkdir`` drift), so a name that clashes with an existing dataset OR an existing file (e.g.
         ``health.db``) is a clean ``FileExistsError`` -> 409 with no members ingested.
      3. Inside the reserved folder, materialize the files (CANONICAL names; zip-slip re-guarded) and THEN
         ``ingest_members`` the already-parsed records — ADDITIVE (``ingest_bundle`` upserts per member;
         existing members stay, a re-used id refreshes that member; no reseed). Folder/disk failures
         therefore happen BEFORE the DB write, so they never strand committed members; on ANY failure the
         reserved folder is removed so the name stays free for a clean retry. The only residual
         side-effect-on-failure is the documented per-member semantic caveat in ``ingest_members``.

    The persisted ``eval_set.jsonl`` / ``lab_panels.csv`` are first-record format-gated here (step 1), then
    kept for later live use (e.g. ``DATASET=<name> make eval``). Returns
    ``{dataset, members, results, ranges, member_ids, files}``; the route adds the auto-scan."""
    dataset = derive_dataset_name(name, filename)

    is_zip = (filename or "").lower().endswith(".zip") or data[:2] == b"PK"
    if not is_zip:
        raise BundleValidationError(
            [
                {
                    "file": None,
                    "row": None,
                    "field": "",
                    "detail": "upload must be a .zip containing a .json (members), a .jsonl (eval set), and a .csv (lab panels)",
                }
            ]
        )
    entries = _zip_entries(
        _open_zip(data)
    )  # bad/odd zip + zip-slip -> ValueError, no side effect

    canonical, members_records, extras = validate_bundle(
        entries
    )  # roles by extension + per-row format, BEFORE any side effect

    dest = create_dataset_dir(
        dataset
    )  # mkdir(exist_ok=False): atomic name reservation, pre-ingest 409
    try:
        files = _write_dataset_folder(dest, canonical, extras)
        summary = ingest_members(
            con, members_records
        )  # DB write LAST; semantic-partial caveat only
        if summary["members"] == 0:
            # Every member row was skipped — all shape-valid at the first-record gate but semantically bad at
            # ingest (e.g. an unparseable reference_range on every row). An upload that ingests NOBODY is a
            # failed upload, not a 200 with an orphan folder: raise so the reserved folder is rolled back
            # below and the route 422s with the per-row detail, rather than persisting an empty dataset that
            # only 409s on retry. ``members_records`` is non-empty here (the gate rejects an empty file), so
            # members==0 ⟺ every row skipped ⟺ ``skipped`` is populated; the fallback is purely defensive.
            raise BundleValidationError(
                [
                    {
                        "file": s["file"],
                        "row": s["row"],
                        "field": "",
                        "detail": f"member {s['member_id']!r}: {s['detail']}",
                    }
                    for s in summary["skipped"]
                ]
                or [
                    {
                        "file": CANONICAL_MEMBERS,
                        "row": None,
                        "field": "",
                        "detail": "no members were ingested from the bundle",
                    }
                ]
            )
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)  # leave the name free for a clean retry
        raise
    return {"dataset": dataset, **summary, "files": files}


def seed_if_empty(con) -> dict:
    """Ingest the active dataset IFF the DB has no members — the idempotent startup path (the api.py
    lifespan / a fresh ephemeral deploy). Returns ``{"seeded": bool, ...counts}``; a no-op
    (``seeded=False``) when members already exist, so a warm restart or a prior ``make seed`` never
    re-ingests, and a deliberately ``DELETE``d member is never resurrected.

    The guard fires only on a TRULY empty DB, which is what makes a clean rollback safe: ``ingest_dataset``
    commits per member, so a mid-loop write failure would otherwise strand a partial set that the
    empty-check then reads as 'seeded' — never re-healing. Instead, on failure we delete the partial
    prefix (every member present is from this attempt, since the DB was empty) and re-raise, so the DB
    returns to empty and the next boot retries cleanly. This rollback is startup-only policy;
    ``ingest_dataset`` itself stays per-member-atomic for the CLI and ``/admin/reseed`` (force-ingest)."""
    if db.list_members(con):
        return {"seeded": False}
    try:
        return {"seeded": True, **ingest_dataset(con)}
    except Exception:
        for mid in db.list_members(
            con
        ):  # all from this failed attempt — the DB was empty before it
            db.delete_member(con, mid)
        raise


def _verify(con) -> None:
    """The Phase-2 runnable: query a member and run the pure core over the persisted data."""
    from health_intelligence.analysis import analyze
    from health_intelligence.config import ANALYSIS_CONFIG

    member_ids = db.list_members(con)
    if not member_ids:
        print("  (no members to verify)")
        return
    mid = member_ids[0]
    member, results, ranges, age, data_version = db.load_for_analysis(con, mid)
    analysis = analyze(
        member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version
    )
    print(
        f"  {mid}: data_version={data_version}  overall_floor={analysis.overall_floor}"
    )
    for mk in analysis.markers:
        flags = ",".join(mk.flags) or "-"
        print(f"    {mk.marker:20s} severity={mk.severity:9s} flags={flags}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Ingest a member bundle into SQLite (the firewall)."
    )
    ap.add_argument(
        "bundle",
        nargs="?",
        default=None,
        help="dataset sub-folder under backend/data (default: DATASET env or training_data)",
    )
    ap.add_argument(
        "--db", default=None, help="SQLite path (default: backend/data/health.db)"
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="after ingest, query one member and run analysis over the DB",
    )
    args = ap.parse_args(argv)

    con = db.connect(args.db)
    try:
        db.init_db(con)
        summary = ingest_dataset(con, args.bundle)
        print(
            f"ingested {summary['members']} members, {summary['results']} results, "
            f"{summary['ranges']} reference ranges"
        )
        # Every ingest path ends in a scan (the lifespan seed, /admin/reseed, /members/upload, and this
        # CLI — the `make seed` behind `make run`), so persisted Observations are always consistent with
        # the loaded data: the local one-command run boots with findings already visible, same as a fresh
        # deploy. Proactive-by-default is the product behavior ("surfaces drift without being asked");
        # the UI's Scan button is the manual re-run of the same idempotent scan, not the trigger.
        # Lazy import: pipeline pulls the LLM seam, which library importers of the firewall never need
        # (the scan itself is deterministic — no LLM call). Best-effort per member (scan_members logs and
        # skips a failing member), so a scan problem can never fail the seed it rides on.
        from health_intelligence import pipeline

        scanned = pipeline.scan_members(con, summary["member_ids"])
        print(f"scanned {scanned} members (observations up to date)")
        if args.verify:
            _verify(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
