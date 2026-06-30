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
import io
import json
import lzma
import os
import pathlib
import re
import shutil
import zipfile
import zlib

from health_intelligence import db
from health_intelligence.config import CONFIG_VERSION, MARKERS
from health_intelligence.models import LabResult, MemberBundle, RangeSex, ReferenceRange
from preprocessing.datasets import (
    create_dataset_dir,
    derive_dataset_name,
    members_path,
)

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


def ingest_bundle(con, bundle: MemberBundle) -> dict:
    """Normalize one member and persist it (replacing any prior version). Returns summary counts."""
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
        con, profile=bundle.profile, results=results, ranges=ranges, notes=bundle.notes
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


def ingest_dataset(con, dataset: str | None = None) -> dict:
    """Load ``members.json`` for the active dataset, validate every bundle (the firewall check —
    ``extra='forbid'`` makes a malformed bundle fail loudly), and write each. Returns summary counts.

    Bundles are validated IN FULL before any write: a malformed bundle raises here while the DB is still
    untouched, rather than after some members have already been committed (``db.replace_member`` is
    per-member-atomic, so an interleaved validate→write loop would strand a partial set on a bad entry).
    This makes the dominant failure mode (malformed data) leave no partial state; ``seed_if_empty`` adds
    the rollback for the rarer write-phase failure on the unattended startup path.

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
    return ingest_members(con, records)


def ingest_members(con, raw: object) -> dict:
    """Validate and write a ``members.json``-shaped array of bundles. The shared core of
    ``ingest_dataset`` (the seed/startup path) and the ``/members/upload`` zip route — both feed it the
    parsed array. Returns ``{members, results, ranges, member_ids}``; ``member_ids`` lets the uploader
    focus the picker on what just landed.

    Two-stage validation, and the partial-set boundary sits between them: every bundle's *Pydantic
    shape* is checked up front (the list-comp below), so a SHAPE-malformed entry raises while the DB is
    still untouched — no partial set. But the firewall's *semantic* parse (reference-range shapes,
    marker-uniqueness) runs per member inside ``ingest_bundle``, which commits as it goes, so a
    semantically-bad bundle LATE in the array raises only after the earlier members are already written
    — leaving a partial set. We deliberately do NOT roll that back here: ``ingest_bundle`` *upserts*, so
    deleting "what this call wrote" could clobber a member that pre-existed the call. The recovery is to
    re-upload the corrected bundle (idempotent upsert) or reseed. The unattended startup path
    (``seed_if_empty``) carries its own delete-the-prefix rollback because there it is provably safe (the
    DB was empty), and that is the one place the partial set must not silently read as 'seeded'."""
    if not isinstance(raw, list):
        raise ValueError(
            "members.json must be a JSON array of member bundles "
            f"(got {type(raw).__name__}); a single bundle goes to POST /members"
        )
    bundles = [
        MemberBundle.model_validate(entry) for entry in raw
    ]  # firewall: validate ALL up front
    # De-dup by member_id (LAST occurrence wins, matching upsert order): a duplicate id in one upload
    # would otherwise ingest the same member twice and inflate the reported members/results counts.
    deduped = list({bundle.member_id: bundle for bundle in bundles}.values())
    member_ids = []
    total_results = 0
    for bundle in deduped:
        summary = ingest_bundle(con, bundle)
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
    }


# --------------------------------------------------------------------------------------------------
# Uploaded-bundle handling — the firewall for a runtime hold-out. The hold-out ships the SAME three
# kinds of file as `training_data` but the NAMES may differ (e.g. `panel.csv`/`members.json`/`eval.json`,
# not guaranteed), so discovery is by TYPE/CONTENT, not name. The bundle is classified, its members are
# ingested through the same `ingest_members` the seed uses, and every file is written to the new dataset
# folder under its CANONICAL name — so every downstream reader (the seed loader, `members_path`, the eval
# adapter) stays name-based and UNCHANGED; this upload boundary is the single place a name is normalized.
# --------------------------------------------------------------------------------------------------

#: The on-disk names every dataset folder uses, whatever the upload called its files.
CANONICAL_MEMBERS = "members.json"
CANONICAL_PANELS = "lab_panels.csv"
CANONICAL_EVAL = "eval_set.jsonl"

#: The record fields that distinguish a members file from an eval file — bound to ``MemberBundle`` so a
#: rename of those model fields fails LOUDLY here at import, not silently at every upload (a members
#: record carries ``profile``/``panels``; an eval case carries ``input``/``category`` and neither).
_MEMBERS_KEYS = ("profile", "panels")
if not all(k in MemberBundle.model_fields for k in _MEMBERS_KEYS):
    raise RuntimeError(
        f"_MEMBERS_KEYS {_MEMBERS_KEYS} drifted from MemberBundle fields "
        f"{tuple(MemberBundle.model_fields)} — update the members-file discriminator"
    )


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


def _try_records(data: bytes) -> list | None:
    """``read_records`` but ``None`` instead of raising — for classification, where a non-JSON file (the
    lab-panels CSV, a README) is expected and is simply not a JSON role rather than an error."""
    try:
        return read_records(data)
    except ValueError:
        return None


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


def _is_members_record(rec: object) -> bool:
    """Whether a parsed record is a member bundle — it carries the :data:`_MEMBERS_KEYS` (``profile`` +
    ``panels``). This record SHAPE, not the filename, is what tells the members file from the eval file
    (the hold-out ships identical schemas under possibly-different names)."""
    return isinstance(rec, dict) and all(k in rec for k in _MEMBERS_KEYS)


def classify_bundle(
    entries: list[tuple[str, bytes]],
) -> tuple[dict[str, bytes], list, list[tuple[str, bytes]]]:
    """Resolve the dataset's roles by TYPE/CONTENT (not filename). Returns
    ``(canonical, members_records, extras)``:

      * ``canonical`` maps the CANONICAL on-disk name -> uploaded bytes: ``members.json`` (always),
        ``lab_panels.csv`` and ``eval_set.jsonl`` (if present);
      * ``members_records`` is the members file ALREADY PARSED (so the caller ingests without re-parsing);
      * ``extras`` are any other files (e.g. a README), kept verbatim.

    Each file is sniffed ONCE with :func:`_try_records`: a file that parses as JSON array/JSONL with
    dict records is a JSON role (the one that's members-shaped is the members file; another is the eval
    set), and a file that doesn't parse as JSON is a non-JSON role (the ``.csv`` is the lab panels, the
    rest are extras). Discovery is by content + extension HINT for the CSV only — never by the members
    file being literally named ``members.json``. Roles are tracked by INDEX, so duplicate entry names in
    the zip never cause a file to be silently dropped.

    Rejects (``ValueError``, NO partial load) a bundle that can't resolve: zero or several members-shaped
    JSON files, more than one eval-shaped JSON, or more than one CSV. The **members file is the only hard
    requirement** (it alone is ingested); panels/eval are optional and a README is tolerated."""
    sniffed = [(n, b, _try_records(b)) for (n, b) in entries]

    def _first_dict(recs: list | None) -> dict | None:
        """The file's first record iff it's a JSON file of dict records, else None — the clean
        (type-narrowing) basis for routing a JSON file by shape; a CSV/README has ``recs is None``."""
        return recs[0] if recs and isinstance(recs[0], dict) else None

    members_idx = [
        i
        for i, (_, _, recs) in enumerate(sniffed)
        if _is_members_record(_first_dict(recs))
    ]
    if len(members_idx) != 1:
        raise ValueError(
            f"could not identify the members file: expected exactly one JSON whose records carry "
            f"{list(_MEMBERS_KEYS)}, found {len(members_idx)} "
            f"(files seen: {[n for n, _, _ in sniffed] or 'none'})"
        )
    mi = members_idx[0]
    members_records = (
        sniffed[mi][2] or []
    )  # non-None by the members_idx filter; `or []` narrows the type
    canonical: dict[str, bytes] = {CANONICAL_MEMBERS: sniffed[mi][1]}

    eval_idx = [
        i
        for i, (_, _, recs) in enumerate(sniffed)
        if i != mi
        and (r := _first_dict(recs)) is not None
        and not _is_members_record(r)
    ]
    if len(eval_idx) > 1:
        raise ValueError(
            f"ambiguous bundle: {len(eval_idx)} eval-shaped JSON files, expected one "
            f"({[sniffed[i][0] for i in eval_idx]})"
        )
    if eval_idx:
        canonical[CANONICAL_EVAL] = sniffed[eval_idx[0]][1]

    csv_idx = [
        i
        for i, (n, _, recs) in enumerate(sniffed)
        if i != mi and recs is None and n.lower().endswith(".csv")
    ]
    if len(csv_idx) > 1:
        raise ValueError(
            f"ambiguous bundle: {len(csv_idx)} CSV files, expected one lab-panels CSV "
            f"({[sniffed[i][0] for i in csv_idx]})"
        )
    if csv_idx:
        canonical[CANONICAL_PANELS] = sniffed[csv_idx[0]][1]

    claimed = {mi, *eval_idx, *csv_idx}
    extras = [(n, b) for i, (n, b, _) in enumerate(sniffed) if i not in claimed]
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
    """The ``POST /members/upload`` firewall: classify an uploaded hold-out bundle by type/content, ingest
    its members ADDITIVELY, and persist the whole bundle as a new dataset folder under the datasets root.

    Accepts a ``.zip`` of a ``training_data``-shaped folder — one CSV (lab panels) + the members JSON +
    an optional eval JSON, under ANY filenames — or a bare members-shaped ``.json``. Order is deliberate
    so a failure leaves the cleanest state (the spec's "no partial load"):

      1. Read the bundle (zip -> entries, with zip-slip + per-entry read errors mapped to a clean
         ``ValueError`` here; or the bare file) and ``classify_bundle`` it by content. An unreadable zip
         or unresolvable bundle raises HERE, before ANY side effect — so it ingests nothing.
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

    The persisted ``eval_set.jsonl`` / ``lab_panels.csv`` are stored opaquely for later live use (e.g.
    ``DATASET=<name> make eval``); their CONTENTS are validated only when consumed, not here. Returns
    ``{dataset, members, results, ranges, member_ids, files}``; the route adds the auto-scan."""
    dataset = derive_dataset_name(name, filename)

    is_zip = (filename or "").lower().endswith(".zip") or data[:2] == b"PK"
    if is_zip:
        entries = _zip_entries(
            _open_zip(data)
        )  # bad/odd zip + zip-slip -> ValueError, no side effect
    else:
        bare_name = pathlib.PurePosixPath(filename or "").name or CANONICAL_MEMBERS
        entries = [(bare_name, data)]

    canonical, members_records, extras = classify_bundle(
        entries
    )  # by content, BEFORE any side effect

    dest = create_dataset_dir(
        dataset
    )  # mkdir(exist_ok=False): atomic name reservation, pre-ingest 409
    try:
        files = _write_dataset_folder(dest, canonical, extras)
        summary = ingest_members(
            con, members_records
        )  # DB write LAST; semantic-partial caveat only
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
        if args.verify:
            _verify(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
