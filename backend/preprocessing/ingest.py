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
import json
import re
from typing import Optional

from health_intelligence import db
from health_intelligence.config import CONFIG_VERSION, MARKERS
from health_intelligence.models import LabResult, MemberBundle, RangeSex, ReferenceRange
from preprocessing.datasets import members_path

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
) -> list[tuple[RangeSex, Optional[float], Optional[float]]]:
    """Parse one printed range string into ``[(sex, ref_low, ref_high)]``.

    Precedence is deliberate — (1) Vitamin-D multi-band, (2) sex-split, (3) scalar — because the
    multi-band string contains ``<20`` / ``20-29`` substrings the scalar patterns would otherwise
    mis-match, and the sex-split wrapper must be peeled before its halves reach the scalar parser.
    """
    del marker  # reserved (architecture's documented signature; dispatch is structural on `raw`)
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


def _parse_graded_deficiency_floor(s: str) -> list[tuple[RangeSex, Optional[float], Optional[float]]]:
    """Reduce the Vitamin-D multi-band to one reference interval: the deficiency floor. ``ref_low`` is
    the threshold from the ``<NN deficient`` clause (below it is unambiguously abnormal -> below_range);
    the sufficient/insufficient/deficient nuance is owned by ``config.graded_bands`` (band-crossing),
    so it is not double-counted here. ``ref_high`` is None (no upper reference for Vitamin D)."""
    m = re.search(r"<\s*([0-9.]+)\s*deficient", s)
    if m:
        return [("any", float(m.group(1)), None)]
    raise ValueError(f"unrecognized graded range string: {s!r}")


def _parse_scalar(s: str) -> tuple[Optional[float], Optional[float]]:
    """Parse a single scalar form into ``(ref_low, ref_high)``. ``<=``/``>=`` are matched before
    ``<``/``>`` so the longer operator wins; ``fullmatch`` rejects anything unexpected loudly."""
    s = s.strip()
    if (m := re.fullmatch(r"<=\s*([0-9.]+)", s)):
        return (None, float(m.group(1)))
    if (m := re.fullmatch(r"<\s*([0-9.]+)", s)):
        return (None, float(m.group(1)))
    if (m := re.fullmatch(r">=\s*([0-9.]+)", s)):
        return (float(m.group(1)), None)
    if (m := re.fullmatch(r">\s*([0-9.]+)", s)):
        return (float(m.group(1)), None)
    if (m := re.fullmatch(r"([0-9.]+)\s*-\s*([0-9.]+)", s)):
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
            results.append(LabResult(
                marker=res.analyte,                 # analyte -> marker is identity (canonical already)
                value=res.value, unit=res.unit,
                panel_id=panel.panel_id, panel_date=panel.collected_date,
            ))
            lab_range_src.setdefault(res.analyte, (res.reference_range, res.unit))
        for vk in VITALS:
            results.append(LabResult(
                marker=vk, value=getattr(panel.vitals, vk),
                unit=_require_unit(vk),             # vitals' unit comes from config (data prints none)
                panel_id=panel.panel_id, panel_date=panel.collected_date,
            ))

    _assert_unique_markers_per_panel(results)
    ranges = _build_ranges(lab_range_src)
    db.replace_member(con, profile=bundle.profile, results=results, ranges=ranges, notes=bundle.notes)
    return {"member_id": bundle.member_id, "results": len(results), "ranges": len(ranges)}


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
            ranges.append(ReferenceRange(
                marker=marker, sex=sex, unit=unit,
                ref_low=ref_low, ref_high=ref_high,
                panic_low=panic_low, panic_high=panic_high,
                config_version=CONFIG_VERSION,
            ))
        # A sex-split marker has only male/female rows. An 'other'/'unknown'-sex member would fall
        # through _range_for (sex row -> 'any') to a non-existent 'any' row and get a `no_reference`
        # flag *before* the panic check — so a sex-independent panic (e.g. Hemoglobin 7.0) would
        # silently never fire for them. If this marker carries a panic, also emit a panic-only 'any'
        # row (ref bounds None — we can't pick a sex's normal range) so the safety floor still fires.
        # "Rather over-escalate than miss" (architecture §6). Only markers WITH a panic get the row,
        # so non-panic sex-split markers keep their honest `no_reference` for an unknown-sex member.
        if all(sex != "any" for sex, _, _ in parsed) and (panic_low is not None or panic_high is not None):
            ranges.append(ReferenceRange(
                marker=marker, sex="any", unit=unit,
                ref_low=None, ref_high=None,
                panic_low=panic_low, panic_high=panic_high,
                config_version=CONFIG_VERSION,
            ))

    for vk in VITALS:
        mcfg = MARKERS[vk]
        ranges.append(ReferenceRange(
            marker=vk, sex="any", unit=_require_unit(vk),
            ref_low=mcfg.ref_low, ref_high=mcfg.ref_high,
            panic_low=mcfg.panic_low, panic_high=mcfg.panic_high,
            config_version=CONFIG_VERSION,
        ))

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

def ingest_dataset(con, dataset: Optional[str] = None) -> dict:
    """Load ``members.json`` for the active dataset, validate every bundle (the firewall check —
    ``extra='forbid'`` makes a malformed bundle fail loudly), and write each. Returns summary counts."""
    raw = json.loads(members_path(dataset).read_text())
    members = 0
    total_results = 0
    for entry in raw:
        bundle = MemberBundle.model_validate(entry)
        summary = ingest_bundle(con, bundle)
        members += 1
        total_results += summary["results"]
    n_ranges = len(db.get_ranges(con))  # via db.py (the one SQLite seam), scoped to the active config
    return {"members": members, "results": total_results, "ranges": n_ranges}


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
    analysis = analyze(member, results, ranges, age, ANALYSIS_CONFIG, data_version=data_version)
    print(f"  {mid}: data_version={data_version}  overall_floor={analysis.overall_floor}")
    for mk in analysis.markers:
        flags = ",".join(mk.flags) or "-"
        print(f"    {mk.marker:20s} severity={mk.severity:9s} flags={flags}")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Ingest a member bundle into SQLite (the firewall).")
    ap.add_argument("bundle", nargs="?", default=None,
                    help="dataset sub-folder under backend/data (default: DATASET env or training_data)")
    ap.add_argument("--db", default=None, help="SQLite path (default: backend/data/health.db)")
    ap.add_argument("--verify", action="store_true",
                    help="after ingest, query one member and run analysis over the DB")
    args = ap.parse_args(argv)

    con = db.connect(args.db)
    try:
        db.init_db(con)
        summary = ingest_dataset(con, args.bundle)
        print(f"ingested {summary['members']} members, {summary['results']} results, "
              f"{summary['ranges']} reference ranges")
        if args.verify:
            _verify(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
