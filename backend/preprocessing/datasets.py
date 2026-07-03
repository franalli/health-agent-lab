"""Dataset location — the one place that resolves *which* synthetic bundle is active, and *where*
new bundles are written.

A dataset is a sub-folder of the datasets root holding one bundle (``members.json``,
``lab_panels.csv``, ``eval_set.jsonl``, ``README.md``). Giving each bundle its own folder lets a new
dataset be dropped in and ingested incrementally without disturbing the shipped one.

Selection is a runtime concern (*which* bundle to read), so it lives here in the ingestion
firewall — never in the pure core (``analysis.py``) or the clinical config (``config.py``).
The active dataset is named by the ``DATASET`` environment variable and defaults to
``training_data`` (the shipped bundle), so the tests and an out-of-box run need no
configuration. The derived SQLite store (``health.db``) is *not* a dataset; it sits at the
datasets-root, outside any bundle.

The datasets root itself is ``backend/data/`` by default, overridable via ``HEALTH_DATA_ROOT`` —
the seam that (1) lets a test point the root at a temp dir and (2) lets a Render persistent disk hold
runtime-uploaded datasets durably (mirror of ``HEALTH_DB_PATH`` for the DB; §15). It is resolved
**per call** (``data_root()``), never frozen at import, so an override set after import still takes.
``POST /members/upload`` *creates* a dataset folder here at runtime — the write helpers
(:func:`create_dataset_dir`, the name guards) live here because folder layout and naming are this
module's concern; the firewall (``ingest.py``) owns parsing the uploaded archive into it.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import shutil

logger = logging.getLogger(__name__)

#: ``backend/data`` — the default datasets root. Each sub-folder is one bundle. Overridable per-call
#: via ``HEALTH_DATA_ROOT`` (see :func:`data_root`); kept as a constant for the override's fallback.
_DEFAULT_DATA_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"

#: Sub-folder used when ``DATASET`` is unset — the bundle shipped with the repo.
DEFAULT_DATASET = "training_data"

#: A safe dataset-folder name: starts alphanumeric, then word chars / dot / dash, ≤64 chars. No path
#: separators and no leading dot — so a name can never traverse out of the root or hide as a dotfile.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def data_root() -> pathlib.Path:
    """The datasets root, resolved fresh each call: ``HEALTH_DATA_ROOT`` if set, else the repo default.
    Per-call (not a module constant) so a test or deploy that sets the env after import still wins."""
    override = os.environ.get("HEALTH_DATA_ROOT")
    return pathlib.Path(override) if override else _DEFAULT_DATA_ROOT


def dataset_dir(name: str | None = None) -> pathlib.Path:
    """Resolve an existing dataset folder under :func:`data_root`.

    Precedence: explicit ``name`` argument, then the ``DATASET`` env var, then
    :data:`DEFAULT_DATASET`. Raises ``FileNotFoundError`` naming the available datasets, so a
    mistyped bundle fails loudly instead of silently ingesting nothing.
    """
    name = name or os.environ.get("DATASET", DEFAULT_DATASET)
    root = data_root()
    path = root / name
    if not path.is_dir():
        available = (
            sorted(p.name for p in root.iterdir() if p.is_dir())
            if root.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"dataset {name!r} not found under {root} (available: {available or 'none'})"
        )
    return path


def members_path(name: str | None = None) -> pathlib.Path:
    """Path to the member bundle (``members.json``) of the active dataset."""
    return dataset_dir(name) / "members.json"


# --------------------------------------------------------------------------------------------------
# Runtime dataset creation — the write side, used by the POST /members/upload firewall. Naming is
# validated HERE (path-safety is a storage-layout concern) before any file is written.
# --------------------------------------------------------------------------------------------------


def sanitize_dataset_name(raw: str | None) -> str:
    """Validate a candidate dataset-folder name against :data:`_NAME_RE` and return it unchanged.

    Rejects (``ValueError``) anything that could escape the root or collide with tooling: empty, ``.``
    / ``..``, a leading dot, any ``/`` or ``\\``, or out-of-charset/over-length. This is the single
    guard between an uploaded/operator-supplied name and a filesystem path — a path-traversal name
    (``../../etc``) never reaches :func:`create_dataset_dir`."""
    name = (raw or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid dataset name {raw!r}: use letters, digits, '.', '_', '-' "
            "(start alphanumeric, <=64 chars, no path separators)"
        )
    return name


def derive_dataset_name(explicit: str | None, filename: str | None) -> str:
    """Resolve the new dataset's folder name: an explicit name if given, else the uploaded file's stem
    (minus a ``.zip``/``.json`` suffix). Always routed through :func:`sanitize_dataset_name`, so a
    junk filename (or none) raises a clear error rather than producing an unsafe folder."""
    if explicit and explicit.strip():
        return sanitize_dataset_name(explicit)
    stem = pathlib.PurePosixPath(
        filename or ""
    ).name  # basename only — drop any client-sent path
    for suffix in (".zip", ".json"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return sanitize_dataset_name(stem)


def remove_uploaded_datasets(keep: str = DEFAULT_DATASET) -> list[str]:
    """Delete every dataset folder under :func:`data_root` EXCEPT ``keep`` (the shipped ``training_data``)
    — the filesystem half of ``POST /admin/reseed``'s factory reset, whose contract is "remove ALL data
    folders except training_data, then reload training_data" (the DB reload is the pinned
    ``ingest_dataset(DEFAULT_DATASET)`` in ``db.reseed_transaction``). ``POST /members/upload`` creates
    ``<root>/<name>/`` folders; a DB-only reseed truncated their members but stranded the folders on disk,
    so a "factory reset" left prior uploads discoverable and re-selectable. This returns the root to just
    the shipped bundle.

    A FACTORY RESET, so the active ``DATASET`` is deliberately NOT spared: reseed always reverts to
    training_data (the re-ingest is pinned to :data:`DEFAULT_DATASET`), so a deploy running ``DATASET=<name>``
    that factory-resets is choosing to discard ``<name>`` — the DB is reloaded from training_data regardless.

    Defensive by construction: only immediate SUB-DIRECTORIES are touched — never a file (``health.db``,
    ``health.db.learn.lock``, ``README.md``, a stray ``.zip``) and never the shipped ``keep`` bundle — and a
    symlink child is skipped (``rmtree`` must never follow a link out of the root). Best-effort AND TOTAL:
    the listing itself AND every per-child stat/delete are guarded, so NO ``OSError`` — an unreadable/missing
    root, a TOCTOU race, an ``EACCES`` on a child ``stat`` (``pathlib`` re-raises those; it swallows only
    ENOENT/ENOTDIR/EBADF/ELOOP) — can escape. The DB reset has already committed by the time the caller runs
    this, so nothing here may fail the reseed or pre-empt its auto-scan (the same after-commit, non-fatal
    contract as ``pipeline.scan_members``). Returns the names removed (sorted)."""
    root = data_root()
    removed: list[str] = []
    try:
        children = list(
            root.iterdir()
        )  # also handles a missing/non-dir root -> OSError -> []
    except OSError as e:
        logger.warning("reseed: could not list the datasets root %s: %s", root, e)
        return []  # nothing removed yet on the listing-failure branch
    for child in children:
        try:
            if child.name == keep or child.is_symlink() or not child.is_dir():
                continue
            shutil.rmtree(child)
            removed.append(child.name)
        except (
            OSError
        ) as e:  # per-child stat/delete failure — log and skip, never fail the reseed
            logger.warning("reseed: could not remove dataset folder %s: %s", child, e)
    return sorted(removed)


def create_dataset_dir(name: str) -> pathlib.Path:
    """Create and return ``<root>/<name>`` for a runtime upload — the SINGLE collision check for an
    upload (the firewall reserves the name with this before the DB write). ``name`` must be pre-sanitized
    (re-guarded here defensively). ``exist_ok=False`` makes a name collision an atomic ``FileExistsError``
    — and because ``mkdir`` raises on ANY existing path (a dir OR a file such as ``health.db``), this is
    a stronger, drift-free guard than a separate ``is_dir`` pre-check would be — so an upload never
    silently overwrites an existing dataset (the shipped ``training_data`` included) and never lands a
    folder atop an existing file. The route maps the ``FileExistsError`` to 409. Parents (the root) are
    created if absent — a fresh ``HEALTH_DATA_ROOT`` or ephemeral deploy self-heals the root."""
    sanitize_dataset_name(
        name
    )  # defense in depth; callers already sanitize via derive_dataset_name
    path = data_root() / name
    path.mkdir(parents=True, exist_ok=False)
    return path
