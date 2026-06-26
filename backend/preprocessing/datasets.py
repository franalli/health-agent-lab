"""Dataset location — the one place that resolves *which* synthetic bundle is active.

A dataset is a sub-folder of ``backend/data/`` holding one bundle (``members.json``,
``lab_panels.csv``, ``eval_set.jsonl``). Giving each bundle its own folder lets a new
dataset be dropped in and ingested incrementally without disturbing the shipped one.

Selection is a runtime concern (*which* bundle to read), so it lives here in the ingestion
firewall — never in the pure core (``analysis.py``) or the clinical config (``config.py``).
The active dataset is named by the ``DATASET`` environment variable and defaults to
``training_data`` (the shipped bundle), so the tests and an out-of-box run need no
configuration. The derived SQLite store (``health.db``) is *not* a dataset; it sits at the
``data/`` root, outside any bundle.
"""

from __future__ import annotations

import os
import pathlib

#: ``backend/data`` — the datasets root. Each sub-folder is one bundle.
DATA_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data"

#: Sub-folder used when ``DATASET`` is unset — the bundle shipped with the repo.
DEFAULT_DATASET = "training_data"


def dataset_dir(name: str | None = None) -> pathlib.Path:
    """Resolve the active dataset folder under :data:`DATA_ROOT`.

    Precedence: explicit ``name`` argument, then the ``DATASET`` env var, then
    :data:`DEFAULT_DATASET`. Raises ``FileNotFoundError`` naming the available datasets, so a
    mistyped bundle fails loudly instead of silently ingesting nothing.
    """
    name = name or os.environ.get("DATASET", DEFAULT_DATASET)
    path = DATA_ROOT / name
    if not path.is_dir():
        available = (
            sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
            if DATA_ROOT.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"dataset {name!r} not found under {DATA_ROOT} (available: {available or 'none'})"
        )
    return path


def members_path(name: str | None = None) -> pathlib.Path:
    """Path to the member bundle (``members.json``) of the active dataset."""
    return dataset_dir(name) / "members.json"
