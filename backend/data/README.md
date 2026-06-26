# Datasets

Each **dataset** is a sub-folder here holding one synthetic bundle:

- `members.json` — members (`member_id`, `profile`, `panels`, `notes`)
- `lab_panels.csv` — the same panels in long (tidy) form
- `eval_set.jsonl` — the labeled reference cases for the eval harness

One bundle per sub-folder lets new datasets be dropped in and ingested **incrementally**
without disturbing the shipped one. `training_data/` is the bundle that ships with the repo;
see its `README.md` for the bundle's field-level shape.

## Selecting the active dataset

`preprocessing/datasets.py` resolves which sub-folder is active from the `DATASET`
environment variable, defaulting to `training_data` when unset — so the tests and an
out-of-box run need no configuration. To work over a different bundle, drop it in a new
sub-folder (e.g. `q3_cohort/`) and set `DATASET=q3_cohort`; an unknown name fails loudly,
naming the folders that do exist. The ingest CLI that consumes the active dataset lands in
Phase 2 — see the root `README.md` for that command.

The derived SQLite store (`health.db`) is **not** a dataset; it sits at this `data/` root,
outside any bundle, and is git-ignored.

Synthetic data only — nothing here is real patient data, and none of it is for clinical use.
