# Health Intelligence Service — the intended interface (CLAUDE.md "Commands"). Run from backend/.
# uv owns the venv + dependency resolution; every target runs inside it.

.PHONY: init-db seed run serve test eval eval-judge lint hooks help
.DEFAULT_GOAL := help

# Shared uvicorn invocation; `run` (dev) and `serve` (prod-mode) differ only in port/reload/workers.
UVICORN := uv run uvicorn api:app --host 0.0.0.0

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*##"}{printf "  %-10s %s\n", $$1, $$2}'

init-db:  ## create the SQLite schema (idempotent)
	uv run python -c "from health_intelligence import db; con = db.connect(); db.init_db(con); con.close()"

seed:  ## ingest the active dataset (DATASET env; default training_data) — inits the schema first
	uv run python -m preprocessing.ingest

run: seed  ## init + seed + serve the API + UI on :8000 (--reload: backend edits restart automatically)
	$(UVICORN) --port 8000 --reload

serve:  ## production-mode serve (no reload; --workers 1; binds $PORT or 8000) — mirrors the Render start command
	$(UVICORN) --port $${PORT:-8000} --workers 1

test:  ## unit tests
	uv run pytest

hooks:  ## install the git pre-commit hook (run once after clone; config is at the repo root)
	cd .. && uv run --project backend pre-commit install

lint:  ## run all pre-commit checks (ruff lint+format, gitleaks) over the whole tree
	cd .. && uv run --project backend pre-commit run --all-files

eval:  ## run the evaluation harness → markdown + JSON report under eval/reports/ (Mode 2 needs ANTHROPIC_API_KEY)
	uv run python -m eval

eval-judge:  ## accuracy gate for the feedback input-judge (real Haiku; needs ANTHROPIC_API_KEY; no key -> SKIP)
	uv run python -m eval.judge_eval
