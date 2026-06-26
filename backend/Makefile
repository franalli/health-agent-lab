# Health Intelligence Service — the intended interface (CLAUDE.md "Commands"). Run from backend/.
# uv owns the venv + dependency resolution; every target runs inside it.

.PHONY: init-db seed run test eval help
.DEFAULT_GOAL := help

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*##"}{printf "  %-10s %s\n", $$1, $$2}'

init-db:  ## create the SQLite schema (idempotent)
	uv run python -c "from health_intelligence import db; con = db.connect(); db.init_db(con); con.close()"

seed:  ## ingest the active dataset (DATASET env; default training_data) — inits the schema first
	uv run python -m preprocessing.ingest

run: seed  ## init + seed + serve the API on :8000 (UI mounts here once frontend/ lands in Phase 6)
	uv run uvicorn api:app --host 0.0.0.0 --port 8000

test:  ## unit tests
	uv run pytest

eval:  ## run the evaluation harness (lands in Phase 5)
	@echo "eval harness is not built yet — it lands in Phase 5 (architecture.md build sequence)."
