"""eval/ — the offline evaluation harness (architecture §8).

Runs the supplied labeled case set (plus a few tagged additions) through BOTH modes against the live
service, scores each dimension with the system's own deterministic-vs-judge discipline, and emits a
markdown + JSON report whose JSON is the regression-gate artifact. The harness imports the serving
library (``health_intelligence``) and the ingestion firewall (``preprocessing``) but nothing imports
it back: eval is a pure consumer, parallel to ``tests/`` — the two-layer purity invariant holds.

Phase 5a ships the deterministic scorers + report + ``make eval``; the LLM judge (5b) and the LangSmith
trace sink (5c) are gated follow-on increments. Run it with ``python -m eval`` (see ``__main__``).
"""
