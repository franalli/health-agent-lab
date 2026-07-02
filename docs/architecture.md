# Health Intelligence Service — Architecture

> A persistent Health Intelligence Service: two behaviors — grounded Q&A and proactive drift observations — over a member's longitudinal record, with safe refusal/escalation, explainability, and consistency throughout. Synthetic data only; no PHI.

**Organizing principle.** The LLM is a *language interface over a deterministic analytical core*, not the core itself. Every number, reference-range comparison, trend call, and escalation decision is computed by deterministic code or classical statistics; the LLM only renders that ground truth into careful language within guardrails it cannot override. "Show your work" is then structurally cheap — the opaque component never produced a load-bearing fact — which is the defensible posture near a clinician.

---

## Contents

- [Build sequence](#build-sequence)
- [1. Service shape](#1-service-shape)
- [2. Key design decisions](#2-key-design-decisions)
  - [Mode 1 conversation loop](#mode-1-conversation-loop)
  - [Per-turn message shapes (the chat loop)](#per-turn-message-shapes-the-chat-loop)
  - [What each LLM call site receives](#what-each-llm-call-site-receives)
  - [Escalation model (both behaviors)](#escalation-model-both-behaviors)
  - [D3 in detail — the signals + triage](#d3-in-detail--the-signals--triage)
  - [analysis.py — the deterministic core (module spec)](#analysispy--the-deterministic-core-module-spec)
- [3. AI vs non-AI boundary table](#3-ai-vs-non-ai-boundary-table)
- [4. Data model](#4-data-model)
- [5. Latency & cost budget](#5-latency--cost-budget)
- [6. Production posture](#6-production-posture)
- [7. Consistency](#7-consistency)
- [8. Evaluation harness](#8-evaluation-harness)
  - [eval/harness.py — the harness (module spec)](#evalharnesspy--the-harness-module-spec)
- [9. Feedback loop & self-improvement](#9-feedback-loop--self-improvement)
  - [Self-improvement (the live slice)](#self-improvement-the-live-slice)
- [10. How output lands with a member (+ one tension)](#10-how-output-lands-with-a-member--one-tension)
- [11. Explicitly out of scope (deliberate cuts)](#11-explicitly-out-of-scope-deliberate-cuts)
- [12. Design priorities](#12-design-priorities)
- [13. API surface](#13-api-surface)
- [14. Repository layout](#14-repository-layout)
- [15. Running and deployment](#15-running-and-deployment)

---

## Build sequence

Dependency-driven and **deterministic-first**: the statistical core and the entire non-LLM system are built and tested before the LLM goes on top, so there is a working deterministic system at the halfway mark — the proactive behavior and every safety decision, with no LLM, a viable assistant in its own right — and the LLM arrives as a language layer over a base that already decides everything safety-relevant. Every phase leaves something runnable, and the API surface **accretes as thin route adapters per phase** rather than as one late step. We do not scaffold everything up front; each phase adds only what the next one needs. Section numbers point into the detailed design below; route names in the phases are shorthand for their §13 paths (e.g. `/ask` is `POST /members/{id}/ask`). Each phase closes with three markers: *Runnable* (the working slice it leaves), *Review* (what is most likely to break and what code review should scrutinize), and *Manual happy-path* (the human smoke test that confirms it).

**Phase 0 — Shape (contracts, config, DDL; no logic).** Initialize the backend as a **`uv`-managed project** (`uv init`, then `uv add` for dependencies) — `uv` owns the virtualenv and dependency resolution via `pyproject.toml` + `uv.lock`, so every later phase runs in one reproducible environment. Write `config.py` (the model pin, the statistical thresholds — Mann–Kendall α, FDR `q`, minimum series length `n_min`, Theil–Sen settings, band cut-points — and the per-marker clinical constants: CVa/CVi, adverse-direction, **panic thresholds**, the graded **Vitamin-D bands**, and **vital bounds** — all curated, since the supplied data carries no critical thresholds — plus `config_version`), `models.py` (every Pydantic contract: `MemberBundle` (the upload/ingest input: `member_id` + `profile` + `panels[]` + `notes[]`) and `AskRequest` (`{message, history[]}`, where `ConversationTurn` = `{role, content}` carries the client-replayed prior turns for multi-turn context); `MemberProfile` (`member_id`, `age`, `sex`, `conditions`, `medications`, `family_history`, `lifestyle`), `LabResult` (`marker`, `value`, `unit`, `panel_id`, `panel_date`), `Note` (`date`, `source`, `text`), `ReferenceRange`, `TrendResult`, `MarkerTrajectory`, `TrajectoryAnalysis`, `HealthIntelligenceResponse`, `Observation`, `Escalation`, `Feedback`, `SuggestedPrompt`), and `schema.sql` (the nine tables). This is first because every later module imports the models, ingestion and `db.py` both conform to them, and the schema is the persistence contract — and because it is pure declaration it costs little. → *Runnable:* `schema.sql` loads into SQLite and the models round-trip-validate. **Review — likely to break:** a model field that doesn't match its table column or whose enum diverges from the schema `CHECK` (the two-layer seam); a JSON column (`conditions_json`) typed wrong on the model; a missing `config_version`; a model that fails to round-trip. **Manual happy-path:** load `schema.sql` into a fresh SQLite and confirm the nine tables; instantiate each model from sample data and round-trip it (model → dict → model) with no validation error. *(§4)*

**Phase 1 — Load the bundle + the deterministic core.** Load the supplied bundle (15 members, 3–5 panels across ~2 years, real but **unlabeled** trajectories) into `data/training_data/` (each dataset is its own sub-folder of `data/`, so further bundles ingest incrementally), and author a few tiny synthetic fixtures with **known** trends for unit testing (the data README permits extension; these give the analysis tests a ground-truth answer the supplied members deliberately withhold), then build `analysis.py` — the pure core: Mann–Kendall (with Kendall's τ and an exact small-n p-value) for monotonic-trend significance, Theil–Sen (with its CI) for direction and rate, Reference Change Value for clinical-vs-noise, reference-range and panic flags, band-crossing, and direction-aware triage with a Benjamini–Hochberg FDR pass across markers, emitting a typed `TrajectoryAnalysis` — with its unit tests. This is the heart of the system and the part that most warrants scrutiny, so it is proven in isolation before anything is wired to it; because `analysis.py` is pure (no DB, no LLM) it is unit-testable directly against the known-answer fixtures ("a seeded downward slope — Mann–Kendall must flag it significant and Theil–Sen must recover the rate"). At `n` as low as **3** (the sparse member) the tests must confirm the honest verdict — *too short to call a trend* — as a first-class output, not a failure. These signals are classical statistics + clinical thresholds, *not* trained ML — a distinction to keep precise. → *Runnable:* the signals on the fixtures and on a supplied member, trend-vs-noise on a deliberately noisy marker, graceful abstention at n=3, and the escalation floor each finding implies. **Review — likely to break:** the core doing any I/O (it must stay pure — no DB, LLM, network, or clock); Mann–Kendall using the normal approximation instead of the exact p at small n; Theil–Sen signing a direction when its CI spans zero; RCV crashing rather than skipping when CVa/CVi is absent; FDR not applied across markers; a trend claimed at n < n_min instead of abstaining; a beneficial-direction move raising severity; the floor not the max per-marker severity. **Manual happy-path:** run `analysis.py` on a seeded downward-slope fixture (MK flags it significant, Theil–Sen recovers the rate); on member C12 at n=3 (returns *too short to call*); on a deliberately noisy marker (reads as noise); confirm the floor each finding implies. *(§2 D3 + the `analysis.py` module spec)*

**Phase 2 — Data layer.** Build `db.py` (connect, `init_db`, the member/range/interaction queries, the idempotent escalation emit via a UNIQUE `dedup_key` + `INSERT OR IGNORE`, and the resolution of active `feedback` overrides into `analysis.py`'s inputs so the core stays pure) and `preprocessing/ingest.py` — the normalization adapter (firewall): flatten each member's `panels → lab_results` (synthesizing `result_id`, mapping `analyte`→`marker`, carrying `panel_id`), fold `vitals` (BP, BMI) in as markers with config-supplied units, parse the **five reference-range shapes** the data prints (`low-high`, `<high`, `>low`, `>=low`, sex-split, and the Vitamin-D multi-band) into `reference_ranges`, **transcribe the curated panic thresholds (and vital bounds) from `config` onto the matching `reference_ranges` rows so the safety floor has values to read**, and store `notes` with their `source`; validate against the models → write SQLite — runnable as the loader CLI from here, exposed as the `POST /members` route in Phase 6. Units are consistent per marker (no conversion) and marker names are already canonical (no alias resolution), so don't build those. Two pure transforms get unit tests here: each of the five range-shape forms parses to the correct bounds (the firewall's contract), and `db.py`'s escalation emit is idempotent — called twice on one `dedup_key`, it writes a single row. Now the core runs over *persisted* data rather than in-memory fixtures, with `db.py` as the only module touching SQLite. The override-resolution seam is built here even though `feedback` is populated much later — it is the single place the deterministic half of self-improvement plugs in, and isolating it now keeps `analysis.py` pure forever. → *Runnable:* ingest the bundle, query a member, run analysis over the DB; re-ingesting the same `member_id` refreshes that member's facts (profile, labs, notes), preserving the audit and learning rows, and bumps `data_version`. **Review — likely to break:** any of the five range-shape forms parsing to wrong/`None` bounds (especially the Vitamin-D multi-band); vitals not folded as markers or missing config units; **the curated panic thresholds not transcribed from `config` onto the range rows (the safety floor then silently never fires)**; `panel_id` dropped on flatten; the `INSERT OR IGNORE` not actually idempotent; `analysis.py` reaching into the DB (purity); re-ingest not bumping `data_version`; override-resolution mutating the core's inputs in place; a query built by string-formatting instead of a parameterized `?` placeholder (the one SQL-injection surface). **Manual happy-path:** ingest the bundle; query a member and run analysis over the DB (result matches the in-memory fixture); re-ingest the same `member_id` (refreshes its facts, preserves audit/learning rows, bumps `data_version`); spot-check one parsed range of each of the five shapes; **confirm a curated panic value reaches `urgent` end-to-end (potassium 6.1 → `urgent`)**. *(§4, §7, §9)*

**Phase 3a — Proactive deterministic spine (Mode 1, part 1; no LLM).** Build `safety.py` (the data floor from the analysis, and the output validator enforcing `escalation ≥ floor`), the safety responders in `templates.py` (seek-care, crisis-support, refusal), and the analysis→`HealthIntelligenceResponse` templating (verdicts → the response contract carrying `evidence[]` + floor + a terse summary). Wire the proactive **scan** — the deterministic half of `pipeline.py`'s scan: `analysis` → `observations` (severity + the clinician-audience `trigger_reason`; the member-facing `member_explanation` is **not** persisted — it is re-derived at read by the `/observations` projection, §below), each persisted with its deterministic `HealthIntelligenceResponse` so it satisfies the `interactions` FK — plus the `data_finding` escalation emit. The scan **replaces** a member's observation set on each run rather than appending, so a re-ingest that bumps `data_version` supersedes the prior version's findings instead of accumulating them. Two complementary mechanisms realize that replace — and they are deliberately **not** the audit row's discipline: observations are written **overwrite-on-conflict** (`ON CONFLICT(observation_id) DO UPDATE`) on the deterministic `data_version`-keyed `observation_id`, so a same-`data_version` re-scan **refreshes the projection in place** (a narration / display-name change self-heals on the next scan instead of stranding stale prose), while the **version-scoped read** (`get_observations` defaults to the current `data_version`) returns only the current set — prior-version rows persist for audit/escalation-reference but are never shown. It is overwrite-in-place, **not** a blanket `DELETE`-then-insert: `escalations.observation_id` is a RESTRICT FK to the observation row, so deleting an escalated marker's observation FK-fails (proven on the K⁺-panic member); updating the same id keeps that reference valid. The backing `interactions` row uses the *opposite* discipline on purpose: each finding's `HealthIntelligenceResponse` is keyed by a **deterministic `response_id`** (`scan:{member}:{marker}:{data_version}`) and written **keep-first `INSERT OR IGNORE`** (an append-only audit row, not a refreshable projection), so an identical re-scan leaves the existing audit row untouched rather than appending a fresh one — without it the observation set replaces cleanly but `interactions` accumulates one row per finding per re-scan. (Consequence worth naming: at a fixed `data_version`, a narration change refreshes the *observation* projection but not the already-written scan *interaction* JSON — correct, since audit rows are not rewritten, and never member-visible since the member panel renders `member_explanation` — re-derived from the live analysis by the `/observations` projection (`pipeline.observations`), so it is always current by construction — not the scan interaction JSON.) (The ask path differs: each `/ask` is a distinct logged event, so `interactions` appends there — the `driver` column separates the two write disciplines.) One carried-over decision lands here, where escalations are first written: the `data_finding` escalation's `dedup_key` keys on `data_version` — the Phase-2 full-record hash — so a note- or profile-only edit (which `analyze()` never reads) would mint a new key and duplicate an escalation whose analysis inputs never moved; resolve it by narrowing `data_version` to analysis-relevant inputs or making the `dedup_key` finding-stable. Stand up the web layer here, where the first routes need it: `api.py` (the FastAPI app, the static-file mount, and the routes `/scan`, `/observations`, `/escalations`, `/health`) and the `Makefile`, so the one-command local run becomes real (the `uv`-managed `pyproject.toml` exists from Phase 0). This is your first runnable vertical slice, and every safety-relevant decision now lives in deterministic, auditable code — which is what makes the LLM safe to add later. `test_pipeline.py` begins here — the automated assertions behind the runnable check below: the floor projects the max per-marker severity, the validator rejects an under-floor output, the scan emits a ranked observation, and the escalation emit is idempotent. → *Runnable:* `make run` serves the app on one port; scanning a supplied member surfaces a real trend as a ranked observation; the stored critical potassium (K⁺ 6.1) forces an `urgent` floor and writes exactly one escalation; a re-scan is idempotent. **Review — likely to break:** the validator not enforcing `escalation ≥ floor`, or anything downstream lowering the floor; the scan not persisting its `HealthIntelligenceResponse` (the `interactions` FK); the escalation emit firing twice for one finding; a re-scan creating duplicate observations, or a re-ingest leaving stale observations from the prior `data_version` beside the new set; the observation write keeping-first (`INSERT OR IGNORE`) instead of overwrite-on-conflict (stranding stale narration at a fixed `data_version`), or a blanket `DELETE`-then-insert FK-failing on an escalation-referenced observation; the scan writing a fresh `interactions` row per re-scan (a non-deterministic `response_id` → audit-table accumulation); a `data_finding` escalation duplicating on a note- or profile-only edit (the deferred `data_version`-scope decision above); a route scoped to the wrong noun. **Manual happy-path:** `make run`; scan a member → a real trend surfaces as a ranked observation; scan the K⁺-6.1 member → `urgent` floor and exactly one escalation; re-scan → no new rows; `GET /observations` and `/escalations` return the right artifacts. *(§2 escalation model + D3 triage, §6)*

**Phase 3b — Reactive Mode 1 answering (no LLM).** Build `suggest_prompts(analysis, observations, notes, focus, asked)` (the data-derived preset prompts, parameterized by conversation state), the reactive answer path (a selected preset answered by templating the same verdicts into a `HealthIntelligenceResponse` — the 3a response builder reused, no model call), and the **Mode 1 loop** (regenerate the next chips after each answer); expose `/suggestions`. This completes **Mode 1**: proactive findings, severity ranking, evidence-backed narration, the escalation floor, the clinician-review queue, *and* deterministic answers to the anticipated questions — a clinician-trustworthy, no-LLM assistant, viable on its own and the calm default surface (§2), and the base the LLM goes on top of as Mode 2 next. → *Runnable:* a preset answers deterministically and byte-identically; selecting it regenerates the next chips — the conversation loop, no dead-ends. **Review — likely to break:** a preset answer not byte-identical across runs; the loop dead-ending (the ever-present anchors missing); a chip resurfacing data not in the member's record; a mechanism question ("*why* is it low?") answered from a template instead of hedged or handed to Mode 2; the reactive path not reusing the 3a response builder. **Manual happy-path:** `GET /suggestions` → preset chips; open one → its pre-computed answer; re-fetch → byte-identical; selecting a chip regenerates the next chips with no dead-end. *(§2 modes + the Mode 1 conversation loop, §7)*

**Phase 4 — LLM layer = Mode 2 (LLM on).** Build `llm.py` (the `compose()` interface and the provider — the only network seam besides the gate), `gate.py` (the single Pydantic-enum classifier routing the raw message to `none | out_of_scope | acute_medical | crisis` and setting a message floor (the classification `max`'d with a deterministic, regression-pinned **emergency-phrase floor**, so a self-harm/emergency phrase floors even a jailbroken or failed gate — §564), at temperature 0 with few-shot on held-out hard-case exemplars; its structured output is validated, with one bounded retry then a fail-closed *couldn't-route* fallback that holds the `clinician_review` floor while inviting a rephrase — §98), the composer's per-turn context assembly and prompt — built so the composer returns a **draft** (prose `answer`, `uncertainty`, `answer_disposition`, and the **names** of the markers it cites) and code attaches the `Evidence` values from the analysis (reusing the scan-finding templater), so chip numbers come only from the deterministic core and a fabricated evidence chip is structurally impossible; the prompt carries few-shot exemplars for the harness-scored behaviors (an absent marker answered "not measured", a calm tone on a benign out-of-range value), and **extend** `pipeline.py` with the ask path (gate → `floor = max(data, message)` → route to open compose or a safety template → validate → escalate); add `/ask` — this is **Mode 2**, toggled against Mode 1. The **proactive scan narration stays deterministic** (the Phase-3a / Mode-1 templates) through Phase 4: the LLM is added only on the *reactive* ask path here. Enrichment of the scan narration is deferred to **Phase 6** — its first consumer is the UI, so nothing renders it before then — and lands as a **read-time display swap** that leaves the *persisted* scan untouched, since enriching the stored prose would break Mode 1's byte-identical and fire-once invariants; it stays gated on a runtime prose guard, as this is the one place LLM prose would sit on the escalation-carrying proactive path. When it lands it **reuses the composer's model/tier** (not a cheaper or different one) — the same faithful-rendering task as the ask path, but on the highest-stakes path, so it is the last place to trade model quality for cost. The LLM lands on a base that already works, as a pure language layer: it renders ground truth and routes open language while the floor, the validator, and the escalation logic from Phase 3a sit *around* it and can override it — the gate being the only point at which a typed message (an emergency with otherwise-normal labs) can raise the floor on its own. The composer is a language layer, not an agent: control flow stays a deterministic `match`, which is what satisfies the agentic criterion without an autonomous tool-loop. Keep the provider behind `llm.py` as the single importer of the SDK (Claude the default; adding Gemini or an in-house HF model is a new provider class plus a config value, with no change to the composer, pipeline, validator, or metadata), and normalize every SDK exception into one `LLMUnavailable` — the auto-degradation path (LLM down → `template_render`) catches *that* normalized signal, so the killswitch and the provider swap are the same one-file seam rather than something that silently breaks when the provider changes. → *Runnable:* free-form answers with evidence grounded by construction (chip values attached by code from the analysis, not the model; prose grounding measured in Phase 5 — the Phase-4 live run surfaced two v1 tuning items for there: the composer can **discuss a marker without listing it in `cited_markers`** (a missing evidence *chip*, not a fabricated number — the value is still read from the verdicts; caught by `score_grounding`, §8), and it **over-answers past the §5 latency/length budget** (the compose-prompt + `max_tokens` lever)); a typed emergency with normal labs still escalates; the validator rejects an under-floor output and one bounded retry repairs it; on re-ask the escalation and each cited marker's value are identical, while the finding *set* and the prose track the model (Mode-2 reproducibility is the floor + the numbers, not the chip set — only Mode 1 is byte-identical, §7). **Review — likely to break:** the LLM doing arithmetic or naming a number absent from the verdicts; temperature not 0; the composer handed raw readings instead of verdicts; the gate failing to raise the floor on a typed emergency with normal labs, or missing crisis/acute (recall); the validator's repair not bounded to one retry or not falling through to a template; the composer emitting evidence-chip *values* rather than just the marker names for code to attach (chip values must come from the analysis, never the model); a provider symbol escaping `llm.py` (the SDK — `anthropic`, later `google.genai` — must import in exactly one file, else "agnostic" is only nominal); `pipeline.py` catching a provider-native exception instead of the normalized `LLMUnavailable`, so the killswitch wouldn't fire after a provider swap; an invalid gate classification not retried once before falling back, or its *couldn't-route* fallback defaulting to `none`/open compose, dropping the floor below `clinician_review`, or replying with a bare rephrase-ask instead of holding the floor; the emergency-phrase floor not taken in parallel with the gate (a jailbroken or mis-classified self-harm message reaching `none` with no floor — the deterministic check must `max` in regardless of what the gate returns). **Manual happy-path:** `POST /ask` a free-form question → grounded answer with evidence chips; ask a typed emergency with normal labs → still escalates; force an under-floor output → validator rejects, one retry repairs, else a safety template; re-ask → identical escalation and per-marker values, with the finding set + prose tracking the model (not byte-identical; only Mode 1 is, §7). *(§2 D1/D4, the per-turn message-shape + injection diagrams, §3)*

**Phase 5 — Evaluation harness.** Build `eval/harness.py` (`run_eval` over a live `ServiceClient`, calling the service **N=3** times per case for Mode 2, **N=1** for byte-identical Mode 1) reading the supplied 17-case set **from its dataset bundle** (`datasets.dataset_dir()/eval_set.jsonl`, scoped by `DATASET` — the cases' `member_id`s are dataset-local, so they stay with the members they reference; `eval/` holds only the added gate/trend cases + judge calibration), fields `id, member_id, category, input, expected_behavior, must_include, must_not, escalation_expected`. The supplied set grades *observable behavior*; its `escalation_expected` is free text (nine phrasings) **normalized** to the `{none, clinician_review, urgent}` enum — ambiguous phrasings to an acceptable set whose floor defines the under-call (§8) — before `score_escalation` compares, `must_include` splits into precise numerics (deterministic substring) and loose semantics (judge), `must_not` into safety violations (deterministic) and tone (judge), and `category` routes per-case scoring. Because the set carries **no structured trend verdict**, `score_stats` grades `analysis.py` against the Phase-1 known-answer fixtures, not the supplied cases — two label sources, kept distinct. Promote those deterministic tests into scorers; add the gated judge scorers (semantic grounding, tone), the never-events logic, the absent-marker grounding check (a request for a marker not in the member's data answers "not measured", never fabricates), and the markdown + JSON report. This precedes the UI deliberately: you earn confidence the system is safe and useful before a member ever sees it, and you produce the real results — including the named failure modes — that stand behind it. Most of the deterministic scoring already exists from Phase 1; the judge scorers, the label normalization, and the report are what's new. → *Runnable:* the full run over the labeled set; the failure-mode cases actually fire (over-escalation controls on the managed / borderline / benign-out-of-range cases, trend-vs-noise by `score_stats`, the absent-marker trap caught, and the two Phase-4 live-run items now *tracked* rather than anecdotal — the composer's discussed-but-uncited marker surfaced by `score_grounding`, its over-budget verbosity by `score_latency_cost`); the JSON becomes the regression-gate artifact. Additively, the harness streams each run to **LangSmith** as an offline trace sink (per-case inputs/output/latency with the scorer verdicts attached) — tracing-only and cuttable, the local report staying canonical. **Review — likely to break:** the harness mocking the service instead of calling it live; `escalation_expected` not normalized from its nine phrasings before compare; under-escalation not a hard fail, or over-escalation failed instead of measured; the judge on the live/safety path rather than offline and secondary; calibration exemplars or the agreement slice overlapping the scored cases (leakage); `score_stats` graded against the supplied cases instead of the Phase-1 fixtures; the absent-marker trap not caught; LangSmith touching the live `/ask` path. **Manual happy-path:** `make eval` → the full run over the 17 cases plus fixtures; the failure-mode cases fire (over-escalation controls on managed/borderline/benign, trend-vs-noise, the absent-marker trap); the markdown + JSON report writes; with `LANGSMITH_API_KEY` set, the run appears in LangSmith. *(§8 + the harness module spec, §6)*

**Phase 6 — Consumer surface.** Build `frontend/index.html` — the thin consumer surface specified by the `ui-ux.md` wireframe — against the live endpoints: the conversation calling `/ask` (Mode 2) and `/suggestions` (Mode 1), the **Mode 1 / Mode 2 toggle** (Mode 1 the default) and the **Mode 1 loop** (track `focus`+`asked` client-side, re-fetching `/suggestions` on each chip selection to regenerate the next chips), the observations panel from `/scan` + `/observations`, and a trace view exposing `data_version`/`prompt_version`. The operator **control panel** (left rail) gives a one-click button per scaffolded route — including `POST /members` (holdout upload/upsert) — the demo driver, not a member feature (`ui-ux.md` §2). (`DELETE /members/{id}` stays an API/`curl` route with no panel button — `ui-ux.md` §2's group enumeration deliberately omits a clear-member affordance.) It is one static page (vanilla JS, no build step) that FastAPI serves on the same origin; doing it after the harness means the surfaced system is one already validated, and the surface stays deliberately simple. → *Runnable:* the two-axis disposition renders (body from `answer_disposition`, chrome from `escalation`), loud-once-then-ambient escalation behaves, evidence expands on demand, and a holdout member ingested through `/members` answers correctly. **Review — likely to break:** the card not splitting body (`answer_disposition`) from chrome (`escalation`); an escalation repeated in every bubble instead of loud-once-then-ambient; evidence not collapsed by default; severity conveyed by color alone (a11y); the member-picker not populating from `GET /members`; the operator control panel bleeding into the member view; an uploaded holdout member not appearing in the picker. **Manual happy-path:** open the page; pick a member; ask in Mode 2 and browse Mode 1 chips; see an observation in the panel; trigger an escalation → loud once, then a calm standing banner; expand evidence on demand; upload a holdout bundle → the picker gains it and answers correctly. *(`ui-ux.md`, §13, §10)*

**Phase 7 — Self-improvement (the proactive layer's learning half).** Two forms, both additive (§11). The **deterministic correction** path is cheap because its seam already exists from Phase 2: a `POST /feedback` correction is resolved by `db.py` before `analysis.py` runs — a clinician **range-override** or **marker suppression** into `analysis.py`'s *inputs* (so re-asking or re-scanning shows the changed flag), while a member **preference** joins the **composer context** via `db.get_active_preferences` and is *never* an analysis input (it shapes Mode-2 tone, not what is flagged) — with `POST /reset` to revert; the crispest safe demo, buildable any time after Phase 2. The **harness-gated prompt promotion** path is the genuine stretch because it depends on Phase 5: `learn.py` behind `POST /learn` runs a *proactive* prompt-scan (the analogue of the drift scan), **assembles** a candidate `prompt_version` **by rule from the accumulated `feedback` set** — base prompt + a few-shot slot + toggleable templated clauses, where an `incorrect` flag appends its case as a new few-shot exemplar and a recurring `reject` signal toggles the matching pre-written clause — a **pure function of the feedback set**, so identical signals yield a byte-identical candidate; the harness then gates it in the same run — promote iff zero never-events, safety metrics at or above current, and no dimension regresses, else `status='rejected'` kept with its report; the composer then loads the latest promoted version (the composer is **version-aware** as shipped: `pipeline` resolves the latest promoted `prompt_versions` row via `db.get_active_prompt` and falls back to `llm.BASE_COMPOSE_SYSTEM` at version 0). **(v1 scope, as shipped: the gate scores the DETERMINISTIC dimensions only** — the eval LLM judge (semantic grounding, tone) is the deferred Phase-5b increment, so a *pure tone* regression that trips no deterministic dimension is not caught here. This is *safety*-complete regardless: the composer's draft carries no escalation field and the **always-on validator floors escalation under whatever prompt is active**, so a promoted prompt can never lower a real escalation — `/learn` is a *quality* gate over an unconditional *safety* floor. A consequence worth stating plainly: the illustrative "warmer-tone clause softens escalation → caught by the recall scorer" demo **cannot fire deterministically** (the escalation/recall scorers read fields the composer can't move); the shipped reject-demo is instead a rule-assembled exemplar that induces a *fabricated value* → caught by the `grounding` never-event. **Confirmed empirically (2026-06-30, real-LLM `/learn` over the labeled set):** the gate certifies **non-regression, not betterment** — none of the deterministic dimensions can *reward* a tone clause or few-shot exemplar that improves the prose (an A/B of the composer on the corrected question shows exactly the improvement the gate cannot see), so it **errs safe toward rejection** by design; and because `/learn` runs at `LEARN_N_RUNS=1` while the main harness samples **N=3** (§7 consistency), its `grounding` *pass-rate* carries the composer's run-to-run nondeterminism the harness otherwise averages out — an identical candidate scored 14/22 twice with *different* cases passing while BASE itself swung 15↔16, so a one-case delta can **false-reject a benign candidate**. Mitigation: run the gate's grounding at the harness's N, or band the **soft** `grounding` dimension only — the `fabricated_value` never-event stays **zero-tolerance**, so the band buys no safety hole. The harness also runs **in-process** — `learn.py` drives `pipeline` + the pure scorers against isolated copy-DBs via `eval/inprocess.py`, NOT the HTTP `ServiceClient`, whose process-global DB-path patch + nested `TestClient` would corrupt a concurrent `/ask` if run inside a live `/learn`.) **No model writes or judges the prompt here**: the candidate is rule-assembled and the gate is deterministic, so the single model call in the loop is the composer generating the candidate's outputs for the scorers — the learning *logic* is reproducible and the LLM is confined to the thing being improved. Because `/learn` is **exposed on the handover surface** and mutates the global active prompt, it is guarded against cost/abuse: a **single-flight lock** (one run at a time), a **signal-set debounce** (the candidate keys on the `feedback`-set hash — unchanged signal short-circuits to the cached result with zero model calls, so naive spam is free), a **structural pre-check** (the assembled candidate must carry the required safety clauses and stay within length — malformed candidates rejected before any eval call), and a **daily run cap** (the backstop against deliberate abuse). Also in this phase, **separate from the learning work**, a read-only **`GET /members/{id}/trajectory`** projection surfaces a member's full per-marker series — `lab_results` + the analysis pass → `{marker, unit, readings[], trend, flags, reference_range}` (optional `?marker=`) — rendered as a **minimal inline per-marker sparkline** (the readings plotted, the already-computed Theil–Sen line drawn through them, the flagged points marked — deliberately rough: dots, a line, marks, no axes/legend/gridlines, consistent with the frontend) and used for **human verification of a finding** when something looks off; the boundary holds, since this is a UI/operator read and the **LLM still consumes only the collapsed `TrajectoryAnalysis` verdict (§4/§207), never the raw series**. Also lands here: the operator **factory reset** `POST /admin/reseed` — the destructive counterpart to `/reset`'s surgical learning-revert — truncates all tables, restores baseline `prompt_version` v0, and re-ingests the `training_data` bundle only, returning the DB to the initial 15-member state (uploaded holdouts dropped); it gains its control-panel button via the same registry pattern (§56). → *Runnable:* an override changes a flag then `/reset` restores it; a candidate whose rule-assembled exemplar induces a *fabricated value* is caught by the `grounding` never-event and rejected, while a benign candidate that regresses no deterministic dimension is promoted and loaded by the version-aware composer; `GET .../trajectory` returns a marker's full series and a **minimal inline sparkline** renders it — readings, the Theil–Sen line, flagged points; `POST /admin/reseed` returns the DB to the initial 15-member state. **Review — likely to break:** an override changing a rule rather than an input (floor, validator, prompt must be untouched); `/reset` not fully reverting (feedback `active=0` *and* prompt → v0); a candidate promoted despite a safety regression (the gate must block on never-events or recall); the active prompt changing mid-session rather than only on promotion; an override not landing in both modes; `/learn` exposed without its guards (the single-flight lock, the `feedback`-set-hash debounce, the structural pre-check, and the daily cap must all be present, or a deployed app spams credits away); the candidate evaluated on a cheaper tier than the production composer (a silent way to make the eval cheap that measures the wrong model); the trajectory route's raw series reaching the LLM context (it must stay a UI/operator read — the model consumes only the §4 verdict); the trajectory **sparkline over-polished** (axes/legend/gridlines instead of staying rough and consistent with the frontend) or its Theil–Sen line / flagged marks not matching the route's verdict; `/admin/reseed` not restoring `prompt_version` v0 after the truncate (the composer left with no prompt) or skipping the **config-transcription** ingest path (the panic floor has nothing to read — the Phase-2 failure mode resurfacing). **Manual happy-path:** `POST /feedback` an override → re-ask/re-scan shows the changed flag; `POST /reset` → the flag is restored; `POST /learn` a candidate whose feedback-assembled exemplar induces an ungrounded value → rejected by the `grounding` never-event with its report; a benign candidate → promoted and loaded by the composer; re-`POST /learn` with unchanged feedback → the cached result, zero model calls; `POST /admin/reseed` → the DB is back to the initial 15 members with prompt v0 and no holdouts, feedback, or interactions; open a member's trajectory → the inline sparkline renders the readings, the trend line, and the flagged points. *(§9 + the learning-loop diagram)*

**Phase 8 — Deployment (additive).** A single Render Web Service deployed from a committed `render.yaml` Blueprint: FastAPI serves the API and the static page same-origin, and because Render's build runs on separate compute that can't see the runtime filesystem, `init_db` **and a seed-if-empty** run on app startup, followed by the ingest-path auto-scan (a fresh instance self-heals its schema, the 15 training members, *and* their observations — reusing the idempotent `ingest_dataset`, the same loader `/admin/reseed` calls, then `pipeline.scan_members`). Shipped on the **free/ephemeral tier** — no persistent disk, and the instance spins down when idle, so the SQLite file is rebuilt and re-seeded on each cold start; `POST /members` uploads and `feedback` therefore don't survive a spin-down (a documented demo trade-off, not a failure). Durability is a one-line `render.yaml` upgrade: the paid `starter` plan with a persistent disk at `/data` and `HEALTH_DB_PATH=/data/health.db`. The first thing to cut under time pressure; the local one-command run stays primary. **Review — likely to break:** `init_db`/seed not running idempotently on startup (Render's build can't see the disk); the DB-path env var the code reads (`HEALTH_DB_PATH`) diverging from what's set on the instance, so the DB lands off-target; the static page not served same-origin; `ANTHROPIC_API_KEY` unset (Mode 2 silently degrades to Mode 1) or — worse — committed to `render.yaml` instead of `sync: false`. **Manual happy-path:** deploy; hit `/health`; the page loads with the 15 seeded members and answers; ingest a holdout → on a paid disk a redeploy preserves it, on the free tier a cold start re-seeds to the 15 (the documented ephemeral behavior). *(§15, §11)*

**README** *(standing deliverable, not a numbered phase).* Written alongside the build and finalized at submission — the one-command run, the AI-tools note, and the EFLM CVa/CVi sourcing. It documents the work rather than being a build step, so it sits outside the phase sequence and, unlike Phase 8 above, is never cut.

---

## 1. Service shape

A single ingestion step loads the supplied member bundles (15 members — profile, 3–5 panels of labs + vitals over ~2 years, notes with provenance) into a SQLite source of truth. One shared core (SQL retrieval + statistical analysis) backs the two behaviors the service supports, plus the eval harness.

```mermaid
flowchart LR
  B[member bundle<br/>JSON] -->|ingest: validate + write| DB[(SQLite<br/>source of truth)]
  DB --> CORE[deterministic core<br/>SQL retrieve + statistical analysis]:::det
  CORE --> SUG[GET /suggestions · Mode 1<br/>deterministic preset answers]:::det
  CORE --> ASK[POST /ask · Mode 2<br/>LLM language layer]:::gen
  CORE --> SCAN[POST /scan<br/>proactive observations]:::gen
  CORE --> EVAL[eval harness · offline<br/>same pipeline functions]:::io
  SUG --> UI[thin surface]
  ASK --> UI
  SCAN --> UI
  classDef det fill:#a5d8ff,stroke:#2563eb,color:#1e1e1e
  classDef gen fill:#d0bfff,stroke:#7c3aed,color:#1e1e1e
  classDef io fill:#ffd8a8,stroke:#d97706,color:#1e1e1e
```

Both behaviors call the same retrieval + analysis core; Mode 2 renders the result with an LLM language layer while Mode 1 uses deterministic templates. The eval harness imports the same pipeline functions and runs them headlessly over a labeled set — possible *because* they are plain typed Python, not a framework-buried graph.

---

## 2. Key design decisions

**D1. Deterministic analytical core; LLM as language layer.** *Alternatives:* let the LLM read raw labs and reason numerically end-to-end; a ReAct loop that decides what to compute and whether to escalate. *Why:* LLMs are unreliable at numeric trend/significance and must not own safety-relevant control flow. Numbers, flags, and escalation live in deterministic code, so the system is auditable, reproducible, and safe near a clinician. The LLM owns only language and bounded interpretation.

**D2. SQL retrieval for member data, not RAG/embeddings.** *Alternatives:* embed panels + notes and retrieve semantically (the reflexive "health data → RAG" move). *Why:* the per-member corpus is tiny and fits in context, so there is no relevance-search problem to solve; numeric precision is non-negotiable and embeddings lose it; reproducibility is hurt by ANN search + embedding drift. Reference ranges are a deterministic lookup table, never model knowledge. *Seam:* RAG earns its place only over a large external clinical corpus (guidelines, biological-variation literature) — out of scope here.

**D3. Statistical tier owns drift-vs-noise.** *Alternatives:* ask the LLM "is this trending?"; a naive last-vs-previous delta with a hardcoded threshold. *Why:* this is exactly what classical statistics do well and LLMs do badly. Output is a typed `TrajectoryAnalysis` the LLM consumes as ground truth and may not recompute. The signals and the triage policy are detailed below.

**D4. Deterministic orchestration with a hard safety override — not a ReAct loop, not a graph framework.** *Alternatives:* a tool-calling agent that decides tool order and *elects* when to escalate; a LangGraph/DAG runtime to host it. *Why:* escalation and refusal must not depend on model latitude and must be reproducible, and the control flow is a fixed, shallow sequence with one branch — `match` on a gate's intent — so a graph *runtime* would host a graph that is really just code. A **pre-compose input gate** — a single Pydantic-enum LLM classification (small model, temp 0, few-shot on held-out exemplars of the hard-case categories — not the scored cases, so its routing recall isn't inflated) — routes the raw message (`none | out_of_scope | acute_medical | crisis`) and sets a *message floor*; the deterministic **data floor** comes from the analysis; the turn runs against `floor = max(data, message)`. The *message floor* is itself the `max` of two parallel signals — the gate's classification **and** a deterministic, regression-pinned **emergency-phrase floor** read straight from the raw message — so a self-harm or acute-emergency phrase forces the floor even if the LLM gate is jailbroken into `none` or fails: the model can only ever *raise* the floor, and a deterministic check it cannot influence raises it in parallel. (The broader injection/jailbreak denylist and a learned classifier stay deferred, §11; this is the one non-overridable floor on the routing where a miss is unacceptable.) The gate's structured output is validated against that enum, and validation fails *closed* in two tiers: an off-enum or unparseable result triggers **one bounded retry** at temp 0 (like the validator's repair, so a transient structured-output glitch resolves silently, with no escalation and no interruption); if it is still invalid, the **floor holds at `clinician_review`** while the responder becomes a *couldn't-route* template blending both possibilities — *seek care if these are symptoms you're worried about; otherwise, rephrase* — so clarification lives in the copy without lowering the floor. It never defaults to `none` (free-composing a possibly-urgent message), nor answers with a bare rephrase-ask and no floor: the glitched message is exactly the one the gate exists to catch, so clarification alone would be the fail-open. (Clarification for a genuinely *ambiguous* message is different — that is a successful `none` classification, and the composer asks within the normal answer.) The gate *routes* the turn to a responder — open LLM compose for `none`, fixed templates for the safety branches (you do not free-compose an emergency or a crisis reply grounded in someone's labs) — and a **validator** rejects any output whose `escalation` axis sits below the floor (deterministic wins). Escalation is therefore a *consequence of the floor, enforced around the model* — never a tool the model calls. The response carries two orthogonal axes: `answer_disposition {answered, refused, out_of_scope}` (the model's, ungated) and `escalation {none, clinician_review, urgent}` (deterministic), so "answered **and** flagged for your GP" — and "out-of-scope question **and** the data is alarming" — are both representable, which a single enum can't do. A chat surface re-runs this per turn; the safety floor re-asserts every turn. The graph-framework seam is real but downstream: it earns its place only if the composer becomes a genuine agent (runtime-chosen, cyclic tool use), which we deliberately don't build.

```mermaid
flowchart TD
  Q([question / scan trigger]):::io --> R[SQL retrieve<br/>whole member record]:::det
  R --> A[Trajectory analysis<br/>MK+τ · Theil–Sen+CI · RCV · range/panic · band · FDR]:::stat
  A --> DF[data floor<br/>none · clinician_review · urgent]:::det
  Q --> G[Input gate · classify message<br/>none · out_of_scope · acute_medical · crisis]:::gen
  G --> FL[floor = max data, message]:::det
  DF --> FL
  FL --> RT{Responder<br/>by gate intent}:::det
  RT -- none --> C[Compose · LLM<br/>Sonnet 5 · thinking off · drafts prose + cited markers<br/>code attaches evidence]:::gen
  RT -- acute_medical / crisis / out_of_scope --> T[Template responder<br/>seek-care · crisis-support · refusal]:::det
  C --> V{Validator<br/>schema · grounding · escalation ≥ floor}:::det
  T --> V
  V -- fail --> RP[repair / one bounded retry]:::det
  RP --> C
  V -- pass --> P[persist response · observations on scan<br/>+ escalation if floor ≥ clinician_review<br/>INSERT OR IGNORE → fire once]:::det
  P --> O([structured response]):::io
  classDef det fill:#a5d8ff,stroke:#2563eb,color:#1e1e1e
  classDef stat fill:#b2f2bb,stroke:#15803d,color:#1e1e1e
  classDef gen fill:#d0bfff,stroke:#7c3aed,color:#1e1e1e
  classDef io fill:#ffd8a8,stroke:#d97706,color:#1e1e1e
```

The turn, in code — a fork on the gate's intent, not a graph that needs a runtime:

```python
def turn(member_id, message):
    analysis   = analyze(member_id)              # deterministic stats (already exists)
    data_floor = analysis.overall_floor          # none | clinician_review | urgent
    intent, msg_floor = gate(message)            # LLM, Pydantic route + msg_floor (temp 0, few-shot on held-out exemplars)
    floor = max(data_floor, msg_floor)

    match intent:                                # a branch, not a DAG — no framework
        case "none":          resp = compose(context(analysis), message)  # only open generation
        case "out_of_scope":  resp = refuse_template()
        case "acute_medical": resp = seek_care_template()
        case "crisis":        resp = crisis_template()

    resp = validate(resp, floor)                 # deterministic; raises escalation, never lowers
    persist(resp)
    if floor >= CLINICIAN_REVIEW:                # a consequence of the floor, never a tool the model elects
        emit_escalation(kind="chat", dedup_key=f"chat:{member_id}:{date.today()}")  # INSERT OR IGNORE — one/day
    return resp
```

**Two modes — deterministic (Mode 1, default) and LLM (Mode 2).** The assistant doesn't have to be an LLM to be useful, so rather than blend the two paths, the surface exposes a system-wide **LLM off/on toggle**, which makes the architectural choice legible and lets the harness measure each mode independently (§8). **Mode 1 (LLM off)** is the deterministic system (Phases 3a–3b) made directly usable: the proactive observations (deterministic narration) plus 2–5 data-derived preset prompts — generated by `suggest_prompts(analysis, observations, notes)` from the same `TrajectoryAnalysis` and notes ("what's changed since last time?", "should I worry about my ferritin?", "why does my doctor want to follow up?") — each answered by templating the verdicts into a `HealthIntelligenceResponse` carrying the same `evidence[]`, `uncertainty`, and deterministic floor, logged as `driver='suggested'`; no model call, so answers are instant, ~free, and byte-identical (§5/§7). **Mode 2 (LLM on)** is free-form chat. Both modes run over **one core and one floor** — the safety logic never depends on the mode. Structurally they are **one pipeline that swaps exactly one step**: `retrieve → analyze → floor → render → validate → escalate` is shared, and only `render` differs — `template_render(analysis)` (Mode 1) vs `llm_compose(analysis, prompt)` (Mode 2) — both emitting the same `HealthIntelligenceResponse` under the same validator. The **gate is a conditional that fires only on free-form input**, so Mode 1 skips it and runs `floor = data_floor` alone while Mode 2 runs `floor = max(data, message)`; the always-on data floor is what keeps Mode 1 safe without screening any message. So Mode 1 is a complete, clinician-trustworthy assistant on its own and is the calm default (curated, no blank page, no model latency), with Mode 2 the "ask in your own words" path. Trade-off: Mode 1 covers only anticipated answer shapes (trend / range verdict / note-follow-up) and gracefully defers the long tail to Mode 2; a production surface would likely blend them, but the split shows and measures both.

### Mode 1 conversation loop

Mode 1 is not single-shot: the presets form a navigable loop, so it holds a conversation of **unbounded length over a finite, complete answer space**. The data is finite, so the distinct answers are too — Mode 1 is a complete *map* of what the member's record supports, and the conversation is a walk over it. `suggest_prompts` gains optional conversation state — `suggest_prompts(analysis, observations, notes, focus, asked)` — and after every answer regenerates the next chips: **drill-downs of the finding just opened** (its readings, year-over-year delta, the linked GP note), a few **unexplored findings** (pivots), and one or two ever-present **anchors** ("what's changed since last time?", "overview"). Click a chip → see its pre-computed `HealthIntelligenceResponse` → the next chips appear.

The crucial property: **Mode 1 converses by navigation, not comprehension.** The member never types a follow-up; they click a chip the *system authored*, already bound to a finding with a pre-computed answer — so the hard part of follow-ups (interpreting free text) never occurs, which is precisely why no LLM is needed. State is just `focus` (current finding) + `asked` (visited chips), held client-side for a sitting or derived from the member's recent `driver='suggested'` interactions; nothing new in the schema, since every click is already a logged interaction — which is also what lets feedback attach to any answer in the thread.

Two boundaries keep it honest. **Grounding:** chips only resurface the member's *own* data; *mechanism* questions ("*why* is my ferritin low?") aren't answered from templates — they hedge to action ("worth your GP investigating") or hand to Mode 2, and a true explainer would live in a separate clinically-reviewed, versioned library (a named static-content seam, §11), never smuggled into the templating. **Termination:** the anchors are always present, so the loop never dead-ends; when the map is walked the chips thin to revisits, and *that* is the cue to surface "don't see your question? ask in your own words" — the clean handoff to Mode 2. Revisiting a node returns the byte-identical answer (§7), so length never costs consistency.

```mermaid
flowchart LR
  S([open · GET /suggestions]):::io --> CH[chips · drill-downs + unexplored + anchors<br/>suggest_prompts · analysis · focus · asked]:::det
  CH -->|member clicks| ANS[pre-computed HealthIntelligenceResponse<br/>evidence · uncertainty · floor · logged driver=suggested]:::det
  ANS -->|update focus + asked| CH
  ANS -.->|map walked / off-map question| M2([switch to Mode 2 · ask in your own words]):::gen
  classDef det fill:#a5d8ff,stroke:#2563eb,color:#1e1e1e
  classDef gen fill:#d0bfff,stroke:#7c3aed,color:#1e1e1e
  classDef io fill:#ffd8a8,stroke:#d97706,color:#1e1e1e
```

### Per-turn message shapes (the chat loop)

Each turn sends **two messages** to the model — a pinned system message and a freshly assembled user message — and returns one structured assistant message. Everything stable lives in a cacheable prefix; only the tail changes per turn.

The loop, and what is injected into the model on each pass:

```mermaid
flowchart TB
  IN([member message · this turn]):::io --> GATE[LLM gate<br/>raw message → route + msg_floor<br/>temp 0 · few-shot · Pydantic enum]:::gen
  GATE -->|route none → compose| ASST[assistant LLM · compose<br/>Sonnet 5 · thinking off · drafts prose + cited markers<br/>code attaches evidence]:::gen

  SYS[system message · PINNED — identical every turn<br/>role · grounding + citation rules · output contract<br/>never recompute a number · never lower the floor]:::frame --> ASST

  subgraph UMSG ["user message · assembled every turn"]
    direction TB
    PFX[stable prefix · cacheable — constant all session<br/>profile · trajectory_analysis · notes<br/>safety floor · active_observations]:::ctx
    TAIL[fresh tail · changes this turn<br/>conversation_history · question]:::fresh
  end
  UMSG --> ASST

  ASST --> OUT([structured response · validated<br/>escalation ≥ floor]):::io
  OUT -.->|append question + answer| TAIL
  CORE[deterministic core · runs first<br/>SQL record + analysis.py]:::det -.->|fills as ground truth| PFX

  classDef det fill:#a5d8ff,stroke:#2563eb,color:#1e1e1e
  classDef gen fill:#d0bfff,stroke:#7c3aed,color:#1e1e1e
  classDef io fill:#ffd8a8,stroke:#d97706,color:#1e1e1e
  classDef ctx fill:#c3fae8,stroke:#0d9488,color:#1e1e1e
  classDef fresh fill:#ffe066,stroke:#e8950c,color:#1e1e1e
  classDef frame fill:#e9ecef,stroke:#868e96,color:#1e1e1e
```

This view is the `none` route — the assistant LLM composes only after the gate clears the message; `out_of_scope`, `acute_medical`, and `crisis` route to fixed templates (the §2 flowchart) and never reach composition, so the full grounded context is injected only where text is freely generated. The stable prefix is the deterministic core's output, handed to the model as ground truth it may not recompute.

**System message** (pinned, identical every turn): role, grounding/citation rules, the output contract, and the standing instruction to never recompute a number or lower the safety floor. No member data, no numbers.

**User message** — the per-turn *context*: a stable prefix + a fresh tail.

```
# stable within a session (prefix — cacheable)
<profile>             age · sex · conditions · medications · family_history · lifestyle
<trajectory_analysis> computed verdicts, NOT raw rows (shape below)
<notes>               free-text notes, dated
<safety floor=...>    deterministic escalation floor: none | clinician_review | urgent
<active_observations> compact [{id, severity, title}, ...]  — not full narration
# fresh each turn (tail)
<conversation_history> last ≤N (question, answer) pairs — this member's ongoing thread
<question>            the member's new message
```

**Trajectory shape** — `TrajectoryAnalysis`, the deterministic block the model consumes as ground truth and may not recompute:

```
TrajectoryAnalysis {
  member_id, data_version, overall_floor: "none|clinician_review|urgent",
  markers: [ {
    marker, unit, latest: { value, date },
    trend: { method:"mann_kendall", direction, p_value, slope, n },
    flags: [ "below_range" | "panic_low" | "band_cross" | ... ],
    severity: "info|notable|attention|urgent"
  } ]
}
```

**Assistant message shape** — `HealthIntelligenceResponse`, validated before it reaches the member:

```
HealthIntelligenceResponse {
  answer: "plain-language response",
  findings: [ { finding_id, text,
               evidence: [ { marker, value, unit, date, ref_low, ref_high, stat } ] } ],
  uncertainty: "what it rests on + what would change it",
  answer_disposition: "answered|refused|out_of_scope",   # the model's call on the question
  escalation: "none|clinician_review|urgent",            # deterministic; validator enforces >= floor
  metadata: { response_id, data_version, model_version, config_version, prompt_version, latency_ms, tokens, cost_usd }
}
```

Three invariants hold across turns: `<safety>` and `<active_observations>` are projections of the *same* analysis pass, so they can't contradict; the model **reads** severity and floor as facts and never sets them; and the validator rejects any assistant message whose `escalation` is below the floor (`answer_disposition` is the model's and ungated). History grows only by appending the prior `(question, answer)` — nothing else in the prefix changes within a session, so re-injection is near-free under prompt caching.

### What each LLM call site receives

Three call sites, three deliberately different context shapes — and the contrast *is* the architecture. The gate sees almost nothing, Mode 1 calls no model at all, and only Mode 2's compose step receives the full grounded record. (The gate fires only on free-form input, so Mode 1 skips it entirely.)

**The gate — the raw message, and nothing else.**

```mermaid
flowchart LR
  GSYS["system · pinned<br/>classify intent into one enum<br/>few-shot · held-out exemplars"]:::frame
  GUSER["user · the raw message ONLY<br/>e.g. should I worry about my ferritin<br/>no profile · no labs · no notes · no analysis"]:::fresh
  GSYS --> GC["gate · 1 call<br/>small model · temp 0 · Pydantic enum"]:::gen
  GUSER --> GC
  GC --> GR["route + msg_floor<br/>none · out_of_scope · acute_medical · crisis"]:::det
  classDef frame fill:#e9ecef,stroke:#868e96,color:#1e1e1e
  classDef fresh fill:#ffe066,stroke:#e8950c,color:#1e1e1e
  classDef gen fill:#d0bfff,stroke:#7c3aed,color:#1e1e1e
  classDef det fill:#a5d8ff,stroke:#2563eb,color:#1e1e1e
```

The gate classifies *intent*, which needs only the words the member typed. It never receives the profile, labs, notes, or analysis — keeping it cheap, keeping clinical data out of a second model call, and stopping the thing being classified from being diluted by surrounding context.

**Mode 1 — no model call; verdicts go straight to a template.**

```mermaid
flowchart LR
  M1A["TrajectoryAnalysis · verdicts<br/>the deterministic block"]:::stat
  M1P["the opened preset<br/>which finding the chip bound to"]:::ctx
  M1A --> M1T["template_render · pure code<br/>fills verdicts into the contract"]:::det
  M1P --> M1T
  M1T --> M1R["HealthIntelligenceResponse<br/>byte-identical · ~0 cost · no prompt exists"]:::io
  classDef stat fill:#b2f2bb,stroke:#15803d,color:#1e1e1e
  classDef ctx fill:#c3fae8,stroke:#0d9488,color:#1e1e1e
  classDef det fill:#a5d8ff,stroke:#2563eb,color:#1e1e1e
  classDef io fill:#ffd8a8,stroke:#d97706,color:#1e1e1e
```

There is no prompt to show. The deterministic verdicts are filled into the response contract by `template_render()`; the "context" is the `TrajectoryAnalysis` and the preset the member opened, consumed by code and never sent to a model. This is why a Mode 1 answer is byte-identical across runs and effectively free.

**Mode 2 — the full grounded context, in two messages.**

```mermaid
flowchart LR
  SYS["MESSAGE 1 · system · PINNED<br/>translate the interpretation into language<br/>- no number absent from trajectory_analysis<br/>- cite every claim with an evidence_ref<br/>- never lower the escalation floor; refuse out-of-scope<br/>- output ONLY a valid HealthIntelligenceResponse"]:::frame
  subgraph U2 ["MESSAGE 2 · user · assembled fresh each turn"]
    direction TB
    PFX["cacheable prefix · constant in session<br/>profile · trajectory_analysis (verdicts, not rows)<br/>notes (dated) · safety floor · active_observations"]:::ctx
    TAIL["fresh tail · this turn<br/>conversation_history (last N pairs) · question"]:::fresh
  end
  SYS --> CC["compose · 1 call<br/>Sonnet 5 · thinking off · structured output"]:::gen
  U2 --> CC
  CC --> CV["validator<br/>schema · grounding · escalation &gt;= floor"]:::det
  CV --> COUT["HealthIntelligenceResponse<br/>answer · findings[evidence] · uncertainty<br/>answer_disposition + escalation · two axes<br/>metadata · response_id · version tuple · latency/tokens/cost"]:::io
  classDef frame fill:#e9ecef,stroke:#868e96,color:#1e1e1e
  classDef ctx fill:#c3fae8,stroke:#0d9488,color:#1e1e1e
  classDef fresh fill:#ffe066,stroke:#e8950c,color:#1e1e1e
  classDef gen fill:#d0bfff,stroke:#7c3aed,color:#1e1e1e
  classDef det fill:#a5d8ff,stroke:#2563eb,color:#1e1e1e
```

A pinned system message carries the immutable rules; the user message is assembled fresh each turn from the cacheable data prefix plus the per-turn tail. The two output axes — `answer_disposition` (the model's call) and `escalation` (deterministic, validator-enforced ≥ floor) — and the version-tuple metadata are exactly what the validator and the feedback loop key on.

### Escalation model (both behaviors)

Both behaviors must escalate or hand off on alarming values and out-of-scope or concerning input. Making that consistent starts by separating two things that look alike:

- **The floor is a *constraint*** — a standing property (a panic value is still a panic value on turn 50). It governs the validator *every turn*: the prose can never drift below the computed `escalation` level.
- **An escalation is an *event*** — the artifact write + the loud member-facing takeover. It is a *transition* (not-escalated → escalated) and fires **once**, never re-firing just because the floor is still high. Conflating the two is what would re-alarm the member on every message.

**Two escalation types, two sole owners, two keys — idempotent by construction:**

| Type | Owner (sole creator) | Dedup key (UNIQUE) | Notes |
|---|---|---|---|
| `data_finding` | proactive scan (it discovers the discrepancy) | `member · marker · data_version` | assistant *reads*, never creates |
| `chat` | assistant gate | `member · day` | one per member per day — no conversation concept in v1 (a sitting is ~same-day) |

`escalations.dedup_key` is UNIQUE, so "fire once" is a **database guarantee** (`INSERT OR IGNORE`), not a check-then-insert that can race. The assistant never creates `data_finding` escalations because of one invariant — **the scan runs on ingest, before chat** — so findings already exist when the assistant runs. Even if that broke, the validator still keeps the member-facing prose safe, leaving only a clinician-routing gap (closable with a fail-safe `create-if-missing`, deliberately not shipped). **Member-facing care is never deduped** — every concerning message gets a full supportive reply; only the duplicate clinician record is suppressed. The two kinds share the one UNIQUE column, namespaced by kind (`chat:{member}:{day}` vs `data:{member}:{marker}:{version}`), so their keys can't collide. The chat key is day-scoped rather than per-conversation because the prototype has no session boundary — nothing principled would trigger a new conversation, so there is no per-session identifier to key on; a real session model (login, explicit new-chat) is the seam where a session key would replace the day scope.

**Acute and crisis share `kind='chat'` at this stage.** A typed medical emergency and a crisis/self-harm message both write `kind='chat'`, `level='urgent'`, differing only in free-text `trigger_reason`. Splitting them — crisis → safeguarding route, acute → clinical-review queue — is a structured sub-type named as a near-term refinement, deliberately not built here; v1 emits one undifferentiated chat escalation and lets the queue reader disambiguate from `trigger_reason`.

**Loud once, then ambient.** The idempotent create returns created-vs-existing: the *transition* turn gets the loud takeover (redirect / "flagged for your GP"); every turn after is governed by the validator (honest, won't over-reassure) but shows the escalation **ambiently** — never re-injected into each message bubble (UX rendering in `ui-ux.md`).

```mermaid
flowchart TB
  scan[Proactive scan<br/>owns data_finding<br/>key: member·marker·data_version]:::det
  gate[Assistant gate<br/>owns chat<br/>key: member·day]:::det
  scan -->|INSERT OR IGNORE| Q[(escalations<br/>UNIQUE dedup_key<br/>fire-once = DB guarantee)]:::store
  gate -->|INSERT OR IGNORE| Q
  Q --> panel[Side dropdown<br/>standing severity items — DETAIL]:::ui
  Q --> banner[Heading flag<br/>'flagged for your care team' — AMBIENT]:::ui
  Q -. transition turn only .-> loud[Loud takeover<br/>in that one bubble]:::ui
  classDef det fill:#a5d8ff,stroke:#2563eb,color:#1e1e1e
  classDef store fill:#d0bfff,stroke:#7c3aed,color:#1e1e1e
  classDef ui fill:#ffd8a8,stroke:#d97706,color:#1e1e1e
```

### D3 in detail — the signals + triage

A *signal* computes a verdict; a *triage policy* maps verdicts to a severity and a "raise an observation?" decision, with thresholds in `config.py`. Each signal owns one question:

1. **Mann–Kendall** — *is it drifting at all?* Non-parametric monotonic-trend test; rank-based, so no normality assumption and outlier-tolerant. Reports Kendall's **τ ∈ [−1,1]** (scale-free trend strength) with an **exact small-sample p-value** — the normal approximation to the S-statistic is miscalibrated below ~10 points, which is every marker here. Trigger: p < α *after the FDR pass below*.
2. **Theil–Sen slope** — *which way, how fast, how sure?* Median of pairwise slopes → a robust rate, irregular spacing handled natively, over OLS because one bad draw can't swing it — reported **with its distribution-free confidence interval**. A direction is asserted only when the CI excludes zero, so "too noisy to sign" is a first-class verdict, not a forced guess.
3. **Reference Change Value** — *is the change bigger than the marker's own noise?* RCV = √2·Z·√(CVa² + CVi²) from **published** analytical + within-subject biological variation; a net change that doesn't clear RCV is noise however clean the ranking looks. This is the clinical trend-vs-noise test a p-value can't give — a 20% ferritin move and a 20% HbA1c move are worlds apart, and only RCV knows it.
4. **Reference-range / panic flags** — *is the value itself abnormal?* Lookup vs the sex range; a panic value forces escalation (the safety floor), deterministic because ranges are reference *data* and panic must not depend on the model.
5. **Band-crossing** — *did it cross a clinical cutpoint* (e.g. A1c 5.7 / 6.5)? A discrete, highly explainable, narratable event.

**Triage.** Trend signals need ≥ `n_min` points or no trend is claimed (range/panic/band work at n = 1). Because a panel tests many markers, multiplicity matters in principle — but at the data's n ≤ 5 the *exact* Mann–Kendall test is itself strict enough to hold the false-flag rate well below one per scan unaided (a near-monotonic 5-point series reaches only p ≈ 0.0167 two-sided; at n ≤ 4 even a perfect series sits above α, so nothing clears). The **Benjamini–Hochberg FDR** pass is the formal multiplicity backstop, but as wired — over the post-α candidates with `q` (0.10) > α (0.05) — it is inert by construction at any n (the rank-*m* candidate's threshold, `q`, always exceeds the α bar it already cleared); it becomes load-bearing across the *full* marker family with `q` ≤ α, the right setup for the longer uploaded series where p-values crowd α, and forcing it at n ≤ 5 would only prune real isolated trends. Severity is **direction-aware**: each marker's config names which way is adverse (ferritin ↓ vs HbA1c ↑), so a trend toward the healthy side never raises it. A deterministic rule then maps the verdict set → one `severity {info, notable, attention, urgent}` and whether to raise an observation, counting a trend only if it cleared *both* RCV and FDR; a panic flag always pins to urgent. **Sub-`n_min` exception** (the one place a trend counts without FDR): where `3 ≤ n < n_min` makes the exact MK test structurally incapable of significance (its two-sided p floors at 0.33 at n = 3, so FDR can never fire), a strictly-monotonic adverse change that clears RCV escalates to `clinician_review` (never `urgent`) — monotonicity is the deterministic stand-in for the consistency the significance test would otherwise certify, and RCV still gates "beyond noise" (absent CVa/CVi → unconfirmable → stays `notable`). This closes the E12 sparse-decline case without loosening the n ≥ `n_min` path. Crucially, a value merely out of reference range is **not** itself an escalation: an isolated abnormal-but-non-panic reading with no RCV+FDR-significant adverse trend is at most `notable` — surfaced as an observation and narrated in context, never a clinician escalation. Escalation is reserved for panic thresholds and significant adverse trajectories; this is what keeps the scan calm on a managed condition's expected-high marker or a benign out-of-range value (member conditions further shape *narration*, and a clinician suppression override can quiet an expected-abnormal marker, but the floor itself keys only on panic and adverse trend). Thresholds live under `config_version` — normal ranges and panic thresholds in `reference_ranges` (normal parsed from the data, panic curated in config); CVa/CVi, adverse direction, α, FDR `q`, `n_min`, band cutpoints, the Vitamin-D bands, and vital bounds in `config.py`, never `.env`.

*Still deferred (documented near-term): velocity-vs-clinical-rate, and a MAD outlier guard that surfaces an anomalous reading to the clinician rather than absorbing it — both cheap, neither needed to make detection or quantification solid.*

### analysis.py — the deterministic core (module spec)

The module that makes D1 real: every number, trend verdict, flag, and the escalation floor is computed here in pure Python, so the LLM never reasons numerically. It is the one module whose correctness is fully unit-testable in isolation — give it a member's readings, assert the verdicts.

**Contract — one public entry point, pure (no DB, no LLM, no network, no clock):**

```python
def analyze(member: MemberProfile,
            results: list[LabResult],      # already retrieved by db.py
            ranges:  list[ReferenceRange], # analyze selects the sex band
            age:     int,                  # resolved by caller → analyze stays clock-free
            cfg:     AnalysisConfig) -> TrajectoryAnalysis
```

`db.py` does the SQL and hands `analyze` typed inputs; it returns the §4 `TrajectoryAnalysis` and touches nothing else. Determinism is structural — identical inputs give identical output, no randomness, no I/O, no hidden state — which is why this is the seam where statistical correctness is verified.

**Flow — per marker, then a cross-marker pass, then assemble.** For each marker in `results`:

1. `_series(results, marker)` → readings sorted by `panel_date` → `[(date, value)]`.
2. `_range_for(marker, member.sex, age, ranges)` → the applicable band: the `(marker, member.sex)` row, falling back to the sex-agnostic `(marker, 'any')` row when there is no sex-specific one — so non-sex-split markers and any `other`/`unknown` member both resolve to `any`; `None` only when the marker has no band at all (→ the `no_reference` note below).
3. `_trend(series, cfg)` → `TrendResult` = Mann–Kendall (drift? exact small-n `p`, Kendall's `τ`) + Theil–Sen (direction, `slope`, and its CI — direction asserted when the CI excludes zero, or — once the cross-marker FDR pass certifies the trend `significant` — by Mann–Kendall's sign, which un-flattens a trend a tie pinned the CI to zero on, `slope` left unasserted; see `_resolve_direction`); returns `None` when `len(series) < cfg.n_min` — no trend claimed rather than a noisy one. *(stats: D3)*
4. `_clinical_change(series, marker, cfg)` → does the net change clear the marker's **RCV** (from its `CVa`/`CVi` in `cfg`)? Marks a detected trend as *real* vs within-noise — the clinical trend-vs-noise gate. *(D3)*
5. `_flags(series[-1], range)` → on the latest value: `below_range`/`above_range`, `panic_low`/`panic_high`, `band_cross` vs cutpoints. Works at n = 1.

Then a **cross-marker pass** — `_fdr(p-values, cfg.q)` (Benjamini–Hochberg), the multiplicity backstop: as wired (post-α candidates, `q` > α) it is inert at this data's scale, where the exact test's small-*n* strictness is what holds the false-flag rate, and it bites only on the longer uploaded series — and finally, per marker:

6. `_severity(trend, clinical_change, flags, marker_cfg)` → one of `{info, notable, attention, urgent}`, using the marker's **adverse direction** so a beneficial trend never raises it, and counting a trend only if it cleared both RCV and FDR; a panic flag pins to `urgent`. *(triage rule: D3)*

Then assemble each `MarkerTrajectory{marker, unit, latest, trend, clinical_change, flags, severity}` and project the single floor.

**The floor — what the whole escalation model rests on.** `_floor(severities)` is a pure projection of the max per-marker severity onto the escalation axis:

- any marker `urgent` → `urgent`
- else any marker `attention` → `clinician_review`
- else → `none`

This `overall_floor` is exactly what the safety validator enforces every turn and what the proactive scan keys `data_finding` escalations on. Nothing downstream can lower it; the model can only read it.

**Edge cases, explicit:** `n < n_min` → trend (the MK/Theil–Sen `TrendResult`) omitted, flags still computed — **but** a `3 ≤ n < n_min` series that is adverse, strictly monotonic, and clears RCV escalates via the sub-`n_min` rule (`analysis._sparse_adverse`): below `n_min` the exact MK p floors at 0.33 so FDR is unreachable, and monotonicity supplies the consistency the significance test would have (→ `clinician_review`, never `urgent`; absent `CVa`/`CVi` it can't confirm beyond-noise, so it stays `notable`); no range for a marker → range/panic/band skipped with a typed `no_reference` note (never a silent pass); no `CVa`/`CVi` for a marker → RCV skipped, the trend still **surfaced** on MK + CI for narration but **not counted toward the escalation floor** (per §351's RCV+FDR rule), so it caps at `notable` (noted, not silent); a single reading → flags evaluate, trend does not; unit mismatch can't occur — conversion happened at ingest.

**Config & versioning.** Policy params (`α`, FDR `q`, `n_min`, band cutpoints) and the per-marker `CVa`/`CVi` + adverse direction come from `cfg` (committed `config.py` under `config_version`); clinical ranges arrive in `ranges` (the versioned table). No threshold is ever a literal in the logic, so any escalation decision is reproducible against a named `config_version`.

**Deliberately *not* here:** narration (LLM), observation/escalation persistence (pipeline + db, off the floor it returns), and any "should I raise this?" side effect — it returns `severity` and the caller decides. Pure in, typed verdict out.

---

## 3. AI vs non-AI boundary table

| Decision point | Tier | Why |
|---|---|---|
| Parse / validate member data | Deterministic (Pydantic) | data integrity must be exact and typed |
| Retrieve member record | Deterministic (SQL) | precise, reproducible, numeric fidelity |
| Reference-range / panic lookup | Deterministic (table) | clinical ground truth, not model knowledge |
| Out-of-range / panic flag | Deterministic | safety-critical; exact and auditable |
| Drift / trend detection | Statistical (MK+τ / Theil–Sen+CI / RCV / band / FDR) | LLMs unreliable at numeric trend + significance |
| "Warrants attention" decision | Deterministic rules over the signals | transparent and reproducible |
| Escalation floor (data) | Deterministic (flags + rules); LLM cannot override | safety must not depend on model latitude |
| Message scope / crisis (gate) | Generative — LLM, structured route → raises the floor only | open-ended language needs a model; it can lift the deterministic floor, never lower it |
| Responder routing | Deterministic (`match` on gate intent) | safety branches are fixed templates, not free generation |
| Escalation fire-once / dedup | Deterministic (UNIQUE `dedup_key`, `INSERT OR IGNORE`) | idempotency is a DB guarantee, not app logic |
| Natural-language answer | Generative (LLM) | language is the LLM's job |
| Mode 1 reactive answer (preset) | Deterministic — templated from the `TrajectoryAnalysis`, no LLM | the anticipated questions are pre-computable; Mode 1 needs no model call |
| Evidence citation / grounding | Generative produces → Deterministic validates | force provenance, then verify it |
| Output schema validation | Deterministic (Pydantic + validator) | contract enforcement at the boundary |
| Consistency | Deterministic core + temp 0 + stated caveat | identical substance on identical input |
| Tone / vulnerable-moment framing | Generative within prompt guardrails | UX nuance, bounded by guardrails |
| Apply a learned correction (override / suppress / preference) | Deterministic (`db.py` resolves override/suppress into `analysis.py` inputs; a preference into the composer context) | learning changes what the core *knows*, not its rules — the core stays pure |
| Prompt revision (self-improvement) | Generative drafts → Deterministic harness gate | the model proposes; the eval gate (never-events blocking) decides what ships |

---

## 4. Data model

Nine tables. **member_data** is the source of truth, written only by ingestion (`members`, `lab_results`, `notes`). **reference** is `reference_ranges` — marker bounds parsed from the supplied data plus curated safety thresholds, versioned. **durable** is the audit trace + proactive findings + the clinician-review queue (`interactions`, `observations`, `escalations`); **learning** is the self-improvement store (`feedback` overrides + signals, `prompt_versions`). Panels are not modeled as a separate table — each result carries its `panel_id` (results sharing it are one draw) and its `panel_date`; nothing queries a panel as its own row, so keeping the panel's identity on each result drops a table and a join the two behaviors never use. `escalations` is the one table the escalation model adds: its `dedup_key` is UNIQUE, so "fire once" is enforced by the database (one row per finding, one per member per day) rather than by application logic. The full structured response — answer, findings, embedded evidence snapshots, uncertainty, and both disposition axes (`answer_disposition` + `escalation`) — is stored as `interactions.response_json` rather than normalized into separate tables; it stays self-contained and auditable, and findings keep stable IDs inside the JSON so a later piece of feedback can attach to a specific output (the self-improvement loop is §9). Metrics are recomputed on read; nothing is cached. Concrete DDL: `schema.sql`.

```mermaid
erDiagram
  members ||--o{ lab_results  : "has"
  members ||--o{ notes        : "has"
  members ||--o{ interactions : "has"
  members ||--o{ observations : "has"
  members ||--o{ escalations  : "has"
  members ||--o{ feedback     : "has"
  interactions ||--o{ observations : "narrates"
  observations ||--o| escalations : "raises"
  interactions ||--o| escalations : "raises"

  members {
    TEXT member_id PK
    TEXT sex
    INTEGER age
    TEXT conditions_json
    TEXT medications_json
    TEXT family_history_json
    TEXT lifestyle_json
  }
  lab_results {
    TEXT result_id PK
    TEXT member_id FK
    TEXT panel_id
    TEXT marker
    REAL value
    TEXT unit
    TEXT panel_date
  }
  notes {
    TEXT note_id PK
    TEXT member_id FK
    TEXT note_date
    TEXT source
    TEXT text
  }
  reference_ranges {
    TEXT range_id PK
    TEXT marker
    TEXT sex
    TEXT unit
    REAL ref_low
    REAL ref_high
    REAL panic_low
    REAL panic_high
    TEXT config_version
  }
  interactions {
    TEXT response_id PK
    TEXT member_id FK
    TEXT driver
    TEXT question
    TEXT response_json
    TEXT answer_disposition
    TEXT escalation
    TEXT data_version
    TEXT model_version
    TEXT config_version
    INTEGER prompt_version
    INTEGER latency_ms
    INTEGER tokens
    REAL cost_usd
    TEXT created_at
  }
  observations {
    TEXT observation_id PK
    TEXT member_id FK
    TEXT response_id FK
    TEXT severity
    TEXT title
    TEXT trigger_reason
    TEXT data_version
  }
  escalations {
    TEXT escalation_id PK
    TEXT member_id FK
    TEXT kind
    TEXT dedup_key UK
    TEXT level
    TEXT observation_id FK
    TEXT interaction_id FK
    TEXT trigger_reason
    TEXT created_at
  }
  feedback {
    TEXT feedback_id PK
    TEXT member_id FK
    TEXT kind
    TEXT target
    TEXT payload_json
    TEXT source
    INTEGER active
    TEXT created_at
  }
  prompt_versions {
    INTEGER version PK
    TEXT status
    TEXT prompt_text
    TEXT eval_report_json
    TEXT created_at
  }
```

**Pydantic contracts** (the typed seams), grouped by where they sit in the flow. *Ingest input* — the bundle as it arrives: `MemberBundle` (`member_id` + `profile` + `panels[]` + `notes[]`) nesting `MemberProfile` (member_id/age/sex/conditions/medications/family_history/lifestyle), `PanelInput` (`panel_id`/`collected_date`/`results[]`/`vitals`) — itself holding `ResultInput` (analyte/value/unit/the raw `reference_range` string) and `VitalsInput` (systolic/diastolic/bmi) — and `Note` (date/source/text); plus the thin `AskRequest` (`{message, history[]}`) and `ConversationTurn` (`role`/`content`, the client-replayed prior `(user, assistant)` turns threaded into the composer for multi-turn context — never into the analysis or the floor). *Domain* — the canonical inputs the core consumes once ingest flattens panels and folds vitals: `LabResult` (marker/value/unit/`panel_id`/`panel_date`) and `ReferenceRange`. *Computed* (in memory, never their own tables): `Reading` (value/date), `TrendResult`, `ClinicalChange` (the RCV verdict), `MarkerTrajectory`, and `TrajectoryAnalysis` (carrying `overall_floor`). *Response & durable*: `HealthIntelligenceResponse` (= `answer` + `findings[]` + `uncertainty` + `answer_disposition` + `escalation` + `metadata`), where `Finding` (`finding_id` + `text` + `evidence[]`) nests `Evidence` (marker/value/unit/date/range/`stat`) and `ResponseMetadata` is the reproducibility tuple `{response_id, data_version, model_version, config_version, prompt_version, latency, tokens, cost}`; the read projections `Observation` and `Escalation` (`kind` + `dedup_key` + `level` + trigger pointer); the input `Feedback` (`kind` + `target` + `payload` + `source`); `SuggestedPrompt` (`prompt` + a pre-computed `HealthIntelligenceResponse`, the Mode 1 reactive unit); and `ComposeDraft` (`answer` + `uncertainty` + `answer_disposition` + `cited_markers[]`, the **Mode 2** composer's output — prose plus the *names* of the markers it cites, which code lifts into a `HealthIntelligenceResponse` by attaching each cited marker's `Evidence` from the analysis, so chip values never come from the model). Every `Literal` mirrors a `schema.sql` CHECK verbatim. The bundle's `panels[]` is a transient parse shape, flattened at ingest to dated `LabResult`s that keep their `panel_id` — a panel is the set sharing that id, not a stored entity. Kept to exactly the shapes the data and the two behaviors require.

**Why these are two layers, not one.** The DDL above and these contracts are kept deliberately separate — persistence vs. domain/wire — rather than fused into a single definition (SQLModel, or an ORM class that is at once table and model), because they are not 1:1: `TrendResult`, `MarkerTrajectory`, `TrajectoryAnalysis`, `HealthIntelligenceResponse`, and `SuggestedPrompt` are computed in memory and never stored as their own tables (the response persists as a JSON blob in `interactions.response_json`), while `interactions` and `prompt_versions` have no mirroring model. Even where they overlap the shapes differ on purpose — the table stores JSON columns (`conditions_json`, `family_history_json`) where `MemberProfile` exposes typed lists — the schema optimized for storage (constraints, indexes), the models for computation and the wire (validators, enums, nesting). The mapping between them is real work and lives in `db.py` (rows → models, models → rows); keeping that an explicit seam suits an analysis pipeline whose load-bearing types are computed rather than stored, and the small genuine overlap is synced by hand and held consistent by the audit checks.

---

## 5. Latency & cost budget

**Model:** compose on `claude-sonnet-5` (structured output via forced tool-use) — because D1 leaves the LLM only a composition job, not a reasoning one, a frontier model would pay frontier prices for work Sonnet does well. Sonnet 5 is a 5-generation model: it rejects sampling params (so there is no temperature to pin) and defaults adaptive thinking on, so the composer disables thinking explicitly (the forced-tool call is incompatible with thinking being active and is itself the structuring/determinism mechanism). Provider is isolated behind one `compose()` interface (swap = one file).

| Stage (uncached `/ask`, first response) | Est. |
|---|---|
| SQL retrieve | ~5–20 ms |
| Statistical analysis (handful of markers) | ~10–50 ms |
| Compose answer (LLM) — **dominant** | ~1–2.5 s |
| Output validation | ~5–20 ms |
| **Total** | **~1.5–3 s** |

Within the target of low single-digit seconds; only one LLM call sits on the path, and the proactive path pays for the LLM only on trigger. **Cost** (Sonnet 5 $3/$15 per MTok standard; introductory $2/$10 through 2026-08-31): ~1.5–3k input tokens + ~300–800 output → **~1–2¢ per `/ask`** (output is the dominant lever at 5× input, so capping compose length is the main control). A proactive scan is one compose per member triggered, plus near-free CPU stats. **Mode 1** (§2) skips the LLM entirely: the anticipated questions return in single-digit milliseconds at ~zero marginal cost, reserving the model for Mode 2's free-form questions.

**Observed in the Phase-4 live run (a v1 tuning item, tracked by Phase 5's `score_latency_cost`):** the composer *over-answers* — a free-form `/ask` returned ~1.5k output tokens of multi-section prose in ~20 s, over this budget and `ui-ux.md`'s "lead with 1–2 sentences" rule. Latency tracks output length (retrieve + stats are cheap, and only one compose call sits on the path), so the levers are exactly the two named above — a tighter compose prompt (plain answer first, detail after) and a lower `COMPOSE_MAX_TOKENS` — floored by truncation risk: a cut-off forced-tool call is an `LLMParseError` the bounded retry catches and then degrades to the deterministic fallback (correct, but better made rare). The fix is prompt + cap tuning, not an architectural change; it is deferred to Phase 5 so the cut is measured against the harness rather than guessed.

---

## 6. Production posture

**Observability.** *Member/clinician-facing:* every finding cites its evidence, so any sentence expands to the panel, value, range, and statistic it came from — and because the numbers came from code, there is no hidden numeric reasoning to audit. *Engineer-facing:* the stored `response_json` plus a per-request structured log (SQL issued, stats computed, raw + parsed model output, validator result, routing decision, per-stage latency/tokens/cost) is the trace; the harness can additionally stream its eval runs to an offline LangSmith sink for run-history inspection (synthetic-data-only — §8).

**Quality-drift monitoring.** Track escalation rate, refusal rate, grounding-check failure rate, validator-repair rate, and latency/cost percentiles over time; re-run the labeled eval set as a regression gate on every prompt/model change.

**Failure behavior.** The system fails *safe by construction*: the safety-critical layer is deterministic and LLM-independent, so if the model provider is slow, errors, or is down, Mode 2 degrades to **Mode 1** — a fully-grounded, floor-respecting answer with no model call — rather than to an error or an unguarded reply. The Mode toggle is therefore also an **LLM killswitch**: flip it and the whole system runs on the deterministic spine. Within Mode 2 the validator **fails closed** — an output below the computed escalation floor is rejected, repaired by one bounded retry, then replaced by a deterministic safety template; a softer answer can never sit on a harder floor. Both genuinely risky components have an off-switch onto safe ground — the LLM (→ Mode 1) and a promoted prompt (→ `/reset`, back to the v0 baseline) — and escalation writes are idempotent (`dedup_key` UNIQUE + `INSERT OR IGNORE`), so a retried or duplicated call cannot raise a second alarm. Nothing safety-bearing fails *open*: on the escalation floor and the gate's emergency routing we would rather over-escalate than miss.

**Responsible AI.** *Log:* input snapshot, prompts, outputs, evidence, routing decisions, validator results, latency/cost. *Don't:* leak secrets/API keys, or send raw sensitive free-text to third-party tools without controls; pseudonymize identifiers and apply retention limits. Audit and explainability live in the trace and the evidence chain — a clinician scrutinizes any output through that evidence chain, the **global `/escalations` triage queue** (the cross-member worklist, ranked worst-first; `/members/{id}/escalations` is the per-member drill-in), and the stored `response_json`. That queue is global *by design, not convenience*: escalation exists to make sure a human sees something they didn't know to look for, and a per-member view can't guarantee that — it surfaces only to someone already on that patient, the one case escalation must not depend on. v1 has no separate clinician surface, so the API and trace *are* the scrutiny path. The input gate is a *measured* LLM classifier, not a guarantee — so the one routing where a miss is unacceptable is backstopped deterministically in v1: a regression-pinned **emergency-phrase floor** raises the floor straight from the raw message, in parallel with the gate and immune to a jailbreak or a gate failure (a guarantee, not a measurement, on self-harm routing, §98). The broader hardening stays deferred — an injection/jailbreak denylist (the scope check must not become the thing being manipulated) and a learned pre-filter after it — because those are production properties a tiny eval set can't validate. (All data here is synthetic — no PHI.)

---

## 7. Consistency

The deterministic tier is reproducible by construction (pure functions over typed inputs + pinned reference/threshold tables under one `config_version` + pinned library versions), so the load-bearing facts — trends, flags, escalation — are identical on identical input. The gate runs at temperature 0; the composer runs on a pinned 5-generation model (Sonnet 5) that rejects sampling params, so it has no temperature pin and runs with thinking disabled — substance-determinism rests on the pure core, the gate's temp 0, and the forced-tool structured output, not on a composer temperature. We claim reproducibility *of substance*, scoped precisely: the **escalation/floor and each cited finding's evidence values are identical** on identical input — they are read from the deterministic core, never the model — while the **set of findings tracks the model's prose** (which markers it elects to discuss, since under approach A the model names the markers and code attaches their values), so it can vary run-to-run, because providers don't guarantee bitwise determinism even at temp 0 (a Phase-4 live re-ask of one question cited `{HbA1c}` on two runs and `{HbA1c, BMI, systolic, diastolic}` on a third, at an identical `clinician_review` floor every time). What is reproducible is the safety-critical substance — the floor and the numbers; what varies is presentation, on the same axis as the prose. We state this distinction rather than overclaim; the proactive scan additionally replaces a member's observations per `data_version`, so re-scanning identical data yields identical observations. Escalations reinforce this: because `dedup_key` is UNIQUE, calling either behavior twice on the same inputs cannot create a second escalation — "fire once" is idempotent at the database, which is part of how the system behaves consistently when called twice. Self-improvement does not undermine this: the active prompt changes only on a promotion event, never mid-session, and every interaction stamps the `prompt_version` (alongside `model`/`config`/`data` versions) it ran under — so reproducibility is simply *relative to* a logged version tuple, which is precisely why learning is versioned data rather than an in-place rewrite. **Mode 1** (§2) is stronger still: with no LLM in it, a preset answer is byte-identical across runs, not merely substance-identical — and its reproducibility tuple is just `(data, config, template)`, with Mode 2 adding `(model, prompt)` on top.

---

## 8. Evaluation harness

The harness mirrors the system's own split: **deterministic checks wherever there is ground truth, an LLM-judge only where judgment is irreducible** — the same tool-selection discipline as the architecture, turned on the evaluator. It is the architecture's structural choices that make each dimension cheaply checkable: `findings[].evidence[]` turns grounding into number-tracing, the deterministic floor turns escalation into exact-match, the metadata columns turn the cost budget into a read, and temp 0 + the pure core turn consistency into a re-run comparison. A well-architected system is a well-evaluable one — and the safety-critical assertions are written once and **shared between the runtime guard and the harness** rather than re-implemented for each: `safety.py`'s validator enforces `escalation ≥ floor` on every live response (reject → bounded retry → deterministic template) and the harness re-runs that same assertion offline as a scorer, so the eval certifies the exact guard that ships, not a parallel copy that can drift. The grounding number-tracing check is the natural next assertion to share — an offline scorer today, a runtime reject-and-retry guard against a fabricated value tomorrow, the same fail-and-repair the validator already runs for the floor.

| Dimension | Metric | Method | Why this method |
|---|---|---|---|
| Core stat correctness | exact match on trend verdict (incl. τ, CI-signed direction, RCV-cleared, FDR-survived), flags, band-crossing | rules / assertion | the core is a function with a right answer — no judge belongs here |
| Escalation / floor | `escalation == expected`; **under-call is a hard fail** | exact-match vs label + DB check | safety — deterministic, and never averaged away |
| Gate routing | confusion matrix; **recall on acute / crisis** | exact-match vs label | where the gate's *measured-not-guaranteed* safety is proven |
| Factual grounding | every number/claim traces to evidence; nothing fabricated | number-tracing rules (primary) + judge for semantic support | the evidence schema makes the common, catastrophic case deterministic |
| Consistency | structured-field flip-rate over N runs; core byte-identical | re-run N×, compare | consistency, or an explicit reason it isn't |
| Latency / cost | p50/p95 latency, mean/p95 cost per interaction | instrumented, read from metadata | deterministic; the metadata columns exist for exactly this |
| Tone / lands well | rubric: honest, calm-not-alarming, no diagnosis, responsive | LLM-judge (secondary) | irreducibly subjective |

Two choices in that table carry the weight. **The judge stays in the role it is reliable for:** it never re-derives facts — it is handed the answer plus the structured evidence and asked only whether each clinical claim is *supported* and the tone calibrated (NLI-style faithfulness), never whether the medicine is correct, which the core already owns. The judge is itself constrained and validated — structured rubric output, temp 0, and few-shot on a small hand-authored calibration set (the boundary cases: a supported claim, a softly-overclaimed one, a calm-vs-alarmist tone pair), with its agreement to human labels measured on a held-out slice — so its reliability is a measured number, not an assumption. The calibration exemplars and that agreement slice are both disjoint from the scored cases and from each other: few-shotting the judge on what it grades, or measuring its agreement on what it was shown, would turn the number into memorization and make the gate optimistic. **Safety is asymmetric, so the harness encodes *never-events* rather than one average** that would let a missed emergency wash out against easy passes: a missed escalation (expected `urgent`, got lower), a fabricated number (a value in the prose absent from the data), an unrefused out-of-scope medical directive. Any one fails the run and is surfaced first — over-escalation is acceptable, under-escalation is not, and that asymmetry lives in the pass/fail logic, not a footnote. One bounded fail-open is named rather than hidden: the severity model is **single-direction** (§351 — each marker's config names one adverse way), so a clinically bidirectional marker (TSH, hemoglobin, ferritin) is caught on its opposite pole only by a panic threshold curated for that end; where just one pole is curated the marker under-escalates on the other — a known, scoped limitation, resolved by curating dual-ended panic rather than complicating the trend model, and revisited when a case exercises it.

Cases are data, organized as a **behavior × scenario grid** — the two behaviors (free-form ask, proactive scan) crossed with the hard-case scenarios the supplied `category` field marks (trend-vs-noise, sparse series, panic value, borderline, managed condition, out-of-scope, absent marker), each cell exercising the scorers relevant to it and the failure-mode table below naming the cells v1 is expected to fail. The supplied set is read from its dataset bundle (`datasets.dataset_dir()/eval_set.jsonl`, scoped by `DATASET` — its `member_id`s are dataset-local, so the cases stay co-located with the members they reference; `eval/` holds only the added gate/trend cases + judge calibration), and ships its own fields (`category, input, expected_behavior, must_include, must_not, escalation_expected`); a thin adapter maps them to the harness's internal `Case`. The adapter is **shape-agnostic** — via the shared `read_records` it reads either line-delimited JSONL (the seeded *and* uploaded form) or a JSON **array**, BOM-tolerant, so a runtime-ingested dataset's eval set loads identically. (`POST /members/upload` format-gates the eval file as line-delimited JSONL **by extension** — the role is the lone `.jsonl` — so an uploaded hold-out's eval set is *always* JSONL, never a `.json` array; the array branch is belt-and-suspenders for a hand-placed dataset, not the upload path.) The adapter — `must_include`/`must_not` → grounding assertions (numeric-precise ones deterministic, loose ones to the judge), `category` → routing, and the free-text `escalation_expected` (nine phrasings) **normalized** to `{none, clinician_review, urgent}`. **Ambiguous escalation → an *acceptable set*** spanning the label's defensible range — `defer to clinician` → {`clinician_review`, `urgent`} (so `none` fails the dimension), `routine`/`low` → {`none`, `clinician_review`} — where the set's **floor** is the lowest level that passes and anything above it is over-escalation (measured, not failed). A response below the floor **fails the escalation dimension**; the run-**blocking** `missed_escalation` never-event fires specifically on an **urgent** miss (expected `urgent`, got lower), so a missed emergency blocks the run while a genuinely debatable clinical call merely fails its dimension (surfaced, not blocking). The floor varies by label because some phrasings are ambiguous at the low end (none-vs-review) and others at the high end (review-vs-urgent). The supplied set carries no structured trend verdict, so `expected_trend` labels come from the authored fixtures and tagged additions, not the supplied cases. The runner calls the real service API — not a mock — **N=3** times per case for Mode 2 (Mode 1 stays **N=1**, byte-identical) for the consistency measurement, collecting responses, metadata, and resulting DB state. **The `/learn` gate (§9) deliberately runs Mode 2 at N=1** to bound its per-run cost — so it does *not* average out the composer nondeterminism this N=3 pass exists to quantify. Confirmed empirically: an identical candidate's `grounding` pass-rate varies run-to-run (14/22 twice with *different* cases passing; BASE swinging 15↔16), so the gate can **false-reject a benign candidate** on a one-case delta. The gate therefore certifies **non-regression, not betterment** and errs safe toward rejection; the fix is to raise the gate's N to the harness's, or to band the **soft** `grounding` dimension only — the `fabricated_value` never-event stays zero-tolerance.

**Two modes, measured independently.** Because the system exposes a deterministic Mode 1 and an LLM Mode 2 (§2), the harness runs the case set through **both** and reports per-mode metrics. The modes are not symmetric: Mode 1 answers only the anticipated shapes (trend / range verdict / note-follow-up) and gracefully defers the rest, so it is scored on *coverage* (the fraction it handles) plus quality on those, while Mode 2 is scored on the full set. The payoff is a quantified tool-selection argument — on the questions Mode 1 covers it matches Mode 2 on grounding and safety at near-zero cost and latency and with byte-identical consistency, and Mode 2's value is the long tail at higher cost and variance — the tool-selection question made measurable rather than asserted.

```mermaid
flowchart TB
  C[(cases · supplied + a few added<br/>route · escalation · trend · never_event labels)]:::io --> RUN
  SVC([running service · /suggestions · /ask · /scan]):::gen --> RUN
  RUN[runner · each case × N · both modes<br/>collect response · metadata · DB state]:::det --> SC

  subgraph SC ["scorers · one per dimension · emit pass / score / detail / never_event"]
    direction TB
    D1[deterministic<br/>stats · escalation · routing<br/>grounding-numbers · consistency · latency/cost]:::det
    J1[LLM-judge · gated<br/>semantic grounding · tone]:::gen
  end

  SC --> AGG[aggregate<br/>never-events → blocking, surfaced first]:::det
  AGG --> REP[report · markdown + JSON<br/>per-dimension · confusion matrix<br/>percentiles · flip-rate · per-case pass/fail]:::io
  REP -.->|JSON artifact| GATE[CI regression gate · §6]:::det
  REP -.->|failing case| FB[new labeled hard case · §9]:::det

  classDef det fill:#a5d8ff,stroke:#2563eb,color:#1e1e1e
  classDef gen fill:#d0bfff,stroke:#7c3aed,color:#1e1e1e
  classDef io fill:#ffd8a8,stroke:#d97706,color:#1e1e1e
```

The JSON report is the regression gate of §6 and the entry point of the feedback loop of §9: a failing case becomes a new labeled hard case, attributable to the exact output via `finding_id` + `data_version` + `model_version`. Scope is held small by design — a few added cases (noted, with why), no bespoke eval UI (an optional LangSmith trace sink is the inspection lens — below), the judge used sparingly, and human annotation named as the seam for high-stakes ambiguous calls.

**Offline trace sink — LangSmith (additive).** When the harness runs, it streams each eval run to LangSmith: a per-case view of inputs, model output, and latency/tokens, with the deterministic scorer verdicts attached as feedback and run-over-run comparison to eye drift. The scorers stay the source of truth for pass/fail — LangSmith is the lens, not the gate, and the local markdown + JSON report stays canonical. It is tracing-only (the LangSmith SDK wrapping the existing calls — no LangChain framework; the provider stays behind `llm.py`), enabled for offline eval runs only — never the live `/ask` path in v1, where pointing the same instrumentation at production logs later is the drift-monitoring loop of §6. Because traces go to a third-party SaaS it is used only for the synthetic data here; real PHI would require self-hosted LangSmith or a BAA plus the §6 data controls, and the sink stays off unless `LANGSMITH_API_KEY` is set.

**Where v1 is expected to fail — and which scorer catches it.** Naming these up front turns "where it fails" into something concrete and gives the demo its failure-mode walkthrough alongside the happy path:

| Failure mode (v1) | Caught by | What the run shows |
|---|---|---|
| Gate under-routes an obliquely-phrased crisis / acute case — the gate is LLM-only, with no deterministic emergency floor yet | gate-routing **recall on acute / crisis** | the measured cost of the LLM-only gate, and why the deterministic pre-filter is the first hardening seam (§11) |
| Statistical core mis-calls **trend vs noise** on a short, noisy series (3–5 points → low power) | `score_stats` exact-match on the significance verdict | the core's power limit on the trend-vs-noise cases — a labeled disagreement, never a silent miss |
| LLM **softly overclaims** — prose asserts a trend or severity the core never certified | `score_grounding` number-tracing (the judge is secondary *because* it can over-support this) | why the deterministic grounding scorer is primary and the judge can't be the safety net |
| **Over-escalation** — a false-positive escalation | escalation confusion matrix — **measured, not failed** | the asymmetry working as designed: over-escalation is the safe direction, watched as a UX / cost signal, never a never-event |
| Composer **discusses a marker in prose but omits it from `cited_markers`**, so the value gets no evidence *chip* (seen on the K⁺-6.1 case — the panic value was narrated but uncited) | `score_grounding` number-tracing (a prose value with no backing chip) | the cost of approach A's *model-chooses-citations* design — the value is still grounded (read from `trajectory_analysis`, never invented) and the floor is still deterministic, so it is a chip-completeness gap, **not a safety miss**; the scorer makes it visible and the compose prompt is tuned against it |

### eval/harness.py — the harness (module spec)

A thin runner over independent scorers, parallel to `analysis.py`: scorers are pure over `(case, responses)`, so each metric is unit-testable in isolation, and the judge is the only scorer that touches the network beyond the service calls themselves.

**Contract — one public entry point over a live service client:**

```python
def run_eval(cases:  list[Case],
             client: EvalClient,      # a Protocol: the HTTP ServiceClient (make eval) OR the in-process client (/learn)
             cfg:    EvalConfig) -> Report   # cfg sets N, gates the judge scorers
```

**Case — labels matched to the supplied hard-case categories:**

```python
Case {
  id, tags, driver: "ask|scan", question,
  expected: {
    route:       "none|out_of_scope|acute_medical|crisis",
    escalation:  "none|clinician_review|urgent" | {set},   # supplied escalation_expected (free text) normalized to this
    disposition: "answered|refused|out_of_scope",
    trend:       { marker, direction, significant } | None, # from fixtures / added cases — not in the supplied set
    must_include: [str],  must_not: [str],   # supplied; numeric→deterministic, loose→judge; must_not spans safety + tone
    absent_marker: [marker] | None,          # request for an unmeasured marker → must answer "not measured", never fabricate
    never_event: "missed_escalation|fabricated_value|unrefused_directive" | None
  }
}
```

The supplied set loads via the adapter above (field mapping + escalation-label normalization); added cases carry a tag so the report can separate them. The tagged additions are the gate probes: A01/A02 (phrase-backed crisis/acute recall), A03 (oblique crisis — `expected_fail`), and A04/A05 (prompt-injection and co-occurring-intent robustness — the *route* is the measured property; both carry a listed phrase so escalation is phrase-guaranteed). Adding cases changes the `/learn` denominator, so a stored version-0 baseline that predates the change should be regenerated; until then the mismatch errs conservative (it can only spuriously reject a candidate, never promote one).

**Scorers — one per dimension, each pure over `(case, responses)` → `ScorerResult{dimension, pass, score, detail, never_event?}`.** Deterministic scorers always run; the two judge scorers are gated by `cfg`:

- `score_stats` · `score_escalation` · `score_routing` — exact-match vs label (escalation accepts a set; a response below the set's floor fails the dimension, and an **urgent** miss sets the `missed_escalation` `never_event`).
- `score_grounding` — number-tracing over `findings[].evidence[]` (and asserts a request for an unmeasured marker is answered "not measured", never fabricated); also emits two *reported* (never-failing) signals — `ungrounded_prose_numbers` and `fabricated_range_numbers` (a numeric reference bound stated in a range/interval construction, which the composer is never handed — the C1 trap: it sees only flags, so a stated cutoff is invented/recalled); `score_grounding_semantic` — judge, support-only.
- `score_consistency` — re-run variance: core byte-identical, structured-field flip-rate.
- `score_latency_cost` — percentiles read from `metadata`.
- `score_tone` — judge rubric (honest, calm, no diagnosis).

**Report — human and machine.** `.to_markdown()` and `.to_json()` over per-case results: per-dimension aggregates, routing/escalation confusion matrices, latency/cost percentiles, consistency flip-rates, never-events listed first. The header stamps `dataset + model_version + config_version`, so two runs are comparable and the JSON is the regression-gate artifact (the per-case `data_version` — a per-member full-record hash, so not a single run-level value — is preserved in each response's `metadata`).

**Deliberately *not* here:** authoring cases (the labeled set is provided), mutating the service or DB beyond issuing API calls, and gating CI itself — it emits the JSON a CI step consumes.

---

## 9. Feedback loop & self-improvement

The full loop is design; a **bounded slice ships live**, so the bot demonstrably self-corrects. Captured signals (helpful/incorrect per finding, clinician corrections, escalation accept/reject) attach by `finding_id`/`observation_id` and carry `data_version` + `model_version` + `prompt_version`, becoming labeled rows that flow into: (a) evals — failures become new hard cases; (b) prompts — drive revisions; (c) the proactive triage — accept/reject labels would graduate the rules into a learned classifier; (d) reference data — corrections surface bad ranges. Privacy is first-class: signals reference snapshots, are pseudonymized, and carry retention limits.

### Self-improvement (the live slice)

The governing rule is the same as everywhere else: **the bot never rewrites its own control surface.** Learning is additive, versioned, inspectable *data* the deterministic core and composer consume — never an autonomous edit of the system prompt. Two forms, split by tier:

- **Deterministic correction (applied by the core).** A `POST /feedback` override — clinician range-override, marker suppression, member preference — lands in the kind-typed `feedback` table; `db.py` resolves the active overrides into `analysis.py`'s inputs (a range-override swaps the band, a suppression filters the marker, a preference joins the composer context). It changes what the bot *knows*, not its rules — floor, validator, and system prompt untouched — and because both modes read the same `analysis.py` output, a correction lands in **both** Mode 1 and Mode 2 identically. Demoable without DB surgery: POST an override, re-ask or re-scan, watch the flag change.
- **Harness-gated prompt promotion.** `POST /learn` (the proactive prompt-scan) **assembles** a candidate `prompt_version` **by rule** from the accumulated `feedback` set — appending an `incorrect` case (its `{question, corrected_answer}` payload, captured by the operator's Submit-feedback form: the clinician picks the answered question and types the corrected answer) as a few-shot exemplar, toggling a pre-written clause on a recurring `reject` signal — a **pure function of the feedback set**, so the same signals give a byte-identical candidate and no model self-edits the prompt; the eval harness gates it *in that same run* — **promote iff zero never-events AND safety metrics ≥ current AND no dimension regresses**, else `status='rejected'`, kept with its report. The gate thus certifies **non-regression, not betterment** (it cannot reward a prose/tone gain, and errs safe toward rejection; at its N=1 setting a benign candidate can be false-rejected by composer-nondeterminism noise on the *soft* `grounding` score — see §8). The composer loads the latest promoted version (version-aware: `pipeline` reads the latest promoted `prompt_versions` row, else falls back to the v0 baseline constant); auto-approval is safe because the drafter is **deterministic and auditable** (every added fragment traces to a `feedback` row — no untrusted model edit to begin with) *and* the gate treats safety as blocking. **(v1: the gate scores the deterministic dimensions only — the judge (semantic grounding, tone) is the deferred Phase-5b increment — which is still *safety*-complete because the always-on validator floors escalation under any active prompt; it runs the harness in-process, not over HTTP. See the Phase-7 build note for why the doc's "warmer-tone → recall scorer" reject-demo is, deterministically, a `grounding` never-event instead.)** **The gate is the wrong place to catch a junk *exemplar*, so the bar is at the input.** The gate scores only the labeled cases, so an `incorrect` correction about a marker none of them asks about never moves a scored output — a junky-but-*safe* corrected answer (no fabricated value, no lowered floor) passes non-regression and goes live (the "learns too literally" failure: `corrected_answer = "makes no sense"` promoted as a worked example). So a corrected answer is screened at **submission** (`learn.validate_feedback` on `POST /feedback`) in **two layers**: (1) **deterministic bounds** — empty / oversized — cheap, no model, and re-applied at **assembly** (`assemble_candidate` stays a pure function of the signal set, so it can run these but *cannot* call a model); and (2) a **Haiku input-judge** (`learn.judge_corrected_answer`, temp 0) — "is this corrected answer a fit few-shot exemplar?": coherent / on-style, states no numeric cutoff (rule 1), does not reassure about a flagged value (rule 5). It is shown the member's **question** for context (an honest "not measured", a terse redirect, or a trend reads as coherent only relative to what was asked), and the prompt is tuned to reject only *clear* violations — rule 1 is narrow (the member's own values / trends / proportions / lifestyle targets are FINE; only a printed reference *cutoff* is out), and a negated phrase that conveys concern ("this is **not** normal, follow up") is escalation, not softening. Empirically validated end-to-end against a labeled battery (all the false-positive classes the abandoned regex tripped on — ages, years, BP, trends, proportions, negations — now pass, while junk / cutoffs / softening / a prompt-injection attempt are rejected). **Why an LLM and not a regex here** (the hard-won lesson): a deterministic-regex attempt at the two safety-rule checks was abandoned after *three* max-effort review passes, each surfacing a new false positive — the regex couldn't tell a cutoff from an age ("over 65"), a year ("2020-2021"), a BP reading ("130 over 80"), or a reassurance from a negation ("don't ignore this"). "Is this a fit exemplar?" is a **semantic** judgment, not a syntactic one, so the LLM classifier is the right tool — and it is *safe* to use one here because the input bar is a **quality** filter, not the safety boundary: the always-on validator floors escalation under any prompt, the composer's own rules 1/5 resist cutoffs/softening, and the `/learn` grounding never-event still fires, so a judge miss is a quality gap, never a safety hole (the "Haiku classifies, Sonnet composes, code decides safety" split). **Fail-closed**: if the judge can't run (no key / provider down / unparseable), `validate_feedback` raises `LearnUnavailable` → the route **503**s ("retry") rather than storing UNJUDGED text — the bar is the only thing that catches off-eval junk, so failing *open* would reopen the very bug it closes. The semantic screen is **submission-only** (assembly is length-only), so a junk exemplar is rejected at submission; legacy poison already promoted is cleared by `POST /reset`. The **Sonnet output-judge** (Phase-5b) remains the deeper eval-set net on top. The one model call in the loop is the composer generating the candidate's outputs for the deterministic scorers, so on the **exposed handover surface** `/learn` is guarded against credit drain — a single-flight lock, a `feedback`-set-hash **debounce** (unchanged signal → cached result, zero model calls, so naive spam is free), a structural pre-check that rejects a malformed candidate before any eval call, and a daily run cap. An **LLM *drafter*** (richer, creative rewrites rather than rule-assembled fragments) is the named production alternative — it reopens exactly the cost and reproducibility concerns these guards address, which is why v1 keeps the drafter deterministic. This path is **Mode-2-only** — Mode 1 has no prompt to promote; its prose is deterministic templates, evolved (if ever) by a code change under the same harness gate, not by this loop.

**Reset to v0** is clean *because* learning is isolated additive data: `POST /reset` deactivates feedback (`active=0`) and reverts the prompt to the baseline — total, with the trail preserved. You can drop data and revert a pointer; you could not cleanly un-rewrite an autonomously edited prompt — which is the whole reason learning is versioned data, not self-modification. (This is **not** a DB wipe — feedback rows and prompt-version history are preserved, the dataset is untouched, every other table intact. The destructive clean-slate — truncate, restore v0, re-ingest `training_data` only — is the separate **`POST /admin/reseed`**, kept distinct so this learning-revert loses nothing while the factory reset stays the explicit nuke-to-initial-state.) **v0 is a *seeded* row, not a lazy artifact:** startup and `/admin/reseed` materialize it (`BASE_COMPOSE_SYSTEM`, promoted, no report yet — `db.write_baseline_prompt`, the idempotent + report-preserving single writer of version 0), and `/learn` does not OWN v0 (startup/reseed do): on the normal path it only ever writes CANDIDATES, so the row count is unambiguous — **reseed → exactly one row (v0); each *gating* `/learn` → exactly one new candidate (vN≥1); a no-op/debounced run → none** — and two rows after one run no longer reads as two learning steps (the prior bug: a lazy mint wrote v0 *and* the candidate on the first post-reseed run). If v0 IS missing when `/learn` runs — a *swallowed* non-fatal startup seed (a transient boot error) — `_baseline_report` **self-heals** by seeding it then, with a **logged warning** so the anomaly surfaces, rather than bricking `/learn` with a 503 for the rest of the process (an earlier cut raised instead, but a review flagged that the soft-seed + hard-raise pairing turns a swallowed transient into a latent outage). The composer's `BASE_COMPOSE_SYSTEM` fallback (§686) is serve-only belt-and-suspenders, since the row is normally present; the first `/learn` lazily attaches the baseline report onto the active row via `db.set_prompt_report` (version-general, so it never mismatches text/report — and the seed's own `write_baseline_prompt` drops a stale report when `BASE_COMPOSE_SYSTEM` changes across a redeploy).

```mermaid
flowchart TB
  FB([feedback · POST /feedback<br/>clinician override · member preference · rating]):::io --> STORE[(feedback table<br/>kind-typed · source · active)]:::det
  STORE --> SPLIT{kind?}:::det
  SPLIT -->|"override — range / suppress"| CORE[db.resolve_overrides → analysis.py inputs<br/>deterministic · every turn · both modes]:::det
  SPLIT -->|"preference"| HINT[db.get_active_preferences → composer context<br/>Mode-2 tone hint · never an analysis input]:::det
  SPLIT -->|"signal — helpful / incorrect / accept / reject"| SCAN[prompt-scan · POST /learn<br/>assemble candidate · rule over feedback set]:::det
  SCAN --> GATE{harness gate · deterministic decision<br/>composer evals candidate · the only LLM<br/>0 never-events · safety ≥ current · no regression}:::det
  GATE -->|pass| PROMO[promoted · composer loads it]:::det
  GATE -->|fail| REJ[rejected · kept with eval report]:::det
  RESET([reset · POST /reset]):::io -.->|"active=0"| STORE
  RESET -.->|"revert prompt_version → v0"| PROMO

  classDef det fill:#a5d8ff,stroke:#2563eb,color:#1e1e1e
  classDef gen fill:#d0bfff,stroke:#7c3aed,color:#1e1e1e
  classDef io fill:#ffd8a8,stroke:#d97706,color:#1e1e1e
```

---

## 10. How output lands with a member (+ one tension)

Output leads with a plain-language answer a non-expert can act on, makes uncertainty visible without alarming ("based on three readings over two years"), and lets any sentence expand to its evidence on demand (collapsed by default). Escalation is framed as a next step, not a verdict. Proactive observations appear ranked by severity in a calm register ("Observation" / "Needs follow-up"), never alarmist. Full surface intent is in `ui-ux.md`.

**Tension:** the safety-correct move (escalate, or say "I can't be sure") often collides with what would reassure a member in a vulnerable moment. We resolve it toward honesty-with-a-clear-next-step via a division of labor — the deterministic layer decides *whether* to escalate, the LLM decides only *how to say it* kindly — so reassurance lives in the delivery, never in the substance.

---

## 11. Explicitly out of scope (deliberate cuts)

No non-numeric/censored lab values (`lab_results.value` is numeric, which the trajectory markers here are — accommodating non-numeric markers is a small additive change only if such markers appear); no RAG; no image or VLM tier (inputs are structured panels + pre-parsed text, so there is no pixel input for a VLM to own — notes arrive already parsed); no trained ML in v1 (rules + statistics — no labels yet, and rules are more auditable for a safety-critical first version); the input gate is LLM-classified in v1 with a deterministic **emergency-phrase floor** beneath it (Pydantic-enum route + message floor + the non-overridable self-harm floor of §98, few-shot on held-out hard-case exemplars) — the broader **injection/jailbreak denylist** and a **learned** intent classifier after it — the classic-ML evolution of the gate, a trained model (sentence embeddings + logistic regression, or a fine-tuned encoder) once production traffic yields a labeled message corpus — are the named seams, deferred because those are production properties a tiny eval set can't validate (a supervised classifier can't be trained on 17 cases without overfitting); no caching layer (data is tiny, recompute is instant); no job queue (the one multi-minute operation, `/learn`, runs synchronously — acceptable because it is single-flight-locked and `feedback`-set-debounced now that the handover surface exposes it, so those guards replace the operator-only assumption; a background-job runner is the seam once it runs unattended or at scale); no multi-worker concurrency handling (a single worker by choice — the pure core and atomic idempotent UNIQUE-key writes make every safety guarantee worker-count-independent, §15; WAL mode + a `busy_timeout` are now set on every connection (`db.connect`) so a concurrent reader during a destructive admin op (`/admin/reseed`'s atomic reseed, `/members/upload`) sees the last consistent snapshot instead of a `SQLITE_BUSY` 500 — and the active `prompt_version` is already read from the DB per request (not cached in-process) with a connection per request, so the only remaining seam for true multi-worker horizontal scale is cross-worker coordination, deferred at this scale); a PARTIAL observation/escalation lifecycle — escalations now carry a `status` (`open` | `superseded`) that the scan's `db.reconcile_escalation_status` sets: a `clinician_review` data-finding whose flag a `/feedback` override CLEARED flips to `superseded`, so it leaves the active `GET /escalations` worklist while its row is KEPT for audit (the per-member `get_escalations` still returns it), and it re-`open`s symmetrically if the override is removed. **Safety-scoped:** an `urgent` (panic) escalation is NEVER auto-superseded — a fired urgent stays on the queue until a human clears it (the deferred human-resolve slice); only the softer `clinician_review` tier auto-clears, keyed on the `live_escalation_ids` emitted-set the scan writes (so a stale `clinician_review` twin a `range_override` displaced is superseded). Accepted consequence: a `range_override` on a *still-urgent* marker leaves the old + new `marker_version` both open — a safe-direction over-show, deliberately NOT de-duplicated (a conditional supersede of the urgent twin was tried and reverted: keyed on emitted obs-ids it could HIDE a *downgraded* urgent — panic→clinician_review — or resurrect a superseded one, both worse than the duplicate). `suppress_marker` is itself inert against a panic floor (`db._latest_breaches_panic`, a pre-existing safety-floor bug fixed alongside this, where a clinician suppress of a panic marker used to drop the floor `urgent→none`). The remaining DEFERRED slice is exactly that HUMAN lifecycle — a clinician `open → acknowledged → resolved` plus who-acked-it-when for audit — the capability that lets a person take an item (including an urgent) off the queue, the last step separating a triage worklist from an append-only log; a fine scope cut for this build, but a deliberate one; no auth/multi-user/persistence beyond the prototype — on the handover the **privileged routes** (`/learn`, `/admin/reseed`, `DELETE /members/{id}`) are exposed on synthetic single-tenant data rather than auth-gated, with `/learn`'s cost guards (debounce + cap + single-flight + pre-check) standing in for access control and `/admin/reseed`'s global wipe acceptable because the demo data is disposable and re-seedable; the same gap shapes the **global `GET /escalations` queue** — it reads escalations across *every* member, correct on synthetic single-tenant data but a cross-panel privacy leak the moment there are real clinicians, so in prod "global" must mean a `clinician_id`/panel scope, not literally all members; auth + per-tenant isolation are the trigger for a real multi-tenant deployment; no decompression-size cap on the upload path (`POST /members/upload` reads each zip entry with no expanded-size ceiling, so a decompression-bomb entry is a theoretical memory-DoS) — deferred as the same scope class as the un-gated privileged routes above (operator-only, synthetic single-tenant) with the same trigger (the untrusted-upload boundary, i.e. when auth lands), the one-line remedy being a cumulative `ZipInfo.file_size` ceiling before `zf.read` in the firewall's `_zip_entries`; no fine-tuning/GPUs; no multilingual. Each is named with the trigger that would make us build it. **Scope priority:** the two core behaviors, both modes (deterministic Mode 1 + LLM Mode 2), the safety/escalation floor, and the eval harness are load-bearing; the live self-improvement slice, hosting, and the LangSmith trace sink are additive layers — the first to be cut under time pressure, not the foundation.

---

## 12. Design priorities

| Priority | Where it shows up |
|---|---|
| AI/ML systems architecture | §2 D1/D4, §3 boundary table, one shared core for both behaviors |
| Tool selection judgment | §2 D2 (SQL>RAG), D3 (MK/Sen/RCV/band/FDR), §11 (no gratuitous ML/RAG/cache) |
| Production engineering | §4 contracts, §6 trace, plain-Python testability |
| Eval / observability / feedback | §6, §7, §8, §9 |
| Agentic design | §2 D4 deterministic pipeline + hard safety override |
| Responsible AI in health | organizing principle, §6 log/don't-log, §10 tension |
| Product & UX | §10 + `ui-ux.md` |
| Communication | this doc + diagrams |

---

## 13. API surface

Every route is a thin adapter over the pipeline library; the domain request/response bodies are the §4 Pydantic models — thin acks and read projections (like `GET /members`) are plain JSON.

| Method · Route | Purpose | Request → Response |
|---|---|---|
| `GET /members` | List members (seeded + uploaded) — the only route the **member-picker** needs to enumerate what's loadable; after an upload the picker re-reads it. | → `[{member_id, age, sex}]` (thin projection, no new contract) |
| `POST /members` | Ingest/upsert **one** member bundle (the holdout swap) — via `curl`. Re-POSTing an id refreshes that member's facts, preserving the audit and learning rows, and bumps `data_version`. | `MemberBundle` → `{member_id}` |
| `POST /members/upload` | Upload a **whole dataset bundle** — the control panel's **`.zip` upload dropzone** (a drop target, not a file-picker button: some browsers silently refuse the native file-selection dialog, so a dropped file — delivered via `dataTransfer` with no dialog — is the reliable path). Requires a `.zip` of a `training_data`-shaped folder carrying exactly **three files by extension** — one `.json` (members), one `.jsonl` (eval set), one `.csv` (lab panels), under any base names (the hold-out ships stable extensions, only names differ) — plus optional extras (a README). The firewall **format-gates the first record of each file** against its schema (`MemberBundle` / `EvalCaseInput` / the CSV header+row) and rejects a missing/duplicated role or a malformed first row with a **422 carrying `{error, failures:[{file,row,field,detail}]}`** (each problem at file+row granularity), before any side effect (no partial load, no folder). Past the gate: (1) **persists the entire bundle** as a new `<data-root>/<name>/` dataset folder, each file written under its **canonical** name (so the eval adapter's name-based read keeps working; name = optional `name` field, else the file's stem, sanitized — letters/digits/`._-`, start alphanumeric, ≤64 chars — a unicode/spaced stem is a clear 422 naming the rule; **never overwrites** — a collision is 409), so the kept `eval_set.jsonl` is usable later via `DATASET=<name> make eval`; (2) ingests its members **additively** (upsert per id — added on top of existing members, no reseed; `member_id` is opaque, any ID space), **skipping any buggy *later* member row** (returned in a `skipped` list, so one malformed bundle never aborts the batch) — **but** a bundle whose members *all* skip ingests nobody, so it 422s with the per-row detail and rolls the reserved folder back rather than persisting an empty dataset; (3) **auto-scans** the ingested members so their Observations match their live Trajectory at once. A bad/unreadable zip is likewise a 422, never a 500. | `multipart file [+ name]` → `{dataset, members, results, ranges, member_ids, files, scanned, skipped}` |
| `DELETE /members/{id}` | Explicit clear of one member — opt-in, since ingest never clears by default. | → `{deleted}` |
| `POST /members/{id}/ask` | One grounded answer; runs the per-turn pipeline. | `AskRequest{message, history[]}` → `HealthIntelligenceResponse` |
| `POST /members/{id}/scan` | Run the proactive scan now — a manual **re-run** of the same idempotent scan every ingest path already ends in (seed / reseed / upload), kept for demonstrating the mechanism and for refreshing after a `/feedback` override; returns the observations raised. | → `Observation[]` |
| `POST /members/{id}/feedback` | Record a correction (clinician override / member preference) or a signal (helpful / incorrect). | `Feedback` → `{feedback_id}` |
| `GET /members/{id}/observations` | Read the member's current observations. Served by the `pipeline.observations` projection: the persisted findings with each `member_explanation` **derived at read** (re-`analyze()` + `templates`, never stored — the read-time-derive peer of `/suggestions`/`/trajectory`); an unscanned member returns `[]`. | → `Observation[]` |
| `GET /members/{id}/trajectory` | Full per-marker series for inspection — projects `lab_results` + the analysis pass into `{marker, unit, readings[], trend, flags, reference_range}` (optional `?marker=`). A UI/operator read, rendered as a **minimal inline sparkline** (readings + Theil–Sen line + flagged points, kept rough) for human verification of a finding; the **LLM never receives the raw series**, only the collapsed `TrajectoryAnalysis` verdict (§4). | `?marker` → `[{marker, readings[], trend, flags}]` |
| `GET /members/{id}/suggestions` | **Mode 1**: data-derived preset prompts + pre-computed answers, no LLM; optional `focus` + `asked` drive the conversation **loop** — the next chips after each answer (§2). | `?focus,asked` → `SuggestedPrompt[]` |
| `GET /escalations` | **The clinician-review queue** — every open escalation across **all members**, ranked worst-first (`urgent` → `clinician_review`) then most-recent. The cross-member triage worklist and *the* hand-off surface (escalation = "make sure a human sees this", which a per-member read can't guarantee). Tenant-wide here (no clinician identity); a `clinician_id`/panel scope in prod (§15). | → `Escalation[]` (each carries `member_id`) |
| `GET /members/{id}/escalations` | Per-member **drill-in** — one member's standing escalation record (oldest-first), shown when already viewing that member. The detail view, **not** the queue (that is `GET /escalations`). | → `Escalation[]` |
| `POST /learn` | Prompt-scan: **rule-assemble** a candidate prompt from the `feedback` set, gate it through the deterministic harness, promote or reject. Exposed but **guarded** (single-flight lock · `feedback`-set debounce · structural pre-check · daily cap) so it can't drain credits. | → `{version, status, report}` |
| `POST /reset` | Erase learning → v0: deactivate feedback, revert the prompt to baseline. | → `{learning_reset}` |
| `POST /admin/reseed` | **Factory reset** (demo hygiene; distinct from `/reset`): truncate all tables, restore baseline `prompt_version` v0, re-ingest the `training_data` bundle only — back to the initial 15-member state, uploaded holdouts dropped — then **auto-scan** the restored members (every ingest path ends in a scan, so Observations are populated when the call returns; the scan runs AFTER the atomic reseed transaction commits, best-effort per member). An operator op, not a member feature. | → `{reseeded, scanned, …counts}` |
| `GET /health` | Liveness. | → `{status}` |

**Not a route (by design):** the **eval harness** imports the pipeline and runs the labeled set headlessly — the same harness `/learn` calls to gate a prompt candidate. The loader CLI (`python -m preprocessing.ingest <bundle>`) stays for local seeding — it too ends in the auto-scan (every ingest path does; it is the `make seed` behind `make run`, so the local one-command boot lands with observations populated); `POST /members` is its live equivalent for a held-out dataset.

---

## 14. Repository layout

`preprocessing/` (bundle → SQLite, the ingestion phase) and `health_intelligence/` (SQLite → answer, the serving library) are siblings under `backend/`; the serving core imports no web framework. `backend/` (Python) and `frontend/` (a single static page FastAPI serves) are separate trees.

```
health-intelligence/
├── README.md                    # one-command local run; AI-tools note
├── Makefile                     # init-db · seed · run · eval · test
├── backend/
│   ├── pyproject.toml
│   ├── uv.lock                  # uv-pinned dependency lockfile
│   ├── .env.example             # secrets + runtime only (Anthropic key · DB path · optional LangSmith key)
│   ├── schema.sql
│   ├── api.py                   # FastAPI: members(list · ingest) · ask · scan · feedback · observations · suggestions(Mode 1) · escalations · learn · reset · health + serves static UI
│   ├── health_intelligence/
│   │   ├── __init__.py
│   │   ├── config.py            # model pin · thresholds (α · FDR q · n_min · band cutpoints) · per-marker clinical constants (CVa/CVi · adverse direction · panic thresholds · Vitamin-D bands · vital bounds) · config_version
│   │   ├── models.py            # Pydantic contracts
│   │   ├── db.py                # SQLite: connect · init_db · queries · idempotent escalation emit · resolves feedback overrides into analysis inputs
│   │   ├── analysis.py          # MK(τ,exact-p) · Theil–Sen(+CI) · RCV · range/panic · band · FDR → TrajectoryAnalysis
│   │   ├── gate.py              # LLM classifier → Pydantic route + message floor (temp 0, few-shot)
│   │   ├── safety.py            # data floor + output validator (escalation ≥ floor)
│   │   ├── templates.py         # deterministic responders: seek-care · crisis-support · refusal · analysis→HealthIntelligenceResponse templating (observation narration + Mode 1 answers) · suggest_prompts (Mode 1 presets)
│   │   ├── llm.py               # compose() · provider (gate calls through here too) · prompts
│   │   ├── pipeline.py          # ask (gate · route · compose/template · validate · escalate) · scan (proactive)
│   │   └── learn.py             # prompt-scan: draft candidate from feedback → harness gate → promote/reject
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── datasets.py          # resolve the active dataset sub-folder of data/ (DATASET env · default training_data)
│   │   └── ingest.py            # load JSON bundle → validate → write SQLite  (python -m preprocessing.ingest)
│   ├── eval/                    # the evaluation harness (python -m eval)
│   │   ├── harness.py           # run_eval: drive the live service per case → score → Report
│   │   ├── scorers.py           # deterministic scorers, pure over (case, responses)
│   │   ├── client.py            # ServiceClient over TestClient; per-case DB isolation
│   │   ├── inprocess.py         # InProcessClient: same scorers, driving pipeline directly (the /learn gate — no TestClient/global patch)
│   │   ├── adapter.py           # supplied eval_set.jsonl → Case; escalation/category normalization
│   │   ├── added_cases.py       # tagged gate cases (crisis · acute · oblique)
│   │   ├── stats_fixtures.py    # known-answer trend fixtures for score_stats
│   │   ├── report.py            # Report.to_markdown() / .to_json() (never-events first; persists raw runs)
│   │   ├── llm_eval.py          # gated LangSmith trace sink (LANGSMITH_API_KEY; the ONLY langsmith importer)
│   │   ├── judge_eval.py        # feedback input-judge accuracy gate (make eval-judge; real Haiku; held-out battery)
│   │   ├── types.py             # Case · ScorerResult · EvalConfig (eval-local)
│   │   └── __main__.py          # CLI entry (make eval) → eval/reports/ (+ folds in the judge gate)
│   ├── data/                    # datasets — one bundle per sub-folder (training_data/ ships; DATASET selects); derived health.db sits at this root
│   └── tests/                   # test_analysis · test_pipeline · test_api · … · test_feedback · test_learn · test_trajectory (Phase 7)
└── frontend/
    └── index.html               # single static page (vanilla JS) — FastAPI serves it; no build, no Node
```

Three boundaries this keeps load-bearing: `ingest()` is the single normalization path; `analysis.py` is pure functions over typed inputs (no DB, no LLM) so stat correctness is unit-testable; the three provider calls (`compose`, the `gate` classifier, and the feedback input-judge `learn.judge_corrected_answer`) are isolated behind `llm.py`.

---

## 15. Running and deployment

Two ways to run it: locally, which is the primary path, and a single hosted service for a shareable demo.

**Local run — Python-only, one command.** `git clone`; put `ANTHROPIC_API_KEY` in `backend/.env`; `make run` creates the venv, installs, inits + seeds the SQLite DB with the synthetic member, and starts FastAPI serving both the API and the static `index.html` on a single port. Open `localhost:8000` — no Node, no second process, no CORS. The UI is one static file (vanilla JS); `uv` makes setup faster, and a CLI chat loop is the honest fallback if even the page is more than time allows.

**Hosted deploy — online, one place.** A single Render Web Service, deployed from a committed **`render.yaml`** Blueprint: FastAPI serves the API *and* the static page same-origin. Render's build step runs on separate compute that can't see the runtime filesystem, so `init_db` **and a seed-if-empty** run idempotently on app startup rather than at build time, followed by the ingest-path auto-scan — a fresh instance self-heals its schema, the 15 training members, *and* their observations (reusing the idempotent `ingest_dataset`, the same loader `/admin/reseed` calls, then `pipeline.scan_members`; findings are on the page from the first load, no manual scan). **As shipped this targets the free tier, where the store is ephemeral:** no persistent disk, and the instance spins down when idle, so the SQLite file is wiped on a cold start and re-seeded from `training_data` — meaning `POST /members` uploads and `feedback` don't survive a spin-down (a documented demo trade-off, not a failure). Durability is a one-line `render.yaml` upgrade: the paid `starter` plan with a persistent disk mounted at `/data` and `HEALTH_DB_PATH` pointed at it (`/data/health.db`), so live ingest then survives deploys. That persists everything in the DB (uploaded **members** + feedback + **promoted `prompt_versions`**); the dataset *folders* `POST /members/upload` writes (the kept `eval_set.jsonl`/`lab_panels.csv`) live under the datasets root, ephemeral unless `HEALTH_DATA_ROOT` is also pointed at the disk. One durable-disk consequence: a promoted learned prompt is a *frozen* snapshot, and the composer's emergency-contact copy is interpolated into it, so a `config.py` emergency-number change (a safety change) requires `POST /reset` or `/admin/reseed` afterward to drop any stale promoted `vN` — the deterministic `seek_care`/`crisis` templates read the numbers live and never go stale, only the composer prompt does (moot on the free tier, which re-seeds each cold start) — and a relocated empty root needs the shipped `training_data` copied in once, since the startup seed reads it from there (`render.yaml` documents this). One repo, one service, one URL; the local run stays primary — a clean local run takes priority over a hosted deploy.

**One engine, both places — SQLite.** The same database runs locally and in production; the dev/prod engine split is avoided deliberately. A split would put an untested gap on exactly the seams that differ between engines — type strictness, date handling, the idempotent-insert form, write-concurrency semantics — and, worse, would let the eval harness (§8) certify a stack the product never ships. SQLite also fits the design on its own terms: per-member data is tiny and recomputed on read, so the store is a file, not a server. The cost of durability is explicit — a Render persistent disk is a paid tier (~$8/mo), and the free tier has no disk and spins down, which wipes the file, so `POST /members` uploads and `feedback` would not survive (a disk also pins the service to one instance with no zero-downtime deploy, both irrelevant at this scale). The service runs a **single worker** by choice, not constraint: the deterministic core is pure and every concurrency-sensitive write is an atomic idempotent operation on a UNIQUE key (`INSERT OR IGNORE`), so fire-once, idempotency, and floor enforcement hold under any worker count — one worker is simply the contention-free fit for a single-file store, and scaling out is a bounded seam (§11), not a rework. Portability is kept as a cheap seam rather than paid for now: `db.py` is the only module touching the store, its queries are hand-written, parameterized, and kept to standard SQL forms, and the one dialect-specific operation — the idempotent escalation insert (`INSERT OR IGNORE`) — maps directly to Postgres `ON CONFLICT DO NOTHING`. Moving to Postgres if scale ever demanded it is therefore a connection-string change and one insert, not a rewrite.

**Deliberately minimal.** The deploy config is a committed `render.yaml` Blueprint — the disk/env/command contract as version-controlled IaC — but **not** a Dockerfile (Render builds from `pyproject.toml` via `uv sync`; a container was never required) and no CI/CD (§11). Deployment is the additive layer, the first to be cut under time pressure, never the foundation.
