-- Health Intelligence Service — SQLite schema (synthetic data only)
-- One file; semantic stores noted in comments:
--   member_data : source of truth (members, lab_results, notes) — written only by ingestion
--   reference   : marker bounds (parsed from supplied data) + safety thresholds (curated config) — versioned
--   durable     : interactions (audit/answer trace) + observations (proactive findings) + escalations (clinician-review queue)
--   learning    : feedback (overrides + signals) + prompt_versions — drives self-improvement
-- Metrics are recomputed on read (the per-member data is tiny); nothing is cached.

PRAGMA foreign_keys = ON;

-- ---- member_data: source of truth (written only by ingestion) -------------

CREATE TABLE members (
    member_id           TEXT PRIMARY KEY,
    sex                 TEXT NOT NULL CHECK (sex IN ('female','male','other','unknown')),
    age                 INTEGER,                -- profile context (risk framing); ranges key on sex alone (no age bands in data)
    conditions_json     TEXT,                   -- small context lists; JSON until they become query keys
    medications_json    TEXT,
    family_history_json TEXT,                   -- e.g. "father: type 2 diabetes" — context for narration
    lifestyle_json      TEXT
);

-- One row per measure reading. No separate panels table: panel_id is the draw's identity —
-- results sharing it are one panel (panel_date is that draw's date, and orders a marker's
-- trajectory across panels) — which keeps panels groupable and source-traceable without a table
-- or a join the two behaviors never use. Vitals (systolic_bp, diastolic_bp, bmi) are stored here
-- as markers too — labs and vitals are one unified measure model, so trend detection covers both;
-- their reference bounds (and units) come from config (the supplied data prints ranges only for labs).
-- value is REAL because the markers here are numeric. A censored or qualitative result
-- ('<5', 'positive') is normalized to a number at ingestion or dropped; a value_text fallback is
-- a deliberate non-feature — one column to add only if such markers appear.
CREATE TABLE lab_results (
    result_id  TEXT PRIMARY KEY,
    member_id  TEXT NOT NULL REFERENCES members(member_id),
    panel_id   TEXT NOT NULL,                  -- the draw's identity: results sharing it are one panel
    marker     TEXT NOT NULL,                  -- canonical key (analyte or vital; normalized at ingestion)
    value      REAL NOT NULL,                  -- numeric only (see note above)
    unit       TEXT NOT NULL,                  -- canonical unit
    panel_date TEXT NOT NULL                   -- the draw's date (human-facing; orders the trajectory)
);
CREATE INDEX idx_results_member_marker ON lab_results(member_id, marker, panel_date);

CREATE TABLE notes (
    note_id    TEXT PRIMARY KEY,
    member_id  TEXT NOT NULL REFERENCES members(member_id),
    note_date  TEXT,
    source     TEXT,                           -- provenance as supplied: 'GP summary' (clinician) | 'in-app'/'onboarding' (member-reported)
    text       TEXT NOT NULL                   -- free-text note
);
CREATE INDEX idx_notes_member ON notes(member_id);

-- ---- reference: marker bounds + safety thresholds -------------------------
-- Normal bounds (ref_low/ref_high) are PARSED from the ranges the supplied data prints per result
-- (constant per marker+sex; nullable for one-sided forms like '<5.7' or '>=90'). Panic thresholds
-- are CURATED in config — the supplied data carries none — as are vital bounds and the graded
-- Vitamin D bands (sufficient/insufficient/deficient). Keyed on sex only; no age bands in the data.
CREATE TABLE reference_ranges (
    range_id       TEXT PRIMARY KEY,
    marker         TEXT NOT NULL,
    sex            TEXT NOT NULL CHECK (sex IN ('female','male','any')),
    unit           TEXT NOT NULL,
    ref_low        REAL,                        -- null = one-sided (e.g. '<5.7' has no lower bound)
    ref_high       REAL,                        -- null = one-sided (e.g. '>=90' has no upper bound)
    panic_low      REAL,                        -- safety-critical thresholds — curated in config, not in the data
    panic_high     REAL,
    config_version TEXT NOT NULL                -- bounds + thresholds are versioned + auditable
);
CREATE INDEX idx_ranges_lookup ON reference_ranges(marker, sex);

-- ---- durable: audit trace + proactive observations ------------------------

-- One row per produced response — /ask answers, /scan observation narrations, and (optional)
-- deterministic 'suggested' fast-path answers.
-- The audit/trace backbone. The full structured response (answer, findings, embedded
-- evidence snapshots, uncertainty, and the two-axis disposition: answer_disposition +
-- escalation) is stored as response_json, so it stays self-contained and auditable.
CREATE TABLE interactions (
    response_id     TEXT PRIMARY KEY,
    member_id       TEXT NOT NULL REFERENCES members(member_id),
    driver          TEXT NOT NULL CHECK (driver IN ('ask','scan','suggested')),
    question        TEXT,                       -- null for proactive
    response_json   TEXT NOT NULL,              -- serialized HealthIntelligenceResponse
    answer_disposition TEXT NOT NULL            -- the model's call on the question (ungated by floor)
                    CHECK (answer_disposition IN ('answered','refused','out_of_scope')),
    escalation      TEXT NOT NULL DEFAULT 'none' -- deterministic floor; validator enforces >= computed floor
                    CHECK (escalation IN ('none','clinician_review','urgent')),
    data_version    TEXT NOT NULL,              -- member-data snapshot this response stood on
    model_version   TEXT NOT NULL,
    config_version  TEXT NOT NULL,              -- threshold set in force (reproducible escalation)
    prompt_version  INTEGER NOT NULL DEFAULT 0, -- prompt_versions.version in force (reproducibility)
    latency_ms      INTEGER,
    tokens          INTEGER,
    cost_usd        REAL,
    created_at      TEXT NOT NULL
);
CREATE INDEX idx_interactions_member ON interactions(member_id, created_at);

-- Proactive observations: severity + narrated summary + which signal fired + the snapshot
-- it stood on. Narration/evidence live via response_id (the interactions row above).
-- Re-scanning a member replaces its observations for the new data_version, so identical
-- inputs yield identical observations. trigger_reason is the clinician/operator explainability (the
-- statistical signal — p-values, RCV), persisted here. The MEMBER-facing plain-language explanation
-- ("what this means for you", no statistics, reference range stated for out-of-range values) is NOT
-- stored: it is a deterministic template (templates.observation_member_explanation) re-derived at read
-- by the /observations projection (pipeline.observations), from the same value+range it states — so
-- there is no stored copy to migrate or go stale.
CREATE TABLE observations (
    observation_id     TEXT PRIMARY KEY,
    member_id          TEXT NOT NULL REFERENCES members(member_id),
    response_id        TEXT NOT NULL REFERENCES interactions(response_id),
    severity           TEXT NOT NULL CHECK (severity IN ('info','notable','attention','urgent')),
    title              TEXT NOT NULL,           -- narrated summary (member-facing headline)
    trigger_reason     TEXT NOT NULL,           -- which statistical signal fired (CLINICIAN/operator audience)
    data_version       TEXT NOT NULL
);
CREATE INDEX idx_obs_member ON observations(member_id, severity);

-- Clinician-review queue: the subset of findings/events that crossed the escalation
-- threshold, from BOTH owners. dedup_key is UNIQUE, so "fire once" is a DB guarantee, not
-- application logic. Data-finding key = member·marker·marker_version (scan owns it, writes via
-- INSERT OR IGNORE), where marker_version hashes ONLY that marker's analyze() inputs (its readings,
-- its resolved range, sex, age) — not notes/profile, and not other markers — so a notes-only edit
-- (or a /feedback override to a DIFFERENT marker) cannot re-fire a finding whose own analysis never
-- moved (architecture §48; db.compute_marker_version); chat key = member·day
-- (assistant owns it — one open escalation per member per day; there is no conversation concept
-- in v1). Member-facing care is never
-- deduped; only this record is. observation_id is set for data findings, interaction_id for
-- chat (the triggering turn). Not every observation escalates — only the ones recorded here.
-- chat covers both acute-medical and crisis at this stage (not distinguished structurally —
-- the difference lives in trigger_reason); a route sub-type is a near-term refinement.
CREATE TABLE escalations (
    escalation_id  TEXT PRIMARY KEY,
    member_id      TEXT NOT NULL REFERENCES members(member_id),
    kind           TEXT NOT NULL CHECK (kind IN ('data_finding','chat')),
    dedup_key      TEXT NOT NULL UNIQUE,         -- idempotency: one per finding / one per member per day
    level          TEXT NOT NULL CHECK (level IN ('clinician_review','urgent')),
    observation_id TEXT REFERENCES observations(observation_id),  -- set when kind='data_finding'
    interaction_id TEXT REFERENCES interactions(response_id),     -- set when kind='chat' (triggering turn)
    trigger_reason TEXT NOT NULL,                -- human-readable why (marker drift / message flag)
    created_at     TEXT NOT NULL,
    -- lifecycle status. 'open' = on the active clinician queue; 'superseded' = a re-scan found the
    -- underlying finding cleared (e.g. a /feedback override removed the flag), so it leaves the active
    -- queue but the row is KEPT for audit (never deleted). Set by the scan's reconcile (db.reconcile_
    -- escalation_status), symmetric (re-opens if the finding returns). The global queue shows only 'open';
    -- an 'urgent' escalation is NEVER superseded (only the softer 'clinician_review' tier auto-clears): a
    -- deliberate range_override panic re-bound can clear the flag, but a fired urgent stays queued until a
    -- human resolves it (suppress itself is panic-inert). §720 lifecycle.
    -- DELIBERATE Literal/CHECK asymmetry: the model's third status, 'acknowledged' (a clinician's active
    -- escalation_accept feedback row targets the escalation), is a READ-TIME overlay over a stored 'open'
    -- (db._ESCALATION_COLUMNS) and is NEVER stored — so this CHECK stays two-valued (no rebuild migration:
    -- SQLite CHECKs are baked into the DDL of every existing DB), /reset reverts it by deactivating the
    -- accept row, and the scan's reconcile needs no knowledge of acknowledgments.
    -- Appended LAST to match the ALTER TABLE ADD COLUMN migration's column order for pre-existing DBs.
    status         TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','superseded'))
);
CREATE INDEX idx_escalations_member ON escalations(member_id, created_at);

-- ---- learning: feedback signals + prompt versions (self-improvement) -------

-- Feedback drives self-improvement. ONE kind-typed store, consumed two ways:
--   deterministic overrides (range_override, suppress_marker = analysis inputs; preference = a compose
--   hint) are resolved by db.py — they change what the bot KNOWS, not its rules;
--   signals (helpful, incorrect, escalation_accept/reject) feed the prompt-scan and the evals.
-- target = marker (for overrides) or finding_id (for signals on a specific output).
-- active=0 is how /reset wipes learning without losing the trail. The ANALYSIS overrides
-- (range_override/suppress_marker) are honored ONLY from clinician/system sources — a member must not be
-- able to change what is flagged (ui-ux §7); a member-sourced one is stored but inert for analysis. Among
-- the honored rows the latest active per marker wins (clinician outranking system), so source drives
-- precedence + provenance. (preference hints accept any source — they shape tone only, never the floor.)
CREATE TABLE feedback (
    feedback_id  TEXT PRIMARY KEY,
    member_id    TEXT NOT NULL REFERENCES members(member_id),
    kind         TEXT NOT NULL CHECK (kind IN (
                     'range_override','suppress_marker','preference',
                     'helpful','incorrect','escalation_accept','escalation_reject')),
    target       TEXT,                          -- marker (overrides) or finding_id (signals)
    payload_json TEXT,                          -- override content / signal detail
    source       TEXT NOT NULL CHECK (source IN ('clinician','member','system')),
    active       INTEGER NOT NULL DEFAULT 1,    -- /reset sets 0; queries filter active=1
    created_at   TEXT NOT NULL                  -- recency: latest active override per target wins
);
CREATE INDEX idx_feedback_member ON feedback(member_id, kind, active);

-- Versioned prompts. The prompt-scan (/learn) drafts a candidate from accumulated feedback;
-- the eval harness gates it IN THE PROACTIVE RUN — promote iff zero never-events AND safety
-- metrics >= current AND no dimension regresses, else status='rejected' (kept with its report).
-- The composer loads the latest status='promoted'; /reset reverts to the v0 baseline.
-- interactions.prompt_version stamps which prompt produced each response (reproducibility).
CREATE TABLE prompt_versions (
    version          INTEGER PRIMARY KEY,        -- ordering + the stamp logged on interactions
    prompt_text      TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('proposed','promoted','rejected','reverted')),
    eval_report_json TEXT,                        -- the harness result that gated this candidate
    created_at       TEXT NOT NULL
);
