"""Pydantic contracts — the typed seams between ingestion, the core, the LLM layer, and the wire.

These are the *domain & wire* layer; ``schema.sql`` is the *persistence* layer. They are deliberately
NOT 1:1 (architecture §4): the computed types (``TrendResult``, ``MarkerTrajectory``,
``TrajectoryAnalysis``, ``HealthIntelligenceResponse``, ``SuggestedPrompt``) are never stored as their
own tables — a response persists as JSON in ``interactions.response_json`` — and the ``interactions`` /
``prompt_versions`` tables have no mirroring model here at all. The row<->model mapping is real work
and lives in ``db.py`` (Phase 2); this file is pure declaration.

Two conventions make the split legible:
  * Every ``Literal`` below mirrors a ``schema.sql`` CHECK constraint *exactly* — the two enumerations
    are kept in lockstep by hand and guarded by the consistency checks.
  * Server-assigned identifiers appear only on *read projections* (``Observation``, ``Escalation``,
    ``ResponseMetadata``). Input contracts (``MemberBundle``, ``Feedback``) and compute inputs
    (``LabResult``, ``Note``) omit them — ``db.py`` synthesizes the storage keys.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------------------------------
# Enumerations — each mirrors a schema.sql CHECK constraint VERBATIM. Change one, change both.
# --------------------------------------------------------------------------------------------------

MemberSex = Literal["female", "male", "other", "unknown"]  # members.sex
RangeSex = Literal["female", "male", "any"]  # reference_ranges.sex
Driver = Literal["ask", "scan", "suggested"]  # interactions.driver
AnswerDisposition = Literal[
    "answered", "refused", "out_of_scope"
]  # interactions.answer_disposition
FloorLevel = Literal[
    "none", "clinician_review", "urgent"
]  # the escalation/floor axis: interactions.escalation + overall_floor
Severity = Literal["info", "notable", "attention", "urgent"]  # observations.severity
EscalationKind = Literal["data_finding", "chat"]  # escalations.kind
EscalationLevel = Literal[
    "clinician_review", "urgent"
]  # escalations.level (no 'none' — only escalated rows exist)
FeedbackKind = Literal[  # feedback.kind
    "range_override",
    "suppress_marker",
    "preference",
    "helpful",
    "incorrect",
    "escalation_accept",
    "escalation_reject",
]
FeedbackSource = Literal["clinician", "member", "system"]  # feedback.source
PromptStatus = Literal[
    "proposed", "promoted", "rejected", "reverted"
]  # prompt_versions.status

# Computed-only enums (no table column — internal to the analysis):
TrendDirection = Literal[
    "increasing", "decreasing", "flat"
]  # "flat" = CI includes zero, can't sign
Flag = Literal[
    "below_range",
    "above_range",
    "panic_low",
    "panic_high",
    "band_cross",
    "no_reference",
]

# Ordinal rank for the two safety axes — the ONE place their order is defined, so no code ever
# string-compares these Literals. Lexicographically "clinician_review" < "none", so a raw
# `escalation >= floor` would INVERT the floor; the validator and the _floor / _severity projections
# (Phase 1/3a) MUST rank through these maps (e.g. max-severity = max(markers, key=lambda m: SEVERITY_ORDER[m.severity])).
FLOOR_ORDER: dict[FloorLevel, int] = {"none": 0, "clinician_review": 1, "urgent": 2}
SEVERITY_ORDER: dict[Severity, int] = {
    "info": 0,
    "notable": 1,
    "attention": 2,
    "urgent": 3,
}

# The ONE severity -> escalation-floor projection (architecture §2 D3 / the _floor rule). Both the
# core's whole-member floor (analysis._floor, the max over markers) and the per-marker escalation level
# (safety.severity_to_level) derive from this single table, so they cannot drift into a per-marker
# escalation that sits below the computed floor (a silent under-escalation). An out-of-range/band value
# is at most `notable` -> `none` here: escalation is reserved for `attention` (RCV+FDR-confirmed adverse
# trend) and `urgent` (panic).
SEVERITY_TO_FLOOR: dict[Severity, FloorLevel] = {
    "info": "none",
    "notable": "none",
    "attention": "clinician_review",
    "urgent": "urgent",
}


# --------------------------------------------------------------------------------------------------
# Ingest input layer — the bundle exactly as it arrives (POST /members and the loader CLI).
# `panels[]` is a transient PARSE shape: ingest flattens it to dated results keyed by panel_id and
# folds `vitals` in as markers; nothing downstream sees PanelInput again.
# --------------------------------------------------------------------------------------------------


class _IngestModel(BaseModel):
    """Base for the upload/ingest contracts: ``extra='forbid'`` makes a mistyped key a clear
    validation error rather than a silently dropped field — the upload doubles as the format check,
    and silently dropping e.g. ``medications``/``conditions`` could quietly change escalation framing."""

    model_config = ConfigDict(extra="forbid")


class VitalsInput(_IngestModel):
    """The three vitals every panel carries; folded into markers at ingest with config-supplied units."""

    systolic_bp: float
    diastolic_bp: float
    bmi: float


class ResultInput(_IngestModel):
    """One measured lab result, with its reference range as the source prints it (one of five string
    shapes). The raw string is kept verbatim and parsed only at ingest (Phase 2 firewall)."""

    analyte: str
    value: float
    unit: str
    reference_range: str


class PanelInput(_IngestModel):
    panel_id: str
    collected_date: str  # ISO date as printed, e.g. "2024-02-12"
    results: list[ResultInput]
    vitals: VitalsInput


class MemberProfile(_IngestModel):
    """Profile context for risk framing and narration. ``age`` is nullable (schema-aligned) and not
    load-bearing for analysis — ranges key on sex alone (no age bands in the data)."""

    member_id: str
    age: int | None = None
    sex: MemberSex
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    lifestyle: dict[str, str] = Field(
        default_factory=dict
    )  # {exercise, alcohol, smoking, sleep}


class Note(_IngestModel):
    """Free-text note with provenance. ``source`` is 'GP summary' (clinician) | 'in-app' | 'onboarding'
    (member-reported); date/source nullable per schema, text required."""

    date: str | None = None
    source: str | None = None
    text: str


class MemberBundle(_IngestModel):
    """The upload/ingest unit — one member's whole record. The format the holdout must match; the
    upload doubles as the format check, so a malformed bundle fails validation with a clear error."""

    member_id: str
    profile: MemberProfile
    panels: list[PanelInput]
    notes: list[Note] = Field(default_factory=list)

    @model_validator(mode="after")
    def _member_ids_agree(self) -> MemberBundle:
        # member_id is carried at the bundle root AND in the profile; a mismatch would cross-wire
        # storage (db.py keys on the root id) with narration (which reads profile.member_id).
        if self.profile.member_id != self.member_id:
            raise ValueError(
                f"member_id mismatch: bundle={self.member_id!r} profile={self.profile.member_id!r}"
            )
        return self


class AskRequest(BaseModel):
    """The thin wire body for POST /members/{id}/ask."""

    message: str


# --------------------------------------------------------------------------------------------------
# Domain layer — post-ingest, the typed inputs the pure core consumes (db.py hands these to analyze).
# Storage PKs (result_id, note_id, member_id FKs) live in the row mapping, not here.
# --------------------------------------------------------------------------------------------------


class LabResult(BaseModel):
    """One reading in canonical form (analyte->marker, vitals unified in). Units are consistent per
    marker, so no conversion ever happens downstream."""

    marker: str
    value: float
    unit: str
    panel_id: str  # the draw's identity: results sharing it are one panel
    panel_date: str  # orders a marker's trajectory


class ReferenceRange(BaseModel):
    """Marker bounds — normal (ref_low/ref_high) parsed from the data, panic curated in config —
    versioned. One-sided forms leave the absent bound ``None``."""

    marker: str
    sex: RangeSex
    unit: str
    ref_low: float | None = None
    ref_high: float | None = None
    panic_low: float | None = None
    panic_high: float | None = None
    config_version: str


# --------------------------------------------------------------------------------------------------
# Computed layer — produced by analysis.py (Phase 1), held in memory, never their own tables. The
# LLM consumes these as GROUND TRUTH it may not recompute.
# --------------------------------------------------------------------------------------------------


class Reading(BaseModel):
    value: float
    date: str


class TrendResult(BaseModel):
    """Mann-Kendall (monotonic trend) + Theil-Sen (direction & rate). ``significant`` is true only
    after clearing alpha AND the cross-marker FDR pass. ``slope`` is ``None`` when too noisy to sign."""

    method: Literal["mann_kendall"] = "mann_kendall"
    direction: TrendDirection
    tau: float  # Kendall's tau in [-1, 1]
    p_value: float  # exact small-sample MK p-value
    slope: float | None = None  # Theil-Sen slope (per unit time)
    slope_ci: tuple[float, float] | None = (
        None  # distribution-free CI; signs direction iff it excludes 0
    )
    n: int  # readings the trend was computed over
    significant: bool = False


class ClinicalChange(BaseModel):
    """Reference Change Value verdict — did the net change clear the marker's own analytical+biological
    noise? ``exceeds_rcv`` is ``None`` when CVa/CVi are absent (the typed-absence skip-path)."""

    rcv: float | None = None
    net_change: float | None = None
    exceeds_rcv: bool | None = None


class MarkerTrajectory(BaseModel):
    """Per-marker assembled verdict. ``trend`` is ``None`` below n_min; ``severity`` is direction-aware
    and a panic flag pins it to 'urgent'."""

    marker: str
    unit: str
    latest: Reading
    trend: TrendResult | None = None
    clinical_change: ClinicalChange | None = None
    flags: list[Flag] = Field(default_factory=list)
    severity: Severity
    n_readings: int | None = (
        None  # series length — the only count available when trend is None (sparse)
    )


class TrajectoryAnalysis(BaseModel):
    """The whole output of the deterministic core for one member. ``overall_floor`` is a pure
    projection of the max per-marker severity onto the escalation axis — the validator enforces it
    every turn and the scan keys data_finding escalations on it. Nothing downstream may lower it."""

    member_id: str
    data_version: str
    overall_floor: FloorLevel
    markers: list[MarkerTrajectory] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# Response / durable layer — the structured answer (persisted as interactions.response_json) and the
# read projections of the proactive findings and the clinician-review queue.
# --------------------------------------------------------------------------------------------------


class Evidence(BaseModel):
    """One load-bearing claim's provenance — shown verbatim because the numbers came from code.
    Range fields mirror how the panel reported it; ``stat`` names the statistic the claim rests on."""

    marker: str
    value: float
    unit: str
    date: str
    ref_low: float | None = None
    ref_high: float | None = None
    stat: str | None = None


class Finding(BaseModel):
    """A single claim with a stable id (so later feedback can attach to it) and its backing evidence."""

    finding_id: str
    text: str
    evidence: list[Evidence] = Field(default_factory=list)


class ResponseMetadata(BaseModel):
    """The reproducibility tuple + cost/latency instrumentation stamped on every response."""

    response_id: str
    data_version: str
    model_version: str
    config_version: str
    prompt_version: int = 0
    latency_ms: int | None = None
    tokens: int | None = None
    cost_usd: float | None = None


class HealthIntelligenceResponse(BaseModel):
    """The validated assistant message — two orthogonal axes: ``answer_disposition`` (the model's call
    on the question) and ``escalation`` (deterministic; the validator enforces it >= the floor, never
    below). The same contract is emitted by Mode 1 templates and Mode 2 compose."""

    answer: str
    findings: list[Finding] = Field(default_factory=list)
    uncertainty: str | None = None
    answer_disposition: AnswerDisposition
    escalation: FloorLevel = "none"
    metadata: ResponseMetadata


class Observation(BaseModel):
    """A proactive finding (read projection of the observations table). Narration/evidence live on the
    linked interaction via ``response_id``.

    Two deterministic narrations, split by AUDIENCE: ``trigger_reason``
    is the clinician/operator explainability — the statistical signal that fired (e.g. "Mann-Kendall
    p=0.017, n=5; clears reference-change value"), PERSISTED and surfaced only on the operator console /
    clinician queue; ``member_explanation`` is the member-facing plain-language "what this means for
    you", never containing test statistics and stating the reference range for an out-of-range value.
    Both come from ``templates`` off the same ``_classify`` signal, so they always describe the same
    finding; the member panel renders ``member_explanation``, never ``trigger_reason``.

    ``member_explanation`` is NOT persisted — it defaults to ``""`` and is re-derived at read by the
    ``/observations`` projection (``pipeline.observations``) from the live analysis, so there is no
    stored copy to migrate or to go stale. A bare ``db.get_observations`` row therefore carries ``""``;
    the member-facing path always fills it."""

    observation_id: str
    member_id: str
    response_id: str
    severity: Severity
    title: str
    trigger_reason: str
    member_explanation: str = (
        ""  # derived at the /observations projection, never stored (see above)
    )
    data_version: str


class Escalation(BaseModel):
    """A clinician-review-queue artifact (read projection). ``observation_id`` is set for data findings,
    ``interaction_id`` for chat; ``dedup_key`` is the UNIQUE idempotency key that makes 'fire once' a
    DB guarantee."""

    escalation_id: str
    member_id: str
    kind: EscalationKind
    dedup_key: str
    level: EscalationLevel
    observation_id: str | None = None
    interaction_id: str | None = None
    trigger_reason: str
    created_at: str


# --------------------------------------------------------------------------------------------------
# Learning layer (input) — the correction/signal the member or a clinician submits. Storage fields
# (feedback_id, active, created_at) are assigned by db.py; member_id comes from the route path.
# --------------------------------------------------------------------------------------------------


class Feedback(BaseModel):
    """A correction (range_override / suppress_marker / preference) or a signal (helpful / incorrect /
    escalation_accept|reject). ``target`` is a marker (overrides) or finding_id (signals)."""

    kind: FeedbackKind
    target: str | None = None
    payload: dict[str, Any] | None = None
    source: FeedbackSource


# --------------------------------------------------------------------------------------------------
# Mode 1 unit — a preset prompt bound to its pre-computed answer (no model call). The conversation
# loop is a walk over these.
# --------------------------------------------------------------------------------------------------


class SuggestedPrompt(BaseModel):
    prompt: str
    response: HealthIntelligenceResponse


# --------------------------------------------------------------------------------------------------
# Mode 2 LLM layer (Phase 4) — computed seams, NOT schema tables. ``ComposeDraft`` is the *only* thing
# the composer LLM is allowed to emit: prose + which markers it relied on. The deterministic core then
# attaches the real ``Evidence`` (numbers come from code, never the model) and sets ``escalation`` to the
# floor, lifting the draft into a full ``HealthIntelligenceResponse``. Keeping the model's surface this
# narrow is what makes "never assert a value the deterministic layer did not produce" (CLAUDE.md's one
# law) structurally true for the evidence chips — the model can choose *which* markers are relevant and
# *how* to phrase the answer, but it cannot mint a number or touch the safety floor.
# --------------------------------------------------------------------------------------------------


class ComposeDraft(BaseModel):
    """The composer LLM's structured output (Mode 2). ``cited_markers`` are CANONICAL marker keys (e.g.
    "HbA1c", "systolic_bp") the answer drew on; ``pipeline`` resolves each to a code-built ``Finding`` +
    ``Evidence`` from the ``TrajectoryAnalysis`` and drops any key not in the member's data (a fabricated
    marker can never produce an evidence chip). ``escalation`` is deliberately absent — it is the
    deterministic floor's to set, never the model's."""

    answer: str
    uncertainty: str | None = None
    answer_disposition: AnswerDisposition
    cited_markers: list[str] = Field(default_factory=list)
