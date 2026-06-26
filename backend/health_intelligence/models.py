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

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------------------------------
# Enumerations — each mirrors a schema.sql CHECK constraint VERBATIM. Change one, change both.
# --------------------------------------------------------------------------------------------------

MemberSex = Literal["female", "male", "other", "unknown"]      # members.sex
RangeSex = Literal["female", "male", "any"]                    # reference_ranges.sex
Driver = Literal["ask", "scan", "suggested"]                   # interactions.driver
AnswerDisposition = Literal["answered", "refused", "out_of_scope"]  # interactions.answer_disposition
FloorLevel = Literal["none", "clinician_review", "urgent"]     # the escalation/floor axis: interactions.escalation + overall_floor
Severity = Literal["info", "notable", "attention", "urgent"]   # observations.severity
EscalationKind = Literal["data_finding", "chat"]               # escalations.kind
EscalationLevel = Literal["clinician_review", "urgent"]        # escalations.level (no 'none' — only escalated rows exist)
FeedbackKind = Literal[                                        # feedback.kind
    "range_override", "suppress_marker", "preference",
    "helpful", "incorrect", "escalation_accept", "escalation_reject",
]
FeedbackSource = Literal["clinician", "member", "system"]      # feedback.source
PromptStatus = Literal["proposed", "promoted", "rejected", "reverted"]  # prompt_versions.status

# Computed-only enums (no table column — internal to the analysis):
TrendDirection = Literal["increasing", "decreasing", "flat"]  # "flat" = CI includes zero, can't sign
Flag = Literal["below_range", "above_range", "panic_low", "panic_high", "band_cross", "no_reference"]

# Ordinal rank for the two safety axes — the ONE place their order is defined, so no code ever
# string-compares these Literals. Lexicographically "clinician_review" < "none", so a raw
# `escalation >= floor` would INVERT the floor; the validator and the _floor / _severity projections
# (Phase 1/3a) MUST rank through these maps (e.g. max-severity = max(markers, key=lambda m: SEVERITY_ORDER[m.severity])).
FLOOR_ORDER: dict[FloorLevel, int] = {"none": 0, "clinician_review": 1, "urgent": 2}
SEVERITY_ORDER: dict[Severity, int] = {"info": 0, "notable": 1, "attention": 2, "urgent": 3}


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
    age: Optional[int] = None
    sex: MemberSex
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    lifestyle: dict[str, str] = Field(default_factory=dict)  # {exercise, alcohol, smoking, sleep}


class Note(_IngestModel):
    """Free-text note with provenance. ``source`` is 'GP summary' (clinician) | 'in-app' | 'onboarding'
    (member-reported); date/source nullable per schema, text required."""

    date: Optional[str] = None
    source: Optional[str] = None
    text: str


class MemberBundle(_IngestModel):
    """The upload/ingest unit — one member's whole record. The format the holdout must match; the
    upload doubles as the format check, so a malformed bundle fails validation with a clear error."""

    member_id: str
    profile: MemberProfile
    panels: list[PanelInput]
    notes: list[Note] = Field(default_factory=list)

    @model_validator(mode="after")
    def _member_ids_agree(self) -> "MemberBundle":
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
    panel_id: str   # the draw's identity: results sharing it are one panel
    panel_date: str  # orders a marker's trajectory


class ReferenceRange(BaseModel):
    """Marker bounds — normal (ref_low/ref_high) parsed from the data, panic curated in config —
    versioned. One-sided forms leave the absent bound ``None``."""

    marker: str
    sex: RangeSex
    unit: str
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    panic_low: Optional[float] = None
    panic_high: Optional[float] = None
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
    tau: float                                       # Kendall's tau in [-1, 1]
    p_value: float                                   # exact small-sample MK p-value
    slope: Optional[float] = None                    # Theil-Sen slope (per unit time)
    slope_ci: Optional[tuple[float, float]] = None   # distribution-free CI; signs direction iff it excludes 0
    n: int                                           # readings the trend was computed over
    significant: bool = False


class ClinicalChange(BaseModel):
    """Reference Change Value verdict — did the net change clear the marker's own analytical+biological
    noise? ``exceeds_rcv`` is ``None`` when CVa/CVi are absent (the typed-absence skip-path)."""

    rcv: Optional[float] = None
    net_change: Optional[float] = None
    exceeds_rcv: Optional[bool] = None


class MarkerTrajectory(BaseModel):
    """Per-marker assembled verdict. ``trend`` is ``None`` below n_min; ``severity`` is direction-aware
    and a panic flag pins it to 'urgent'."""

    marker: str
    unit: str
    latest: Reading
    trend: Optional[TrendResult] = None
    clinical_change: Optional[ClinicalChange] = None
    flags: list[Flag] = Field(default_factory=list)
    severity: Severity


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
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    stat: Optional[str] = None


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
    latency_ms: Optional[int] = None
    tokens: Optional[int] = None
    cost_usd: Optional[float] = None


class HealthIntelligenceResponse(BaseModel):
    """The validated assistant message — two orthogonal axes: ``answer_disposition`` (the model's call
    on the question) and ``escalation`` (deterministic; the validator enforces it >= the floor, never
    below). The same contract is emitted by Mode 1 templates and Mode 2 compose."""

    answer: str
    findings: list[Finding] = Field(default_factory=list)
    uncertainty: Optional[str] = None
    answer_disposition: AnswerDisposition
    escalation: FloorLevel = "none"
    metadata: ResponseMetadata


class Observation(BaseModel):
    """A proactive finding (read projection of the observations table). Narration/evidence live on the
    linked interaction via ``response_id``."""

    observation_id: str
    member_id: str
    response_id: str
    severity: Severity
    title: str
    trigger_reason: str
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
    observation_id: Optional[str] = None
    interaction_id: Optional[str] = None
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
    target: Optional[str] = None
    payload: Optional[dict[str, Any]] = None
    source: FeedbackSource


# --------------------------------------------------------------------------------------------------
# Mode 1 unit — a preset prompt bound to its pre-computed answer (no model call). The conversation
# loop is a walk over these.
# --------------------------------------------------------------------------------------------------

class SuggestedPrompt(BaseModel):
    prompt: str
    response: HealthIntelligenceResponse
