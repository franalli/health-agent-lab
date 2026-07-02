"""Clinical & statistical constants — the versioned, auditable knobs the deterministic core runs on.

Pure declaration: plain Python constants, no environment, no I/O, no clock. Secrets and runtime
config live in ``.env``; *nothing* clinical does. Every threshold the analysis applies is named
here so it is never a literal buried in logic, and any escalation decision is reproducible against
``CONFIG_VERSION``.

Curation discipline (see the plan): a clinical number is filled ONLY where a test pins it, with its
source named inline at the point of use. Phase 0 anchored the three values the supplied materials fix
directly; Phase 1 curates the eval-forced + demo-prominent subset (the markers the labeled cases and
the live scan actually exercise). Anything no case touches stays **typed absence** (``None`` / empty)
ON PURPOSE: ``analysis.py``'s documented skip-paths fire on these ``None``s (no CVa/CVi -> RCV skipped,
trend judged on Mann-Kendall + Theil-Sen CI alone; no panic -> that flag is not evaluated; no range ->
a typed ``no_reference`` note), so a deferred value is an *exercised edge case*, not unfinished work.

What Phase 0 anchored (unchanged):
  - statistical policy (alpha, FDR q, n_min, Theil-Sen CI)  -> standard defaults the architecture names as config (D3)
  - HbA1c band cut-points 5.7 / 6.5                          -> eval case E01 + architecture
  - Vitamin D status bands (>=30 / 20-29 / <20)              -> printed in the supplied data
  - Potassium critical-high 6.0                              -> eval case E07 pins K+ 6.1 -> urgent

What Phase 1 sources (each cited inline in MARKERS below):
  - CVa / CVi           -> EFLM Biological Variation Database (biologicalvariation.eu) for the ~14 trended
                           labs the eval exercises (representative within-subject CVi; CVa = the desirable
                           analytical spec = 0.5*CVi, Fraser), plus systolic BP from BP-variability
                           literature (demo-prominent). Potassium (panic-gated), diastolic and BMI stay
                           deferred -> live RCV skip-path (their trends surface at notable, not escalated).
  - adverse_direction   -> standard clinical interpretation (which way is pathological). All labs + vitals;
                           Potassium left None (genuinely bidirectional -> safety via panic, not trend).
  - panic thresholds    -> standard adult laboratory critical-value tables. The small set the eval/demo
                           need (K+ high anchored P0; + K+ low, glucose hypo/hyper, Hb low); rest deferred.
  - vital normal bounds -> AHA 2017 (BP) / WHO (BMI).

CONFIG_VERSION stays ``v0``: Phase 1 *completes* the deferred set rather than changing an anchored
value, and no escalation has yet been produced against the absent config (the core lands here, the DB
in Phase 2). Bump only when a value that has already keyed a stored escalation changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --------------------------------------------------------------------------------------------------
# Versioning
# --------------------------------------------------------------------------------------------------

#: Stamped onto every reference-range row and every interaction, so an escalation is reproducible
#: against the exact threshold set that produced it. Bump when a constant changes that has already
#: keyed a stored escalation; Phase 1 filling the deferred set stays v0 (no escalation exists yet —
#: the DB lands in Phase 2). Full rationale in the module docstring.
CONFIG_VERSION = "v0"

# --------------------------------------------------------------------------------------------------
# Model pins (consumed only by the LLM layer in Phase 4 — listed here so the version tuple is whole)
# --------------------------------------------------------------------------------------------------

#: Composition model (architecture §5). D1 leaves the LLM a *language* job, not a reasoning one, so
#: the mid/balanced tier is the right tool. Pinned verbatim — do not silently upgrade.
#: Sonnet 5 is a "5-generation" model: it rejects sampling params (temperature/top_p/top_k → 400) and
#: defaults adaptive thinking ON. The composer therefore omits temperature and explicitly disables
#: thinking (see ``llm._ADAPTIVE_ONLY_MODELS``) — the forced-tool structured-output call is incompatible
#: with thinking being active, and that forced call IS our determinism/structuring mechanism.
COMPOSE_MODEL = "claude-sonnet-5"

#: Input-gate intent classifier (architecture §2: "a small model, temp 0, few-shot"). The doc names
#: a small model without pinning one; confirm this choice when the gate is built (Phase 4).
GATE_MODEL = "claude-haiku-4-5"

#: Feedback input-judge (Phase 7 self-improvement): a small model that classifies whether a clinician's
#: corrected answer is a FIT few-shot exemplar (coherent / on-style / states no numeric cutoff / does not
#: reassure about a flagged value). Same Haiku tier as the gate — a bounded temp-0 classifier — and
#: DELIBERATELY not the deterministic core: the input bar is a QUALITY filter, not the safety boundary
#: (the always-on validator floors escalation under any prompt regardless). Same id as GATE_MODEL, so the
#: PRICING entry below already covers it.
JUDGE_MODEL = "claude-haiku-4-5"

#: Models on the "5-generation" request surface: they reject sampling params (temperature/top_p/top_k
#: with a 400) and default adaptive thinking ON. ``llm.structured`` keys its request shape on membership
#: here (omit temperature + explicitly disable thinking) — a per-model CAPABILITY, deliberately NOT the
#: compose/gate ROLE, so it lives next to the model pins it classifies. This is an explicit allow-list, NOT
#: derived from ``COMPOSE_MODEL`` (the compose role is 5-gen TODAY, but that is a coincidence, not a rule):
#: swapping ``COMPOSE_MODEL`` above to another 5-gen id that is NOT listed here would 400 on temperature, so
#: a model swap MUST update BOTH the pin and this set (both now in this one file). Extend when a new model
#: joins that surface (Opus 4.7+, Fable 5, …); an old-gen id must NOT be added (it would lose its temp 0).
ADAPTIVE_ONLY_MODELS: frozenset[str] = frozenset({"claude-sonnet-5"})

#: ``model_version`` stamped on a Mode-1 response — no model ran, but the schema column is NOT NULL and
#: the reproducibility tuple must be whole. Mode 1's tuple is (data, config, template); this sentinel
#: makes "no LLM was involved" explicit in the trace rather than leaving an empty string (architecture §7).
MODEL_VERSION_DETERMINISTIC = "deterministic"

# --------------------------------------------------------------------------------------------------
# LLM runtime knobs (Phase 4 — consumed only by gate.py / llm.py). Pure constants under config_version,
# not .env: temperature is a reproducibility setting (architecture §7), the token caps are the cost
# lever (§5: output is ~5x input, so capping compose length is the main control), and PRICING is the
# published per-MTok rate used to stamp ``cost_usd`` on every interaction.
# --------------------------------------------------------------------------------------------------

#: The gate's sampling temperature, pinned at 0 (it is the determinism-sensitive enum classifier, and
#: Haiku 4.5 still honors sampling params). The composer no longer takes a temperature: Sonnet 5 rejects
#: sampling params (5-gen surface), so ``llm.structured`` omits it for that model and disables thinking
#: instead. The substance-determinism claim now rests on the pure core + the gate's temp 0 + the
#: forced-tool structured output; only the composer's prose phrasing is free to vary (§7).
LLM_TEMPERATURE = 0

#: The gate emits one enum value, so a tiny budget suffices; the composer emits a short answer +
#: uncertainty + the cited-marker keys (the deterministic core already did the analysis, §5). The
#: compose budget carries headroom over the ~300–800-token target so a thorough multi-marker answer
#: plus the forced-tool JSON overhead doesn't truncate mid-structure (a truncated tool call would be an
#: ``LLMParseError`` — handled by the retry/fallback, but cheaper to make rare). Sonnet 5 uses a heavier
#: tokenizer (~30% more tokens than Sonnet 4.6 for the same text), so the compose budget is widened to
#: keep that headroom over the ~300–800-token target; with thinking disabled there are no thinking tokens
#: competing for the budget, so this caps the structured answer alone.
GATE_MAX_TOKENS = 256
#: The feedback judge emits a bool + a one-line reason, so a small budget suffices (like the gate).
JUDGE_MAX_TOKENS = 256
COMPOSE_MAX_TOKENS = 2048

#: Per-model ($/MTok input, $/MTok output) — published rates, used by ``llm.cost_usd`` to stamp
#: each interaction's cost. Keyed by the pinned model ids above.
PRICING: dict[str, tuple[float, float]] = {
    COMPOSE_MODEL: (
        3.0,
        15.0,
    ),  # Claude Sonnet 5 (standard rate; intro $2/$10 through 2026-08-31)
    GATE_MODEL: (1.0, 5.0),  # Claude Haiku 4.5
}

#: The deterministic, regression-pinned **emergency-phrase floor** (architecture §98/§564): a fixed
#: denylist read straight from the raw message, in PARALLEL with the LLM gate, so a self-harm or
#: acute-emergency phrase forces the floor to ``urgent`` even if the gate is jailbroken into ``none``
#: or the provider is down — the one routing where a miss is unacceptable is a *guarantee*, not a
#: *measurement*. Matched lower-cased, with smart apostrophes folded to ASCII and on WORD boundaries
#: (so "want to die" does not fire on "want to diet", and a curly-quote "don't" still matches) — see
#: ``gate.emergency_phrase_floor``. Deliberately conservative (over-escalation is the safe direction,
#: §562). NOT exhaustive — the broader injection/jailbreak denylist stays deferred (§11); this is only
#: the non-overridable self-harm/acute floor. Because matching is word-bounded, list the inflected forms
#: that matter (e.g. "self-harming") rather than relying on substring containment. Changing this set is a
#: safety change — pin it with a regression test (test_gate.py), never trim it casually.
EMERGENCY_PHRASES: tuple[str, ...] = (
    # self-harm / crisis
    "kill myself",
    "killing myself",
    "end my life",
    "ending my life",
    "ending it all",
    "take my own life",
    "want to die",
    "wanna die",
    "don't want to live",
    "do not want to live",
    "don't want to be alive",
    "suicidal",
    "suicide",
    "hurt myself",
    "hurting myself",
    "harm myself",
    "harming myself",
    "self-harm",
    "self-harming",
    # acute medical emergency
    "chest pain",
    "crushing chest",
    "can't breathe",
    "cannot breathe",
    "can't breath",
    "trouble breathing",
    "struggling to breathe",
    "having a stroke",
    "face is drooping",
    "slurred speech",
    "overdose",
    "overdosed",
    "severe bleeding",
    "bleeding heavily",
    "coughing up blood",
    "vomiting blood",
    "anaphylaxis",
    "anaphylactic",
)

#: Locale emergency contacts (**Switzerland**) surfaced in URGENT responses: the deterministic
#: ``seek_care``/``crisis`` templates (the *guaranteed* acute path — they fire off the
#: ``EMERGENCY_PHRASES`` floor even when the LLM is down or jailbroken) and, as softer copy on top, the
#: composer's urgent branch. **Presentation-layer only** — consumed by ``llm``/``templates``, NEVER by
#: ``analyze`` — so this is not a clinical threshold and does NOT bump ``CONFIG_VERSION`` (the analytical
#: contract is untouched). Curated + regression-pinned here like ``EMERGENCY_PHRASES`` all the same,
#: because an emergency reply that prints the WRONG number is a safety defect. Numbers verified against
#: official Swiss guidance (ch.ch): 144 = ambulance/medical, 112 = pan-European (valid in CH), 143 = Die
#: Dargebotene Hand (24/7 emotional-support / crisis line). Changing these is a safety change.
#:
#: OPERATIONAL NOTE (emergency-number change): the deterministic ``seek_care``/``crisis`` templates read
#: these live, so the GUARANTEED acute path is always current. The composer's urgent-branch copy is DIFF-
#: erent: ``EMERGENCY_MEDICAL_CONTACT`` is interpolated into ``llm.BASE_COMPOSE_SYSTEM`` and frozen into
#: each promoted ``prompt_versions`` row. Startup/reseed re-syncs v0, but a promoted learned vN is a frozen
#: snapshot, so on a DURABLE-disk deploy a stale number can persist in an active vN. After changing a
#: number, run ``POST /reset`` (reverts learned prompts to v0, which re-syncs) or ``POST /admin/reseed`` so
#: no promoted vN carries the old copy. (No-op on the free/ephemeral tier — the DB re-seeds each cold start.)
EMERGENCY_MEDICAL_NUMBER = "144"  # Swiss ambulance / medical emergency (Sanitätsnotruf)
EMERGENCY_EU_NUMBER = "112"  # pan-European emergency, also valid in Switzerland
CRISIS_LINE_NUMBER = (
    "143"  # Die Dargebotene Hand — Swiss 24/7 crisis / emotional-support line
)

#: Ready-to-render contact sentences, composed once so the template copy and the prompt copy can't drift
#: to different numbers. The medical line leads with 144 (ambulance); the crisis line leads with 143 but
#: still points to 144/112 for immediate physical danger (a crisis caller may also be in acute danger).
EMERGENCY_MEDICAL_CONTACT = (
    f"In Switzerland, call {EMERGENCY_MEDICAL_NUMBER} for an ambulance, "
    f"or {EMERGENCY_EU_NUMBER} (the Europe-wide emergency number)."
)
CRISIS_CONTACT = (
    f"In Switzerland you can reach Die Dargebotene Hand on {CRISIS_LINE_NUMBER} any time, "
    f"or call {EMERGENCY_MEDICAL_NUMBER} (or {EMERGENCY_EU_NUMBER}) if you are in immediate danger."
)

# --------------------------------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------------------------------

#: A marker's direction of clinical concern — which way is *bad* (HbA1c up, ferritin down). Used by
#: severity triage so a trend toward the healthy side never raises alarm. ``None`` => not yet curated.
Direction = Literal["up", "down"]


@dataclass(frozen=True)
class StatConfig:
    """Statistical policy for the trajectory analysis (architecture §2 D3). Defaults are the
    standard small-sample choices the architecture names; all live here, never inline."""

    alpha: float = 0.05
    """Mann-Kendall significance level, applied to the exact small-sample p-value AFTER the FDR pass."""

    fdr_q: float = 0.10
    """Benjamini-Hochberg false-discovery rate across a panel's markers — controls multiplicity so a
    many-marker panel can't throw a false trend by chance."""

    n_min: int = 4
    """Minimum readings before any trend is claimed. Set to 4 so the sparse member (C12, 3 panels)
    yields the honest 'too short to call a trend' verdict — a correct output, not a failure."""

    ts_ci_level: float = 0.95
    """Theil-Sen confidence level. A direction is asserted only when this CI excludes zero, so
    'too noisy to sign' stays a first-class verdict."""

    rcv_z: float = 1.96
    """Z for the Reference Change Value (RCV = sqrt(2)*Z*sqrt(CVa^2 + CVi^2)); 1.96 = two-sided 95%.
    Lives here, not as a literal in analysis.py, so an RCV-gated escalation is reproducible against
    config_version — the same discipline as alpha / ts_ci_level."""


@dataclass(frozen=True)
class GradedBand:
    """One labelled segment of a multi-band marker (e.g. Vitamin D 'insufficient'). Bounds are
    half-open ``[low, high)``; ``None`` is an open end."""

    label: str
    low: float | None  # inclusive lower bound; None => open below
    high: float | None  # exclusive upper bound; None => open above


@dataclass(frozen=True)
class MarkerConfig:
    """Per-marker clinical constants. Every field is PRESENT; deferred ones are typed absence
    (``None`` / empty), which the analysis is designed to degrade on rather than fail. Only ``unit``
    and the explicitly-anchored fields are populated in Phase 0."""

    unit: str | None = None
    """Config-supplied unit — VITALS ONLY (the data prints no vital units, so config is the source).
    Labs get their unit from the data (ResultInput.unit), so leaving it ``None`` here avoids a second
    source of truth that could drift — the same 'vitals only' treatment as ref_low/ref_high below."""

    # ---- Normal reference bounds: VITALS ONLY ----
    # Labs get ref_low/ref_high parsed from the ranges the data prints per result; vitals do not, so
    # their normal bounds come from here. Deferred (AHA / WHO) -> curated Phase 1.
    ref_low: float | None = None
    ref_high: float | None = None

    # ---- Direction-aware severity ----
    adverse_direction: Direction | None = (
        None  # deferred Phase 1 (consumed by _severity)
    )

    # ---- Reference Change Value (clinical trend-vs-noise) ----
    # RCV = sqrt(2)*Z*sqrt(CVa^2 + CVi^2). Either being None => analysis skips RCV and judges the
    # trend on Mann-Kendall + Theil-Sen CI alone (noted, not silent). Deferred (EFLM) -> Phase 1.
    cva: float | None = None  # analytical CV, %
    cvi: float | None = None  # within-subject biological CV, %

    # ---- Safety-critical panic thresholds (curated here, never parsed from data) ----
    # None => the corresponding panic flag is simply not evaluated. Only the Potassium high is
    # anchored in Phase 0 (eval E07); all other panic bounds are deferred -> Phase 1.
    panic_low: float | None = None
    panic_high: float | None = None

    # ---- Discrete clinical cut-points (band-crossing is a highly explainable event) ----
    band_cutpoints: tuple[
        float, ...
    ] = ()  # e.g. HbA1c (5.7, 6.5); () => no band-cross signal
    graded_bands: tuple[
        GradedBand, ...
    ] = ()  # labelled multi-band (Vitamin D); () => none


@dataclass(frozen=True)
class AnalysisConfig:
    """The single ``cfg`` object ``analysis.analyze(...)`` receives — versioned policy + per-marker
    constants, bundled so an analysis run is reproducible against one named version."""

    version: str
    stats: StatConfig
    markers: dict[str, MarkerConfig]


# --------------------------------------------------------------------------------------------------
# Per-marker table — 16 labs + 3 vitals (vitals fold in as markers). Keyed by canonical marker name.
# Phase 1 curated adverse_direction, CVa/CVi, a small panic set and vital bounds for the eval-forced +
# demo-prominent subset; what no case exercises stays typed absence (the live skip-paths). Per-field
# sourcing is in the module docstring.
# --------------------------------------------------------------------------------------------------

MARKERS: dict[str, MarkerConfig] = {
    # Labs omit `unit` — it comes from the data per reading (see MarkerConfig.unit). Phase-1 curation
    # fills adverse_direction (standard clinical interpretation), CVa/CVi (EFLM; CVa = 0.5*CVi desirable
    # APS), and a small panic set; constants no case exercises stay typed absence (the skip-path).
    # --- Glycaemic ---
    "HbA1c": MarkerConfig(
        adverse_direction="up",  # higher = worse glycaemic control (ADA)
        cva=0.6,
        cvi=1.2,  # EFLM — very tight within-subject variation
        band_cutpoints=(
            5.7,
            6.5,
        ),  # 5.7 prediabetes / 6.5 diabetes — eval E01 + architecture
    ),
    "Fasting glucose": MarkerConfig(
        adverse_direction="up",  # higher = hyperglycaemia
        cva=2.5,
        cvi=4.9,  # EFLM
        panic_low=50.0,
        panic_high=500.0,  # critical hypo-/hyperglycaemia (critical-value tables, mg/dL)
    ),
    # --- Lipids ---
    "LDL cholesterol": MarkerConfig(
        adverse_direction="up",  # higher = atherogenic risk
        cva=4.2,
        cvi=8.3,  # EFLM
    ),
    "HDL cholesterol": MarkerConfig(
        adverse_direction="down",  # protective — LOWER is the adverse move
        cva=3.7,
        cvi=7.3,  # EFLM
    ),
    "Triglycerides": MarkerConfig(
        adverse_direction="up",
        cva=10.0,
        cvi=20.0,  # EFLM — high within-subject variation
    ),
    "Total cholesterol": MarkerConfig(
        adverse_direction="up",
        cva=2.9,
        cvi=5.8,  # EFLM
    ),
    # --- Renal ---
    "eGFR": MarkerConfig(
        adverse_direction="down",  # lower = worse renal function (KDIGO)
        cva=2.65,
        cvi=5.3,  # EFLM — derived from creatinine; CVi taken from creatinine
    ),
    "Creatinine": MarkerConfig(
        adverse_direction="up",  # higher = worse renal function
        cva=2.65,
        cvi=5.3,  # EFLM
    ),
    # --- Hepatic ---
    "ALT": MarkerConfig(
        adverse_direction="up",  # higher = hepatocellular injury
        cva=6.0,
        cvi=12.0,  # EFLM
    ),
    "AST": MarkerConfig(
        adverse_direction="up",
        cva=6.0,
        cvi=12.0,  # EFLM
    ),
    # --- Inflammation ---
    "CRP": MarkerConfig(
        adverse_direction="up",  # higher = more inflammation
        cva=21.0,
        cvi=42.0,  # EFLM — very high within-subject variation
    ),
    # --- Haematology / iron ---
    "Hemoglobin": MarkerConfig(
        adverse_direction="down",  # dataset/eval concern is anaemia (C04); clinically bidirectional
        cva=1.4,
        cvi=2.8,  # EFLM
        panic_low=7.0,  # critical anaemia / transfusion threshold (g/dL)
    ),
    "Ferritin": MarkerConfig(
        adverse_direction="down",  # low = iron deficiency (C04); clinically bidirectional
        cva=7.1,
        cvi=14.2,  # EFLM
    ),
    # --- Vitamin D: graded status bands printed in the supplied data (ng/mL) ---
    "Vitamin D (25-OH)": MarkerConfig(
        adverse_direction="down",  # low = deficiency
        cva=6.15,
        cvi=12.3,  # EFLM
        graded_bands=(
            GradedBand("deficient", None, 20.0),  # <20
            GradedBand("insufficient", 20.0, 30.0),  # 20-29
            GradedBand("sufficient", 30.0, None),  # >=30
        ),
    ),
    # --- Thyroid ---
    "TSH": MarkerConfig(
        adverse_direction="up",  # eval concern is rising TSH -> hypothyroidism (C05); bidirectional
        cva=9.85,
        cvi=19.7,  # EFLM — high within-subject variation
    ),
    # --- Electrolyte: panic-gated, genuinely bidirectional -> the RCV + direction skip-path ---
    "Potassium": MarkerConfig(
        # adverse_direction stays None: both hyper- and hypokalaemia are dangerous, so there is no single
        # adverse TREND direction — K+ safety is the panic floor, not a trend verdict. CVa/CVi are left
        # absent too, so RCV is skipped for K+ (the documented typed-absence path, exercised by a test).
        panic_low=2.8,  # severe hypokalaemia (critical-value tables, mmol/L)
        panic_high=6.0,  # hyperkalaemia — eval E07: C07's 6.1 -> urgent (anchored P0)
    ),
    # --- Vitals: the data prints no vital units, so config IS the source. Normal bounds AHA/WHO.
    #     systolic carries CVa/CVi (RCV-gated, demo-prominent); diastolic + BMI stay RCV-free, so their
    #     trends surface at notable but never escalate (BMI is real trajectory, not setpoint noise). ---
    "systolic_bp": MarkerConfig(
        unit="mmHg",
        adverse_direction="up",
        ref_low=90.0,
        ref_high=120.0,  # AHA 2017: normal <120; <90 hypotension
        # CVa/CVi from within-subject BP-variability literature (NOT EFLM — BP is not a lab analyte);
        # systolic is in every panel (demo-prominent), so it is RCV-gated and can trend-escalate first-class.
        cva=2.85,
        cvi=5.7,
    ),
    "diastolic_bp": MarkerConfig(
        unit="mmHg",
        adverse_direction="up",
        ref_low=60.0,
        ref_high=80.0,  # AHA 2017: normal <80
    ),
    "bmi": MarkerConfig(
        unit="kg/m2",
        adverse_direction="up",
        ref_low=18.5,
        ref_high=25.0,  # WHO: normal 18.5-24.9
    ),
}


#: The assembled, ready-to-use config the core consumes. ``analyze(..., cfg=ANALYSIS_CONFIG)``.
ANALYSIS_CONFIG = AnalysisConfig(
    version=CONFIG_VERSION, stats=StatConfig(), markers=MARKERS
)
