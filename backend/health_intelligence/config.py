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
from typing import Literal, Optional

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
#: a mid-tier model is the right tool. Pinned verbatim — do not silently upgrade.
COMPOSE_MODEL = "claude-sonnet-4-6"

#: Input-gate intent classifier (architecture §2: "a small model, temp 0, few-shot"). The doc names
#: a small model without pinning one; confirm this choice when the gate is built (Phase 4).
GATE_MODEL = "claude-haiku-4-5"

#: ``model_version`` stamped on a Mode-1 response — no model ran, but the schema column is NOT NULL and
#: the reproducibility tuple must be whole. Mode 1's tuple is (data, config, template); this sentinel
#: makes "no LLM was involved" explicit in the trace rather than leaving an empty string (architecture §7).
MODEL_VERSION_DETERMINISTIC = "deterministic"

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
    low: Optional[float]   # inclusive lower bound; None => open below
    high: Optional[float]  # exclusive upper bound; None => open above


@dataclass(frozen=True)
class MarkerConfig:
    """Per-marker clinical constants. Every field is PRESENT; deferred ones are typed absence
    (``None`` / empty), which the analysis is designed to degrade on rather than fail. Only ``unit``
    and the explicitly-anchored fields are populated in Phase 0."""

    unit: Optional[str] = None
    """Config-supplied unit — VITALS ONLY (the data prints no vital units, so config is the source).
    Labs get their unit from the data (ResultInput.unit), so leaving it ``None`` here avoids a second
    source of truth that could drift — the same 'vitals only' treatment as ref_low/ref_high below."""

    # ---- Normal reference bounds: VITALS ONLY ----
    # Labs get ref_low/ref_high parsed from the ranges the data prints per result; vitals do not, so
    # their normal bounds come from here. Deferred (AHA / WHO) -> curated Phase 1.
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None

    # ---- Direction-aware severity ----
    adverse_direction: Optional[Direction] = None  # deferred Phase 1 (consumed by _severity)

    # ---- Reference Change Value (clinical trend-vs-noise) ----
    # RCV = sqrt(2)*Z*sqrt(CVa^2 + CVi^2). Either being None => analysis skips RCV and judges the
    # trend on Mann-Kendall + Theil-Sen CI alone (noted, not silent). Deferred (EFLM) -> Phase 1.
    cva: Optional[float] = None  # analytical CV, %
    cvi: Optional[float] = None  # within-subject biological CV, %

    # ---- Safety-critical panic thresholds (curated here, never parsed from data) ----
    # None => the corresponding panic flag is simply not evaluated. Only the Potassium high is
    # anchored in Phase 0 (eval E07); all other panic bounds are deferred -> Phase 1.
    panic_low: Optional[float] = None
    panic_high: Optional[float] = None

    # ---- Discrete clinical cut-points (band-crossing is a highly explainable event) ----
    band_cutpoints: tuple[float, ...] = ()       # e.g. HbA1c (5.7, 6.5); () => no band-cross signal
    graded_bands: tuple[GradedBand, ...] = ()    # labelled multi-band (Vitamin D); () => none


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
        adverse_direction="up",                  # higher = worse glycaemic control (ADA)
        cva=0.6, cvi=1.2,                        # EFLM — very tight within-subject variation
        band_cutpoints=(5.7, 6.5),               # 5.7 prediabetes / 6.5 diabetes — eval E01 + architecture
    ),
    "Fasting glucose": MarkerConfig(
        adverse_direction="up",                  # higher = hyperglycaemia
        cva=2.5, cvi=4.9,                        # EFLM
        panic_low=50.0, panic_high=500.0,        # critical hypo-/hyperglycaemia (critical-value tables, mg/dL)
    ),
    # --- Lipids ---
    "LDL cholesterol": MarkerConfig(
        adverse_direction="up",                  # higher = atherogenic risk
        cva=4.2, cvi=8.3,                        # EFLM
    ),
    "HDL cholesterol": MarkerConfig(
        adverse_direction="down",                # protective — LOWER is the adverse move
        cva=3.7, cvi=7.3,                        # EFLM
    ),
    "Triglycerides": MarkerConfig(
        adverse_direction="up",
        cva=10.0, cvi=20.0,                      # EFLM — high within-subject variation
    ),
    "Total cholesterol": MarkerConfig(
        adverse_direction="up",
        cva=2.9, cvi=5.8,                        # EFLM
    ),
    # --- Renal ---
    "eGFR": MarkerConfig(
        adverse_direction="down",                # lower = worse renal function (KDIGO)
        cva=2.65, cvi=5.3,                       # EFLM — derived from creatinine; CVi taken from creatinine
    ),
    "Creatinine": MarkerConfig(
        adverse_direction="up",                  # higher = worse renal function
        cva=2.65, cvi=5.3,                       # EFLM
    ),
    # --- Hepatic ---
    "ALT": MarkerConfig(
        adverse_direction="up",                  # higher = hepatocellular injury
        cva=6.0, cvi=12.0,                       # EFLM
    ),
    "AST": MarkerConfig(
        adverse_direction="up",
        cva=6.0, cvi=12.0,                       # EFLM
    ),
    # --- Inflammation ---
    "CRP": MarkerConfig(
        adverse_direction="up",                  # higher = more inflammation
        cva=21.0, cvi=42.0,                      # EFLM — very high within-subject variation
    ),
    # --- Haematology / iron ---
    "Hemoglobin": MarkerConfig(
        adverse_direction="down",                # dataset/eval concern is anaemia (C04); clinically bidirectional
        cva=1.4, cvi=2.8,                        # EFLM
        panic_low=7.0,                           # critical anaemia / transfusion threshold (g/dL)
    ),
    "Ferritin": MarkerConfig(
        adverse_direction="down",                # low = iron deficiency (C04); clinically bidirectional
        cva=7.1, cvi=14.2,                       # EFLM
    ),
    # --- Vitamin D: graded status bands printed in the supplied data (ng/mL) ---
    "Vitamin D (25-OH)": MarkerConfig(
        adverse_direction="down",                # low = deficiency
        cva=6.15, cvi=12.3,                      # EFLM
        graded_bands=(
            GradedBand("deficient", None, 20.0),     # <20
            GradedBand("insufficient", 20.0, 30.0),  # 20-29
            GradedBand("sufficient", 30.0, None),    # >=30
        ),
    ),
    # --- Thyroid ---
    "TSH": MarkerConfig(
        adverse_direction="up",                  # eval concern is rising TSH -> hypothyroidism (C05); bidirectional
        cva=9.85, cvi=19.7,                      # EFLM — high within-subject variation
    ),
    # --- Electrolyte: panic-gated, genuinely bidirectional -> the RCV + direction skip-path ---
    "Potassium": MarkerConfig(
        # adverse_direction stays None: both hyper- and hypokalaemia are dangerous, so there is no single
        # adverse TREND direction — K+ safety is the panic floor, not a trend verdict. CVa/CVi are left
        # absent too, so RCV is skipped for K+ (the documented typed-absence path, exercised by a test).
        panic_low=2.8,                           # severe hypokalaemia (critical-value tables, mmol/L)
        panic_high=6.0,                          # hyperkalaemia — eval E07: C07's 6.1 -> urgent (anchored P0)
    ),
    # --- Vitals: the data prints no vital units, so config IS the source. Normal bounds AHA/WHO.
    #     systolic carries CVa/CVi (RCV-gated, demo-prominent); diastolic + BMI stay RCV-free, so their
    #     trends surface at notable but never escalate (BMI is real trajectory, not setpoint noise). ---
    "systolic_bp": MarkerConfig(
        unit="mmHg", adverse_direction="up",
        ref_low=90.0, ref_high=120.0,            # AHA 2017: normal <120; <90 hypotension
        # CVa/CVi from within-subject BP-variability literature (NOT EFLM — BP is not a lab analyte);
        # systolic is in every panel (demo-prominent), so it is RCV-gated and can trend-escalate first-class.
        cva=2.85, cvi=5.7,
    ),
    "diastolic_bp": MarkerConfig(
        unit="mmHg", adverse_direction="up",
        ref_low=60.0, ref_high=80.0,             # AHA 2017: normal <80
    ),
    "bmi": MarkerConfig(
        unit="kg/m2", adverse_direction="up",
        ref_low=18.5, ref_high=25.0,             # WHO: normal 18.5-24.9
    ),
}


#: The assembled, ready-to-use config the core consumes. ``analyze(..., cfg=ANALYSIS_CONFIG)``.
ANALYSIS_CONFIG = AnalysisConfig(version=CONFIG_VERSION, stats=StatConfig(), markers=MARKERS)
