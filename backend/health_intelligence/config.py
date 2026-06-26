"""Clinical & statistical constants — the versioned, auditable knobs the deterministic core runs on.

Pure declaration: plain Python constants, no environment, no I/O, no clock. Secrets and runtime
config live in ``.env``; *nothing* clinical does. Every threshold the analysis applies is named
here so it is never a literal buried in logic, and any escalation decision is reproducible against
``CONFIG_VERSION``.

Phase-0 curation discipline (see the plan): values are filled ONLY where anchored in the supplied
materials. Externally-sourced clinical numbers — EFLM CVa/CVi, panic thresholds beyond the one the
eval set pins, adverse directions, and vital normal bounds — ship as **typed absence** (``None`` /
empty), NOT as guesses. This is deliberate, not unfinished: ``analysis.py``'s documented skip-paths
fire on these ``None``s (no CVa/CVi -> RCV skipped; no panic -> flag skipped; no range -> a typed
``no_reference`` note), so a deferred value is an *exercised edge case*. Each is curated in Phase 1
beside the test that pins it, with its source named inline at that point.

What's filled now and why:
  - statistical policy (alpha, FDR q, n_min, Theil-Sen CI)  -> standard defaults the architecture names as config (D3)
  - HbA1c band cut-points 5.7 / 6.5                          -> eval case E01 + architecture
  - Vitamin D status bands (>=30 / 20-29 / <20)              -> printed in the supplied data
  - Potassium critical-high threshold                        -> eval case E07 pins K+ 6.1 -> urgent
What's deferred to Phase 1 (typed absence):
  - CVa / CVi for every marker (EFLM Biological Variation Database)
  - panic thresholds for every marker except the Potassium high anchored above (standard critical-value tables)
  - adverse directions (clinical standard; consumed by _severity)
  - vital normal bounds for BP / BMI (AHA / WHO)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# --------------------------------------------------------------------------------------------------
# Versioning
# --------------------------------------------------------------------------------------------------

#: Stamped onto every reference-range row and every interaction, so an escalation is reproducible
#: against the exact threshold set that produced it. Bump on any clinical-constant change.
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
# Filled: HbA1c band cut-points, Vitamin D graded bands, Potassium critical-high. Everything else is
# typed absence by design (see the module docstring).
# --------------------------------------------------------------------------------------------------

MARKERS: dict[str, MarkerConfig] = {
    # Labs omit `unit` — it comes from the data per reading (see MarkerConfig.unit). Only the three
    # Phase-0 anchored values appear; every deferred constant is left as typed absence.
    # --- Glycaemic ---
    "HbA1c": MarkerConfig(
        # 5.7 (prediabetes) / 6.5 (diabetes) — eval E01 + architecture. Adverse direction "up" deferred.
        band_cutpoints=(5.7, 6.5),
    ),
    "Fasting glucose": MarkerConfig(),
    # --- Lipids ---
    "LDL cholesterol": MarkerConfig(),
    "HDL cholesterol": MarkerConfig(),
    "Triglycerides": MarkerConfig(),
    "Total cholesterol": MarkerConfig(),
    # --- Renal ---
    "eGFR": MarkerConfig(),
    "Creatinine": MarkerConfig(),
    # --- Hepatic ---
    "ALT": MarkerConfig(),
    "AST": MarkerConfig(),
    # --- Inflammation ---
    "CRP": MarkerConfig(),
    # --- Haematology / iron ---
    "Hemoglobin": MarkerConfig(),
    "Ferritin": MarkerConfig(),
    # --- Vitamin D: graded status bands printed in the supplied data (ng/mL) ---
    "Vitamin D (25-OH)": MarkerConfig(
        graded_bands=(
            GradedBand("deficient", None, 20.0),     # <20
            GradedBand("insufficient", 20.0, 30.0),  # 20-29
            GradedBand("sufficient", 30.0, None),    # >=30
        ),
    ),
    # --- Thyroid ---
    "TSH": MarkerConfig(),
    # --- Electrolyte: the one Phase-0 panic anchor ---
    "Potassium": MarkerConfig(
        # eval E07 expects URGENT for a K+ of 6.1 (C07's latest panel already reads 6.1); a
        # critical-high of 6.0 mmol/L (standard hyperkalemia panic value) makes 6.1 trip the floor.
        # The critical-LOW side is not anchored by any case, so it stays deferred -> Phase 1.
        panic_high=6.0,
    ),
    # --- Vitals: the data prints no vital units, so config IS the source (normal bounds deferred) ---
    "systolic_bp": MarkerConfig(unit="mmHg"),
    "diastolic_bp": MarkerConfig(unit="mmHg"),
    "bmi": MarkerConfig(unit="kg/m2"),
}


#: The assembled, ready-to-use config the core consumes. ``analyze(..., cfg=ANALYSIS_CONFIG)``.
ANALYSIS_CONFIG = AnalysisConfig(version=CONFIG_VERSION, stats=StatConfig(), markers=MARKERS)
