"""The firewall's pure contract — every reference-range shape parses to the correct bounds.

``parse_reference_range`` is the one transform the architecture mandates a unit test for (Phase 2):
the five shapes the supplied data prints must each reduce to the right ``(sex, ref_low, ref_high)``
rows, with the Vitamin-D multi-band — the flagged #1 parse risk — pinned explicitly. Pure functions,
no SQLite.
"""

from preprocessing.ingest import parse_reference_range


def test_parse_reference_range_all_five_shapes():
    # upper-bound only (<high): HbA1c, CRP, LDL, Total chol, Triglycerides
    assert parse_reference_range("<5.7", "HbA1c") == [("any", None, 5.7)]
    # bounded low-high: Fasting glucose, TSH, AST, ALT, Potassium
    assert parse_reference_range("70-99", "Fasting glucose") == [("any", 70.0, 99.0)]
    # lower-bound only (>=low): eGFR
    assert parse_reference_range(">=90", "eGFR") == [("any", 90.0, None)]
    # sex-split wrapping a bounded form: Creatinine (also Hemoglobin, Ferritin)
    assert parse_reference_range("0.74-1.35 (male) / 0.59-1.04 (female)", "Creatinine") == [
        ("male", 0.74, 1.35),
        ("female", 0.59, 1.04),
    ]
    # sex-split wrapping a one-sided '>low' form: HDL
    assert parse_reference_range(">40 (male) / >50 (female)", "HDL cholesterol") == [
        ("male", 40.0, None),
        ("female", 50.0, None),
    ]


def test_vitamin_d_multiband_reduces_to_deficiency_floor():
    # The graded string reduces to the deficiency floor (ref_low=20); the suff/insuff/defic nuance is
    # owned by config.graded_bands (band-crossing), so it is not double-counted as a range flag.
    assert parse_reference_range(
        ">=30 sufficient; 20-29 insufficient; <20 deficient", "Vitamin D (25-OH)"
    ) == [("any", 20.0, None)]
