"""Shared test builders — the MemberBundle / panel / result dict shapes and a fresh-DB connection, in
ONE place so the test modules don't each re-spell them (a member-shape change is then a single edit).

pytest's default import mode (no ``tests/__init__.py``) puts this directory on ``sys.path``, so test
modules import these as ``from builders import ...``.
"""

from health_intelligence import db
from health_intelligence.models import MemberBundle


def fresh_con(path=":memory:"):
    """A connection to a freshly-initialized DB. In-memory by default; pass a file path for the API
    ``TestClient`` seam (which can't share an in-memory DB across per-request connections)."""
    con = db.connect(path)
    db.init_db(con)
    return con


def make_result(analyte, value, unit, reference_range):
    return {
        "analyte": analyte,
        "value": value,
        "unit": unit,
        "reference_range": reference_range,
    }


def make_panel(panel_id, date, results, vitals=None):
    # default vitals sit in-range and stable, so single-marker tests never accidentally raise the floor
    return {
        "panel_id": panel_id,
        "collected_date": date,
        "results": results,
        "vitals": vitals or {"systolic_bp": 118, "diastolic_bp": 76, "bmi": 22.5},
    }


def bundle_dict(
    member_id,
    panels=None,
    *,
    sex="male",
    age=50,
    conditions=None,
    medications=None,
    family_history=None,
    lifestyle=None,
    notes=None,
):
    """The MemberBundle as a RAW dict (un-validated) — what the API tests POST and mutate to exercise
    the route's own validation. With no ``panels``, supplies one in-range HbA1c panel so a caller that
    only needs 'some valid member' need not spell labs."""
    if panels is None:
        panels = [
            make_panel(
                f"{member_id}-P1",
                "2024-01-15",
                [make_result("HbA1c", 5.3, "%", "<5.7")],
            )
        ]
    return {
        "member_id": member_id,
        "profile": {
            "member_id": member_id,
            "age": age,
            "sex": sex,
            "conditions": conditions or [],
            "medications": medications or [],
            "family_history": family_history or [],
            "lifestyle": lifestyle or {},
        },
        "panels": panels,
        "notes": notes or [],
    }


def make_bundle(member_id, panels, **kwargs):
    """The validated ``MemberBundle`` the core / db / pipeline tests consume."""
    return MemberBundle.model_validate(bundle_dict(member_id, panels, **kwargs))
