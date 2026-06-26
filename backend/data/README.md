# Founding AI Engineer Mini Project: Sample Data Bundle

Synthetic data for the Health Intelligence Service mini project. Nothing here is real patient data.

## Contents
- `members.json`     15 synthetic members. Each: { member_id, profile, panels, notes }.
                     Each panel result carries its reference range, as a lab report prints it.
- `lab_panels.csv`   All panels across all members in long (tidy) form, for quick reading.
- `eval_set.jsonl`   A labeled reference set (one case per line). Each case names the `member_id`
                     it applies to and covers grounded Q&A, trend-versus-noise, alarming values,
                     out-of-scope and unsafe requests, grounding/hallucination traps, borderline and
                     managed-condition nuance, improving trajectories, sparse history, and uncertainty.

## members.json shape
Each member has 3 to 5 panels spanning up to two years. The 15 members span a deliberate range:
healthy baselines, improving trajectories, and a variety of developing patterns. We do not label
each member's "story" for you; reading the trajectories is part of the exercise.

## eval_set.jsonl fields
`id`, `member_id`, `category`, `input`, `expected_behavior`, `must_include`, `must_not`, `escalation_expected`.
Some cases inject a hypothetical new value (e.g. an alarming potassium) on top of a member's history; the case text says so.

## Notes
- Reference ranges are illustrative and synthetic. Do not use any of this for real clinical decisions.
- Extend the data if it helps; note briefly what you added and why.
- The labeled set is a starting point for your evaluation harness, not the full bar.
