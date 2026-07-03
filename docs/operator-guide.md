# Health Intelligence Service: Reviewer Guide

*How to drive the demo: the operator console, the happy path, and the two feedback loops that show the system learning. Every member is synthetic — no real health data anywhere. The [architecture writeup](writeup.md) explains why it's built this way.*

**Where:** the hosted demo at **https://health-intelligence-jfuf.onrender.com** — nothing to install, no login, shared by all reviewers. Local runs work too (see the README), but without an `ANTHROPIC_API_KEY`, Ask mode degrades to a deterministic fallback by design — use the hosted instance for the full model path.

## The two surfaces

The center chat and the right-hand Observations and Trajectory tabs are the member's experience; the left rail is the **Operator console**, a clinician and debug surface no member would see. Across the top sit the **Member selector** (a reviewer tool) and the **Mode toggle** — **Guided** (Mode 1: deterministic preset answers, no LLM) or **Ask** (Mode 2: free-form questions via the LLM). Both modes read the same analysis; only the phrasing differs.

## The operator console

**Data**

- **Upload a new valid .zip dataset** is a drag-and-drop target: drop a bundle of your own (a JSON of members, a JSONL eval set, a CSV of panels) and its members are format-checked, added on top of the existing set, and scanned.
- **Reseed (factory)** is the destructive counterpart: it wipes the shared demo back to the fifteen seeded members and the baseline prompt for everyone — use it to hand the next reviewer a clean state.

**Proactive**

- **Scan** re-runs the trajectory analysis and refreshes Observations; every data load already scans automatically, so this is the manual re-run.
- **Member escalations** is the selected member's own escalation record.

**Clinician**

- **Clinician queue** is the global, cross-member triage worklist, worst-first; seeing other members there is intended, not a leak.
- **Submit feedback** records a correction (range-override, marker suppression, or preference) or a signal (helpful, incorrect, escalation-accept, escalation-reject). For an incorrect answer you pick the question and type the correction — screened on submit, so a nonsense correction is rejected with the reason.
- **Member feedback** is the read-only audit trail for that member.

**Learning**

- **Run learn** assembles a new composer prompt from the accumulated feedback and adopts it only if it passes the evaluation gate. It runs a real evaluation — give it a minute or two.
- **Prompt history** lists every prompt version with its gate verdict, flagging the one in use.
- **Reset learning** reverts to the baseline prompt and deactivates all feedback; uploaded data stays.

**Route response** (bottom of the rail) always holds the raw JSON from the last call.

## The happy path

1. Pick a member and read **Observations**: what the proactive scan raised, worst-first, split between needs-follow-up and routine notes.
2. Open the **Trajectory** tab to see the series behind a finding — the numbers the analysis ran on.
3. In **Guided**, click a suggested question: a deterministic, grounded answer with evidence attached, no model involved.
4. Switch to **Ask** and type something free-form ("what's changed since last time?"); expand **Evidence** and **trace** under the answer for the exact values and routing behind it.
5. Ask something alarming, or something outside the member's data, to see refusal and escalation — and the concrete next step each gives.

## Two feedback loops

**Correction → observations.** Pick a flagged Observation — say HbA1c rising, now above range — and submit a clinician range-override that widens the range, or a suppression of that marker. A correction re-scans the member on submit, so the flag clears on the spot, the item leaves the queue, and both Guided and Ask answers reflect it immediately. Two deliberate exceptions, both safety features: a member-sourced override is inert (a member can't lobby the safety logic), and suppressing a *critically*-flagged value does nothing — only an explicit range change can clear a critical finding, and one that has already fired stays queued for human triage rather than ever being hidden.

**Signal → prompt.** In Ask, find an answer you would phrase differently, submit it as *incorrect*, and type the corrected answer. Run learn turns the accumulated signals into a candidate prompt and gates it; Prompt history shows the verdict. If adopted, future answers pick up your framing while every safety check still holds — the escalation floor is enforced outside the prompt, so learning can never weaken it. Reset learning reverts both loops.
