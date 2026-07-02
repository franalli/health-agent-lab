# Health Intelligence Service — UI/UX Intent

> Companion to `architecture.md`. Describes the thin consumer surface and, above all, **how output lands with a member in a vulnerable moment**. The surface is deliberately rough; the *judgment* about what a member sees, when, and in what tone is the point.

---

## 1. Design principles

Five principles, in priority order. When they conflict, the earlier one wins.

1. **Honest before reassuring.** Never trade truth for comfort. If the system can't be sure, it says so; if something warrants a clinician, it says that plainly. Reassuring phrasing must never soften a real flag (the central tension — §6).
2. **Calm, not alarming.** Health anxiety is real. Even an escalation is delivered in a steady, supportive register, framed as a *next step*, never a scare. No red klaxons, no all-caps "CRITICAL".
3. **Evidence on demand, never in your face.** Every claim is backed and inspectable, but the member sees a clean plain-language answer first; evidence is one tap away, collapsed by default.
4. **Show what's uncertain and what would change it.** Let the member see the edges of what the system knows — "this is based on three readings over two years" — so they can calibrate trust themselves.
5. **No diagnosis, no decisions for them.** The system describes patterns and surfaces what's worth raising; it does not diagnose, prescribe, or tell the member what to do beyond "this is worth raising with X."

---

## 2. The member surface

A single page: two **member** regions, plus an **operator** control panel held visually apart on the left (not part of the member experience).

- **Conversation** (center): the member asks free-form questions; answers render as cards (§3).
- **Observations** (right): proactive findings raised without being asked, ranked by severity (§5). Present before the member types anything — the "persistent intelligence" idea made visible. The right panel is **two tabs** — **Observations** (default) and **Trajectory** — sharing the selected member; the Trajectory tab is the member-facing per-marker sparkline view (§5).
- **Control panel** (left, operator-only): a one-click button for every scaffolded API route, kept out of the chat — for driving demos, never member-facing (detailed below).

```
+----------------------------------------------------------------------+
| (!) Some results flagged for your care team.                  [view] |  (only when active)
+----------------------------------------------------------------------+
| Health Intelligence       (!) Member [C07 v]  Mode [ Guided | Ask ]   |
+------------------+------------------------------+--------------------+
| CONTROL PANEL    | CONVERSATION                 | OBSERVATIONS       |
| (operator)       |                              |                    |
| Data             | member: should I worry       | (•) Needs          |
|  (drop .zip)     |       about my ferritin?     |     follow-up      |
|  [Reseed]        |                              |     Ferritin       |
| Proactive        | +--------------------------+ |     below range    |
|  [Scan member]   | | Ferritin drifted down,   | |     [why?]         |
|  [Observations]  | | now just below normal.   | |                    |
|  [Suggestions]   | | Worth raising with GP.   | | ( ) Vitamin D dip  |
|  [Clinician q]   | | > why · v evidence       | |                    |
|  [Trajectory]    | +--------------------------+ | .   LDL stable     |
| Learning         |                              |                    |
|  [Run learn]     | [ Ask a question...      ^ ] |                    |
|  [Submit fdbk]   |                              |                    |
|  [Sample ovr]    |                              |                    |
|  [Reset learn]   |                              |                    |
| System           |                              |                    |
|  [Health]        |                              |                    |
| ---------------- |                              |                    |
| {route response} |                              |                    |
+------------------+------------------------------+--------------------+
```

The **sample-data notice** ("synthetic only — not medical advice") is a persistent warning glyph `(!)` in the header next to **Member**, rather than a standing banner — it stays out of the way but reveals the full text on hover/focus (and carries it as an `aria-label` for screen readers). The ambient escalation summary lives as a severity-coloured flag badge on the **Observations** tab label (so it stays visible even from the Trajectory tab), appearing only when an escalation is active — so the header stays uncluttered.

The two seams between these regions are **drag-resizable gutters**: the operator can widen the control panel, the member can give Observations more room, and the centre conversation flexes to fill what's left. Each side is clamped (never past 40% of the viewport, so the conversation is never squeezed out) and the chosen split is held in memory only — no storage, reset on reload (double-click a gutter, press Home, or just reload to restore the default). The gutters are keyboard-operable separators (focus, then ←/→). Below ~820px the three regions stack into one scrolling column and the gutters drop away. *(Build note, Phase 6.)*

The member never configures, tunes, or sees system internals — the conversation and the right panel (Observations + Trajectory tabs) are the entire member experience. The **control panel** (top-left, labeled operator-only) is the demo surface: a button for every scaffolded route, grouped **Data** (a `.zip` **upload dropzone** · factory reseed), **Proactive** (scan · sample clinician override · member escalations), **Clinician** (clinician queue · submit feedback), and **Learning** (run learn · reset learning) — each route's JSON response shown in a readout below (the **submit-feedback** form is the exception, drawn as inline UI in the readout instead of JSON). The **Clinician** group is the human-in-the-loop triage surface: read the global queue, then file the corrections against a finding on it. **Clinician queue** is the *global* triage worklist — `GET /escalations`, every open escalation across all members ranked worst-first, so it needs no member selected (escalation means "make sure a human sees this", which a per-member view can't guarantee); **Member escalations** (`GET /members/{id}/escalations`, under Proactive) is the per-member drill-in — the selected member's own escalation record, the detail view to the queue's worklist. The remaining per-member GET reads — observations, suggestions, trajectory — are deliberately **not** surfaced as operator buttons: those endpoints still exist and drive the member-facing surfaces (the right-panel Observations + Trajectory tabs, the ambient escalation flag, the Mode-1 chips), reached through the member's own view. The **submit-feedback** button opens a small form over the full `/feedback` surface: pick a `kind` (the three correction kinds — `range_override` · `suppress_marker` · `preference` — or the four signal kinds), a `source`, and the kind-appropriate target/payload. The **target is a dropdown, never free text** — markers from the member's trajectory for the correction kinds; for the signal kinds, a finding from the current observations *or an observation-less (chat-driven) escalation* — so an escalation raised only in an `/ask` turn, which has no observation row, can still receive an `escalation_accept`/`escalation_reject` (`target` is audit-only for signals) — **except `incorrect`**, which targets an **answered question** (the member's suggested prompts, each a `{question, answer}` pair) and shows a **Corrected answer** field: picking a question auto-fills it read-only and the clinician types only the correction, posted as the `{question, corrected_answer}` payload `learn.py` folds into a few-shot exemplar (§58). So a non-matching target (which the API accepts with a 200 and then leaves silently inert on the next scan) can't be submitted; together with the per-kind payload contract the form encodes (the permissive `dict` schema does not), a submission can never silently no-op on a stray key or a typo'd target. Then submit against the selected member. *(Two paths, one table: the correction kinds feed the **deterministic core** on the next Scan/Ask via `resolve_overrides` (a `preference` joins the composer context instead — never an analysis input); only the four **signal** kinds feed `POST /learn`. **Sample override** therefore sits under **Proactive**, beside Scan — its effect surface — not by the Learning controls, so a correction never reads as a `/learn` input. Applying a `range_override` and then running **Run learn** is expected to report* no signals *— and `/learn`'s no-op now names the active corrections so that reads as "on the other path", not "picks up nothing".)* Every non-destructive button fires its route DIRECTLY — no confirm, no client-side preconditions — an unrestricted operator API console; the two destructive actions (reset learning, reseed) are the exception: each carries a red visual cue **and** opens a confirm popup — a custom modal overlay (never native `confirm()`, in keeping with the panel's own-dialog style), dismissable by Cancel / Escape / backdrop-click — *but not while the confirmed request is in flight*, since there is no way to abort it and dismissing would read as "cancelled" while the wipe proceeds — that fires the route only on explicit confirm, since both are irreversible from the UI. It is deliberately *not* in the chat and adds no new behaviour: it only fires routes the API already exposes, so the whole system is exercisable end-to-end from the page (run a scan and watch observations update; post an override then reset it; gate a `/learn` candidate) without curl.

*(Build note: the panel is driven by a route registry — a button per **scaffolded** route — so each phase's routes appear simply by being added to the registry. As of **Phase 7** the learning + clinician routes (`/learn` · `/reset` in **Learning**; `/feedback` in **Clinician**) have landed alongside the **factory reseed** (`/admin/reseed`, Data); **Reset learning** and **Reseed** are destructive — a red cue *and* a confirm popup (a custom modal overlay, dismissable by Cancel / Escape / backdrop-click) gate them before firing; every other button fires its call directly, by design. The **Submit feedback** button is the one non-JSON readout — drawn as an inline form over the `/feedback` surface instead of a raw-JSON response. (`GET /members/{id}/trajectory` also landed in Phase 7, but it is **not** an operator button: its minimal per-marker sparkline — readings · the already-computed Theil–Sen line · flagged points · the reference bound(s) with their numeric labels in a left gutter, deliberately rough, no legend/tooltips — is the member-facing **Trajectory** tab, §131, not an operator readout.))*

**Upload** is the holdout path — how an unseen dataset is brought in at runtime. The holdout is data we've never seen, so it can't be a pre-baked option: a **drag-and-drop dropzone** at the top of the **Data** group takes a **`.zip` of a `training_data`-shaped folder** carrying exactly three files matched **by extension** — one `.json` (members), one `.jsonl` (eval set), one `.csv` (lab panels), under any base names (`roster.json`, `cases.jsonl`, `panel.csv` all work; the hold-out ships stable extensions, only the names differ) — plus optional extras, and `POST`s it as multipart to `/members/upload`. The server **format-gates the first row of each file** against the Pydantic models / CSV schema, so a missing/duplicated file or a malformed first row returns a clear **per-row reason** in the same readout (`{file, row, field, detail}` for each problem): the upload *is* the format check. Past that gate it (1) **saves the whole bundle** as a new dataset folder under `backend/data/` (named after the zip, so the kept `eval_set.jsonl` is usable later via `DATASET=<name> make eval`; a name collision is a clear 409, never an overwrite), (2) ingests the members **additively** — added on top of the existing data, no reseed, **skipping any buggy later row** (reported in a `skipped` list) — and (3) **auto-scans** the new members so their Observations are populated at once (as every ingest path now does — the startup seed and the factory reseed too, so Observations are always consistent with whatever data is loaded). On success the header **member-picker** gains the uploaded members and switches to the last one, and the surface repopulates for the new data. So on the deployment, an unseen dataset can be loaded and demoed immediately — no redeploy, no curl — and the picker selects among the 15 seeded members and anything uploaded. The single-bundle `POST /members` is documented in the README as a `curl` for a scriptable equivalent. Upload is a **dropzone rather than a file-picker button** deliberately: some browsers (certain Chrome instances) silently refuse to open the native file-selection dialog, so a click-to-pick control looks dead, whereas a dropped file arrives via `dataTransfer` with no dialog. Clicking the dropzone falls back to the picker where it works.

A production member build simply omits this column — together with the header's member-picker and the per-answer trace (the stored `response_json` / logs) — as operator affordances, never member features.

**Two modes — Mode 1 (deterministic, default) and Mode 2 (LLM).** A single toggle flips the whole surface between an LLM-off and an LLM-on experience (architecture §2). **Mode 1** is the default and the calm landing: the proactive observations plus 2–5 data-derived preset prompts (common example questions among them), each answered instantly and fully-grounded with no model call — which solves the blank-page problem in a vulnerable moment and keeps the common path fast and consistent. **Mode 2** is the "ask in your own words" chat. Both render with the same answer-card anatomy (§3), evidence chips and all, so the two modes are indistinguishable in quality to the member; Mode 1 is simply faster and bounded to anticipated questions, gracefully pointing to Mode 2 for anything else.

---

## 3. Anatomy of an answer card

Reading order is the design, top to bottom:

1. **Plain answer** — one or two sentences a non-expert can act on. Lead with this, always.
2. **Uncertainty line** (when relevant) — what the answer rests on and what would change it: *"Based on 3 ferritin readings since 2023; a current panel would sharpen this."*
3. **Evidence chips** (collapsed) — each load-bearing claim links to its evidence: the marker, value, date, reference range (shown as the panel reported it), and trend statistic it came from. Because those numbers came from deterministic code, the chip shows them verbatim with confidence.
4. **Disposition & escalation treatment** — the card's body follows `answer_disposition` and its chrome/banner follows `escalation` (§4). The member never sees the enums; they see the consequence.

---

## 4. Disposition & escalation → UI treatment

The response carries **two axes** (see `architecture.md`): `answer_disposition` (did we engage the question) and `escalation` (does a human need to act, and how fast). The member never sees either enum — only its consequence. The **body** of the card follows `answer_disposition`; the **chrome/banner** follows `escalation`. They're independent, so "answered **and** flagged for your GP" renders as a normal answer *with* a calm escalation banner.

**Answer body** (`answer_disposition`):

| Value | Treatment | Microcopy stance |
|---|---|---|
| `answered` | Standard answer card, evidence chips. | Direct, plain, grounded. |
| `out_of_scope` | Friendly redirect, no evidence chips. | "That's outside what I can help with — here's who can." |
| `refused` | Gentle, still useful. | Names the limit, points to a clinician / their note. |

**Escalation chrome** (`escalation` — deterministic, set by the data floor or the message gate):

| Value | Treatment |
|---|---|
| `none` | No banner. |
| `clinician_review` | Calm banner with a concrete next step ("worth raising with your GP, not an emergency"). |
| `urgent` · acute | Prominent-but-calm takeover: the unmissable action leads ("contact emergency services / urgent care now"); the normal answer is withheld if it would dilute. |
| `urgent` · crisis | The crisis-support responder, not a clinical frame: warm, present, resources; never disengages. |

The two `urgent` rows come from the gate's `acute_medical` vs `crisis` intents — same floor level, deliberately different responders (a crisis reply must not read like a lab result). The rule the member can't see but always benefits from: **a softer-toned body can never sit on top of a harder escalation** — the validator guarantees the words never under-sell the floor the safety layer set.

### Three rendering layers — and one hard rule

An escalation is **loud exactly once**, then becomes ambient. The hard rule: **an escalation never lives inside every message bubble** — repeating it each turn is both annoying and dishonest (it implies a fresh event). It lives in the conversation chrome instead:

1. **Side dropdown (detail).** The Observations panel (§5): standing severity items, persistent, expandable to their evidence. Where a member goes to see *what* is flagged.
2. **Tab-label flag (ambient summary).** A calm severity-coloured flag badge on the **Observations** tab label whenever ≥1 escalation is active (so it's visible from either tab); its hover/focus message is *"Some results have been flagged for your care team."* (also an `aria-label`, kept short so it folds cleanly into the tab's accessible name). This is the standing state; it's where "we've contacted your physician" lives, and it does **not** re-alarm. Clicking the Observations tab while flagged scrolls to and pulses the flagged items.
3. **Per-turn bubble (event, once).** Only the single *transition* turn — when an escalation first fires, or a concerning message arrives — gets the loud in-bubble takeover. Every later turn has a clean bubble; the flag carries the status and the validator keeps the prose honest.

Two sources feed the same chrome (see `architecture.md`): the **proactive scan** (standing data findings → dropdown + flag) and a **concerning chat message** (→ the transition-turn takeover, then flag). **Member-facing care is never deduped:** if a member sends ten distressed messages, each still gets a full, present reply — only the *clinician record* is created once. We never answer a repeated cry with "already logged."

---

## 5. Proactive observations

Severity drives visual weight, and that's the whole model — there is no dismiss/acknowledge/resolve workflow in v1. Member-facing copy uses a calm register: `info`/`notable` read as **"Observation"**, `attention`/`urgent` as **"Needs follow-up"** paired with a next step. The four-tier `severity` enum is the internal truth that also sets the safety floor; the member just sees the gentler words.

| Severity | Visual |
|---|---|
| `info` | Quiet feed item, low contrast. |
| `notable` | Feed item, slightly emphasized. |
| `attention` | Amber card, prominent; tied to a "raise with your GP" next step. |
| `urgent` | Red severity flag on the Observations tab, immediate next step + clinician handoff. |

Each observation carries a **"why am I seeing this?"** expansion — its `member_explanation`, a deterministic plain-language account of what the finding means for the member, and for an out-of-range value the **actual reference range** so they can see how far off they are ("…above the normal range (under 100 mg/dL)") — so a proactive nudge is never a black box. **This is the member audience; the statistical `trigger_reason` (which test fired — Mann–Kendall *p*, RCV) is clinician/operator data that surfaces only on the operator console's raw-JSON readouts, never to the member.** Both narrations are deterministic and derived from the same `_classify` signal, and the member text never carries the statistics. *(Build note, Phase 8: `member_explanation` is **derived at read** by the `/observations` projection (`pipeline.observations` re-runs the analysis and composes it via `templates.observation_member_explanation`) — not stored beside the clinician `trigger_reason`, so it never goes stale and always matches the value+range the Trajectory tab shows; the LLM is deliberately **not** used — it is forbidden to state a numeric bound (`llm.py` rule 1: "The system prints the exact bounds as evidence"), so only the deterministic layer may print the real range. The referral nudge is severity-gated — a bare out-of-range value gets none (Escalation ≠ out-of-range). The full evidence chips for that marker are one click away via its Mode-1 chip in the conversation, since the `Observation` projection carries no `evidence[]` and §13 adds no observation→interaction read route.)* The panel shows the member's current observations; a re-scan on new data replaces them, so the member isn't shown stale findings. This panel is the **detail layer** of the three-layer escalation rendering (§4); the severity flag on its tab label is the ambient summary, and only a transition turn ever touches a chat bubble.

The right panel's second tab, **Trajectory**, is the member-facing per-marker sparkline view: one rough per-marker sparkline (readings · Theil–Sen line · flagged points · the reference bound(s); no legend/tooltips) over `GET /members/{id}/trajectory`, ordered flagged-markers-first to mirror the Observations ranking. The reference range is drawn as **shaded regions**: the good (in-range) area is **green** and the bad (out-of-range) area is **red**, both at 0.1 opacity, with a hairline divider + the bound's **numeric value** in a small left-axis gutter at each boundary. "Good = inside the range" is read straight from the range's *shape* — no adverse-direction config: a **two-sided** range (e.g. BMI 18.5–25) is green between the bounds and red above/below them; a **one-sided** range shades from its single bound — only-`ref_high` (LDL `<100`) is green below the line and red above, only-`ref_low` (Vitamin D `≥20`) is green above and red below. Because every reading and both bounds are inside the plot's value domain, a comfortably-in-range marker shows only a thin red sliver at the far edge — the shading is honest about how much headroom remains. Correspondingly the flag text adapts to the range shape: an out-of-range value reads `above/below range` for a two-sided marker but `above/below threshold` for a one-sided one. A marker that is **within its bound(s) *and* benign per the core** (deterministic `severity` = `info`) gets the positive counterpart — a **circled green check** (icon only, no words; an accessible `title`/`aria-label` carries "within range"/"within threshold"). The check is severity-gated, not merely range-gated: an in-range value with an escalating adverse trend (severity ≥ `notable`) shows **no** check, since *in-range can still escalate* (§ escalation law) and a calm check there would falsely reassure. A boundless marker (no reference bound at all) gets neither flag nor check. Each card carries an **"All N readings"** disclosure (`<details>`, mirroring the Observations "why am I seeing this?" affordance) that expands to the **exact dated value of every reading** in the series (date · value, oldest→newest to match the chart) — so the sparkline isn't the only way to read the numbers. That list is a **verbatim echo** of the `readings[]` the route returns (the member's own measured inputs); it does **not** re-classify any reading against the range — the one `severity` per marker stays the deterministic core's to assign, so the UI asserts nothing the core didn't produce. It is **presentation only** — it shows the raw per-marker series for human eyes, while the **LLM still consumes only the collapsed `TrajectoryAnalysis` verdict, never this series** (architecture §4/§207). The tab lazy-loads on first open and caches per member (invalidated on member switch and on a re-scan).

**Only the Observations tab gates on the scan; the Trajectory tab is scan-independent.** The Observations tab is the *proactive* surface ("we flagged this for you"), so until a scan has produced observations it shows `"No results to show yet — insights appear here once your latest lab results have been reviewed."` (deliberately *not* "Not analyzed yet" — that would be false for a scanned-clean member whose scan correctly writes zero observations, and would contradict an active care-team flag; and deliberately not "run a scan" — a member instruction to an operator control). Since every ingest path auto-scans (seed / reseed / upload), that empty state is normally the *scanned-clean* case — a healthy member with nothing flagged — not an unscanned member. The Trajectory tab is the *inspection* surface and does **not** wait for a scan: `GET /members/{id}/trajectory` recomputes from the live analysis on every call (it never reads the `observations` table), so the tab renders a member's per-marker sparklines as soon as it is opened — including for a *healthy* member who would raise nothing and leaves no scan trace. The two surfaces are deliberately **decoupled**: a pre-scan Trajectory can surface an out-of-range/panic flag that the Observations tab will only echo once scanned. That is intentional, not a leak — the Trajectory flags come straight from the deterministic analysis (always truthful, never fabricated) and the sparkline surfaces panic/out-of-range as icon+text (the *narration-must-surface-core-flags* invariant), so showing the live series earlier is more honest, not less. (We deliberately do **not** backfill the Trajectory tab with anything from the scan-populated observations — its data is its own live series.) *(Build note, Phase 8; decoupled post-Phase-8.)*

---

## 6. Microcopy guidance

Tone: a calm, well-informed friend who is also scrupulously honest — not a chirpy chatbot, not a clinical report.

- **Uncertainty.** ✗ "Your ferritin is fine." ✓ "Your ferritin has drifted down over about 18 months and is now just below the normal range. It's worth raising with your GP — it isn't an emergency."
- **Escalation (calm-direct).** ✗ "⚠️ WARNING: ABNORMAL VALUE." ✓ "One of your recent results is outside the usual range in a way that's worth a clinician looking at. I'd contact your GP in the next few days."
- **Never diagnosing.** ✗ "You have iron-deficiency anaemia." ✓ "This pattern is often linked to low iron, but your clinician can interpret it properly."
- **Refusal that still helps.** ✗ "I can't answer that." ✓ "I can't tell that from your data — but your GP could, and your last note already mentions a follow-up on this."

---

## 7. The central tension (and how we resolve it)

The safety-correct move — escalate, or admit "I can't be sure" — frequently collides with what would most reassure a member in a vulnerable moment. We resolve it toward **honesty with a clear next step**, via a division of labor: the deterministic layer decides *whether* to escalate (the member experience cannot lobby the safety logic), and the LLM decides only *how to say it* kindly. The member gets warmth and a calm tone, but never at the cost of the truth — the reassurance is in the delivery, never the substance.

---

## 8. Accessibility & vulnerable-moment care

- Severity, disposition, and escalation are **never conveyed by colour alone** — icon + text label back every amber/red treatment (colour-blind safe, and calmer).
- Plain language by default; medical terms glossed inline on first use.
- One clear primary action per card (the "next step"), so a worried member isn't hunting.
- No dark patterns, no urgency manufactured for engagement; the only urgency shown is clinical and real. The system never thanks the member for "reaching out" or nudges them to keep chatting — it answers, surfaces what matters, and gets out of the way.

---

## 9. Out of scope

A polished design system, theming, animations, a multi-breakpoint responsive grid, auth, and account management are out of scope — beyond the drag-resizable side panels and the single narrow-screen stack (§2), this surface exists to demonstrate *how output lands*, not to ship. The microcopy and the disposition/severity treatments carry the UX judgment; everything else is intentionally rough, and a CLI chat loop is an honest fallback if time is better spent on the eval harness.
