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
- **Observations** (right): proactive findings raised without being asked, ranked by severity (§5). Present before the member types anything — the "persistent intelligence" idea made visible.
- **Control panel** (left, operator-only): a one-click button for every scaffolded API route, kept out of the chat — for driving demos, never member-facing (detailed below).

```
+----------------------------------------------------------------------+
| (i) Sample data — does not constitute medical advice.                |
| (!) Some results flagged for your care team.                  [view] |
+------------------+------------------------------+--------------------+
| CONTROL PANEL    | CONVERSATION                 | OBSERVATIONS       |
| (operator)       |                              |                    |
| Data             | member: should I worry       | (•) Needs          |
|  [Seed member]   |       about my ferritin?     |     follow-up      |
|  [Upload bundle] |                              |     Ferritin       |
|  [Clear member]  | +--------------------------+ |     below range    |
| Proactive        | | Ferritin drifted down,   | |     [why?]         |
|  [Run scan]      | | now just below normal.   | |                    |
|  [Observations]  | | Worth raising with GP.   | | ( ) Vitamin D dip  |
|  [Suggestions]   | | > why · v evidence       | |                    |
|  [Clinician q]   | +--------------------------+ | .   LDL stable     |
| Learning         |                              |                    |
|  [Run learn]     | [ Ask a question...      ^ ] |                    |
|  [Sample ovr]    |                              |                    |
|  [Reset learn]   |                              |                    |
| System           |                              |                    |
|  [Health]        |                              |                    |
| ---------------- |                              |                    |
| {route response} |                              |                    |
+------------------+------------------------------+--------------------+
```

The member never configures, tunes, or sees system internals — the conversation and observations are the entire member experience. The **control panel** (top-left, labeled operator-only) is the demo surface: a button for every scaffolded route, grouped **Data** (seed · **upload bundle** · clear member), **Proactive** (run scan · observations · suggestions · clinician queue), **Learning** (run learn · sample clinician override · reset learning), and **System** (health) — each route's JSON response shown in a readout below, the two destructive actions (clear, reset) behind a confirm. It is deliberately *not* in the chat and adds no new behaviour: it only fires routes the API already exposes, so the whole system is exercisable end-to-end from the page (run a scan and watch observations update; post an override then reset it; gate a `/learn` candidate) without curl.

**Upload bundle** is the holdout path — how an unseen dataset is brought in at runtime. The holdout is data we've never seen, so it can't be a pre-baked option: the button reads a `MemberBundle` JSON file, `POST`s it to `/members` (the same route "Seed member" uses), and the server validates it against the Pydantic models — a malformed file returns a clear error in the same readout, so the upload *is* the format check. On success the header **member-picker** gains the uploaded member and switches to it, and the surface repopulates for the new data. So on the deployment, unseen data can be loaded and demoed immediately — no redeploy, no curl — and the picker selects among the 15 seeded members and anything uploaded. The same `POST /members` is documented in the README as a `curl` for a scriptable equivalent; the sample bundle ships in the repo as the format template.

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
2. **Floating banner (ambient summary).** A calm, persistent one-liner whenever ≥1 escalation is active — *"Some results have been flagged for your care team."* This is the standing state; it's where "we've contacted your physician" lives, and it does **not** re-alarm.
3. **Per-turn bubble (event, once).** Only the single *transition* turn — when an escalation first fires, or a concerning message arrives — gets the loud in-bubble takeover. Every later turn has a clean bubble; the banner carries the status and the validator keeps the prose honest.

Two sources feed the same chrome (see `architecture.md`): the **proactive scan** (standing data findings → dropdown + banner) and a **concerning chat message** (→ the transition-turn takeover, then banner). **Member-facing care is never deduped:** if a member sends ten distressed messages, each still gets a full, present reply — only the *clinician record* is created once. We never answer a repeated cry with "already logged."

---

## 5. Proactive observations

Severity drives visual weight, and that's the whole model — there is no dismiss/acknowledge/resolve workflow in v1. Member-facing copy uses a calm register: `info`/`notable` read as **"Observation"**, `attention`/`urgent` as **"Needs follow-up"** paired with a next step. The four-tier `severity` enum is the internal truth that also sets the safety floor; the member just sees the gentler words.

| Severity | Visual |
|---|---|
| `info` | Quiet feed item, low contrast. |
| `notable` | Feed item, slightly emphasized. |
| `attention` | Amber card, prominent; tied to a "raise with your GP" next step. |
| `urgent` | Calm banner at top, immediate next step + clinician handoff. |

Each observation carries a **"why am I seeing this?"** expansion — its `trigger_reason` (which statistical signal fired) plus the evidence — so a proactive nudge is never a black box. The panel shows the member's current observations; a re-scan on new data replaces them, so the member isn't shown stale findings. This panel is the **detail layer** of the three-layer escalation rendering (§4); the floating banner is its ambient summary, and only a transition turn ever touches a chat bubble.

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

A polished design system, theming, animations, responsive breakpoints, auth, and account management are out of scope — this surface exists to demonstrate *how output lands*, not to ship. The microcopy and the disposition/severity treatments carry the UX judgment; everything else is intentionally rough, and a CLI chat loop is an honest fallback if time is better spent on the eval harness.
