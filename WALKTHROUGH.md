# Loom / Live Walkthrough Script (~10 min)

This is your speaking guide for the demo + code walkthrough. Sovann said the review
spends most time on the **agent loop** and the **trace**, so that's where the time goes.
Speak to *why* each decision was made — that's what he's evaluating.

Target: ~10 minutes. Rough budget in [brackets].

---

## 0. One-line framing [0:30]

> "This is an agent that sits on an audit inbox, reads messy PBC email threads with
> attachments, and keeps a structured request tracker current — with a full reasoning
> trace behind every status decision, because in audit everything has to be defensible
> to a PCAOB inspector."

Show the deployed UI tracker populated. Point at: statuses, evidence, confidence.

---

## 1. The agent loop — the core [3:30]  ← spend the most time here

Open `agent/loop.py`. Walk `run_agent` → `process_email` (grep the function name).

Say the three-phase shape out loud:

> "For each email the agent does three things, and it *decides* each one — this is not a
> hardcoded pipeline."

**a) Plan (`plan_email`, line 182).** Haiku call, forced to the `plan_decision` tool.
> "First a cheap model decides: is this email even relevant, which PBC items, and do we
> need to process or skip. That decision is the dynamic control flow — skipped emails
> cost almost nothing."

**b) Act (`run_tool_loop`, line 242).** This is the native tool-use loop.
> "Here the model chooses tools itself via Anthropic tool-use — parse_pdf, parse_excel,
> ocr_image, parse_zip, classify_document, extract_fields, update_item_status. I don't
> sequence these in Python; the model calls what it needs and I feed results back until
> it stops. That's the difference between an agent and a prompt-stitched pipeline."

Point at the `for iteration in range(max_iterations)` loop and the `tool_results`
feedback. This is the code Sovann will ask you to walk — know it cold.

**c) Verify (`verify_item`, line 311).** Separate Sonnet call.
> "A distinct verifier checks whether the collected evidence actually satisfies the
> acceptance criteria. Crucially it sees only the *structured extracted fields* and the
> criteria — not the raw document — so it can't invent new facts. It returns sufficient
> / insufficient / not_started."

---

## 2. Two similar emails, different paths [1:30]  ← he explicitly asks for this

Have two traces ready in the Traces tab (find them by subject):
- **"Re: FY2026 audit kickoff…"** (Jenna sends Excel/PDF attachments) → agent parses each,
  classifies, extracts with citations, verifies → PBC-07 / PBC-10 **Received**.
- **"PBC-05 bank reconciliations"** (Sam sends a whiteboard *photo*, IMG_2847) → agent runs
  **ocr_image**, sees it's an informal handwritten photo, and the verifier marks PBC-05
  **Insufficient** because a formal typed reconciliation was explicitly required.

> "Same shape — an email with an attachment — but the agent chose different tools
> (parse_excel vs ocr_image) and reached different verdicts, and the trace shows exactly
> why."

(Also good: PBC-04 — Jenna sends *one* bank statement but the criteria require *all*
accounts → Insufficient with the specific missing accounts named. Great insufficiency demo.)

---

## 3. Hallucination guardrails [2:00]  ← the regulated-domain story

Open `tools/citation_verify.py`.

> "Agents hallucinate. I don't try to eliminate that — I make it visible and unable to
> silently corrupt a decision. Three layers:"

1. **Citation re-verification (deterministic).** Every extracted value carries a citation
   — page, cell, or bbox. This module *greps the claimed value against the actual parsed
   document*. A fabricated number fails the check. No LLM, unit-tested, zero cost.
2. **Extract/verify separation** (point back at verify_item) — verifier can't re-hallucinate.
3. **Confidence-based degradation** (`tools/tracker_ops.py`) — if a citation can't be
   verified, the item can't be marked Received; it's downgraded to *Under review* with the
   reason in the trace. "Needs a human" is the correct answer under low confidence in audit.

---

## 4. Cost routing + evals [1:30]

> "Cost lever is model routing: Haiku for the per-email planning fan-out, Sonnet only for
> extraction, verification, and drafting — the steps where accuracy matters."

Show the cost in the UI header. Then `eval/run_eval.py`:

> "Evals are first-class. Against the ground truth: overall accuracy ~97–100%,
> insufficiency-detection recall 1.0 — we never miss a genuinely insufficient item — at
> about 60 cents a run. temperature=0 keeps it stable across runs."

Mention honestly: PBC-26 is a judgment-boundary case; you deliberately didn't overfit.

---

## 5. Close: scale + what's next [0:30]

> "It's stateless per email so it scales horizontally; at 10–100 concurrent audits the
> real work is moving run state to Redis/Postgres and a rate-limit-aware queue — that's in
> the README. The PBC list is pure config; nothing here is hardcoded to this engagement,
> which is why it runs on a swapped list cold."

---

# Q&A prep — likely questions

**"Show me where the model picks a tool."**
→ `run_tool_loop`, the `client.messages.create(..., tools=TOOL_DEFINITIONS)` and the
`for block in response.content: if block.type == "tool_use"` handling.

**"What stops it from hallucinating a citation?"**
→ `citation_verify.verify_citation` — deterministic grep against parsed doc; failure sets
`has_fabrication` and downgrades status in `tracker_ops._do_update`.

**"What if the held-out set has a corrupt/unknown attachment?"**
→ Every parser returns `{"error": ...}` instead of throwing; `parse_zip` fails per-member;
`execute_tool` wraps handlers in try/except. The run degrades, it doesn't crash.

**"Why free-form PBC parsing with an LLM — isn't that a cost/latency risk?"**
→ One Haiku call per *list*, cached to disk by content hash. It doesn't scale with mailbox
size and is free on re-runs from a clean checkout.

**"Why is it slow / how fast can it go?"**
→ Emails are processed concurrently (thread pool, `max_workers`). 520s → ~165s. Next lever
is batching plan calls and a shared parse cache across identical attachments.

**"Where's the state? Is this multi-tenant?"**
→ Currently a single in-memory run (fine for one audit / the review). Multi-tenancy =
per-engagement store keyed by id + queue; the loop itself is already stateless per email.

**Honest weak spots to own before he finds them:**
- Single global `current_run` in `main.py` — one audit at a time right now.
- PBC-26 judgment boundary; run-to-run variance exists even at temp 0 (small).
- No auth on the deployed app yet (synthetic data, but worth noting for a real deploy).
