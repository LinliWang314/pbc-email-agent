# PBC Email Agent

An agentic system that ingests a messy PBC (Prepared-By-Client) audit email thread
with attachments and produces a live, structured audit-request tracker — with a full
reasoning trace behind every status decision, defensible to a PCAOB inspector.

---

## Quick start (clean checkout → running app)

```bash
git clone https://github.com/LinliWang314/pbc-email-agent.git
cd pbc-email-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # core deps only — all prebuilt wheels
# optional: pip install -r requirements-ocr.txt   # image OCR (needs tesseract binary)

export ANTHROPIC_API_KEY=sk-ant-...        # required for a real run
# PBC_DATA_DIR is optional — defaults to the bundled sample_data/
export PBC_DATA_DIR=/path/to/data          # dir with PBC_List PDF + sample/ mailbox

python main.py                             # serves UI at http://localhost:8000
```

Then open `http://localhost:8000`, click **Run Agent**, and inspect the Tracker,
Agent Traces, and Follow-ups tabs. Without an API key the app still boots and parses
deterministically (mock/offline mode) but does not classify. OCR is optional — without
the OCR extras the `ocr_image` tool degrades to a "needs manual review" result rather
than failing.

Run the deterministic test suite (no API key, no cost):

```bash
python tests/test_citation_verify.py
python tests/test_versioning.py
python tests/test_parsing.py
python tests/test_extraction.py
```

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │  PBC list PDF (free-form)  +  client profile │
                         └───────────────────┬─────────────────────────┘
                                             │  parse_pbc_list_llm (1 Haiku call, cached)
                                             ▼
   .eml / .mbox ──ingest──▶  Emails (thread-reconstructed via In-Reply-To)
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    │           AGENT LOOP  (per email)                │
                    │                                                  │
   [PLAN]  Haiku ── decides: process / skip? which items? which tools? │
                    │                                                  │
   [ACT]  Sonnet ── native tool-use loop, model picks tools:           │
                    │   parse_pdf · parse_excel · ocr_image            │
                    │   classify_document · extract_fields             │
                    │   update_item_status · get_item_status           │
                    │                                                  │
                    │   ↓ every parse output cached on the run         │
                    │   ↓ every citation deterministically re-checked  │
                    │     against the parsed doc (anti-hallucination)  │
                    │                                                  │
   [VERIFY] Sonnet ─ independent verifier: does the evidence actually  │
                    │  satisfy this item's acceptance criteria?        │
                    │  (sees ONLY structured fields + criteria, not    │
                    │   the raw doc → cannot re-hallucinate)           │
                    └────────────────────────┬─────────────────────────┘
                                             ▼
                        Tracker state  +  full trace per item
                                             │
                    [DRAFT]  Sonnet ── grouped follow-ups, one per recipient
                                             ▼
                         UI: Tracker · Agent Trace · Follow-up review
```

Core agent loop lives in a single inspectable file: **`agent/loop.py`** (~300 lines).

## The agentic workflow (how it actually works)

This is a real agent — the model decides control flow via **native Anthropic tool-use** —
not a hardcoded pipeline or a prompt-stitching chain. Per email, three phases:

**1. PLAN (`plan_email`, Haiku).** The cheap model is forced to call a `plan_decision`
tool and returns `{action: process|skip, relevant_items, reasoning}`. This is the dynamic
control-flow gate: irrelevant emails are skipped for almost no cost. A deterministic
`_fast_skip` runs first and drops pure acknowledgements (no attachment, no PBC terms)
before any LLM call at all.

**2. ACT (`run_tool_loop`, Sonnet).** The real agent loop. The model is given the full
tool set and **chooses which tools to call itself**; we execute them and feed results back:

```python
for _ in range(max_iterations):
    resp = client.messages.create(model=..., system=cached_system(...),
                                   messages=messages, tools=TOOL_DEFINITIONS)
    if resp.stop_reason == "end_turn":
        break                                  # model decides it's done
    for block in resp.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input, run, email)
            tool_results.append({... "content": result})
    messages += [assistant(resp.content), user(tool_results)]   # feed back, loop
```

The model picks `parse_pdf` vs `parse_excel` vs `ocr_image` vs `parse_zip` based on the
attachment, extracts fields with citations, and calls `update_item_status`. Nothing about
the order is hardcoded — that's what makes it an agent. Two similar-looking emails take
different tool paths (an Excel → `parse_excel` → Received; a whiteboard photo →
`ocr_image` → Insufficient), and the trace shows exactly why.

**3. VERIFY (`verify_item`, Sonnet).** A *separate* call decides sufficient / insufficient /
under_review / not_started against the acceptance criteria. It sees only the structured
extracted fields + criteria — never the raw document — so it cannot introduce new "facts."

Tools are the abstraction (schemas in `tools/registry.py`): `parse_pdf`, `parse_excel`,
`ocr_image`, `parse_zip`, `classify_document`, `extract_fields`, `update_item_status`,
`get_item_status`. Every plan, tool call, and verdict is recorded on the per-email
`AgentTrace`, which is what the UI's Agent Traces tab renders.

### On hallucination (honest)

The design assumption is that the LLM *will* hallucinate; the system makes it visible and
unable to silently corrupt a status decision (see "Hallucination guardrails" below):
citation re-verification against the real document, extract/verify separation, and
confidence-based degradation to *Under review*. Measured: overall accuracy 96.7–100% and
**insufficiency recall is always 1.0** — a genuinely incomplete item is never missed. The
residual is the occasional judgment-boundary call (e.g. PBC-26), which we surface with a
defensible trace rather than hide.

## Model choices per step

| Step | Model | Why |
|---|---|---|
| PBC list structuring | Haiku | One call per list, not per email. Cheap, cached to disk. |
| Plan (process/skip, routing) | Haiku | High volume (once per email), simple classification. |
| Extraction w/ citations | Sonnet | Needs careful reading + precise field/citation output. |
| Verification | Sonnet | Judgment call against acceptance criteria; accuracy matters most here. |
| Follow-up drafting | Sonnet | Customer-facing prose; quality matters. |

Routing is the cost lever: the cheap model handles the per-email fan-out; the
expensive model only runs on emails that actually carry evidence.

## Tool schemas

Tools the model calls via native Anthropic tool-use (full schemas in `tools/registry.py`):

- `parse_pdf(filename)` → pages + text
- `parse_excel(filename, sheet_name?)` → sheets, rows, cell refs
- `ocr_image(filename)` → OCR text + bounding boxes (graceful fallback if OCR unavailable)
- `parse_zip(filename)` → lists archive contents and parses inner PDF/Excel/image files (batched submissions)
- `classify_document(filename, content_summary, document_type)` → candidate PBC items + scores
- `extract_fields(filename, pbc_item_id, content, fields_to_extract?)` → fields + citations
- `update_item_status(item_id, status, reasoning, evidence_filename?, citations?, ...)`
- `get_item_status(item_id)`

## Hallucination guardrails

Three layers. The design assumption is that the LLM *will* hallucinate; the system
makes it visible and unable to silently corrupt a status decision.

1. **Citation re-verification (deterministic, `tools/citation_verify.py`).**
   Every value the model claims to have extracted carries a citation (page / cell / bbox).
   We deterministically grep the cited value against the *actual parsed document*. A
   fabricated value fails the check. No LLM, fully unit-tested, zero cost.
2. **Extract/verify separation.** The verifier sees only the structured extracted fields
   and the acceptance criteria — never the raw document — so it cannot re-hallucinate new
   "facts." It only reasons over already-checked evidence.
3. **Confidence-based degradation.** If any citation fails verification, an item cannot be
   marked *Received/Complete* on that evidence — it is downgraded to *Under review* with the
   reason recorded in the trace. In a regulated domain, "needs a human" is the correct
   answer when confidence is low, not a confident wrong call.
4. **Self-consistency voting on the verifier (optional, default off).** Implemented as a
   tunable knob (`AgentConfig.verifier_votes`): run N verifier votes and take the majority.
   Measured on the sample it did *not* reduce variance on the genuinely-ambiguous boundary
   items — the diversity temperature just re-introduced noise at ~30% higher cost — so the
   default is a single deterministic shot (temp 0), the most stable config measured. Left in
   as an honest negative result and a lever for harder future cases.

## Eval strategy

`eval/run_eval.py` scores a run against labeled ground truth (`sample_groundtruth.json`):

- Status classification precision / recall / F1 per status, plus overall accuracy
- **Insufficiency-detection F1** (the hardest and most valuable signal) with TP/FP/FN detail
- Follow-up grouping: recipient match + item coverage
- Cost per PBC list processed (USD)

Ground truth for the sample: 5 items *Insufficient*, 3 *Received*, rest *Not started* —
so the eval is weighted toward the Received-vs-Insufficient boundary, which is where the
product's value lives.

## Measured results (sample set, 15 emails / 30 PBC items)

Full plan → act → verify → draft run against the sample, scored vs. ground truth:

| Metric | Result |
|---|---|
| Overall status accuracy | **96.7%** (29/30) |
| Insufficiency-detection recall | **1.00** (every genuinely-insufficient item flagged) |
| Insufficiency-detection precision | 0.83 (one stable judgment-boundary case, PBC-26) |
| Follow-up recipient match | 2/2, item coverage 100% |
| **Cost per PBC list** | **~$0.38** with prompt caching (~$0.58 without), vs. the $2 target |
| Wall-clock | ~165s (emails processed concurrently) |

**Performance levers.** Three are in place:
1. **Concurrency** — emails are processed in a thread pool (520s → ~165s on the sample).
2. **Prompt caching** — the large static prefix (the full PBC list, embedded in the
   planning/extraction system prompts) is marked with `cache_control` and reused across
   every call. On the sample: ~124k tokens served from cache vs ~9.5k written, cost ~$0.58
   → ~$0.38. Cache read/write counts are in `/api/cost`.
3. **Deterministic fast-skip** — emails with no attachments and no PBC-relevance hints
   (brief acknowledgements like "Working on it — J.") are skipped before any LLM call,
   avoiding a whole plan→act→verify chain. Conservative by design (any attachment or any
   PBC term defers to the planner), so on the tight 15-email sample it skips nothing; its
   payoff is on large, noisy mailboxes where a meaningful fraction of mail is chatter. Every
   skip is recorded in the trace with its reason.

The single miss (PBC-26 going-concern memo) is a defensible judgment disagreement, not a
parsing error — the agent's trace argues the referenced 12-month forecast attachment
wasn't actually sent, which a real auditor might well flag. We deliberately did not
over-fit the verifier prompt to force this one case, to preserve generalization to the
held-out set. Setting temperature=0 removed almost all run-to-run variance.

Cost is surfaced live in the UI header and via `GET /api/cost`.

## What breaks at 10 and 100 concurrent audits

- **Current state:** single-process, in-memory `current_run` — fine for one audit at a time
  (the review scenario), not concurrent.
- **At 10 concurrent:** move run state out of the module global into a per-engagement store
  (Redis / Postgres), key traces + tracker by engagement id. Anthropic rate limits become
  the bottleneck before compute does — batch/queue the per-email calls.
- **At 100 concurrent:** the per-email Haiku fan-out dominates the rate-limit budget. Needs
  a work queue (SQS/Celery) with per-tenant concurrency caps, a shared embedding/parse cache
  (identical attachments recur across audits), and provisioned throughput. The agent loop
  itself is stateless per email, so it scales horizontally; the shared state and rate limits
  are what require real infrastructure.

## Layout

```
agent/loop.py          core agent loop (plan → act → verify → draft)
agent/pbc_parser.py    free-form PBC list → structured items (LLM, cached)
agent/ingest.py        .eml/.mbox + PBC/profile parsing (deterministic)
agent/models.py        data models (PBCItem, TrackerState, AgentTrace, ...)
agent/config.py        model routing + cost config
tools/registry.py      tool schemas + dispatch (+ parse cache for citation check)
tools/parsers.py       PDF / Excel / OCR (deterministic)
tools/classification.py document → PBC item matching
tools/extraction.py    field + citation extraction
tools/citation_verify.py  anti-hallucination citation re-check (deterministic)
tools/tracker_ops.py   status updates, versioning, confidence degradation
eval/run_eval.py       eval harness
main.py                FastAPI server + REST API
ui/index.html          Tracker / Trace / Follow-up UI
tests/                 deterministic tests (parsing, versioning, extraction, citations)
```
