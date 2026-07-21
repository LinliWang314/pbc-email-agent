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
| **Cost per PBC list** | **~$0.58** (well under the $2 target) |
| Wall-clock | ~165s (emails processed concurrently) |

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
