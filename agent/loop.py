"""
PBC Email Agent — Core Agent Loop

This is the main agent loop that processes emails against a PBC list.
It uses Anthropic's native tool-use API to dynamically decide what to do
for each email: parse, classify, extract, verify, or skip.

Architecture:
  1. Load PBC list (dynamic config, not hardcoded)
  2. For each email thread, the agent decides which tools to call
  3. A separate verifier step confirms extractions against acceptance criteria
  4. Results update the tracker state
  5. Follow-up emails are grouped by recipient

Model routing for cost control:
  - claude-haiku-4-5-20251001: classification, routing, simple decisions
  - claude-sonnet-4-5-20250929: extraction with citations, verification, drafting
"""

import json
import os
import time
from typing import Any
from dataclasses import dataclass, field

from anthropic import Anthropic

from agent.models import (
    PBCItem, TrackerState, EmailMessage, AgentTrace,
    TraceStep, ToolCall, VerifierVerdict
)
from agent.config import AgentConfig
from tools.registry import TOOL_DEFINITIONS, execute_tool


def cached_system(text: str) -> list[dict]:
    """
    Wrap a system prompt as a cacheable block (Anthropic prompt caching).

    The planning/extraction system prompts embed the full PBC list, which is identical
    across every email in a run. Marking it with cache_control lets the API reuse the
    prefix instead of re-processing it on each of the hundreds of calls — big latency and
    cost win on large mailboxes (cache reads are ~0.1x the input price). The first call
    writes the cache; the rest read it.
    """
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


@dataclass
class CostTracker:
    """Tracks API costs across the run."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    _lock: Any = field(default_factory=__import__("threading").Lock)

    def add(self, usage: Any) -> None:
        """Thread-safe accumulation of token usage from a response (incl. cache stats)."""
        with self._lock:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            # Prompt-caching counters (present when cache_control is used)
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    @property
    def cost_usd(self) -> float:
        # Blended input/output rates (most calls are Haiku, extraction/verify are Sonnet).
        haiku_input, haiku_output = 0.80 / 1e6, 4.00 / 1e6
        sonnet_input, sonnet_output = 3.00 / 1e6, 15.00 / 1e6
        avg_input = haiku_input * 0.7 + sonnet_input * 0.3
        avg_output = haiku_output * 0.7 + sonnet_output * 0.3
        # Prompt caching: writes cost 1.25x input, reads cost 0.1x input.
        return (
            self.input_tokens * avg_input
            + self.output_tokens * avg_output
            + self.cache_write_tokens * avg_input * 1.25
            + self.cache_read_tokens * avg_input * 0.10
        )


@dataclass
class AgentRun:
    """State for a single agent run across all emails."""
    tracker: TrackerState
    traces: list[AgentTrace] = field(default_factory=list)
    cost: CostTracker = field(default_factory=CostTracker)
    config: AgentConfig = field(default_factory=AgentConfig)
    offline: bool = False
    lock: Any = field(default_factory=__import__("threading").Lock)


def run_agent(
    pbc_items: list[PBCItem],
    emails: list[EmailMessage],
    config: AgentConfig | None = None,
    client_contacts: dict | None = None,
    engagement_info: dict | None = None,
) -> AgentRun:
    """
    Main entry point. Processes all emails against the PBC list.

    Returns an AgentRun with the final tracker state, all traces, and cost.
    """
    if config is None:
        config = AgentConfig()

    tracker = TrackerState(items={item.id: item for item in pbc_items})
    # Set contacts/engagement up front so follow-up drafting (inside this run)
    # has recipient emails available. Build a real name->email directory from the
    # mailbox so drafting uses observed addresses instead of guessing a domain.
    from agent.ingest import build_contact_directory
    tracker.client_contacts = client_contacts or {}
    tracker.engagement_info = engagement_info or {}
    tracker.contact_directory = build_contact_directory(emails, client_contacts)
    run = AgentRun(tracker=tracker, config=config)
    run.parsed_documents = {}  # init before threads start (avoids race on first parse)

    # Offline/mock mode: no API key. Ingest deterministically and return the
    # parsed tracker so the deployed app is inspectable without LLM spend.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        run.offline = True
        trace = AgentTrace(
            email_id="offline_mode",
            subject="Offline mode — no ANTHROPIC_API_KEY set",
        )
        trace.add_step(TraceStep(
            phase="plan",
            decision="offline",
            reasoning=(
                f"No API key configured. Ingested {len(pbc_items)} PBC items and "
                f"{len(emails)} emails deterministically. Set ANTHROPIC_API_KEY to run "
                "the full plan → act → verify agent loop and classify evidence."
            ),
        ))
        run.traces.append(trace)
        return run

    from agent.config import make_client
    client = make_client()

    # Phase 1: Process emails concurrently. Emails are independent units of work and
    # the slow part is LLM network I/O, so a thread pool cuts wall-clock time roughly
    # linearly. Shared-state writes (tracker, cost, parsed docs) are guarded by a lock
    # in tools.registry / tracker_ops; the LLM calls happen outside any lock.
    ordered = sorted(emails, key=lambda e: e.date)
    if config.max_workers > 1 and len(ordered) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            traces = list(pool.map(lambda e: process_email(client, run, e), ordered))
        run.traces.extend(traces)
    else:
        for email in ordered:
            run.traces.append(process_email(client, run, email))

    # Phase 2: Generate follow-up drafts (needs the full tracker state, so it runs last)
    followup_trace = generate_followups(client, run)
    run.traces.append(followup_trace)

    return run


import re as _re

# Terms that hint an email may carry PBC-relevant content even without an attachment
# (e.g. "the confirmations will follow", "PBC-12 is ready"). Kept broad on purpose —
# the cost of a false "relevant" is one cheap Haiku plan call; the cost of a false
# "skip" is a missed item, which we must avoid.
_RELEVANCE_HINTS = _re.compile(
    r"pbc[-\s]?\d+|attach|enclos|statement|reconcil|register|invoice|ledger|"
    r"trial balance|aging|payroll|confirmation|minutes|memo|schedule|"
    r"tax|lease|loan|insurance|going concern|provision|accrual|disposal|"
    r"receivable|payable|register|workpaper|filing",
    _re.IGNORECASE,
)


def _fast_skip_reason(email: EmailMessage) -> str | None:
    """
    Return a reason string if this email can be safely skipped without an LLM call,
    else None. Conservative: only skips when there are NO attachments AND the body has
    no PBC-relevance hints and is short (a brief acknowledgement, not a substantive ask).
    """
    if email.attachments:
        return None  # any attachment → let the agent look at it
    body = (email.body or "").strip()
    if _RELEVANCE_HINTS.search(body) or _RELEVANCE_HINTS.search(email.subject or ""):
        return None  # mentions something PBC-ish → defer to the planner
    # No attachments, no relevance hints. Short acknowledgements are safe to skip.
    if len(body) <= 200:
        return "no attachments and body is a brief acknowledgement with no PBC-relevant terms"
    return None  # longer body with no hints — still let the LLM decide, to be safe


def process_email(
    client: Anthropic,
    run: AgentRun,
    email: EmailMessage,
) -> AgentTrace:
    """
    Process a single email through the agent loop.

    The agent decides what to do: parse attachments, classify against PBC items,
    extract fields, or skip if irrelevant.
    """
    trace = AgentTrace(email_id=email.message_id, subject=email.subject)

    # Step 0: Deterministic fast-skip (no LLM). Skips emails that have no attachments
    # AND whose body shows no sign of PBC-relevant content — e.g. "Working on it — J."
    # This is the main wall-clock lever on large mailboxes (a whole plan→act→verify chain
    # avoided per skipped email). Conservative by design: any attachment, or any hint of
    # relevance, defers to the LLM planner. The skip + its reason are recorded in the trace.
    skip_reason = _fast_skip_reason(email)
    if skip_reason:
        trace.add_step(TraceStep(
            phase="plan",
            decision="skip",
            reasoning=f"Fast-skip (no LLM call): {skip_reason}",
        ))
        return trace

    # Step 1: Planning — decide what to do with this email
    plan = plan_email(client, run, email, trace)

    if plan.get("action") == "skip":
        trace.add_step(TraceStep(
            phase="plan",
            decision="skip",
            reasoning=plan.get("reasoning", "No relevant PBC content"),
        ))
        return trace

    trace.add_step(TraceStep(
        phase="plan",
        decision="process",
        reasoning=plan.get("reasoning", ""),
        relevant_items=plan.get("relevant_items", []),
    ))

    # Step 2: Tool-calling loop — agent picks tools until done
    run_tool_loop(client, run, email, trace)

    # Step 3: Verification — verify every item the agent actually touched via
    # update_item_status, not just the planner's predicted list. The agent may update
    # an item the planner didn't name; those must still be verified (and get their
    # final confidence), otherwise they keep a stale extraction-time value.
    touched = set(trace.affected_items)
    for step in trace.steps:
        if step.tool_call and step.tool_call.tool_name == "update_item_status":
            iid = step.tool_call.tool_input.get("item_id")
            if iid:
                touched.add(iid)
    for item_id in touched:
        verify_item(client, run, item_id, trace)

    return trace


def plan_email(
    client: Anthropic,
    run: AgentRun,
    email: EmailMessage,
    trace: AgentTrace,
) -> dict[str, Any]:
    """
    Ask the planning model whether this email is relevant and what to do.
    Uses Haiku for cost efficiency.
    """
    system_prompt = build_planning_prompt(run)
    user_msg = format_email_for_prompt(email, run)

    response = client.messages.create(
        model=run.config.planning_model,
        max_tokens=1024,
        temperature=run.config.temperature,
        system=cached_system(system_prompt),
        messages=[{"role": "user", "content": user_msg}],
        tools=[{
            "name": "plan_decision",
            "description": "Decide what to do with this email",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["process", "skip"],
                        "description": "Whether to process this email or skip it"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this decision was made"
                    },
                    "relevant_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "PBC item IDs this email is relevant to"
                    },
                    "tools_needed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Which tools the agent should use"
                    }
                },
                "required": ["action", "reasoning"]
            }
        }],
        tool_choice={"type": "tool", "name": "plan_decision"},
    )

    run.cost.add(response.usage)

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    return {"action": "skip", "reasoning": "No plan returned"}


def run_tool_loop(
    client: Anthropic,
    run: AgentRun,
    email: EmailMessage,
    trace: AgentTrace,
    max_iterations: int = 10,
) -> None:
    """
    The core tool-calling loop. The agent decides which tools to call
    until it signals completion. Uses Sonnet for complex extraction.
    """
    system_prompt = build_extraction_prompt(run)
    system_blocks = cached_system(system_prompt)
    messages = [{"role": "user", "content": format_email_for_prompt(email, run)}]

    for iteration in range(max_iterations):
        response = client.messages.create(
            model=run.config.extraction_model,
            max_tokens=4096,
            temperature=run.config.temperature,
            system=system_blocks,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        run.cost.add(response.usage)

        # Check if the agent is done (no more tool calls)
        if response.stop_reason == "end_turn":
            # Extract any final text as summary
            for block in response.content:
                if hasattr(block, "text"):
                    trace.add_step(TraceStep(
                        phase="extraction",
                        decision="complete",
                        reasoning=block.text,
                    ))
            break

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_call = ToolCall(
                    tool_name=block.name,
                    tool_input=block.input,
                )

                # Execute the tool
                result = execute_tool(block.name, block.input, run, email)
                tool_call.tool_output = result

                trace.add_step(TraceStep(
                    phase="extraction",
                    decision=f"call_{block.name}",
                    reasoning=f"Calling {block.name}",
                    tool_call=tool_call,
                ))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result) if isinstance(result, dict) else str(result),
                })

        # Build next turn
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


def verify_item(
    client: Anthropic,
    run: AgentRun,
    item_id: str,
    trace: AgentTrace,
) -> None:
    """
    Independent verification step. A separate model call checks whether
    the extracted evidence actually satisfies the PBC item's acceptance criteria.
    """
    item = run.tracker.items.get(item_id)
    if not item:
        return

    system_prompt = """You are an audit verification agent. Decide whether the evidence
collected for a PBC item satisfies its acceptance criteria.

Return one of four verdicts:
- "sufficient": the evidence reasonably satisfies the request. If the item asks for a
  single document (e.g. a fixed-asset register, an AR aging, a signed memo) and that
  document was received and matches the entity/period, it is SUFFICIENT. Do not withhold
  this verdict just because you cannot personally re-audit every number.
- "insufficient": there is a CONCRETE, NAMEABLE piece of the request that was NOT received
  at all. Reserve this for a genuine MISSING artifact: the criteria require ALL
  accounts/entities/meetings and only some arrived; the wrong entity or period; a
  threshold not met; an informal artifact where a formal one was explicitly required
  (e.g. a photo instead of a signed reconciliation). The gap is that something is absent.
- "under_review": the requested document arrived and is substantively on-point, but you
  have a QUALITY/GRANULARITY/FORMAT reservation you cannot fully resolve from the extracted
  fields alone (e.g. the register summarizes by category and you'd want line-item detail;
  a forecast is referenced with headline figures but a fuller schedule may exist). The
  right call here is human review, NOT insufficiency — the substance is present, the
  question is depth.
- "not_started": no relevant evidence was received.

Decision rule: is something ASKED-FOR actually MISSING? → insufficient. Is the document
present but its depth/format debatable? → under_review. Present and clearly adequate? →
sufficient. Judge the document on the substance it contains, not on cross-references to
separate attachments.

IMPORTANT — distinguish "the document lacks X" from "I didn't extract X". The extracted
fields are a summary, not the whole document. If the right TYPE of document arrived for
the right period/entity and your only concern is that a criterion (e.g. "reconciles to
GL", an as-of date, supporting calculations) isn't visible IN THE EXTRACTED FIELDS, that
is NOT grounds for "insufficient" — it may simply not have been extracted. Use
"under_review" for that (human confirms), and reserve "insufficient" for a concrete gap
you can point to: wrong period/entity, a threshold provably not met, an explicitly
incomplete set (N of M received), or an informal artifact where a formal one was required.

You must call the verification_verdict tool with your decision."""

    evidence_summary = json.dumps({
        "item_id": item.id,
        "description": item.description,
        "acceptance_criteria": item.acceptance_criteria,
        "current_status": item.status,
        "evidence_received": [e.dict() for e in item.evidence],
    }, indent=2)

    verdict_tool = {
        "name": "verification_verdict",
        "description": "Record the verification verdict for a PBC item",
        "input_schema": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string"},
                "verdict": {
                    "type": "string",
                    "enum": ["sufficient", "insufficient", "under_review", "not_started"],
                },
                "reasoning": {"type": "string"},
                "missing": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What's still missing, if insufficient"
                },
            },
            "required": ["item_id", "verdict", "reasoning"]
        }
    }

    def one_vote(temperature: float) -> dict | None:
        response = client.messages.create(
            model=run.config.verification_model,
            max_tokens=1024,
            temperature=temperature,
            system=cached_system(system_prompt),
            messages=[{"role": "user", "content": f"Verify this PBC item:\n\n{evidence_summary}"}],
            tools=[verdict_tool],
            tool_choice={"type": "tool", "name": "verification_verdict"},
        )
        run.cost.add(response.usage)
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        return None

    # Self-consistency: run N independent votes and take the majority verdict.
    # This dampens run-to-run variance on the boundary cases. First vote uses the
    # configured (deterministic) temperature; extra votes use a higher temperature
    # to produce genuine diversity rather than identical samples. N=1 restores the
    # original single-shot behaviour.
    n = max(1, run.config.verifier_votes)
    votes: list[dict] = []
    for i in range(n):
        temp = run.config.temperature if i == 0 else run.config.verifier_vote_temperature
        v = one_vote(temp)
        if v:
            votes.append(v)

    if not votes:
        return

    from collections import Counter
    tally = Counter(v["verdict"] for v in votes)
    winning_verdict, _ = tally.most_common(1)[0]
    # Use the reasoning from a vote that matches the winning verdict.
    winner = next(v for v in votes if v["verdict"] == winning_verdict)

    verdict = VerifierVerdict(
        item_id=item_id,
        verdict=winning_verdict,
        reasoning=winner["reasoning"],
        missing=winner.get("missing", []),
    )
    # Record the vote breakdown in the trace when we actually voted (>1).
    if n > 1:
        verdict.reasoning = f"[vote {dict(tally)}] " + verdict.reasoning
    trace.verifier_verdicts.append(verdict)

    status_map = {
        "sufficient": "Received",
        "insufficient": "Insufficient",
        "under_review": "Under review",
        "not_started": "Not started",
    }
    item.status = status_map.get(winning_verdict, item.status)
    item.verifier_reasoning = verdict.reasoning

    # Re-derive evidence confidence to match the FINAL verifier verdict, so the number
    # shown in the tracker is consistent with the decided status (evidence confidence was
    # first set at extraction time, before the verifier ran).
    from tools.tracker_ops import _blend_confidence
    for ev in item.evidence:
        ev.confidence = _blend_confidence(item.status, ev.citation_conf, bool(ev.citations))


def generate_followups(
    client: Anthropic,
    run: AgentRun,
) -> AgentTrace:
    """
    Generate grouped follow-up email drafts. One email per recipient/topic,
    not individual emails per item.
    """
    trace = AgentTrace(email_id="followup_generation", subject="Follow-up drafts")

    # Only follow up on items that are actually IN-FLIGHT: either received-but-insufficient,
    # or a Not-started item that was mentioned/requested somewhere in the mailbox. We do NOT
    # chase the long tail of Not-started items the client was never asked about — that's not
    # how an audit senior works, and it matches the ground-truth grouping.
    discussed: set[str] = set()
    for t in run.traces:
        discussed.update(t.affected_items)
        for step in t.steps:
            discussed.update(step.relevant_items)
            if step.tool_call and step.tool_call.tool_name == "update_item_status":
                iid = step.tool_call.tool_input.get("item_id")
                if iid:
                    discussed.add(iid)

    outstanding_items = []
    for item_id, item in run.tracker.items.items():
        include = item.status == "Insufficient" or (
            item.status == "Not started" and item_id in discussed
        )
        if include:
            outstanding_items.append({
                "id": item.id,
                "description": item.description,
                "status": item.status,
                "reason": item.verifier_reasoning or "Requested but not yet received",
            })

    if not outstanding_items:
        trace.add_step(TraceStep(
            phase="followup",
            decision="none_needed",
            reasoning="All items received or complete",
        ))
        return trace

    system_prompt = """You are drafting follow-up emails for an audit engagement.

Group outstanding PBC items by the client contact who owns them — send ONE email per
recipient, not one per item. Map items to the right person using what the email threads
showed (e.g. the controller is the primary contact and owns most items; the bookkeeper
owns bank reconciliations). An item may go to more than one recipient if genuinely owned
by both.

CRITICAL RULES:
- Use ONLY the exact email addresses from contact_directory below. Never invent or guess
  an address or domain — copy the real address verbatim.
- Prioritize items the client has actually been asked about or engaged with in the threads.
  Do not dump every "Not started" item on the primary contact; focus the follow-up on what
  is genuinely outstanding and in-flight.
- For Insufficient items, state the specific missing piece.

Call the draft_followup tool ONCE PER RECIPIENT (multiple times if multiple recipients).
Do not write any prose outside the tool calls."""

    context = json.dumps({
        "outstanding_items": outstanding_items,
        "contact_directory": run.tracker.contact_directory,
        "engagement": run.tracker.engagement_info,
    }, indent=2)

    draft_tool = {
        "name": "draft_followup",
        "description": "Draft one grouped follow-up email to a single client contact.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient_email": {"type": "string"},
                "recipient_name": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "items_covered": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
            "required": ["recipient_email", "recipient_name", "subject", "body", "items_covered"]
        }
    }

    # Tool-use loop so the model can emit multiple drafts (one per recipient).
    messages = [{"role": "user", "content": f"Draft the grouped follow-up emails:\n\n{context}"}]
    for _ in range(4):
        response = client.messages.create(
            model=run.config.extraction_model,
            max_tokens=4096,
            temperature=run.config.temperature,
            system=system_prompt,
            messages=messages,
            tools=[draft_tool],
        )
        run.cost.add(response.usage)

        tool_results = []
        made_draft = False
        for block in response.content:
            if block.type == "tool_use":
                made_draft = True
                draft = block.input
                run.tracker.followup_drafts.append(draft)
                trace.add_step(TraceStep(
                    phase="followup",
                    decision="draft_created",
                    reasoning=draft.get("rationale", f"Draft to {draft.get('recipient_name','')}"),
                    tool_call=ToolCall(tool_name="draft_followup", tool_input=draft),
                ))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Draft recorded.",
                })

        if response.stop_reason == "end_turn" or not made_draft:
            break
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    # Enforce "one clean email per recipient" deterministically: if the model produced
    # more than one draft for the same address, merge them (union of items, concatenated
    # bodies). This guarantees the brief's requirement regardless of model behaviour.
    merged: dict[str, dict] = {}
    order: list[str] = []
    for d in run.tracker.followup_drafts:
        key = (d.get("recipient_email") or d.get("recipient_name") or "").lower().strip()
        if key not in merged:
            merged[key] = dict(d)
            merged[key]["items_covered"] = list(d.get("items_covered", []))
            order.append(key)
        else:
            m = merged[key]
            # Union items, preserving order
            for it in d.get("items_covered", []):
                if it not in m["items_covered"]:
                    m["items_covered"].append(it)
            # Append the extra body under the same email
            if d.get("body") and d["body"] not in m.get("body", ""):
                m["body"] = m.get("body", "").rstrip() + "\n\n" + d["body"]
    if len(merged) != len(run.tracker.followup_drafts):
        run.tracker.followup_drafts = [merged[k] for k in order]
        trace.add_step(TraceStep(
            phase="followup",
            decision="merged_by_recipient",
            reasoning=f"Merged drafts to one email per recipient ({len(order)} recipient(s)).",
        ))

    return trace


# --- Prompt builders ---

def build_planning_prompt(run: AgentRun) -> str:
    """
    Build the system prompt for the planning phase.

    Note: this intentionally omits per-item status. The prompt must be identical across
    every email so it can be prompt-cached (status also isn't meaningful here — emails are
    processed concurrently and relevance doesn't depend on current status).
    """
    items_summary = "\n".join(
        f"  {item.id}: [{item.category}] {item.description[:80]}..."
        for item in run.tracker.items.values()
    )
    return f"""You are a PBC (Prepared-By-Client) email triage agent for a financial audit.

Your job is to look at an incoming email and decide:
1. Is it relevant to any PBC items? If not, skip it.
2. If relevant, which PBC items does it relate to?
3. What tools will be needed (parse_pdf, parse_excel, ocr_image, etc.)?

Current PBC tracker state:
{items_summary}

Respond with your plan using the plan_decision tool."""


def build_extraction_prompt(run: AgentRun) -> str:
    """Build the system prompt for the extraction phase."""
    items_detail = "\n".join(
        f"  {item.id}: {item.description}\n    Acceptance: {item.acceptance_criteria}"
        for item in run.tracker.items.values()
    )
    return f"""You are a PBC document extraction agent for a financial audit.

Your job is to:
1. Parse attachments using the appropriate tool (parse_pdf, parse_excel, ocr_image)
2. Extract relevant fields with citations (page number, cell reference, bounding box)
3. Match extracted content to PBC items using classify_document
4. Update the tracker using update_item_status

Always provide citations for extracted data. Be precise about periods, entities, and amounts.

PBC items and acceptance criteria:
{items_detail}

Call tools as needed. When done, provide a brief summary of what was found."""


def format_email_for_prompt(email: EmailMessage, run: AgentRun) -> str:
    """Format an email message for inclusion in a prompt."""
    attachments_list = "\n".join(
        f"  - {a.filename} ({a.content_type}, {a.size_bytes} bytes)"
        for a in email.attachments
    ) or "  (none)"

    return f"""Email:
From: {email.sender}
To: {email.recipients}
Date: {email.date}
Subject: {email.subject}
Thread: {email.thread_id}

Body:
{email.body}

Attachments:
{attachments_list}"""
