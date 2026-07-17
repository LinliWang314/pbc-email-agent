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
  - claude-sonnet-4-20250514: extraction with citations, verification, drafting
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


@dataclass
class CostTracker:
    """Tracks API costs across the run."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        haiku_input = 0.80 / 1_000_000
        haiku_output = 4.00 / 1_000_000
        sonnet_input = 3.00 / 1_000_000
        sonnet_output = 15.00 / 1_000_000
        # Approximate as weighted average (most calls are haiku)
        avg_input = (haiku_input * 0.7 + sonnet_input * 0.3)
        avg_output = (haiku_output * 0.7 + sonnet_output * 0.3)
        return self.input_tokens * avg_input + self.output_tokens * avg_output


@dataclass
class AgentRun:
    """State for a single agent run across all emails."""
    tracker: TrackerState
    traces: list[AgentTrace] = field(default_factory=list)
    cost: CostTracker = field(default_factory=CostTracker)
    config: AgentConfig = field(default_factory=AgentConfig)
    offline: bool = False


def run_agent(
    pbc_items: list[PBCItem],
    emails: list[EmailMessage],
    config: AgentConfig | None = None,
) -> AgentRun:
    """
    Main entry point. Processes all emails against the PBC list.

    Returns an AgentRun with the final tracker state, all traces, and cost.
    """
    if config is None:
        config = AgentConfig()

    tracker = TrackerState(items={item.id: item for item in pbc_items})
    run = AgentRun(tracker=tracker, config=config)

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

    client = Anthropic()

    # Phase 1: Process each email through the planning agent
    for email in sorted(emails, key=lambda e: e.date):
        trace = process_email(client, run, email)
        run.traces.append(trace)

    # Phase 2: Generate follow-up drafts
    followup_trace = generate_followups(client, run)
    run.traces.append(followup_trace)

    return run


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

    # Step 3: Verification — separate model verifies extractions
    for item_id in trace.affected_items:
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
        system=system_prompt,
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

    run.cost.input_tokens += response.usage.input_tokens
    run.cost.output_tokens += response.usage.output_tokens

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
    messages = [{"role": "user", "content": format_email_for_prompt(email, run)}]

    for iteration in range(max_iterations):
        response = client.messages.create(
            model=run.config.extraction_model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        run.cost.input_tokens += response.usage.input_tokens
        run.cost.output_tokens += response.usage.output_tokens

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

    system_prompt = """You are an audit verification agent. Your job is to check whether
the evidence collected for a PBC item actually satisfies its acceptance criteria.

Be strict: if the acceptance criteria say "all accounts" and only one was received,
that is Insufficient. If the period doesn't match, that is Insufficient.

You must call the verification_verdict tool with your decision."""

    evidence_summary = json.dumps({
        "item_id": item.id,
        "description": item.description,
        "acceptance_criteria": item.acceptance_criteria,
        "current_status": item.status,
        "evidence_received": [e.dict() for e in item.evidence],
    }, indent=2)

    response = client.messages.create(
        model=run.config.verification_model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Verify this PBC item:\n\n{evidence_summary}"}],
        tools=[{
            "name": "verification_verdict",
            "description": "Record the verification verdict for a PBC item",
            "input_schema": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["sufficient", "insufficient", "not_started"],
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
        }],
        tool_choice={"type": "tool", "name": "verification_verdict"},
    )

    run.cost.input_tokens += response.usage.input_tokens
    run.cost.output_tokens += response.usage.output_tokens

    for block in response.content:
        if block.type == "tool_use":
            verdict_data = block.input
            verdict = VerifierVerdict(
                item_id=item_id,
                verdict=verdict_data["verdict"],
                reasoning=verdict_data["reasoning"],
                missing=verdict_data.get("missing", []),
            )
            trace.verifier_verdicts.append(verdict)

            # Update tracker status based on verdict
            status_map = {
                "sufficient": "Received",
                "insufficient": "Insufficient",
                "not_started": "Not started",
            }
            item.status = status_map.get(verdict.verdict, item.status)
            item.verifier_reasoning = verdict.reasoning


def generate_followups(
    client: Anthropic,
    run: AgentRun,
) -> AgentTrace:
    """
    Generate grouped follow-up email drafts. One email per recipient/topic,
    not individual emails per item.
    """
    trace = AgentTrace(email_id="followup_generation", subject="Follow-up drafts")

    outstanding_items = []
    for item_id, item in run.tracker.items.items():
        if item.status in ("Not started", "Insufficient"):
            outstanding_items.append({
                "id": item.id,
                "description": item.description,
                "status": item.status,
                "reason": item.verifier_reasoning or "Not yet received",
            })

    if not outstanding_items:
        trace.add_step(TraceStep(
            phase="followup",
            decision="none_needed",
            reasoning="All items received or complete",
        ))
        return trace

    system_prompt = """You are drafting follow-up emails for an audit engagement.
Group outstanding PBC items by recipient — send ONE email per person, not one per item.
Be professional, specific about what's needed, and include deadlines.

Use the draft_followup tool for each email you want to send."""

    context = json.dumps({
        "outstanding_items": outstanding_items,
        "client_contacts": run.tracker.client_contacts,
        "engagement": run.tracker.engagement_info,
    }, indent=2)

    response = client.messages.create(
        model=run.config.extraction_model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Draft follow-up emails:\n\n{context}"}],
        tools=[{
            "name": "draft_followup",
            "description": "Draft a follow-up email to a client contact",
            "input_schema": {
                "type": "object",
                "properties": {
                    "recipient_email": {"type": "string"},
                    "recipient_name": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "items_covered": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["recipient_email", "recipient_name", "subject", "body", "items_covered"]
            }
        }],
    )

    run.cost.input_tokens += response.usage.input_tokens
    run.cost.output_tokens += response.usage.output_tokens

    for block in response.content:
        if block.type == "tool_use":
            draft = block.input
            run.tracker.followup_drafts.append(draft)
            trace.add_step(TraceStep(
                phase="followup",
                decision="draft_created",
                reasoning=draft.get("rationale", ""),
                tool_call=ToolCall(
                    tool_name="draft_followup",
                    tool_input=draft,
                ),
            ))

    return trace


# --- Prompt builders ---

def build_planning_prompt(run: AgentRun) -> str:
    """Build the system prompt for the planning phase."""
    items_summary = "\n".join(
        f"  {item.id}: [{item.category}] {item.description[:80]}... Status: {item.status}"
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
