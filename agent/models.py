"""
Data models for the PBC agent.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    """A piece of evidence linked to a PBC item."""
    filename: str
    source_email_id: str
    extracted_fields: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0

    def dict(self) -> dict:
        return {
            "filename": self.filename,
            "source_email_id": self.source_email_id,
            "extracted_fields": self.extracted_fields,
            "citations": self.citations,
            "confidence": self.confidence,
        }


@dataclass
class PBCItem:
    """A single line item from the PBC list."""
    id: str
    category: str
    description: str
    acceptance_criteria: str
    priority: str = "Medium"
    expected_documents: list[str] = field(default_factory=list)
    status: str = "Not started"
    evidence: list[Evidence] = field(default_factory=list)
    verifier_reasoning: str | None = None
    versions: list[dict[str, Any]] = field(default_factory=list)

    def dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "acceptance_criteria": self.acceptance_criteria,
            "priority": self.priority,
            "expected_documents": self.expected_documents,
            "status": self.status,
            "evidence": [e.dict() for e in self.evidence],
            "verifier_reasoning": self.verifier_reasoning,
            "versions": self.versions,
        }


@dataclass
class Attachment:
    """An email attachment."""
    filename: str
    content_type: str
    size_bytes: int
    file_path: str


@dataclass
class EmailMessage:
    """A parsed email message."""
    message_id: str
    thread_id: str
    sender: str
    recipients: str
    subject: str
    date: str
    body: str
    attachments: list[Attachment] = field(default_factory=list)
    in_reply_to: str | None = None


@dataclass
class ToolCall:
    """Record of a tool call made by the agent."""
    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None

    def dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": str(self.tool_output)[:500] if self.tool_output else None,
        }


@dataclass
class TraceStep:
    """A single step in the agent trace."""
    phase: str
    decision: str
    reasoning: str
    relevant_items: list[str] = field(default_factory=list)
    tool_call: ToolCall | None = None
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    def dict(self) -> dict:
        result = {
            "phase": self.phase,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "relevant_items": self.relevant_items,
            "timestamp": self.timestamp,
        }
        if self.tool_call:
            result["tool_call"] = self.tool_call.dict()
        return result


@dataclass
class VerifierVerdict:
    """Result of the verification step for a PBC item."""
    item_id: str
    verdict: str
    reasoning: str
    missing: list[str] = field(default_factory=list)

    def dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "missing": self.missing,
        }


@dataclass
class AgentTrace:
    """Full trace of agent processing for one email."""
    email_id: str
    subject: str
    steps: list[TraceStep] = field(default_factory=list)
    verifier_verdicts: list[VerifierVerdict] = field(default_factory=list)
    affected_items: list[str] = field(default_factory=list)

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)
        if step.relevant_items:
            for item_id in step.relevant_items:
                if item_id not in self.affected_items:
                    self.affected_items.append(item_id)

    def dict(self) -> dict:
        return {
            "email_id": self.email_id,
            "subject": self.subject,
            "steps": [s.dict() for s in self.steps],
            "verifier_verdicts": [v.dict() for v in self.verifier_verdicts],
            "affected_items": self.affected_items,
        }


@dataclass
class TrackerState:
    """The full state of the PBC tracker."""
    items: dict[str, PBCItem] = field(default_factory=dict)
    followup_drafts: list[dict[str, Any]] = field(default_factory=list)
    client_contacts: dict[str, str] = field(default_factory=dict)
    engagement_info: dict[str, str] = field(default_factory=dict)

    def dict(self) -> dict:
        return {
            "items": {k: v.dict() for k, v in self.items.items()},
            "followup_drafts": self.followup_drafts,
        }
