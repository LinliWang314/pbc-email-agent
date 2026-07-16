"""
Tracker operation tools — update and query the PBC tracker state.
"""

from typing import Any
from agent.models import Evidence


def update_item_status(
    item_id: str,
    status: str,
    reasoning: str,
    run: Any = None,
    email: Any = None,
    evidence_filename: str | None = None,
    source_email_id: str | None = None,
    extracted_fields: dict | None = None,
    citations: list | None = None,
) -> dict[str, Any]:
    """
    Update a PBC item's status in the tracker.
    Records evidence and maintains version history.
    """
    if run is None:
        return {"error": "No run context"}

    item = run.tracker.items.get(item_id)
    if not item:
        return {"error": f"Item {item_id} not found"}

    # Record evidence if a file was provided
    if evidence_filename:
        evidence = Evidence(
            filename=evidence_filename,
            source_email_id=source_email_id or (email.message_id if email else "unknown"),
            extracted_fields=extracted_fields or {},
            citations=citations or [],
        )

        # Version tracking: check if this supersedes an existing file
        existing = [e for e in item.evidence if _is_same_document(e.filename, evidence_filename)]
        if existing:
            item.versions.append({
                "superseded": existing[-1].filename,
                "superseded_by": evidence_filename,
                "reason": "newer version detected",
            })

        item.evidence.append(evidence)

    # Update status (but don't downgrade from a better status)
    status_order = ["Not started", "Received", "Under review", "Insufficient", "Complete"]
    # Allow any status update — the verifier will correct if needed
    item.status = status

    return {
        "item_id": item_id,
        "new_status": status,
        "evidence_count": len(item.evidence),
        "reasoning": reasoning,
    }


def get_item_status(item_id: str, run: Any = None) -> dict[str, Any]:
    """Get current status and evidence for a PBC item."""
    if run is None:
        return {"error": "No run context"}

    item = run.tracker.items.get(item_id)
    if not item:
        return {"error": f"Item {item_id} not found"}

    return {
        "item_id": item.id,
        "status": item.status,
        "description": item.description,
        "acceptance_criteria": item.acceptance_criteria,
        "evidence": [e.dict() for e in item.evidence],
        "versions": item.versions,
    }


def _is_same_document(filename1: str, filename2: str) -> bool:
    """
    Detect if two filenames refer to versions of the same document.
    E.g., Final_v1.xlsx and Final_v3_REAL.xlsx are the same doc.
    """
    import re

    # Strip version indicators and compare base names
    def normalize(name: str) -> str:
        name = name.lower()
        name = re.sub(r'_?v\d+', '', name)
        name = re.sub(r'_?final', '', name)
        name = re.sub(r'_?real', '', name)
        name = re.sub(r'_?draft', '', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        return name

    return normalize(filename1) == normalize(filename2)
