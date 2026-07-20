"""
Tracker operation tools — update and query the PBC tracker state.
"""

from typing import Any
from agent.models import Evidence
from tools.citation_verify import verify_extraction


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

    # Guard tracker mutation — emails are processed concurrently.
    lock = getattr(run, "lock", None)
    if lock is not None:
        lock.acquire()
    try:
        return _do_update(
            item, status, reasoning, run, email, evidence_filename,
            source_email_id, extracted_fields, citations,
        )
    finally:
        if lock is not None:
            lock.release()


def _do_update(
    item, status, reasoning, run, email, evidence_filename,
    source_email_id, extracted_fields, citations,
) -> dict[str, Any]:
    item_id = item.id
    # Record evidence if a file was provided
    citation_check = None
    if evidence_filename:
        # --- Anti-hallucination guardrail: verify claimed citations against parsed content ---
        confidence = 0.5  # default when we have no document to check against
        parsed_doc = getattr(run, "parsed_documents", {}).get(evidence_filename)
        if parsed_doc and citations:
            citation_check = verify_extraction(extracted_fields or {}, citations, parsed_doc)
            confidence = citation_check["citation_confidence"]

        evidence = Evidence(
            filename=evidence_filename,
            source_email_id=source_email_id or (email.message_id if email else "unknown"),
            extracted_fields=extracted_fields or {},
            citations=citations or [],
            confidence=confidence,
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

        # --- Confidence-based degradation ---
        # If a citation was fabricated (value not found in the document), we cannot
        # trust the extraction. Don't let the agent mark it Received/Complete on the
        # strength of unverifiable evidence — downgrade to Under review for a human.
        if citation_check and citation_check["has_fabrication"] and status in ("Received", "Complete"):
            status = "Under review"
            reasoning += (
                f" [GUARDRAIL: {citation_check['total_citations'] - citation_check['verified_citations']} "
                f"of {citation_check['total_citations']} citations could not be verified against the "
                f"document; downgraded to Under review for human confirmation.]"
            )

    item.status = status

    result = {
        "item_id": item_id,
        "new_status": status,
        "evidence_count": len(item.evidence),
        "reasoning": reasoning,
    }
    if citation_check:
        result["citation_verification"] = {
            "verified": citation_check["verified_citations"],
            "total": citation_check["total_citations"],
            "confidence": citation_check["citation_confidence"],
            "has_fabrication": citation_check["has_fabrication"],
        }
    return result


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
