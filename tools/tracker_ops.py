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
        # Confidence blends two signals: (a) did the agent's citations actually check out
        # against the document, and (b) how decisive the status itself is. A clean document
        # match with no explicit citations should still read as reasonably confident, not 0%.
        parsed_doc = getattr(run, "parsed_documents", {}).get(evidence_filename)
        citation_conf = None
        if parsed_doc and citations:
            citation_check = verify_extraction(extracted_fields or {}, citations, parsed_doc)
            citation_conf = citation_check["citation_confidence"]

        confidence = _blend_confidence(status, citation_conf, bool(citations))

        evidence = Evidence(
            filename=evidence_filename,
            source_email_id=source_email_id or (email.message_id if email else "unknown"),
            extracted_fields=extracted_fields or {},
            citations=citations or [],
            confidence=confidence,
            citation_conf=citation_conf,
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


def _blend_confidence(status: str, citation_conf: float | None, had_citations: bool) -> float:
    """
    Derive a display confidence for an item's evidence.

    Combines the status decision with citation verification so the number is intuitive:
    - A Received/Complete item reads as confident (the verifier judged it sufficient),
      nudged up when its citations also grep-verified and down when they didn't.
    - Insufficient/Under review sit in the middle — a real but not-final judgement.
    - When the agent supplied no explicit citations we can't raise confidence on citation
      grounds, but we don't punish a clean Received match down to 0% either.

    citation_conf is the fraction of cited values that matched the document (0..1) or None
    if no citation check ran.
    """
    base = {
        "Received": 0.9,
        "Complete": 0.95,
        "Insufficient": 0.7,
        "Under review": 0.6,
        "Not started": 0.0,
    }.get(status, 0.5)

    if citation_conf is None:
        # No citations to verify against — return the status-based confidence,
        # slightly tempered so it doesn't overclaim certainty.
        return round(base if not had_citations else base * 0.9, 2)

    # Blend: 60% status decisiveness, 40% how well citations verified.
    return round(0.6 * base + 0.4 * citation_conf, 2)


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
