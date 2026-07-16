"""
Document classification tool — matches documents to PBC items.

Uses simple keyword/semantic matching rather than an LLM call,
keeping costs low. The agent itself does the reasoning about whether
a match is correct.
"""

from typing import Any


def classify_document(
    filename: str,
    content_summary: str,
    document_type: str,
    run: Any = None,
) -> dict[str, Any]:
    """
    Classify a document against PBC items using keyword matching.
    Returns candidate matches with confidence scores.

    This is a deterministic tool — the agent decides the final classification.
    """
    if run is None:
        return {"error": "No run context available"}

    candidates = []
    content_lower = content_summary.lower()
    filename_lower = filename.lower()

    for item_id, item in run.tracker.items.items():
        score = 0.0
        reasons = []

        desc_lower = item.description.lower()
        criteria_lower = item.acceptance_criteria.lower()

        # Keyword overlap between content and item description
        desc_words = set(desc_lower.split())
        content_words = set(content_lower.split())
        overlap = desc_words & content_words
        keyword_score = len(overlap) / max(len(desc_words), 1)
        score += keyword_score * 0.5

        if keyword_score > 0.2:
            reasons.append(f"keyword overlap: {', '.join(list(overlap)[:5])}")

        # Filename heuristics
        if "aging" in filename_lower and "receivable" in desc_lower:
            score += 0.4
            reasons.append("filename suggests AR aging")
        if "asset" in filename_lower and "asset" in desc_lower:
            score += 0.4
            reasons.append("filename suggests fixed assets")
        if "invoice" in filename_lower and "invoice" in desc_lower:
            score += 0.4
            reasons.append("filename suggests invoice")
        if "minutes" in filename_lower and "minutes" in desc_lower:
            score += 0.4
            reasons.append("filename suggests board minutes")
        if "concern" in filename_lower and "concern" in desc_lower:
            score += 0.4
            reasons.append("filename suggests going concern")
        if "bank" in filename_lower and "bank" in desc_lower:
            score += 0.3
            reasons.append("filename suggests bank document")
        if "recon" in filename_lower and "reconcil" in desc_lower:
            score += 0.4
            reasons.append("filename suggests reconciliation")
        if "confirm" in filename_lower and "confirm" in desc_lower:
            score += 0.4
            reasons.append("filename suggests confirmation")
        if "operating" in filename_lower and "statement" in desc_lower:
            score += 0.3
            reasons.append("filename suggests bank statement")

        if score > 0.2:
            candidates.append({
                "item_id": item_id,
                "item_description": item.description[:100],
                "confidence": min(score, 1.0),
                "reasons": reasons,
            })

    # Sort by confidence
    candidates.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "filename": filename,
        "document_type": document_type,
        "candidates": candidates[:5],  # Top 5 matches
        "total_candidates": len(candidates),
    }
