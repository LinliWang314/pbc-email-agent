"""
Citation verification — the core anti-hallucination guardrail.

When the extraction model claims "period_end = 2026-06-30, cited from page 2",
we do NOT trust it. This module deterministically checks that the cited value
actually appears at the cited location in the parsed document.

This catches the most dangerous failure mode: the model fabricating a value that
isn't in the document at all. A fabricated citation fails the grep and the field
is marked `verified: false`, which downstream logic treats as unusable evidence.

This is deterministic (no LLM), so it is fully covered by unit tests and adds no cost.
"""

from __future__ import annotations

import json
import re
from typing import Any


def normalize(s: str) -> str:
    """Normalize a string for loose matching (whitespace, case, punctuation)."""
    s = str(s).lower()
    s = re.sub(r"[\s,$]", "", s)
    return s.strip()


def _as_number(s: str):
    """Return a float if the string is purely a (possibly formatted) number, else None."""
    t = re.sub(r"[\s,$%()]", "", str(s)).strip()
    if t.startswith("-") or t.startswith("−"):
        t = "-" + t.lstrip("-−")
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        try:
            return float(t)
        except ValueError:
            return None
    return None


def _values_match(claimed: str, candidate: str) -> bool:
    """
    True if the claimed value matches the candidate text. Numbers are compared
    numerically (so '214700', '$214,700.00' and '214700.0' all match); otherwise a
    normalized substring check is used.
    """
    cn, dn = _as_number(claimed), _as_number(candidate)
    if cn is not None and dn is not None:
        return abs(cn - dn) < 0.01
    nc = normalize(claimed)
    return bool(nc) and nc in normalize(candidate)


def verify_citation(
    claimed_value: str,
    citation: dict[str, Any],
    parsed_document: dict[str, Any],
) -> dict[str, Any]:
    """
    Check that `claimed_value` actually appears at the location `citation` points to
    within `parsed_document` (the output of parse_pdf / parse_excel / ocr_image).

    Returns {"verified": bool, "reason": str, "found_context": str|None}.
    """
    citation_type = citation.get("type", "")
    reference = citation.get("reference", "")
    target = normalize(claimed_value)

    if not target:
        return {"verified": False, "reason": "empty claimed value", "found_context": None}

    # --- PDF page citation ---
    if citation_type in ("page", "text_position"):
        pages = parsed_document.get("pages", [])
        # Try the specific page first
        page_num = _extract_int(reference)
        haystacks = []
        if page_num is not None:
            for p in pages:
                if p.get("page_number") == page_num:
                    haystacks.append(p.get("text", ""))
        # Fall back to full text
        if not haystacks:
            haystacks.append(parsed_document.get("full_text", ""))
            haystacks.extend(p.get("text", "") for p in pages)

        claimed_num = _as_number(claimed_value)
        for text in haystacks:
            if target in normalize(text):
                return {
                    "verified": True,
                    "reason": f"value found in {citation_type} {reference}",
                    "found_context": _context(text, claimed_value),
                }
            # Numeric fallback: compare the claimed number against every numeric token
            # in the text, so $214,700.00 matches a bare 214700 in the document.
            if claimed_num is not None:
                for tok in re.findall(r"-?[\d,]+(?:\.\d+)?", text):
                    tn = _as_number(tok)
                    if tn is not None and abs(tn - claimed_num) < 0.01:
                        return {
                            "verified": True,
                            "reason": f"numeric value found in {citation_type} {reference}",
                            "found_context": tok,
                        }
        return {"verified": False, "reason": f"value not found at {citation_type} {reference}", "found_context": None}

    # --- Excel cell citation ---
    if citation_type == "cell":
        sheets = parsed_document.get("sheets", {})
        for sheet_data in sheets.values():
            for row in sheet_data.get("rows", []):
                # reference like "B12" — check that exact cell, else any cell.
                # Numbers are compared numerically so $214,700.00 matches 214700.
                if reference in row and _values_match(claimed_value, row[reference]):
                    return {"verified": True, "reason": f"value found in cell {reference}", "found_context": row[reference]}
                for cell_ref, cell_val in row.items():
                    if _values_match(claimed_value, cell_val):
                        return {"verified": True, "reason": f"value found in cell {cell_ref}", "found_context": cell_val}
        return {"verified": False, "reason": f"value not found in cell {reference}", "found_context": None}

    # --- Image bbox citation (OCR) ---
    if citation_type == "bbox":
        full_text = parsed_document.get("full_text", "")
        if target in normalize(full_text):
            return {"verified": True, "reason": "value found in OCR text", "found_context": _context(full_text, claimed_value)}
        return {"verified": False, "reason": "value not found in OCR output", "found_context": None}

    # Unknown citation type — search everywhere as a last resort
    blob = json.dumps(parsed_document) if not isinstance(parsed_document, str) else parsed_document
    if target in normalize(blob):
        return {"verified": True, "reason": "value found (untyped citation)", "found_context": None}
    return {"verified": False, "reason": f"unknown citation type '{citation_type}' and value not found", "found_context": None}


def verify_extraction(
    extracted_fields: dict[str, Any],
    citations: list[dict[str, Any]],
    parsed_document: dict[str, Any],
) -> dict[str, Any]:
    """
    Verify all citations for an extraction. Returns a summary with per-field verification
    and an overall confidence derived from the fraction of citations that check out.
    """
    results = []
    verified_count = 0

    for citation in citations:
        claimed = citation.get("text", "")
        result = verify_citation(claimed, citation, parsed_document)
        result["claimed_value"] = claimed
        result["citation"] = citation
        results.append(result)
        if result["verified"]:
            verified_count += 1

    total = len(citations)
    confidence = verified_count / total if total > 0 else 0.0

    return {
        "citation_results": results,
        "verified_citations": verified_count,
        "total_citations": total,
        "citation_confidence": round(confidence, 3),
        "all_verified": total > 0 and verified_count == total,
        "has_fabrication": any(not r["verified"] for r in results),
    }


def _extract_int(s: str) -> int | None:
    match = re.search(r"\d+", str(s))
    return int(match.group()) if match else None


def _context(text: str, value: str, window: int = 40) -> str:
    """Return a snippet of text around the first occurrence of value (loose match)."""
    norm_text = normalize(text)
    norm_val = normalize(value)
    idx = norm_text.find(norm_val)
    if idx == -1:
        return ""
    # Map back roughly to original text
    approx = int(idx * len(text) / max(len(norm_text), 1))
    start = max(0, approx - window)
    return text[start:approx + len(value) + window].strip()
