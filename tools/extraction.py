"""
Field extraction tool — extracts structured data with citations.

This is a deterministic tool that pulls specific fields from parsed content.
The LLM agent decides WHAT to extract; this tool handles the mechanics.
"""

import re
from typing import Any


def extract_fields(
    filename: str,
    pbc_item_id: str,
    content: str,
    fields_to_extract: list[str] | None = None,
) -> dict[str, Any]:
    """
    Extract fields from document content relevant to a PBC item.
    Returns structured data with citations (page/cell/position references).
    """
    if fields_to_extract is None:
        fields_to_extract = ["period", "entity", "date", "amount"]

    extracted = {}
    citations = []

    # Date extraction
    if "date" in fields_to_extract or "period" in fields_to_extract or "period_end" in fields_to_extract:
        dates = re.findall(
            r'(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|'
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}|'
            r'(?:Jun|Jul|Aug|Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May)\s+\d{1,2},?\s*\d{4}|'
            r'(?:Q[1-4])\s*(?:FY)?\d{2,4}|'
            r'FY\s*\d{4})',
            content
        )
        if dates:
            extracted["dates_found"] = dates[:10]
            for date in dates[:3]:
                pos = content.find(date)
                citations.append({
                    "type": "text_position",
                    "reference": f"char {pos}",
                    "text": date,
                    "context": content[max(0, pos-30):pos+len(date)+30],
                })

    # Entity extraction
    if "entity" in fields_to_extract:
        entities = re.findall(
            r'(Northwind\s+(?:Beverages|Distribution)|Cascade\s+Cold\s+Brew|'
            r'Silverline|Peak\s+National|consolidated)',
            content, re.IGNORECASE
        )
        if entities:
            extracted["entities_found"] = list(set(entities))

    # Amount extraction
    if "amount" in fields_to_extract or "total_amount" in fields_to_extract:
        amounts = re.findall(
            r'\$[\d,]+(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b',
            content
        )
        if amounts:
            extracted["amounts_found"] = amounts[:20]

    # Period extraction (fiscal year, quarter)
    if "period" in fields_to_extract:
        periods = re.findall(
            r'FY\s*20\d{2}|fiscal\s+year\s+20\d{2}|'
            r'year\s+ended?\s+\w+\s+\d{1,2},?\s*\d{4}|'
            r'Q[1-4]\s*(?:FY)?\s*20\d{2}|'
            r'(?:Jul(?:y)?|Aug|Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun(?:e)?)\s*20\d{2}\s*(?:to|through|[-–])\s*(?:Jul(?:y)?|Aug|Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun(?:e)?)\s*20\d{2}',
            content, re.IGNORECASE
        )
        if periods:
            extracted["periods_found"] = list(set(periods))

    return {
        "filename": filename,
        "pbc_item_id": pbc_item_id,
        "extracted_fields": extracted,
        "citations": citations,
        "fields_requested": fields_to_extract,
        "content_length": len(content),
    }
