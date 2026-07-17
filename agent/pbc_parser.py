"""
Free-form PBC list parser.

The held-out PBC list at review time is free-form text, not the structured
`Acceptance: key=value` format of the sample. So we cannot rely on regex.

Strategy (two layers, so the deterministic part is testable and the LLM part
is isolated + cheap):
  1. Deterministic: extract raw text from the PDF (pdftotext / PyPDF2).
  2. LLM (one Haiku call): turn free-form text into structured PBCItem objects,
     each with normalized, machine-checkable acceptance criteria.

The LLM structuring runs ONCE per PBC list (not per email), so it is cheap and
does not scale with mailbox size. Results are cached to disk keyed by a hash of
the text, so re-runs from a clean checkout are free and deterministic.

If no API key is available, falls back to the regex parser in ingest.py so the
app still runs (degraded) in mock/offline mode.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from agent.models import PBCItem


CACHE_DIR = Path(__file__).parent.parent / ".pbc_cache"


def extract_pdf_text(pdf_path: str) -> str:
    """Deterministically extract text from the PBC PDF."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback to pdftotext CLI
    import subprocess
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True, text=True
    )
    return result.stdout


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_cache(key: str) -> list[dict] | None:
    cache_file = CACHE_DIR / f"pbc_{key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def _save_cache(key: str, items: list[dict]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"pbc_{key}.json"
    with open(cache_file, "w") as f:
        json.dump(items, f, indent=2)


PBC_STRUCTURING_PROMPT = """You are parsing a PBC (Prepared-By-Client) list for a financial statement audit.

A PBC list enumerates the documents the audit team requests from the client. The text
may be free-form: numbered items, bullet points, or prose. Your job is to extract EACH
distinct request as a structured item.

For each item, produce:
  - id: a stable identifier. Use the item's own numbering if present (e.g. "PBC-01", "1",
    "3.a"). If none, assign sequential IDs "ITEM-01", "ITEM-02", ...
  - category: the audit area (e.g. Cash, Fixed Assets, Revenue & AR, Governance, Tax).
    Infer it if not stated.
  - description: the full text of what is being requested.
  - acceptance_criteria: a normalized, machine-checkable statement of what makes this item
    SATISFIED. Extract concrete conditions the client's submission must meet — periods,
    entities, thresholds, signatures, completeness ("all accounts", "every meeting"),
    document types. This is what a verifier will check evidence against, so be specific and
    literal. Do NOT invent criteria not implied by the text.
  - priority: High / Medium / Low if stated or clearly implied, else "Medium".
  - expected_documents: likely file types (pdf, excel, docx, zip, image), best guess.

Call the emit_pbc_items tool exactly once with the full list.
Do not skip items. Do not merge distinct requests. Do not hallucinate items not in the text."""


EMIT_TOOL = {
    "name": "emit_pbc_items",
    "description": "Emit the full structured list of PBC items parsed from the text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "category": {"type": "string"},
                        "description": {"type": "string"},
                        "acceptance_criteria": {"type": "string"},
                        "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                        "expected_documents": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["id", "category", "description", "acceptance_criteria"],
                },
            }
        },
        "required": ["items"],
    },
}


def parse_pbc_list_llm(
    pdf_path: str,
    model: str = "claude-haiku-4-5-20251001",
    use_cache: bool = True,
) -> list[PBCItem]:
    """
    Parse a free-form PBC list PDF into structured PBCItem objects using one LLM call.

    Falls back to the deterministic regex parser if no API key is set.
    """
    text = extract_pdf_text(pdf_path)
    key = _cache_key(text)

    # Cache hit — free and deterministic across clean checkouts
    if use_cache:
        cached = _load_cache(key)
        if cached is not None:
            return [_dict_to_item(d) for d in cached]

    # No API key — degrade gracefully to the regex parser
    if not os.environ.get("ANTHROPIC_API_KEY"):
        from agent.ingest import _parse_pbc_text
        return _parse_pbc_text(text)

    from anthropic import Anthropic
    client = Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=PBC_STRUCTURING_PROMPT,
        messages=[{"role": "user", "content": f"PBC list text:\n\n{text}"}],
        tools=[EMIT_TOOL],
        tool_choice={"type": "tool", "name": "emit_pbc_items"},
    )

    items_data = []
    for block in response.content:
        if block.type == "tool_use":
            items_data = block.input.get("items", [])
            break

    if use_cache and items_data:
        _save_cache(key, items_data)

    return [_dict_to_item(d) for d in items_data]


def _dict_to_item(d: dict[str, Any]) -> PBCItem:
    return PBCItem(
        id=d["id"],
        category=d.get("category", "Uncategorized"),
        description=d.get("description", ""),
        acceptance_criteria=d.get("acceptance_criteria", ""),
        priority=d.get("priority", "Medium"),
        expected_documents=d.get("expected_documents", []),
    )
