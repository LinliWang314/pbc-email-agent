"""
Tests for the citation verification guardrail — the core anti-hallucination check.

These prove that fabricated citations (values not actually in the document) are
caught deterministically, while genuine citations pass.
"""

import sys
sys.path.insert(0, '/local/home/linliw/pbc-agent/project')

from tools.citation_verify import verify_citation, verify_extraction, normalize


# --- Sample parsed documents ---

PDF_DOC = {
    "filename": "statement.pdf",
    "total_pages": 2,
    "pages": [
        {"page_number": 1, "text": "Northwind Beverages Inc. Balance Sheet"},
        {"page_number": 2, "text": "Period ended 2026-06-30. Total assets $1,234,567."},
    ],
    "full_text": "Northwind Beverages Inc. Balance Sheet\nPeriod ended 2026-06-30. Total assets $1,234,567.",
}

EXCEL_DOC = {
    "filename": "aging.xlsx",
    "sheet_names": ["Sheet1"],
    "sheets": {
        "Sheet1": {
            "headers": [{"ref": "A1", "value": "Customer"}, {"ref": "B1", "value": "Balance"}],
            "rows": [
                {"A1": "Customer", "B1": "Balance"},
                {"A2": "Acme Corp", "B2": "50000"},
            ],
        }
    },
}


def test_genuine_pdf_citation_passes():
    result = verify_citation(
        "2026-06-30",
        {"type": "page", "reference": "page 2", "text": "2026-06-30"},
        PDF_DOC,
    )
    assert result["verified"] is True


def test_fabricated_pdf_citation_fails():
    # Model claims a value that is NOT in the document
    result = verify_citation(
        "2025-12-31",
        {"type": "page", "reference": "page 2", "text": "2025-12-31"},
        PDF_DOC,
    )
    assert result["verified"] is False


def test_genuine_excel_cell_passes():
    result = verify_citation(
        "50000",
        {"type": "cell", "reference": "B2", "text": "50000"},
        EXCEL_DOC,
    )
    assert result["verified"] is True


def test_fabricated_excel_cell_fails():
    result = verify_citation(
        "999999",
        {"type": "cell", "reference": "B2", "text": "999999"},
        EXCEL_DOC,
    )
    assert result["verified"] is False


def test_amount_matches_despite_formatting():
    # "$1,234,567" claimed as "1234567" should still match after normalization
    result = verify_citation(
        "$1,234,567",
        {"type": "page", "reference": "page 2", "text": "$1,234,567"},
        PDF_DOC,
    )
    assert result["verified"] is True


def test_verify_extraction_all_genuine():
    citations = [
        {"type": "page", "reference": "page 2", "text": "2026-06-30"},
        {"type": "page", "reference": "page 2", "text": "$1,234,567"},
    ]
    result = verify_extraction({}, citations, PDF_DOC)
    assert result["all_verified"] is True
    assert result["has_fabrication"] is False
    assert result["citation_confidence"] == 1.0


def test_verify_extraction_with_fabrication():
    citations = [
        {"type": "page", "reference": "page 2", "text": "2026-06-30"},   # real
        {"type": "page", "reference": "page 2", "text": "fabricated"},   # fake
    ]
    result = verify_extraction({}, citations, PDF_DOC)
    assert result["all_verified"] is False
    assert result["has_fabrication"] is True
    assert result["citation_confidence"] == 0.5


def test_normalize():
    assert normalize("$1,234,567") == "1234567"
    assert normalize("  Hello World  ") == "helloworld"
    assert normalize("2026-06-30") == "2026-06-30"


if __name__ == "__main__":
    test_genuine_pdf_citation_passes()
    test_fabricated_pdf_citation_fails()
    test_genuine_excel_cell_passes()
    test_fabricated_excel_cell_fails()
    test_amount_matches_despite_formatting()
    test_verify_extraction_all_genuine()
    test_verify_extraction_with_fabrication()
    test_normalize()
    print("All citation verification tests passed ✓")
