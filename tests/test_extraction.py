"""
Tests for field extraction logic.
"""

import sys
sys.path.insert(0, '/local/home/linliw/pbc-agent/project')

from tools.extraction import extract_fields


def test_date_extraction():
    content = "Report for period ending 2026-06-30. Prepared January 15, 2026."
    result = extract_fields("test.pdf", "PBC-01", content, ["date"])
    assert "2026-06-30" in result["extracted_fields"]["dates_found"]


def test_entity_extraction():
    content = "Northwind Beverages consolidated financial statements"
    result = extract_fields("test.pdf", "PBC-01", content, ["entity"])
    assert any("Northwind" in e for e in result["extracted_fields"]["entities_found"])


def test_amount_extraction():
    content = "Total assets: $1,234,567.89. Revenue was $500,000."
    result = extract_fields("test.pdf", "PBC-01", content, ["amount"])
    assert "$1,234,567.89" in result["extracted_fields"]["amounts_found"]


def test_period_extraction():
    content = "For the fiscal year FY2026 ended June 30, 2026"
    result = extract_fields("test.pdf", "PBC-01", content, ["period"])
    assert any("FY2026" in p for p in result["extracted_fields"]["periods_found"])


def test_citation_generation():
    content = "Date: 2026-06-30 is the period end."
    result = extract_fields("test.pdf", "PBC-01", content, ["date"])
    assert len(result["citations"]) > 0
    assert result["citations"][0]["type"] == "text_position"


if __name__ == "__main__":
    test_date_extraction()
    test_entity_extraction()
    test_amount_extraction()
    test_period_extraction()
    test_citation_generation()
    print("All extraction tests passed ✓")
