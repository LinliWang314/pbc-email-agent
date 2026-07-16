"""
Tests for email parsing and PBC list ingestion.
"""

import sys
sys.path.insert(0, '/local/home/linliw/pbc-agent/project')

from agent.ingest import _parse_pbc_text_simple, _compute_thread_id


def test_thread_id_from_filename():
    assert _compute_thread_id("Re: test", "<123>", "<456>", "thread01_msg03.eml") == "thread_01"
    assert _compute_thread_id("Re: test", "<123>", "<456>", "thread04_msg01.eml") == "thread_04"


def test_thread_id_from_subject():
    id1 = _compute_thread_id("FY2026 audit kickoff", "<a>", "", "msg.eml")
    id2 = _compute_thread_id("Re: FY2026 audit kickoff", "<b>", "<a>", "msg2.eml")
    assert id1 == id2  # Same thread


def test_pbc_parsing():
    sample_text = """
PBC-01 [Financial Statements] Priority: High
   Adjusted trial balance as of fiscal year-end.
   Acceptance: period_end=2026-06-30, entity=consolidated
   Expected documents: excel, pdf

PBC-02 [Financial Statements] Priority: Medium
   Draft income statement.
   Acceptance: period=FY2026
   Expected documents: excel
"""
    items = _parse_pbc_text_simple(sample_text)
    assert len(items) == 2
    assert items[0].id == "PBC-01"
    assert items[0].category == "Financial Statements"
    assert items[0].priority == "High"
    assert "trial balance" in items[0].description
    assert "period_end=2026-06-30" in items[0].acceptance_criteria


if __name__ == "__main__":
    test_thread_id_from_filename()
    test_thread_id_from_subject()
    test_pbc_parsing()
    print("All parsing tests passed ✓")
