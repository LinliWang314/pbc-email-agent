"""
Tests for document version detection logic.
"""

import sys
sys.path.insert(0, '/local/home/linliw/pbc-agent/project')

from tools.tracker_ops import _is_same_document


def test_version_detection_basic():
    assert _is_same_document("Report_v1.xlsx", "Report_v2.xlsx")
    assert _is_same_document("Final_v1.xlsx", "Final_v3_REAL.xlsx")
    assert _is_same_document("Report_draft.xlsx", "Report_final.xlsx")


def test_version_detection_different_docs():
    assert not _is_same_document("AR_Aging.xlsx", "FixedAssetRegister.xlsx")
    assert not _is_same_document("Invoice_001.pdf", "Board_Minutes.pdf")


def test_version_detection_edge_cases():
    assert _is_same_document("data_v1.xlsx", "data_v99.xlsx")
    assert _is_same_document("Final_Report.pdf", "Final_Report_v2.pdf")
    assert not _is_same_document("Q1_Report.pdf", "Q2_Report.pdf")


if __name__ == "__main__":
    test_version_detection_basic()
    test_version_detection_different_docs()
    test_version_detection_edge_cases()
    print("All versioning tests passed ✓")
