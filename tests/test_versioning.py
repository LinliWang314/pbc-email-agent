"""
Tests for document version detection logic.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_supersession_marks_old_evidence_and_keeps_lineage():
    # End-to-end: v1 then v3 of the same doc. Old version must be flagged superseded,
    # the newest is current, and the lineage is recorded (auditors need the trail).
    from agent.loop import AgentRun
    from agent.models import TrackerState, PBCItem
    from tools.tracker_ops import update_item_status
    item = PBCItem(id="PBC-14", category="AP & Accruals", description="accruals",
                   acceptance_criteria="as_of=2026-06-30")
    run = AgentRun(tracker=TrackerState(items={"PBC-14": item}))
    run.parsed_documents = {}
    update_item_status("PBC-14", "Received", "draft", run=run, evidence_filename="Accruals_v1.xlsx")
    update_item_status("PBC-14", "Received", "final", run=run, evidence_filename="Accruals_Final_v3_REAL.xlsx")
    evs = run.tracker.items["PBC-14"].evidence
    assert len(evs) == 2
    assert evs[0].superseded is True and evs[0].filename == "Accruals_v1.xlsx"
    assert evs[1].superseded is False
    assert len(run.tracker.items["PBC-14"].versions) == 1


if __name__ == "__main__":
    test_version_detection_basic()
    test_version_detection_different_docs()
    test_version_detection_edge_cases()
    test_supersession_marks_old_evidence_and_keeps_lineage()
    print("All versioning tests passed ✓")
