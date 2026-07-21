"""
Tests for TF-IDF document→PBC-item classification.

Guards against the mis-binding that a keyword heuristic produced at scale
(e.g. AP aging vs AR aging), and checks accounting-abbreviation expansion.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.pbc_parser import parse_pbc_list_llm
from agent.models import TrackerState
from tools.classification import classify_document, _tokenize

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Run:
    def __init__(self, items):
        self.tracker = TrackerState(items={i.id: i for i in items})


def _run():
    os.environ.pop("ANTHROPIC_API_KEY", None)
    pbc = parse_pbc_list_llm(os.path.join(PROJECT, "sample_data", "PBC_List_FY2026.pdf"))
    return _Run(pbc)


def test_synonym_expansion():
    toks = _tokenize("AR_Aging")
    assert "receivable" in toks and "accounts" in toks
    assert "payable" in _tokenize("AP")


def test_ar_vs_ap_not_confused():
    # The key property: AP (payable) must NOT be pulled toward the AR (receivable)
    # items, and vice versa. As a candidate-retrieval step the correct item must be
    # in the top candidates; the LLM agent picks the final one with fuller context.
    run = _run()
    ar_ids = [c["item_id"] for c in classify_document(
        "AR_Aging_YE_2026.xlsx", "aged accounts receivable buckets current 30 60 90",
        "excel", run=run)["candidates"][:3]]
    ap_ids = [c["item_id"] for c in classify_document(
        "AP_Aging_2026.xlsx", "aged accounts payable vendor balances", "excel", run=run)["candidates"][:3]]
    assert ar_ids[0] == "PBC-10"         # AR aging is the top pick for the AR doc
    assert ap_ids[0] == "PBC-13"         # AP aging is the top pick for the AP doc


def test_bank_statement_binds_to_cash_item():
    run = _run()
    r = classify_document("Silverline_Operating_Q4_FY26.pdf",
                          "Q4 bank statement operating account", "pdf", run=run)
    assert r["candidates"][0]["item_id"] == "PBC-04"


def test_returns_separated_candidates():
    run = _run()
    r = classify_document("FixedAssetRegister_FY26.xlsx", "cost depreciation NBV by asset", "excel", run=run)
    assert r["method"] == "tfidf_cosine"
    assert r["candidates"][0]["item_id"] == "PBC-07"


if __name__ == "__main__":
    test_synonym_expansion()
    test_ar_vs_ap_not_confused()
    test_bank_statement_binds_to_cash_item()
    test_returns_separated_candidates()
    print("All classification tests passed ✓")
