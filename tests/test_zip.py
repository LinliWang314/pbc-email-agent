"""
Tests for ZIP attachment parsing and graceful failure.
"""

import os
import sys
import tempfile
import zipfile

sys.path.insert(0, '/local/home/linliw/pbc-agent/project')

from tools.parsers import set_attachments_dir, parse_zip


def test_parse_real_sample_zip():
    set_attachments_dir('/local/home/linliw/pbc-agent/project/sample_data/sample/attachments')
    r = parse_zip('Customer_Confirmations_Batch1.zip')
    assert r.get("member_count") == 3
    assert len(r.get("members", [])) == 3
    # Each inner PDF should have been parsed to text
    assert any("Confirmation" in (m.get("parsed", {}).get("full_text", "")) for m in r["members"])


def test_missing_zip_graceful():
    set_attachments_dir('/local/home/linliw/pbc-agent/project/sample_data/sample/attachments')
    r = parse_zip('does_not_exist.zip')
    assert "error" in r


def test_corrupt_zip_graceful():
    d = tempfile.mkdtemp()
    bad = os.path.join(d, "corrupt.zip")
    with open(bad, "w") as f:
        f.write("this is not a zip file")
    set_attachments_dir(d)
    r = parse_zip("corrupt.zip")
    assert "error" in r
    assert "manual review" in r["error"].lower() or "valid zip" in r["error"].lower()


if __name__ == "__main__":
    test_parse_real_sample_zip()
    test_missing_zip_graceful()
    test_corrupt_zip_graceful()
    print("All zip tests passed ✓")
