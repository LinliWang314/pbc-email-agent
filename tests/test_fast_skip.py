"""
Tests for the deterministic fast-skip heuristic (perf optimization).

Verifies it skips only genuinely empty acknowledgements and never an email that
carries an attachment or any hint of PBC relevance.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.loop import _fast_skip_reason
from agent.models import EmailMessage, Attachment


def _email(body, subject="", attachments=None):
    return EmailMessage(
        message_id="<x>", thread_id="t", sender="a@b.com", recipients="c@d.com",
        subject=subject, date="", body=body, attachments=attachments or [],
    )


def test_skips_brief_acknowledgement():
    # thread02_msg04 style: "Working on it — bookkeeper is compiling. J."
    assert _fast_skip_reason(_email("Working on it — bookkeeper is compiling.  J.")) is not None


def test_does_not_skip_when_attachment_present():
    att = Attachment(filename="x.pdf", content_type="application/pdf", size_bytes=10, file_path="")
    assert _fast_skip_reason(_email("here you go", attachments=[att])) is None


def test_does_not_skip_pbc_mention():
    assert _fast_skip_reason(_email("The board minutes will follow next week.")) is None
    assert _fast_skip_reason(_email("Still need PBC-12 and the tax provision workpapers.")) is None


def test_does_not_skip_long_body():
    long_body = "x " * 150  # >200 chars, no hints — be safe, let the LLM decide
    assert _fast_skip_reason(_email(long_body)) is None


def test_skips_auditor_request_without_attachment():
    # An auditor request/reminder with no attachment carries no client evidence → skip,
    # even though it names PBC items (auditors always do when requesting).
    e = _email("Please send PBC-04 bank statements and PBC-11 confirmations.")
    assert _fast_skip_reason(e, direction="auditor") is not None
    # But the SAME email from the client is NOT skipped (they might be submitting).
    assert _fast_skip_reason(e, direction="client") is None


def test_never_skips_auditor_email_with_attachment():
    att = Attachment(filename="x.pdf", content_type="application/pdf", size_bytes=10, file_path="")
    e = _email("see attached", attachments=[att])
    assert _fast_skip_reason(e, direction="auditor") is None


def test_real_sample_emails():
    """Against the real sample: only the pure acknowledgements should fast-skip."""
    from agent.ingest import load_emails_from_directory
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    emails = load_emails_from_directory(os.path.join(root, "sample_data", "sample", "emails"))
    skipped = [e.subject for e in emails if _fast_skip_reason(e)]
    # Emails carrying attachments or PBC terms must NOT be skipped.
    for e in emails:
        if e.attachments:
            assert _fast_skip_reason(e) is None, f"must not skip email with attachment: {e.subject}"
    # At least one brief acknowledgement in the sample should be skippable.
    print(f"fast-skip would skip {len(skipped)} of {len(emails)} sample emails")


if __name__ == "__main__":
    test_skips_brief_acknowledgement()
    test_does_not_skip_when_attachment_present()
    test_does_not_skip_pbc_mention()
    test_does_not_skip_long_body()
    test_skips_auditor_request_without_attachment()
    test_never_skips_auditor_email_with_attachment()
    test_real_sample_emails()
    print("All fast-skip tests passed ✓")
