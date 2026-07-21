"""
Regression test on REAL emails (a slice of the public python-dev list archive).

Synthetic data is always clean; real archives have obfuscated senders
("user at domain.com (Name)"), long quote chains, odd encodings, and multipart
bodies. This fixture caught a real bug: email_direction returned "unknown" on the
"user at domain.com" format. Guards ingestion + direction against real-world mess.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mailbox
from agent.ingest import _parse_email, email_direction

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "real_emails_sample.mbox")


def test_parses_all_real_emails_without_error():
    msgs = list(mailbox.mbox(FIXTURE))
    assert len(msgs) > 0
    parsed = [_parse_email(m, f"real_{i}.eml") for i, m in enumerate(msgs)]
    # Every real email parses; none has an empty body or missing sender.
    assert all(p.sender for p in parsed)
    assert all(p.body.strip() for p in parsed)


def test_thread_reconstruction_on_real_replies():
    msgs = list(mailbox.mbox(FIXTURE))
    parsed = [_parse_email(m, f"real_{i}.eml") for i, m in enumerate(msgs)]
    # Real reply chains carry In-Reply-To; at least some should be detected.
    assert any(p.in_reply_to for p in parsed)


def test_direction_handles_obfuscated_sender():
    # "user at domain.com (Name)" — the list-archive format that broke the naive regex.
    from agent.models import EmailMessage
    e = EmailMessage(message_id="x", thread_id="t",
                     sender="jenna at northwindbev.example.com (Jenna)",
                     recipients="", subject="", date="", body="")
    assert email_direction(e, {"northwindbev.example.com"}) == "client"


if __name__ == "__main__":
    test_parses_all_real_emails_without_error()
    test_thread_reconstruction_on_real_replies()
    test_direction_handles_obfuscated_sender()
    print("All real-email tests passed ✓")
