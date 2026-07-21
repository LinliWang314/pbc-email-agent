"""
Tests for email-direction detection (auditor request vs client submission).

An auditor's request for an item must not be mistaken for received evidence — a
common false-positive source at held-out scale. Domains are derived from the profile,
never hardcoded, so a swapped engagement still works.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ingest import derive_client_domains, email_direction
from agent.models import EmailMessage


def _email(sender):
    return EmailMessage(message_id="<x>", thread_id="t", sender=sender,
                        recipients="", subject="", date="", body="")


def test_derives_client_domain_not_hardcoded():
    emails = [_email("Jenna <jenna@clientco.example.com>"),
              _email("Marcus <marcus@auditfirm.example.com>")]
    domains = derive_client_domains(emails, {"controller": "Jenna"})
    assert "clientco.example.com" in domains
    assert "auditfirm.example.com" not in domains


def test_direction_client_vs_auditor():
    domains = {"clientco.example.com"}
    assert email_direction(_email("Jenna <jenna@clientco.example.com>"), domains) == "client"
    assert email_direction(_email("Marcus <marcus@auditfirm.example.com>"), domains) == "auditor"


def test_direction_unknown_without_domains():
    assert email_direction(_email("x <x@y.com>"), set()) == "unknown"


if __name__ == "__main__":
    test_derives_client_domain_not_hardcoded()
    test_direction_client_vs_auditor()
    test_direction_unknown_without_domains()
    print("All direction tests passed ✓")
