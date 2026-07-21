"""
Ingestion layer — loads emails, attachments, PBC list, and client profile.

Handles both .eml files and .mbox format.
"""

import email
import os
import re
from email import policy
from pathlib import Path
from typing import Any

from agent.models import PBCItem, EmailMessage, Attachment


def load_emails_from_directory(eml_dir: str) -> list[EmailMessage]:
    """Load all .eml files from a directory."""
    messages = []
    for filename in sorted(os.listdir(eml_dir)):
        if not filename.endswith(".eml"):
            continue
        filepath = os.path.join(eml_dir, filename)
        with open(filepath, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        messages.append(_parse_email(msg, filename))
    return messages


def load_emails_from_mbox(mbox_path: str) -> list[EmailMessage]:
    """Load emails from an mbox file."""
    import mailbox
    mbox = mailbox.mbox(mbox_path)
    messages = []
    for i, msg in enumerate(mbox):
        messages.append(_parse_email(msg, f"mbox_msg_{i:03d}"))
    return messages


def _parse_email(msg: Any, source_file: str) -> EmailMessage:
    """Parse an email.message.Message into our EmailMessage model."""
    # Extract body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="replace")
                break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode("utf-8", errors="replace")

    # Extract attachments metadata
    attachments = []
    if msg.is_multipart():
        for part in msg.walk():
            filename = part.get_filename()
            if filename:
                attachments.append(Attachment(
                    filename=filename,
                    content_type=part.get_content_type(),
                    size_bytes=len(part.get_payload(decode=True) or b""),
                    file_path="",  # Will be resolved against attachments dir
                ))

    # Determine thread ID from subject or In-Reply-To
    message_id = msg.get("Message-ID", "")
    in_reply_to = msg.get("In-Reply-To", "")
    thread_id = _compute_thread_id(msg.get("Subject", ""), message_id, in_reply_to, source_file)

    return EmailMessage(
        message_id=message_id,
        thread_id=thread_id,
        sender=msg.get("From", ""),
        recipients=msg.get("To", ""),
        subject=msg.get("Subject", ""),
        date=msg.get("Date", ""),
        body=body.strip(),
        attachments=attachments,
        in_reply_to=in_reply_to or None,
    )


def _compute_thread_id(subject: str, message_id: str, in_reply_to: str, source_file: str) -> str:
    """Compute a thread ID for grouping emails."""
    # If source file has thread info, use it
    match = re.match(r"thread(\d+)_", source_file)
    if match:
        return f"thread_{match.group(1)}"

    # Otherwise use normalized subject
    normalized = re.sub(r"^(Re:\s*|Fwd:\s*)+", "", subject, flags=re.IGNORECASE).strip()
    return f"thread_{hash(normalized) % 10000:04d}"


def load_pbc_list(pdf_path: str) -> list[PBCItem]:
    """
    Parse the PBC list PDF into structured PBCItem objects.
    Handles the format: PBC-XX [Category] Priority: X\n  Description\n  Acceptance: ...
    """
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        # Fallback to pdftotext
        import subprocess
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True, text=True
        )
        text = result.stdout

    return _parse_pbc_text(text)


def _parse_pbc_text(text: str) -> list[PBCItem]:
    """Parse PBC list text into structured items."""
    items = []

    # Pattern: PBC-XX [Category] Priority: X
    pattern = re.compile(
        r'(PBC-\d+)\s*\[([^\]]+)\]\s*Priority:\s*(\w+)\s*\n'
        r'\s*(.+?)(?=\n\s*Acceptance:)'
        r'\s*Acceptance:\s*(.+?)(?=\n\s*Expected documents:)'
        r'\s*Expected documents:\s*(.+?)(?=\n\s*(?:PBC-\d+|Confidential|$))',
        re.DOTALL
    )

    for match in pattern.finditer(text):
        item_id = match.group(1)
        category = match.group(2).strip()
        priority = match.group(3).strip()
        description = match.group(4).strip().replace("\n", " ").replace("  ", " ")
        acceptance = match.group(5).strip().replace("\n", " ").replace("  ", " ")
        expected_docs = [d.strip() for d in match.group(6).strip().split(",")]

        items.append(PBCItem(
            id=item_id,
            category=category,
            description=description,
            acceptance_criteria=acceptance,
            priority=priority,
            expected_documents=expected_docs,
        ))

    # If regex didn't catch all items, try a simpler approach
    if len(items) < 10:
        items = _parse_pbc_text_simple(text)

    return items


def _parse_pbc_text_simple(text: str) -> list[PBCItem]:
    """Simpler PBC parser that handles format variations."""
    items = []
    lines = text.split("\n")
    current_item = None
    current_text = []

    for line in lines:
        # New item starts
        match = re.match(r'\s*(PBC-\d+)\s*\[([^\]]+)\]\s*Priority:\s*(\w+)', line)
        if match:
            # Save previous item
            if current_item:
                items.append(_finalize_item(current_item, current_text))
            current_item = {
                "id": match.group(1),
                "category": match.group(2).strip(),
                "priority": match.group(3).strip(),
            }
            current_text = []
        elif current_item:
            current_text.append(line)

    # Don't forget the last item
    if current_item:
        items.append(_finalize_item(current_item, current_text))

    return items


def _finalize_item(item_data: dict, text_lines: list[str]) -> PBCItem:
    """Finalize a PBC item from collected text lines."""
    full_text = " ".join(line.strip() for line in text_lines if line.strip())

    # Split on "Acceptance:" and "Expected documents:"
    description = full_text
    acceptance = ""
    expected_docs = []

    if "Acceptance:" in full_text:
        parts = full_text.split("Acceptance:", 1)
        description = parts[0].strip()
        rest = parts[1]
        if "Expected documents:" in rest:
            acc_parts = rest.split("Expected documents:", 1)
            acceptance = acc_parts[0].strip()
            expected_docs = [d.strip() for d in acc_parts[1].strip().split(",")]
        else:
            acceptance = rest.strip()

    return PBCItem(
        id=item_data["id"],
        category=item_data["category"],
        description=description,
        acceptance_criteria=acceptance,
        priority=item_data["priority"],
        expected_documents=expected_docs,
    )


def build_contact_directory(
    emails: list[EmailMessage],
    profile_contacts: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """
    Build a name -> real email directory from actual mailbox addresses.

    The client profile lists contact NAMES but not addresses, so drafting must use
    the real addresses observed in the mailbox rather than guessing a domain. We only
    include client-side contacts, and tag each with their role from the profile when
    it can be matched by name.
    """
    profile_contacts = profile_contacts or {}
    # name (lowercased) -> role
    name_to_role = {v.lower(): k for k, v in profile_contacts.items()}

    seen: dict[str, dict[str, str]] = {}
    for e in emails:
        # Parse "Name <addr>" from the From header
        m = re.match(r"\s*(.*?)\s*<([^>]+)>", e.sender or "")
        if not m:
            continue
        name, addr = m.group(1).strip(), m.group(2).strip().lower()
        if addr in seen:
            continue
        role = name_to_role.get(name.lower(), "")
        seen[addr] = {"name": name, "email": addr, "role": role}

    # Keep only client-side contacts (those matched to a profile role), if any matched;
    # otherwise return everyone (defensive — unknown engagements).
    client_side = [c for c in seen.values() if c["role"]]
    return client_side if client_side else list(seen.values())


def derive_client_domains(
    emails: list[EmailMessage],
    profile_contacts: dict[str, str] | None = None,
) -> set[str]:
    """
    Derive the client's email domain(s) from the profile contacts' addresses in the
    mailbox — WITHOUT hardcoding any domain (the engagement is swapped at review).

    Client contacts (CFO/Controller/Bookkeeper names from the profile) are matched to
    their observed addresses; their domains are the "client side". Everything else
    (the audit firm, third parties) is treated as non-client. Used to tell whether an
    email is a client SUBMISSION (can produce evidence) vs an auditor REQUEST (cannot).
    """
    directory = build_contact_directory(emails, profile_contacts)
    domains = set()
    for c in directory:
        addr = c.get("email", "")
        if "@" in addr and c.get("role"):
            domains.add(addr.split("@", 1)[1].lower())
    return domains


def email_direction(email: EmailMessage, client_domains: set[str]) -> str:
    """
    Classify an email as 'client' (a submission from the client side) or 'auditor'
    (a request/other from the audit firm or outside). Falls back to 'unknown' when we
    can't tell, which the agent treats conservatively.
    """
    sender = email.sender or ""
    # Normalize obfuscated addresses ("user at domain.com" → "user@domain.com"),
    # seen in list archives and anti-scrape formats. Real .eml uses "Name <addr>".
    sender_norm = re.sub(r"\s+at\s+", "@", sender)
    m = (re.search(r"<([^>]+)>", sender_norm)
         or re.search(r"([^\s<]+@[^\s>()]+)", sender_norm))
    if not m:
        return "unknown"
    domain = m.group(1).split("@")[-1].lower().strip(">.")
    if not client_domains:
        return "unknown"
    return "client" if domain in client_domains else "auditor"


def load_client_profile(pdf_path: str) -> dict[str, Any]:
    """Parse client profile PDF into structured data."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        import subprocess
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True, text=True
        )
        text = result.stdout

    profile = {
        "entity_name": "",
        "fiscal_year_end": "",
        "subsidiaries": [],
        "contacts": {},
    }

    # Extract key fields
    for line in text.split("\n"):
        if "Legal Entity Name:" in line:
            profile["entity_name"] = line.split(":", 1)[1].strip()
        elif "Fiscal Year End:" in line:
            profile["fiscal_year_end"] = line.split(":", 1)[1].strip()
        elif "Controller:" in line:
            profile["contacts"]["controller"] = line.split(":", 1)[1].strip()
        elif "CFO:" in line:
            profile["contacts"]["cfo"] = line.split(":", 1)[1].strip()
        elif "Bookkeeper:" in line:
            profile["contacts"]["bookkeeper"] = line.split(":", 1)[1].strip()

    # Extract subsidiaries
    in_subs = False
    for line in text.split("\n"):
        if "Consolidated Entities:" in line:
            in_subs = True
            continue
        if in_subs:
            if line.strip().startswith("-"):
                profile["subsidiaries"].append(line.strip().lstrip("- "))
            elif line.strip() and not line.strip().startswith("-"):
                in_subs = False

    return profile
