"""
Synthetic stress dataset generator — approximates the held-out scale/shape.

Produces ~90 emails / ~12 threads / ~40 attachments covering all 30 PBC items, plus
adversarial traps drawn from the brief (wrong period, wrong entity, threshold-not-met,
incomplete set, Q2-when-Q3-asked, version supersession) and noise emails (chatter with
no PBC content). Writes a ground-truth file so eval can score it.

This is NOT the real held-out set — it's a self-built proxy to check the pipeline holds
at scale (esp. document→item classification) before the live review. Deterministic: no
randomness, so runs are reproducible.

Usage: python -m eval.generate_stress_set  → writes to stress_data/
"""

import email.utils
import os
import zipfile
from pathlib import Path

OUT = Path(__file__).parent.parent / "stress_data"
EMAILS = OUT / "sample" / "emails"
ATTACH = OUT / "sample" / "attachments"

SENDER = "Jenna Alvarez <jenna.alvarez@northwindbev.example.com>"
BOOKKEEPER = "Sam Whitaker <sam.whitaker@northwindbev.example.com>"
CFO = "David Okafor <david.okafor@northwindbev.example.com>"
AUDITOR = "Marcus Chen <marcus.chen@fairmontcole.example.com>"


def _eml(idx, thread, msgno, frm, to, subject, body, in_reply_to=None, attachments=None):
    mid = f"<stress.{idx}.{thread}.{msgno}@northwind.example.com>"
    # Fixed timestamps (no Date.now — deterministic); space them out by index.
    date = email.utils.formatdate(1_780_000_000 + idx * 3600)
    lines = [f"From: {frm}", f"To: {to}", f"Subject: {subject}",
             f"Date: {date}", f"Message-ID: {mid}", "MIME-Version: 1.0"]
    if in_reply_to:
        lines.append(f"In-Reply-To: {in_reply_to}")
    if attachments:
        # Minimal multipart with attachment filename references (bodies live in ATTACH/)
        boundary = f"bnd{idx}"
        lines.append(f'Content-Type: multipart/mixed; boundary="{boundary}"')
        parts = [f"--{boundary}", "Content-Type: text/plain", "", body]
        for fn in attachments:
            parts += [f"--{boundary}",
                      f'Content-Type: application/octet-stream; name="{fn}"',
                      f'Content-Disposition: attachment; filename="{fn}"', "", "(binary)"]
        parts.append(f"--{boundary}--")
        content = "\r\n".join(lines) + "\r\n\r\n" + "\r\n".join(parts) + "\r\n"
    else:
        lines += ["Content-Type: text/plain", ""]
        content = "\r\n".join(lines) + "\r\n" + body + "\r\n"
    return mid, content


# (filename, content, doc_type) → written into ATTACH; content is what parsers will read.
def _pdf(path, text):
    # Write a real minimal PDF so pdftotext/PyPDF2 can read it.
    # Simple single-stream PDF.
    stream = f"BT /F1 10 Tf 40 750 Td ({text[:1500]}) Tj ET"
    objs = [
        "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj",
        "2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj",
        "3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj",
        f"4 0 obj<</Length {len(stream)}>>stream\n{stream}\nendstream endobj",
        "5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj",
    ]
    pdf = "%PDF-1.4\n"
    offsets = []
    for o in objs:
        offsets.append(len(pdf))
        pdf += o + "\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n"
    pdf += f"trailer<</Size {len(objs)+1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF"
    with open(path, "w", encoding="latin-1") as f:
        f.write(pdf)


def _xlsx(path, rows):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    wb.save(path)


def build():
    EMAILS.mkdir(parents=True, exist_ok=True)
    ATTACH.mkdir(parents=True, exist_ok=True)

    truth = {}          # PBC-XX -> {status, evidence?}
    idx = 0

    def emit(thread, msgno, frm, to, subj, body, atts=None, reply=None):
        nonlocal idx
        idx += 1
        mid, content = _eml(idx, thread, msgno, frm, to, subj, body, reply, atts)
        (EMAILS / f"thread{thread:02d}_msg{msgno:02d}.eml").write_text(content, encoding="utf-8")
        return mid

    # ---- Clean "Received" items (doc arrives, matches) ----
    _xlsx(ATTACH / "FixedAssetRegister_FY26.xlsx",
          [["Asset", "Cost", "Accum Depr", "NBV"], ["Machinery", 500000, 120000, 380000],
           ["Vehicles", 90000, 30000, 60000]])
    emit(1, 1, SENDER, AUDITOR, "PBC-07 fixed asset register",
         "Attached the fixed asset register at year end.", ["FixedAssetRegister_FY26.xlsx"])
    truth["PBC-07"] = {"status": "Received"}

    _xlsx(ATTACH / "AR_Aging_YE_2026-06-30.xlsx",
          [["Customer", "Current", "1-30", "31-60", "61-90", "90+", "Total"],
           ["Rocky Mountain", 184200, 22400, 8100, 0, 0, 214700]])
    emit(2, 1, SENDER, AUDITOR, "PBC-10 AR aging",
         "Aged accounts receivable listing attached.", ["AR_Aging_YE_2026-06-30.xlsx"])
    truth["PBC-10"] = {"status": "Received"}

    _xlsx(ATTACH / "AP_Aging_YE_2026.xlsx",
          [["Vendor", "Current", "1-30", "Total"], ["Acme Supply", 44000, 12000, 56000]])
    emit(3, 1, SENDER, AUDITOR, "PBC-13 AP aging",
         "Accounts payable aging as of year end attached.", ["AP_Aging_YE_2026.xlsx"])
    truth["PBC-13"] = {"status": "Received"}

    _xlsx(ATTACH / "Payroll_Register_FY26.xlsx",
          [["Employee", "Gross Wages"], ["E. Smith", 82000], ["Total", 4200000]])
    emit(4, 1, SENDER, AUDITOR, "PBC-16 payroll register",
         "FY26 payroll register reconciled to GL wages.", ["Payroll_Register_FY26.xlsx"])
    truth["PBC-16"] = {"status": "Received"}

    _pdf(ATTACH / "Going_Concern_Memo_FY26.pdf",
         "Going Concern Assessment FY2026 Prepared by David Okafor CFO. 12-month cash flow "
         "forecast operating cash flow 7.2M. Substantial doubt does not exist. Signed David Okafor CFO")
    emit(5, 1, CFO, AUDITOR, "PBC-26 going concern memo",
         "Signed going concern memo attached, includes 12-month forecast.", ["Going_Concern_Memo_FY26.pdf"])
    truth["PBC-26"] = {"status": "Received"}

    # ---- TRAP: wrong period (asked Q4, sent Q2) → Insufficient ----
    _pdf(ATTACH / "BankStmt_Q2_Silverline.pdf",
         "Bank Statement Silverline Operating Account Period October 1 2025 to December 31 2025 Q2 FY2026")
    emit(6, 2, SENDER, AUDITOR, "PBC-04 bank statements",
         "Attaching a bank statement for the operating account.", ["BankStmt_Q2_Silverline.pdf"])
    truth["PBC-04"] = {"status": "Insufficient", "trap": "wrong period (Q2 not Q4) + only one account"}

    # ---- TRAP: wrong entity (asked consolidated, sent subsidiary only) → Insufficient ----
    _xlsx(ATTACH / "TB_Cascade_only.xlsx",
          [["Account", "Balance"], ["Cash", 120000], ["Entity", "Cascade Cold Brew LLC only"]])
    emit(7, 3, SENDER, AUDITOR, "PBC-01 trial balance",
         "Trial balance attached.", ["TB_Cascade_only.xlsx"])
    truth["PBC-01"] = {"status": "Insufficient", "trap": "wrong entity (subsidiary only, not consolidated)"}

    # ---- TRAP: threshold not met (asked >$10k, sent $4.2k) → Insufficient ----
    _pdf(ATTACH / "Invoice_office_chairs.pdf",
         "Invoice office chairs total 4200 dollars fixed asset addition FY2026")
    emit(8, 4, SENDER, AUDITOR, "PBC-08 fixed asset additions invoice",
         "Here is a supporting invoice for a fixed asset addition.", ["Invoice_office_chairs.pdf"])
    truth["PBC-08"] = {"status": "Insufficient", "trap": "threshold not met ($4.2k < $10k)"}

    # ---- TRAP: incomplete set (asked all top-10 confirmations, sent 3 in a zip) → Insufficient ----
    conf_zip = ATTACH / "Customer_Confirmations_Batch1.zip"
    with zipfile.ZipFile(conf_zip, "w") as z:
        for name in ["RockyMtn_signed.txt", "Aurora_signed.txt", "Sunset_signed.txt"]:
            z.writestr(name, "Customer balance confirmation signed. Returned to Fairmont & Cole.")
    emit(9, 5, SENDER, AUDITOR, "PBC-11 customer confirmations",
         "First three signed confirmations attached; chasing the other seven.", ["Customer_Confirmations_Batch1.zip"])
    truth["PBC-11"] = {"status": "Insufficient", "trap": "incomplete set (3 of 10)"}

    # ---- TRAP: informal artifact (photo instead of formal recon) → Insufficient ----
    # (image OCR degrades gracefully; verifier should still flag informality from context)
    (ATTACH / "IMG_2847_wb_recon.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)  # tiny fake jpg
    emit(10, 6, BOOKKEEPER, AUDITOR, "PBC-05 bank rec photo",
         "Photo of the working recon for Silverline operating. Will type it up properly this week.",
         ["IMG_2847_wb_recon.jpg"])
    truth["PBC-05"] = {"status": "Insufficient", "trap": "informal photo, not typed reconciliation"}

    # ---- Version supersession: v1 then v3 of the same doc ----
    _xlsx(ATTACH / "Accruals_YE_v1.xlsx", [["Accrual", "Amount"], ["Bonus", 100000]])
    emit(11, 7, SENDER, AUDITOR, "PBC-14 accruals draft", "Draft accruals schedule.", ["Accruals_YE_v1.xlsx"])
    _xlsx(ATTACH / "Accruals_YE_Final_v3_REAL.xlsx",
          [["Accrual", "Amount"], ["Bonus", 120000], ["Vacation", 45000]])
    emit(12, 7, SENDER, AUDITOR, "Re: PBC-14 accruals FINAL",
         "Corrected accruals schedule — this supersedes the earlier draft.",
         ["Accruals_YE_Final_v3_REAL.xlsx"])
    truth["PBC-14"] = {"status": "Received", "note": "v3 supersedes v1"}

    # ---- Noise emails (no PBC content) — should be skipped, not misclassified ----
    noise = [
        ("Lunch next week?", "Are you free for lunch on Thursday? — J"),
        ("Out of office", "I'll be OOO Friday, back Monday."),
        ("Re: parking", "The garage will be closed this weekend for maintenance."),
        ("Thanks", "Thanks so much for your help earlier, really appreciated it."),
        ("Quick hello", "Great to meet you at the conference last month!"),
    ]
    for i, (subj, body) in enumerate(noise):
        emit(20 + i, 8, SENDER, AUDITOR, subj, body)

    # ---- More clean Received items with RICH documents (so complete items pass) ----
    _pdf(ATTACH / "scan0042.pdf",   # messy filename on purpose
         "Officer and Director Compensation Schedule FY2026. CEO David Okafor base salary "
         "420000 bonus 150000. CFO base 310000 bonus 90000. Director fees 40000 each. Total "
         "officer compensation FY2026 1250000. Period July 2025 to June 2026.")
    emit(40, 10, SENDER, AUDITOR, "PBC-17 officer comp",
         "Officer and director compensation schedule for FY26 attached.", ["scan0042.pdf"])
    truth["PBC-17"] = {"status": "Received"}

    _xlsx(ATTACH / "headcount_recon_FINAL.xlsx",
          [["Metric", "Count"], ["Opening headcount Jul 2025", 210], ["Hires", 34],
           ["Terminations", 19], ["Closing headcount Jun 2026", 225]])
    emit(41, 10, SENDER, AUDITOR, "PBC-18 headcount reconciliation",
         "Headcount reconciliation opening hires terminations closing for FY2026.", ["headcount_recon_FINAL.xlsx"])
    truth["PBC-18"] = {"status": "Received"}

    _pdf(ATTACH / "Form5500_401k_FY26.pdf",
         "Form 5500 filing confirmation 401(k) plan. Plan contributions FY2026 total 1850000 "
         "employer match 620000. Filed with DOL. Reconciliation to payroll attached.")
    emit(42, 10, SENDER, AUDITOR, "PBC-19 401k form 5500",
         "401k plan contribution reconciliation and Form 5500 filing confirmation.", ["Form5500_401k_FY26.pdf"])
    truth["PBC-19"] = {"status": "Received"}

    _pdf(ATTACH / "Prior_Year_1120_FY2025.pdf",
         "Form 1120 U.S. Corporation Income Tax Return FY2025 as filed. Taxable income 8400000 "
         "federal tax 1764000. State returns Colorado California attached. Prior year 2025.")
    emit(43, 11, SENDER, AUDITOR, "PBC-22 prior year tax return",
         "Prior-year federal Form 1120 and state returns as filed.", ["Prior_Year_1120_FY2025.pdf"])
    truth["PBC-22"] = {"status": "Received"}

    _pdf(ATTACH / "Insurance_Schedule_YE.pdf",
         "Insurance Policy Schedule as of June 30 2026. General liability 5M coverage. Property "
         "12M. D&O 10M. Workers comp statutory. Cyber 3M. All policies in effect at year end.")
    emit(44, 11, SENDER, AUDITOR, "PBC-30 insurance schedule",
         "Insurance policy schedule with coverage summaries for all policies at year-end.", ["Insurance_Schedule_YE.pdf"])
    truth["PBC-30"] = {"status": "Received"}

    _xlsx(ATTACH / "Lease_Inventory_ASC842.xlsx",
          [["Lease", "ROU Asset", "Liability", "Term"], ["Denver HQ", 2400000, 2400000, "10yr"],
           ["Warehouse", 890000, 890000, "5yr"], ["Note", "ASC 842 worksheets included", "", ""]])
    emit(45, 11, SENDER, AUDITOR, "PBC-28 lease inventory",
         "Complete lease inventory with ASC 842 lease accounting worksheets.", ["Lease_Inventory_ASC842.xlsx"])
    truth["PBC-28"] = {"status": "Received"}

    # ---- TRAP: Q2 sent when Q3 asked (right family, wrong document) → Insufficient ----
    _pdf(ATTACH / "Final_v3_REAL.pdf",   # deliberately unhelpful name
         "Statement of Cash Flows FY2026 indirect method. Operating 7200000 investing -3100000 "
         "financing -1800000. Net change in cash 2300000. Consolidated all entities.")
    emit(46, 12, SENDER, AUDITOR, "PBC-03 cash flows (also re PBC-02?)",
         "Attaching what you asked for — think this covers the income statement request too.",
         ["Final_v3_REAL.pdf"])
    truth["PBC-03"] = {"status": "Received"}
    truth["PBC-02"] = {"status": "Insufficient",
                       "trap": "client claims this covers PBC-02 but it's a cash flow stmt, not income stmt/BS"}

    # ---- Forward chain / multi-recipient thread ----
    fwd = emit(47, 13, AUDITOR, f"{SENDER}, {CFO}", "Fwd: outstanding items week 3",
               "Forwarding the running list of what's still open. Please prioritize the cash items.")
    emit(48, 13, SENDER, AUDITOR, "Re: Fwd: outstanding items week 3",
         "Acknowledged — Sam is pulling the bank recs, I'll handle the rest.", reply=fwd)

    # ---- Noise emails (no PBC content) — should be skipped, not misclassified ----
    noise = [
        ("Lunch next week?", "Are you free for lunch on Thursday? — J"),
        ("Out of office", "I'll be OOO Friday, back Monday."),
        ("Re: parking", "The garage will be closed this weekend for maintenance."),
        ("Thanks", "Thanks so much for your help earlier, really appreciated it."),
        ("Quick hello", "Great to meet you at the conference last month!"),
        ("Coffee chat", "Loved catching up. Let's do it again soon."),
        ("Reminder: all-hands", "Company all-hands is Friday at 2pm in the main room."),
        ("Re: birthday", "Happy birthday!! Hope you have a great one."),
        ("Fwd: newsletter", "Thought you might find this industry newsletter interesting."),
        ("Re: expense report", "Your Q2 expense report was approved, nothing needed from you."),
    ]
    for i, (subj, body) in enumerate(noise):
        emit(50 + i, 14, SENDER, AUDITOR, subj, body)

    # ---- Filler acknowledgements referencing items (in-flight, no attachment) ----
    for i, iid in enumerate(["PBC-06", "PBC-23", "PBC-20", "PBC-21", "PBC-24", "PBC-25", "PBC-27", "PBC-29"]):
        emit(60 + i, 15, SENDER, AUDITOR, f"Re: {iid} status",
             f"Still working on {iid}, expect to send it by the end of next week.")
        truth.setdefault(iid, {"status": "Not started", "note": "requested, not yet received"})

    # Remaining items: Not started (never mentioned)
    for n in range(1, 31):
        iid = f"PBC-{n:02d}"
        truth.setdefault(iid, {"status": "Not started"})

    # ---- Write ground truth ----
    import json
    gt = {"set": "stress-synthetic", "expected_status": truth}
    (OUT / "sample" / "sample_groundtruth.json").write_text(json.dumps(gt, indent=2))

    # Copy the real PBC list + profile so the run uses the same config
    import shutil
    src = Path(__file__).parent.parent / "sample_data"
    shutil.copy(src / "PBC_List_FY2026.pdf", OUT / "PBC_List_FY2026.pdf")
    shutil.copy(src / "Client_Profile.pdf", OUT / "Client_Profile.pdf")

    # ---- Also produce a single uploadable ZIP bundle (for the UI upload path) ----
    bundle = OUT / "stress_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / "PBC_List_FY2026.pdf", "PBC_List_FY2026.pdf")
        z.write(OUT / "Client_Profile.pdf", "Client_Profile.pdf")
        for p in EMAILS.glob("*.eml"):
            z.write(p, f"emails/{p.name}")
        for p in ATTACH.glob("*"):
            z.write(p, f"attachments/{p.name}")

    n_emails = len(list(EMAILS.glob("*.eml")))
    n_att = len(list(ATTACH.glob("*")))
    print(f"Wrote stress set: {n_emails} emails, {n_att} attachments, {len(truth)} items to {OUT}")
    traps = {k: v for k, v in truth.items() if "trap" in v}
    print(f"Traps: {list(traps.keys())}")
    print(f"Uploadable bundle: {bundle}")


if __name__ == "__main__":
    build()
