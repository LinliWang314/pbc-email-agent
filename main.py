"""
PBC Email Agent — Main entry point.

FastAPI server that provides:
  - POST /api/run — process a mailbox against a PBC list
  - GET /api/tracker — get current tracker state
  - GET /api/traces — get agent traces
  - GET /api/followups — get follow-up drafts
  - POST /api/followups/{id}/approve — approve a follow-up draft
  - POST /api/upload — upload a new mailbox + PBC list
"""

import json
import os
import tempfile
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from agent.loop import run_agent, AgentRun
from agent.config import AgentConfig
from agent.ingest import (
    load_emails_from_directory,
    load_emails_from_mbox,
    load_pbc_list,
    load_client_profile,
)
from agent.pbc_parser import parse_pbc_list_llm
from tools.parsers import set_attachments_dir

load_dotenv()

app = FastAPI(title="PBC Email Agent", version="0.1.0")

# Global state for the current run
current_run: AgentRun | None = None


class RunConfig(BaseModel):
    data_dir: str = ""
    max_cost_usd: float = 2.00


@app.post("/api/run")
async def start_run(config: RunConfig):
    """Process a mailbox against the PBC list."""
    global current_run

    # Resolve data dir: explicit arg > env var > bundled sample_data/
    bundled = str(Path(__file__).parent / "sample_data")
    data_dir = config.data_dir or os.environ.get("PBC_DATA_DIR", "") or bundled
    if not os.path.isdir(data_dir):
        raise HTTPException(400, f"Data directory not found: {data_dir}")

    # Load PBC list
    pbc_path = os.path.join(data_dir, "PBC_List_FY2026.pdf")
    if not os.path.exists(pbc_path):
        # Try to find any PBC list PDF
        for f in os.listdir(data_dir):
            if f.lower().startswith("pbc") and f.endswith(".pdf"):
                pbc_path = os.path.join(data_dir, f)
                break

    # Free-form PBC lists require LLM structuring; falls back to regex if no API key
    pbc_items = parse_pbc_list_llm(pbc_path)

    # Load client profile
    profile_path = os.path.join(data_dir, "Client_Profile.pdf")
    client_profile = {}
    if os.path.exists(profile_path):
        client_profile = load_client_profile(profile_path)

    # Load emails
    emails_dir = os.path.join(data_dir, "sample", "emails")
    mbox_path = os.path.join(data_dir, "sample", "sample_mailbox.mbox")

    if os.path.isdir(emails_dir):
        emails = load_emails_from_directory(emails_dir)
    elif os.path.exists(mbox_path):
        emails = load_emails_from_mbox(mbox_path)
    else:
        raise HTTPException(400, "No emails found. Expected sample/emails/ or sample/sample_mailbox.mbox")

    # Set attachments directory
    attachments_dir = os.path.join(data_dir, "sample", "attachments")
    set_attachments_dir(attachments_dir)

    # Configure and run agent
    agent_config = AgentConfig(max_cost_usd=config.max_cost_usd)

    current_run = run_agent(
        pbc_items=pbc_items,
        emails=emails,
        config=agent_config,
        client_contacts=client_profile.get("contacts", {}),
        engagement_info={
            "entity": client_profile.get("entity_name", ""),
            "fiscal_year_end": client_profile.get("fiscal_year_end", ""),
        },
    )

    return {
        "status": "complete",
        "offline": current_run.offline,
        "note": (
            "Offline mode: no ANTHROPIC_API_KEY set. Ingested deterministically; "
            "set the key to run the full agent loop and classify evidence."
            if current_run.offline else "Full agent run complete."
        ),
        "items_processed": len(pbc_items),
        "emails_processed": len(emails),
        "cost_usd": current_run.cost.cost_usd,
        "traces_count": len(current_run.traces),
    }


@app.get("/api/tracker")
async def get_tracker():
    """Get the current tracker state."""
    if current_run is None:
        raise HTTPException(404, "No run has been executed yet")
    return current_run.tracker.dict()


@app.get("/api/traces")
async def get_traces():
    """Get all agent traces."""
    if current_run is None:
        raise HTTPException(404, "No run has been executed yet")
    return [t.dict() for t in current_run.traces]


@app.get("/api/traces/{email_id}")
async def get_trace(email_id: str):
    """Get trace for a specific email."""
    if current_run is None:
        raise HTTPException(404, "No run has been executed yet")
    for trace in current_run.traces:
        if trace.email_id == email_id:
            return trace.dict()
    raise HTTPException(404, f"No trace for email {email_id}")


@app.get("/api/followups")
async def get_followups():
    """Get follow-up email drafts."""
    if current_run is None:
        raise HTTPException(404, "No run has been executed yet")
    return current_run.tracker.followup_drafts


@app.post("/api/followups/{index}/approve")
async def approve_followup(index: int):
    """Approve a follow-up draft (mock send)."""
    if current_run is None:
        raise HTTPException(404, "No run has been executed yet")
    if index >= len(current_run.tracker.followup_drafts):
        raise HTTPException(404, "Follow-up not found")
    current_run.tracker.followup_drafts[index]["approved"] = True
    return {"status": "approved", "note": "Send is mocked — no email was actually sent."}


@app.post("/api/followups/{index}/reject")
async def reject_followup(index: int):
    """Reject a follow-up draft."""
    if current_run is None:
        raise HTTPException(404, "No run has been executed yet")
    if index >= len(current_run.tracker.followup_drafts):
        raise HTTPException(404, "Follow-up not found")
    current_run.tracker.followup_drafts[index]["rejected"] = True
    return {"status": "rejected"}


@app.get("/api/cost")
async def get_cost():
    """Get the cost breakdown for the current run."""
    if current_run is None:
        raise HTTPException(404, "No run has been executed yet")
    return {
        "input_tokens": current_run.cost.input_tokens,
        "output_tokens": current_run.cost.output_tokens,
        "total_cost_usd": current_run.cost.cost_usd,
        "budget_remaining_usd": current_run.config.max_cost_usd - current_run.cost.cost_usd,
    }


@app.post("/api/upload")
async def upload_data(
    pbc_list: UploadFile = File(...),
    mailbox: UploadFile = File(...),
    attachments: list[UploadFile] = File(default=[]),
):
    """Upload a new mailbox + PBC list for processing."""
    # Create temp directory for uploaded data
    upload_dir = tempfile.mkdtemp(prefix="pbc_upload_")
    sample_dir = os.path.join(upload_dir, "sample")
    emails_dir = os.path.join(sample_dir, "emails")
    attach_dir = os.path.join(sample_dir, "attachments")
    os.makedirs(emails_dir, exist_ok=True)
    os.makedirs(attach_dir, exist_ok=True)

    # Save PBC list
    pbc_path = os.path.join(upload_dir, pbc_list.filename)
    with open(pbc_path, "wb") as f:
        f.write(await pbc_list.read())

    # Save mailbox (could be .mbox or .zip of .eml files)
    mailbox_path = os.path.join(sample_dir, mailbox.filename)
    with open(mailbox_path, "wb") as f:
        f.write(await mailbox.read())

    # If it's a zip, extract emails
    if mailbox.filename.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(mailbox_path, "r") as z:
            z.extractall(emails_dir)

    # Save attachments
    for attachment in attachments:
        att_path = os.path.join(attach_dir, attachment.filename)
        with open(att_path, "wb") as f:
            f.write(await attachment.read())

    return {"upload_dir": upload_dir, "status": "uploaded"}


# Serve the frontend
@app.get("/")
async def serve_frontend():
    """Serve the main UI page."""
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return HTMLResponse("<h1>PBC Email Agent</h1><p>UI not built yet. Use /docs for API.</p>")


# Mount static files for UI assets
ui_dir = Path(__file__).parent / "ui" / "static"
if ui_dir.exists():
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
