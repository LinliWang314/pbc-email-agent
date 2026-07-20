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
import threading
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
# Run status for the async job: idle | running | complete | error
run_status: dict = {"state": "idle", "message": "", "started": False}
_run_lock = threading.Lock()


class RunConfig(BaseModel):
    data_dir: str = ""
    max_cost_usd: float = 2.00


def _execute_run(config: RunConfig) -> None:
    """
    Heavy agent run — executed in a background thread so the HTTP request returns
    immediately (the full run takes minutes and would otherwise hit the platform's
    request timeout). Progress is exposed via /api/status.
    """
    global current_run, run_status
    try:
        bundled = str(Path(__file__).parent / "sample_data")
        data_dir = config.data_dir or os.environ.get("PBC_DATA_DIR", "") or bundled
        if not os.path.isdir(data_dir):
            raise RuntimeError(f"Data directory not found: {data_dir}")

        pbc_path = os.path.join(data_dir, "PBC_List_FY2026.pdf")
        if not os.path.exists(pbc_path):
            for f in os.listdir(data_dir):
                if f.lower().startswith("pbc") and f.endswith(".pdf"):
                    pbc_path = os.path.join(data_dir, f)
                    break

        pbc_items = parse_pbc_list_llm(pbc_path)

        profile_path = os.path.join(data_dir, "Client_Profile.pdf")
        client_profile = load_client_profile(profile_path) if os.path.exists(profile_path) else {}

        emails_dir = os.path.join(data_dir, "sample", "emails")
        mbox_path = os.path.join(data_dir, "sample", "sample_mailbox.mbox")
        if os.path.isdir(emails_dir):
            emails = load_emails_from_directory(emails_dir)
        elif os.path.exists(mbox_path):
            emails = load_emails_from_mbox(mbox_path)
        else:
            raise RuntimeError("No emails found (expected sample/emails/ or sample/sample_mailbox.mbox)")

        set_attachments_dir(os.path.join(data_dir, "sample", "attachments"))

        agent_config = AgentConfig(max_cost_usd=config.max_cost_usd)
        # Allow the platform to cap concurrency (lower memory footprint on small dynos)
        workers_env = os.environ.get("PBC_MAX_WORKERS")
        if workers_env:
            agent_config.max_workers = int(workers_env)

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
        run_status = {
            "state": "complete",
            "offline": current_run.offline,
            "items_processed": len(pbc_items),
            "emails_processed": len(emails),
            "cost_usd": current_run.cost.cost_usd,
            "message": "Offline mode — set ANTHROPIC_API_KEY to classify."
                       if current_run.offline else "Full agent run complete.",
        }
    except Exception as e:
        import traceback
        # Log the full trace server-side; do not return it (it can contain secrets).
        print("RUN ERROR:\n" + traceback.format_exc(), flush=True)
        run_status = {"state": "error", "message": f"{type(e).__name__}: {e}"}


@app.post("/api/run")
async def start_run(config: RunConfig):
    """Kick off an agent run in the background; returns immediately."""
    global run_status
    with _run_lock:
        if run_status.get("state") == "running":
            return {"status": "already_running"}
        run_status = {"state": "running", "message": "Agent run in progress…"}
    threading.Thread(target=_execute_run, args=(config,), daemon=True).start()
    return {"status": "started"}


@app.get("/api/status")
async def get_status():
    """Poll the status of the current/last run."""
    return run_status


@app.get("/api/diag")
async def diagnostics():
    """
    Connectivity diagnostics — used to isolate why API calls fail in a given
    hosting environment. Checks: API key presence, DNS resolution of the Anthropic
    host, a raw HTTPS socket connect, and a minimal live API call. Never returns
    the key itself.
    """
    import socket
    import ssl
    import time

    result = {}

    # 1. Is the key present (and what shape)? Never echo the key itself.
    raw_key = os.environ.get("ANTHROPIC_API_KEY", "")
    key = "".join(raw_key.split())  # sanitized copy used for the diag call below
    result["api_key_present"] = bool(raw_key)
    result["api_key_prefix"] = (key[:7] + "…") if key else None
    result["api_key_had_whitespace"] = raw_key != key  # flags the mangled-paste case

    def _redact(msg: str) -> str:
        return msg.replace(raw_key, "<KEY>").replace(key, "<KEY>") if raw_key else msg

    host = "api.anthropic.com"

    # 2. DNS resolution
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        addrs = sorted({i[4][0] for i in infos})
        result["dns"] = {"ok": True, "addresses": addrs}
    except Exception as e:
        result["dns"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 3. Raw TLS socket connect
    try:
        ctx = ssl.create_default_context()
        t0 = time.time()
        with socket.create_connection((host, 443), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=host):
                pass
        result["tls_connect"] = {"ok": True, "seconds": round(time.time() - t0, 2)}
    except Exception as e:
        result["tls_connect"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 4a. Raw httpx GET to the API host (bypasses the SDK) — surfaces the true error.
    try:
        import httpx
        t0 = time.time()
        r = httpx.get(f"https://{host}/v1/models", timeout=10,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        result["httpx_get"] = {"ok": True, "status": r.status_code,
                               "seconds": round(time.time() - t0, 2)}
    except Exception as e:
        import traceback
        result["httpx_get"] = {"ok": False, "error": _redact(f"{type(e).__name__}: {str(e)[:200]}"),
                               "cause": _redact(str(getattr(e, "__cause__", ""))[:200])}

    # 4b. httpx GET forced to IPv4 transport
    try:
        import httpx
        tr = httpx.HTTPTransport(local_address="0.0.0.0")
        with httpx.Client(transport=tr, timeout=10) as c:
            t0 = time.time()
            r = c.get(f"https://{host}/v1/models",
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        result["httpx_ipv4"] = {"ok": True, "status": r.status_code,
                                "seconds": round(time.time() - t0, 2)}
    except Exception as e:
        result["httpx_ipv4"] = {"ok": False, "error": _redact(f"{type(e).__name__}: {str(e)[:200]}"),
                                "cause": _redact(str(getattr(e, "__cause__", ""))[:200])}

    # 5. Minimal live API call via the SDK
    try:
        from agent.config import make_client
        client = make_client()
        t0 = time.time()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": "ok"}],
        )
        result["api_call"] = {"ok": True, "seconds": round(time.time() - t0, 2),
                              "text": resp.content[0].text if resp.content else ""}
    except Exception as e:
        result["api_call"] = {"ok": False, "error": _redact(f"{type(e).__name__}: {str(e)[:200]}"),
                              "cause": _redact(str(getattr(e, "__cause__", ""))[:200])}

    return result


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
