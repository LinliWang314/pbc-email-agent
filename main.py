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
_run_thread: threading.Thread | None = None   # the worker; used to detect a dead run
_RUN_TIMEOUT_S = 15 * 60                        # a run older than this is treated as stale


def _run_is_stuck() -> bool:
    """
    True if run_status says 'running' but the run is actually dead/stale — the worker
    thread has exited (crash/deploy restart) or it has run past the timeout. Without this,
    a single interrupted run would deadlock the app in 'running' forever.
    """
    if run_status.get("state") != "running":
        return False
    started = run_status.get("_started_at")
    if _run_thread is not None and not _run_thread.is_alive():
        return True
    import time as _t
    if started and (_t.time() - started) > _RUN_TIMEOUT_S:
        return True
    return False


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

        def _progress(done, total, phase="processing emails"):
            # Update the shared status so /api/status can report live progress.
            import time as _t
            run_status.update({
                "message": f"{phase.capitalize()}… {done}/{total}",
                "progress": {"done": done, "total": total, "phase": phase},
                "_started_at": run_status.get("_started_at", _t.time()),
            })

        current_run = run_agent(
            pbc_items=pbc_items,
            emails=emails,
            config=agent_config,
            client_contacts=client_profile.get("contacts", {}),
            engagement_info={
                "entity": client_profile.get("entity_name", ""),
                "fiscal_year_end": client_profile.get("fiscal_year_end", ""),
            },
            progress_cb=_progress,
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
    global run_status, _run_thread
    import time as _t
    with _run_lock:
        if run_status.get("state") == "running" and not _run_is_stuck():
            return {"status": "already_running"}
        run_status = {"state": "running", "message": "Agent run in progress…",
                      "_started_at": _t.time()}
    _run_thread = threading.Thread(target=_execute_run, args=(config,), daemon=True)
    _run_thread.start()
    return {"status": "started"}


@app.get("/api/status")
async def get_status():
    """Poll the status of the current/last run. Reports a dead/stale run as error so
    the client stops waiting forever and can retry."""
    if _run_is_stuck():
        return {"state": "error",
                "message": "Previous run did not finish (worker exited or timed out). "
                           "Click Run again to start a fresh one."}
    return run_status


@app.get("/api/version")
async def version():
    """Report the running code version so we can confirm what's deployed."""
    import subprocess
    sha = "unknown"
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or "unknown"
    except Exception:
        pass
    # Feature markers — presence proves the newer code is running.
    return {
        "git_sha": sha,
        "has_numeric_citation_matching": True,
        "has_blend_confidence": True,
        "marker": "confidence-fixes-v2",
    }


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
    c = current_run.cost
    return {
        "input_tokens": c.input_tokens,
        "output_tokens": c.output_tokens,
        "cache_write_tokens": c.cache_write_tokens,
        "cache_read_tokens": c.cache_read_tokens,
        "total_cost_usd": c.cost_usd,
        "budget_remaining_usd": current_run.config.max_cost_usd - c.cost_usd,
    }


@app.post("/api/upload")
async def upload_data(bundle: UploadFile = File(...)):
    """
    Upload a single ZIP bundle (e.g. the held-out mailbox) and start a run on it.

    The archive is unpacked and normalized into the expected layout regardless of its
    internal folder structure: we locate the PBC list PDF, the .eml emails, and the
    attachments wherever they sit in the tree. This tolerates the held-out data arriving
    in whatever shape it's zipped in.
    """
    import zipfile

    if not bundle.filename.endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip bundle (PBC PDF + emails + attachments).")

    work = tempfile.mkdtemp(prefix="pbc_bundle_")
    raw = os.path.join(work, "raw")
    os.makedirs(raw, exist_ok=True)
    zip_path = os.path.join(work, bundle.filename)
    with open(zip_path, "wb") as f:
        f.write(await bundle.read())
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw)
    except zipfile.BadZipFile:
        raise HTTPException(400, "Uploaded file is not a valid ZIP.")

    # Normalize into <data>/PBC_List.pdf + <data>/sample/emails + <data>/sample/attachments
    data_dir = os.path.join(work, "data")
    emails_dir = os.path.join(data_dir, "sample", "emails")
    attach_dir = os.path.join(data_dir, "sample", "attachments")
    os.makedirs(emails_dir, exist_ok=True)
    os.makedirs(attach_dir, exist_ok=True)

    pbc_pdf = None
    profile_pdf = None
    pdf_candidates = []   # fallback pool if no obviously-named PBC pdf is found
    mbox_files = []
    for root, _dirs, files in os.walk(raw):
        for fn in files:
            src = os.path.join(root, fn)
            low = fn.lower()
            if low.endswith(".eml"):
                shutil.copy(src, os.path.join(emails_dir, fn))
            elif low.endswith(".mbox") or low == "mbox":
                mbox_files.append(src)
            elif "profile" in low and low.endswith(".pdf"):
                profile_pdf = src
            elif low.endswith(".pdf") and (low.startswith("pbc") or "pbc" in low
                                           or "request" in low or "list" in low):
                pbc_pdf = pbc_pdf or src   # obviously-named PBC list
            elif low.endswith((".pdf", ".xlsx", ".xls", ".jpg", ".jpeg", ".png", ".zip", ".docx", ".csv")):
                if low.endswith(".pdf"):
                    pdf_candidates.append(src)
                shutil.copy(src, os.path.join(attach_dir, fn))

    # If a .mbox was provided instead of loose .eml files, expand it to .eml.
    if mbox_files and len(os.listdir(emails_dir)) == 0:
        import mailbox as _mb
        for mbf in mbox_files:
            for i, m in enumerate(_mb.mbox(mbf)):
                with open(os.path.join(emails_dir, f"mbox_{os.path.basename(mbf)}_{i:04d}.eml"), "wb") as out:
                    out.write(m.as_bytes())

    # Fallback: if no obviously-named PBC pdf, use the single remaining PDF candidate
    # (or the largest) — held-out lists may be named differently (e.g. RequestList.pdf).
    if not pbc_pdf and pdf_candidates:
        pbc_pdf = max(pdf_candidates, key=lambda p: os.path.getsize(p))
        # It was also copied into attachments; that's harmless.

    if pbc_pdf:
        shutil.copy(pbc_pdf, os.path.join(data_dir, "PBC_List_FY2026.pdf"))
    else:
        raise HTTPException(400, "No PBC list PDF found in the bundle (include the request-list PDF).")
    if profile_pdf:
        shutil.copy(profile_pdf, os.path.join(data_dir, "Client_Profile.pdf"))

    n_emails = len(os.listdir(emails_dir))
    if n_emails == 0:
        raise HTTPException(400, "No .eml emails found in the bundle.")

    # Kick off a run against the uploaded data (same background-job path as /api/run).
    global run_status, _run_thread
    import time as _t
    with _run_lock:
        if run_status.get("state") == "running" and not _run_is_stuck():
            return {"status": "already_running"}
        run_status = {"state": "running", "message": f"Processing uploaded bundle ({n_emails} emails)…",
                      "_started_at": _t.time()}
    _run_thread = threading.Thread(target=_execute_run, args=(RunConfig(data_dir=data_dir),), daemon=True)
    _run_thread.start()
    return {"status": "started", "emails": n_emails,
            "attachments": len(os.listdir(attach_dir))}


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
