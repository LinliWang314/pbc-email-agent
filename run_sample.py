"""
End-to-end runner: run the agent on the sample data, save results, run eval.

Usage:
    ANTHROPIC_API_KEY=... python run_sample.py
"""

import json
import os
import sys
import time
from pathlib import Path

from agent.loop import run_agent
from agent.config import AgentConfig
from agent.ingest import load_emails_from_directory, load_client_profile
from agent.pbc_parser import parse_pbc_list_llm
from tools.parsers import set_attachments_dir
from eval.run_eval import run_evaluation, print_report

DATA = Path(__file__).parent / "sample_data"


def main():
    t0 = time.time()

    print("Loading PBC list (free-form LLM parse)...")
    pbc_items = parse_pbc_list_llm(str(DATA / "PBC_List_FY2026.pdf"))
    print(f"  {len(pbc_items)} items")

    print("Loading emails + profile...")
    emails = load_emails_from_directory(str(DATA / "sample" / "emails"))
    profile = load_client_profile(str(DATA / "Client_Profile.pdf"))
    set_attachments_dir(str(DATA / "sample" / "attachments"))
    print(f"  {len(emails)} emails")

    print("Running agent (plan -> act -> verify -> draft)...")
    run = run_agent(
        pbc_items, emails, AgentConfig(),
        client_contacts=profile.get("contacts", {}),
        engagement_info={
            "entity": profile.get("entity_name", ""),
            "fiscal_year_end": profile.get("fiscal_year_end", ""),
        },
    )

    elapsed = time.time() - t0

    # Save run results
    result = run.tracker.dict()
    result["cost_usd"] = run.cost.cost_usd
    result["elapsed_s"] = round(elapsed, 1)
    result["input_tokens"] = run.cost.input_tokens
    result["output_tokens"] = run.cost.output_tokens

    out_dir = Path(__file__).parent / "runs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "sample_run.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "sample_traces.json", "w") as f:
        json.dump([t.dict() for t in run.traces], f, indent=2)

    print(f"\nRun complete in {elapsed:.1f}s")
    print(f"Cost: ${run.cost.cost_usd:.4f} ({run.cost.input_tokens} in / {run.cost.output_tokens} out tokens)")

    # Print status summary
    print("\nStatus summary:")
    from collections import Counter
    statuses = Counter(item.status for item in run.tracker.items.values())
    for status, count in statuses.items():
        print(f"  {status}: {count}")

    # Run eval
    gt_path = str(DATA / "sample" / "sample_groundtruth.json")
    report = run_evaluation(result, gt_path)
    print_report(report, cost_usd=run.cost.cost_usd)


if __name__ == "__main__":
    main()
