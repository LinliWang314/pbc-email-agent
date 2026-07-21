import json, time
from agent.loop import run_agent
from agent.config import AgentConfig
from agent.ingest import load_emails_from_directory, load_client_profile
from agent.pbc_parser import parse_pbc_list_llm
from tools.parsers import set_attachments_dir
from eval.run_eval import run_evaluation, print_report
D="stress_data"
pbc=parse_pbc_list_llm(f"{D}/PBC_List_FY2026.pdf")
emails=load_emails_from_directory(f"{D}/sample/emails")
prof=load_client_profile(f"{D}/Client_Profile.pdf")
set_attachments_dir(f"{D}/sample/attachments")
run=run_agent(pbc,emails,AgentConfig(),client_contacts=prof.get("contacts",{}))
res=run.tracker.dict(); res["cost_usd"]=run.cost.cost_usd
rep=run_evaluation(res,f"{D}/sample/sample_groundtruth.json")
print("Overall:", rep["status_classification"]["overall_accuracy"])
print("Insuff P/R:", rep["insufficiency_detection"]["precision"], rep["insufficiency_detection"]["recall"])
for iid in ["PBC-07","PBC-14","PBC-16"]:
    print(f"  {iid}: {run.tracker.items[iid].status}")
