"""Public task CLI. Default is offline inventory; Provider access is opt-in."""
from __future__ import annotations
import argparse,asyncio,json,os
from pathlib import Path
import yaml

class PublicConfigError(ValueError):pass

def parser():
    p=argparse.ArgumentParser(description="44-tool typed evidence agent source candidate")
    p.add_argument("--processor-mode",choices=["baseline"],default="baseline")
    sub=p.add_subparsers(dest="command")
    sub.add_parser("check",help="Offline import/registry/typed inventory, no evidence or Provider")
    r=sub.add_parser("run",help="Run one operator-supplied task after explicit Provider opt-in")
    r.add_argument("--allow-provider",action="store_true")
    r.add_argument("--config",type=Path,default=Path("config.example.yaml"))
    r.add_argument("--case",type=Path,required=True)
    r.add_argument("--source-id",required=True)
    r.add_argument("--task-file",type=Path,required=True)
    r.add_argument("--run-dir",type=Path,required=True)
    return p

def inventory():
    from .tool_registry import create_tool_registry
    from .capsules.registry import load_profile0_capsules
    from .capsules.typed_graph import load_typed_graph
    reg=create_tool_registry(None,mode="baseline");caps=load_profile0_capsules();graph=load_typed_graph()
    return {"mode":"OFFLINE_CHECK","tools":len(reg.tools),"capsules":len(caps),"typed_nodes":len(graph.nodes),"confirmed_can_feed":len(graph.edges),"Provider_requests":0}

async def run_task(args):
    if not args.allow_provider:raise PublicConfigError("Provider access is disabled; run requires --allow-provider")
    key=os.environ.get("MODEL_API_KEY","");base=os.environ.get("MODEL_BASE_URL","");model=os.environ.get("MODEL_NAME","")
    if not all([key,base,model]):raise PublicConfigError("Provider is not configured: set MODEL_API_KEY, MODEL_BASE_URL and MODEL_NAME")
    from urllib.parse import urlsplit
    u=urlsplit(base)
    if u.scheme!="https" or not u.hostname or u.username or u.password or u.query or u.fragment:raise PublicConfigError("MODEL_BASE_URL must be a credential-free HTTPS URL")
    if args.run_dir.exists():raise PublicConfigError("Use a new run directory; private historical runs cannot be resumed")
    from .case import load_case
    from .runtime import RuntimeContext
    from .trace import TraceRecorder
    from .tool_registry import create_tool_registry
    from .llm_client import LLMClient
    from .primitive_agent import PrimitiveEvidenceAgent
    from .capsules.context_v2 import build_capsule_retrieval_v2_provider
    config=yaml.safe_load(args.config.read_text()) or {}
    if config.get("processor_mode","baseline")!="baseline":raise PublicConfigError("Only baseline output mode is distributed")
    # API credentials remain in the SDK client, never in serialized config.
    config.setdefault("agent",{}).update(model=model,vision_model=os.environ.get("MODEL_VISION_MODEL") or model)
    config["processor_mode"]="baseline"
    case=load_case(args.case);source=case.get_source(args.source_id)
    if source is None:raise PublicConfigError("Unknown source ID in the operator-supplied case")
    task=args.task_file.read_text(encoding="utf-8")
    trace=TraceRecorder(args.run_dir,run_id=args.run_dir.name)
    ctx=RuntimeContext(case,args.run_dir,trace,config);ctx.set_active_source(source)
    registry=create_tool_registry(ctx,mode="baseline")
    agent_config=config.get("agent",{})
    client=LLMClient(api_key=key,base_url=base,model=model,max_tokens=agent_config.get("max_tokens",8192),proxy=None)
    ctx.vision_client=client.client
    try:
        provider=build_capsule_retrieval_v2_provider(config,registry,client)
        agent=PrimitiveEvidenceAgent(client,ctx,registry,max_iterations=agent_config.get("max_iterations",40),capsule_context_provider=provider)
        answer,_=await agent.run(task,"operator_task")
        print(answer)
    finally:await client.close()

def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command in [None,"check"]:print(json.dumps(inventory()));return 0
        asyncio.run(run_task(args));return 0
    except Exception as exc:
        # Configuration errors are safe; SDK/data exception details remain local.
        known=isinstance(exc,PublicConfigError)
        print(str(exc) if known else "Operation failed ("+type(exc).__name__+"); no remote response or credential details printed",file=__import__("sys").stderr)
        return 2

if __name__=="__main__":raise SystemExit(main())
