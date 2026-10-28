"""Synthetic-only, credential-free smoke tests. Never calls a forensic executor."""
from __future__ import annotations
import ast,asyncio,importlib,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
network_denials=[]
def network_guard(event,args):
    if event=='socket.__new__' and int(args[1])==1:return  # asyncio local socketpair only
    if event in ['socket.__new__','socket.connect','socket.connect_ex','socket.getaddrinfo','socket.sendto']:
        network_denials.append(event)
        raise PermissionError('NETWORK_DISABLED_FOR_OFFLINE_SMOKE')
sys.addaudithook(network_guard)
import yaml
from mfpro_jax.tool_registry import create_tool_registry
from mfpro_jax.capsules.registry import load_profile0_capsules
from mfpro_jax.capsules.typed_graph import load_typed_graph
from mfpro_jax.capsules.retrieval_graph import SeedToolSelection
from mfpro_jax.capsules.context_v2 import build_capsule_retrieval_v2_provider
from mfpro_jax.llm_client import LLMClient
from mfpro_jax.primitive_agent import PrimitiveEvidenceAgent
from mfpro_jax.case import Case,EvidenceSource
from mfpro_jax.runtime import RuntimeContext
from mfpro_jax.trace import TraceRecorder
from mfpro_jax.result_store import ResultStore
from mfpro_jax.representation_runtime import RepresentationRuntime
from mfpro_jax.context_lifecycle import RepresentationViewStore,SharedContextLifecycle
from mfpro_jax.task_evidence_graph import TaskEvidenceGraph
from mfpro_jax.main import inventory,parser
cases=[]
def passed(name,**evidence):cases.append({'case':name,'status':'PASS',**evidence})
# Every shipped module compiles and imports from this tree, never another checkout.
for p in sorted((ROOT/'mfpro_jax').rglob('*.py')):
    compile(p.read_bytes(),p.name,'exec')
    name='.'.join(p.relative_to(ROOT).with_suffix('').parts)
    if name.endswith('.__init__'):name=name[:-9]
    module=importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(ROOT)
passed('compile_and_independent_imports')
reg=create_tool_registry(None,mode='baseline');caps=load_profile0_capsules();graph=load_typed_graph()
assert len(reg.tools)==len(caps)==len(graph.nodes)==44 and len(graph.edges)==74
assert set(reg.tools)==set(caps.by_tool_name)==set(graph.by_tool_name)
assert {'read_windows_registry_value','parse_e01_header','analyze_image'}<=set(reg.tools)
assert all(callable(t.executor) and t.input_schema.get('type')=='object' for t in reg.tools.values())
assert not reg.output_profile_error
assert len({e.edge_id for e in graph.edges})==74 and all(e.metadata['status']=='CONFIRMED' for e in graph.edges)
for name,t in reg.tools.items():assert graph.ports[name]['input_schema']==t.input_schema
passed('genuine_registry_capsules_and_typed_ports',tools=44,nodes=44,edges=74)
for name in reg.tools:
    f={e.edge_id for e in graph.edges if e.producer==name};b={e.edge_id for e in graph.edges if e.consumer==name}
    assert {e.edge_id for e in graph.query(name,'forward')}==f
    assert {e.edge_id for e in graph.query(name,'backward')}==b
    assert {e.edge_id for e in graph.query(name,'both')}==f|b
passed('all_tools_forward_backward_both',direction_queries=132)
try:create_tool_registry(None,mode='selected')
except ValueError:pass
else:raise AssertionError('UNSUPPORTED_OUTPUT_MODE_ACCEPTED')
assert parser().parse_args(['check']).command=='check'
assert inventory()['tools']==44
passed('baseline_only_cli_and_inventory')
try:LLMClient(api_key='',base_url='',model='',proxy=None)
except ValueError as e:assert str(e).startswith('Provider is not configured')
else:raise AssertionError('UNCONFIGURED_PROVIDER_ACCEPTED')
passed('missing_credentials_fail_before_SDK_client')

class FakeClient(LLMClient):
    def __init__(self):
        super().__init__(api_key='PLACEHOLDER',base_url='https://offline.invalid/v1',model='offline-fake',proxy=None)
        self.main_calls=0;self.seed_calls=0
    async def _call(self,messages,tools,**kwargs):
        if not tools:
            self.seed_calls+=1
            return json.dumps({'selected_tools':[{'tool_name':'read_text','query_type':'forward','reason':'synthetic navigation'}]}),None,[],None,{}
        self.main_calls+=1
        if self.main_calls==1:
            return '',None,[{'id':'synthetic_call','name':'read_text','arguments':json.dumps({'file_path':'synthetic.txt'})}],None,{}
        return 'ANSWER: synthetic-only',None,[],None,{}

async def exercise(root):
    config=yaml.safe_load((ROOT/'config.example.yaml').read_text())
    source=EvidenceSource('src-synthetic','Synthetic empty tree','directory',str(root),'tree')
    case=Case('synthetic-case','Synthetic',[source])
    trace=TraceRecorder(root/'agent',run_id='synthetic-run')
    ctx=RuntimeContext(case,root/'agent',trace,config);ctx.set_active_source(source)
    registry=create_tool_registry(ctx,mode='baseline')
    executions=[]
    async def synthetic_executor(**kwargs):
        executions.append('synthetic');return 'SYNTHETIC_ONLY_CONTENT'
    # Intentional injection replaces only the test-bound executor; source tools remain intact.
    registry.tools['read_text'].executor=synthetic_executor
    client=FakeClient()
    try:
        provider=build_capsule_retrieval_v2_provider(config,registry,client)
        assert provider is not None
        graph=TaskEvidenceGraph.create(ctx=ctx,task_id='synthetic-projection',question_id='synthetic-task')
        assert graph is not None and graph.settings.mode=='active' and graph.projection is not None
        assert graph.projection.settings.presentation=='evidence_cards_v2_1';graph.seal('synthetic-check')
        answer,history=await PrimitiveEvidenceAgent(client,ctx,registry,max_iterations=3,capsule_context_provider=provider).run('Use only the synthetic observation','synthetic-task')
        assert answer=='ANSWER: synthetic-only' and executions==['synthetic']
        assert client.main_calls==2 and client.seed_calls>=1
        assert any(e.event_type=='iteration_retrieval_completed' for e in trace.events)
        passed('real_Agent_A_B_loop_with_injected_synthetic_executor',fake_main_calls=client.main_calls,fake_seed_calls=client.seed_calls,synthetic_executions=len(executions),real_forensic_executions=0)
    finally:await client.close()
    store=ResultStore(root/'foundation',run_id='synthetic-foundation')
    env=store.store_raw_result(question_id='synthetic-task',iteration=1,call_index=0,tool_call_id='synthetic-call',tool_name='read_text',arguments={},status='success',raw_result='synthetic '*3000)
    runtime=RepresentationRuntime(store)
    life=SharedContextLifecycle(runtime=runtime,view_store=RepresentationViewStore(root/'views'))
    first,persistent=life.materialize_first_persistent(env)
    assert store.get_raw_result(env.result_id)=='synthetic '*3000
    assert first.text and persistent.text and not first.processor_applied and not persistent.processor_applied
    recall=life.request_recovery(result_id=env.result_id,tool_call_id='synthetic-recall')
    assert recall.status.value=='REQUESTED'
    passed('baseline_raw_storage_bounded_views_and_recovery')
    callbacks=[]
    def forbidden_processor(*args,**kwargs):
        callbacks.append(True);raise AssertionError('PROCESSOR_MUST_NOT_EXECUTE')
    for operation in [lambda:runtime.generate_canonical_first(env.result_id,processor=forbidden_processor),
                      lambda:runtime.render_persistent_from_canonical(runtime.generate_identity_canonical_first(env.result_id),processor=forbidden_processor)]:
        try:operation()
        except ValueError as e:assert str(e)=='External output processors are not distributed'
        else:raise AssertionError('EXTERNAL_PROCESSOR_ACCEPTED')
    assert callbacks==[]
    passed('external_processor_callbacks_rejected_before_execution')
with tempfile.TemporaryDirectory(prefix='public-agent-smoke-') as tmp:asyncio.run(exercise(Path(tmp)))
# An exception that mimics a known message prefix must not bypass CLI redaction.
import contextlib,io
from unittest.mock import patch
from mfpro_jax import main as cli
async def injected_failure(args):raise RuntimeError('Provider is not configured: SYNTHETIC_PRIVATE_ERROR_BODY')
err=io.StringIO()
with patch.object(cli,'run_task',injected_failure),contextlib.redirect_stderr(err):
    assert cli.main(['run','--case','not-read.yaml','--source-id','src-synthetic','--task-file','not-read.txt','--run-dir','not-created'])==2
assert 'SYNTHETIC_PRIVATE_ERROR_BODY' not in err.getvalue() and 'RuntimeError' in err.getvalue()
passed('CLI_untrusted_exception_body_not_printed')
assert not network_denials
print(json.dumps({'status':'PASS','cases':cases,'Provider_requests':0,'external_network_requests':0,'real_forensic_executions':0,'real_question_runs':0,'independent_public_imports':True,'network_denials':len(network_denials)}))
