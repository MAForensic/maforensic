"""Thin runtime-neutral adapters. No Agent/ToolRegistry/ResultStore imports."""
from __future__ import annotations
from copy import deepcopy
from .contracts import FrontierSnapshot, ProjectionSettings, SourceScope, ProjectionInputError
from .sidecar import project_observed_result
from .renderer import render_evidence_view


def project_from_observation(*,event,node,settings=None,tool_contract=None):
    settings=settings or ProjectionSettings()
    if not settings.enabled:return None, None
    try:
        if event.get('tool_name')=='recall_tool_result':return None,'RECOVERY_EXCLUDED'
        env=event.get('envelope')
        if env is None:return None,'MISSING_ENVELOPE'
        if isinstance(env,dict):metadata=dict(env)
        else:
            from .sidecar import metadata_dict
            metadata=metadata_dict(env)
        for key,node_key in (('result_id','result_id'),('tool_name','tool_name'),('iteration','iteration'),('status','status')):
            if metadata.get(key)!=node.get(node_key):return None,'OBSERVATION_NODE_IDENTITY_MISMATCH'
        for key in ('tool_name','iteration','call_index','status','tool_call_id'):
            if event.get(key) is not None and event[key]!=metadata.get(key):return None,'OBSERVATION_ENVELOPE_MISMATCH'
        for key in ('execution_started','execution_completed','error_origin'):
            metadata[key]=event.get(key)
        ctx=event.get('input_context',{})
        scope=SourceScope(ctx.get('source_id'),str(ctx['partition_id']) if ctx.get('partition_id') is not None else None)
        first=event.get('first_view');text=first.get('text') if isinstance(first,dict) else getattr(first,'text',None)
        return project_observed_result(raw_text=event.get('raw_result'),first_text=text,envelope=metadata,
             tool_contract=tool_contract,source_scope=scope,observation_seq=node['created_seq'],
             evidence_id=node['node_id'],settings=settings),None
    except (ProjectionInputError,ValueError,TypeError,KeyError,AttributeError,RecursionError):
        return None,'OBSERVATION_PROJECTION_REJECTED'


def frontier_from_graph_snapshot(*,nodes,result_order,cursor,edges,keep_ids,iteration,used_tools=None):
    """Snapshot already-observed graph data. Reject future nodes rather than hiding them."""
    mapping=deepcopy(nodes if isinstance(nodes,dict) else {n['node_id']:n for n in nodes})
    if type(iteration) is not int or iteration<1:raise ProjectionInputError('INVALID_SNAPSHOT_ITERATION')
    if type(cursor) is not int or not 0<=cursor<=len(result_order):raise ProjectionInputError('INVALID_GRAPH_CURSOR')
    if len(set(result_order))!=len(result_order):raise ProjectionInputError('DUPLICATE_RESULT_ORDER')
    previous_seq=-1
    for nid in result_order:
        n=mapping.get(nid)
        if n is None or n.get('node_id')!=nid or n.get('kind')!='RESULT':raise ProjectionInputError('INVALID_RESULT_ORDER_NODE')
        seq=n.get('created_seq');it=n.get('iteration')
        if type(seq) is not int or seq<=previous_seq:raise ProjectionInputError('UNORDERED_OBSERVATIONS')
        if type(it) is not int or not 0<=it<iteration:raise ProjectionInputError('FUTURE_OR_INVALID_OBSERVATION')
        previous_seq=seq
    new=tuple(result_order[cursor:]);selected_edges=tuple(deepcopy(e) for e in edges if e['to_node_id'] in new)
    parents=tuple(dict.fromkeys(e['from_node_id'] for e in selected_edges))
    recent=list(result_order[-4:]);used=list(used_tools) if used_tools is not None else list(dict.fromkeys(mapping[n]['tool_name'] for n in result_order))
    usage={'used_tool_names':used,'last_calls':[{'tool_name':mapping[n]['tool_name'],'status':mapping[n]['status'],'node_id':n} for n in recent],
           'recent_failure_flags':[{'node_id':n,'status':mapping[n]['status']} for n in recent if mapping[n]['status']!='SUCCESS']}
    return FrontierSnapshot(mapping,new,parents,tuple(keep_ids),selected_edges,iteration,
                            max([mapping[n].get('created_seq',0) for n in result_order]+[0]),len(result_order),usage)


def build_view_adapter(*,frontier_snapshot=None,sidecars=None,task_context=None,keep_ids=None,
                       settings=None,legacy_builder,record_projection=None):
    """Gate-3 seam: preserve TaskEvidenceGraph.build_view's (view,metrics,cursor).

    Default/failed projections invoke the supplied existing legacy builder once.
    Never constructs sidecars or performs I/O. Core wiring remains external.
    """
    settings=settings or ProjectionSettings()
    if not settings.enabled:return legacy_builder()
    result=render_evidence_view(frontier_snapshot=frontier_snapshot,sidecars=sidecars or {},
              task_context=task_context,keep_ids=keep_ids,settings=settings)
    if record_projection is not None:
        try:record_projection(result)
        except Exception:pass  # observational callback cannot affect the business path
    if result.use_legacy:return legacy_builder()
    return result.view,result.metrics,frontier_snapshot.cursor_end
