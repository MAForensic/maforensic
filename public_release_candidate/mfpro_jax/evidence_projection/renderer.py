"""Pure frontier rendering: atomic facts, full identity, explicit incomplete views."""
from __future__ import annotations
from dataclasses import asdict
from collections import Counter
import json
import time
from .contracts import (ProjectionSettings, ProjectionResult, ProjectionInputError, SCHEMA_VERSION,
                        BOUNDARY, POLICY_VERSION, SourceScope, compact, encoded_chars)
from .ranking import order_facts, score_fact
from .budget import serialization_metrics


def validate_keep_update(previous, proposed, nodes, limit=8):
    """Already-parsed optional IDs only; no model-response parser or ID correction.

    Matches existing semantics: omission/all-invalid preserve, [] clears, mixed
    list filters invalid entries; duplicate valid entries collapse, cap enforced.
    """
    if (len(previous)>limit or len(set(previous))!=len(previous) or
        any(not isinstance(nid,str) or not nid.startswith('ev:') or nid not in nodes or nodes[nid].get('kind')!='RESULT' for nid in previous)):
        raise ProjectionInputError('INVALID_PREVIOUS_KEEP_STATE')
    if proposed is None:return tuple(previous),()
    if not isinstance(proposed,(list,tuple)):return tuple(previous),('KEEP_NOT_LIST',)
    out=[];reasons=[]
    for item in proposed:
        if not isinstance(item,str) or not item.startswith('ev:') or item not in nodes or nodes[item].get('kind')!='RESULT':reasons.append('UNKNOWN_OR_OUT_OF_SCOPE')
        elif item in out:reasons.append('DUPLICATE')
        elif len(out)>=limit:reasons.append('OVER_LIMIT')
        else:out.append(item)
    return (tuple(previous) if proposed and not out else tuple(out)),tuple(reasons)


def _wire_fact(f):
    o=f.origin
    return [f.fact_id,f.kind,f.object_key,f.value,
            [o.layer,o.kind,list(o.pointers),o.transform,json.loads(o.selector_json)]]


def render_evidence_view(*,frontier_snapshot,sidecars,task_context=None,keep_ids=None,settings=None):
    settings=settings or ProjectionSettings()
    if not settings.enabled:
        return ProjectionResult(None,{'status':'LEGACY_BYPASS','sidecars_built':0},False,None,True)
    if settings.presentation in ('evidence_cards_v2','evidence_cards_v2_1'):
        from .renderer_v2 import render_v2
        return render_v2(frontier_snapshot=frontier_snapshot,sidecars=sidecars,task_context=task_context,keep_ids=keep_ids,settings=settings)
    started=time.process_time_ns();ctx=task_context or {};snap=frontier_snapshot
    metrics={'schema_version':SCHEMA_VERSION,'provider_tokens':'NOT_MEASURED','identities_omitted':0,
             'relations_omitted':0,'bindings_inline':0,'fallback_scope':'projection_only','keep_rejections':[]}
    def fallback(reason,view=None):
        m={**metrics,'status':'EVIDENCE_VIEW_INCOMPLETE','projection_not_delivered':True,
           'identities_omitted':metrics.get('identity_count','UNKNOWN'),
           'relations_omitted':metrics.get('edge_count','UNKNOWN'),
           'render_cpu_ns':time.process_time_ns()-started}
        if view is not None:
            measured=serialization_metrics(view,ctx);measured.pop('task');m.update(measured)
        return ProjectionResult(None,m,True,reason,True)
    try:
        keeps,rejections=validate_keep_update(snap.kept_ids,keep_ids,snap.nodes,settings.max_keep_ids)
        metrics['keep_rejections']=list(rejections)
        # Usage is also A-visible evidence. Validate references even when the
        # future/foreign node is not selected for a card.
        for key in ('last_calls','recent_failure_flags'):
            for item in snap.usage.get(key,[]):
                n=snap.nodes.get(item.get('node_id'))
                if (n is None or n.get('kind')!='RESULT' or type(n.get('iteration')) is not int or
                    n['iteration']>=snap.iteration or type(n.get('created_seq')) is not int or n['created_seq']>snap.observation_seq):
                    return fallback('INVALID_OR_FUTURE_USAGE_REFERENCE')
                if item.get('status',n['status'])!=n['status'] or item.get('tool_name',n['tool_name'])!=n['tool_name']:
                    return fallback('USAGE_NODE_IDENTITY_MISMATCH')
        observed_tools={n.get('tool_name') for n in snap.nodes.values() if n.get('kind')=='RESULT' and
                        type(n.get('iteration')) is int and n['iteration']<snap.iteration}
        if any(name not in observed_tools for name in snap.usage.get('used_tool_names',[])):return fallback('UNOBSERVED_USAGE_TOOL')
        roles={}
        for role,ids in [('NEW',snap.new_ids),('PARENT',snap.parent_ids),('KEPT',keeps)]:
            for nid in ids:
                if nid not in snap.nodes:return fallback('MISSING_FRONTIER_NODE')
                if role not in roles.setdefault(nid,[]):roles[nid].append(role)
        # Do not silently drop any edge participant even on malformed snapshots.
        for e in snap.edges:
            if e.get('from_node_id') not in roles or e.get('to_node_id') not in roles:
                return fallback('EDGE_PARTICIPANT_OUTSIDE_FRONTIER')
            if e.get('to_node_id') not in snap.new_ids:return fallback('EDGE_CHILD_OUTSIDE_NEW_FRONTIER')
        ids=list(roles);index={nid:i for i,nid in enumerate(ids)}
        for field in ('question_uid','run_id','task_id'):
            scopes_seen={snap.nodes[nid].get(field,snap.nodes[nid].get('provenance',{}).get(field)) for nid in ids}
            scopes_seen.discard(None)
            if len(scopes_seen)>1:return fallback('FRONTIER_TASK_SCOPE_MISMATCH')
        metrics['identity_count']=len(ids);metrics['edge_count']=len(snap.edges)
        scopes=[];cards=[];validated={};core_omissions=0
        for nid in ids:
            n=snap.nodes[nid]
            if n.get('node_id')!=nid:return fallback('NODE_MAPPING_IDENTITY_MISMATCH')
            if type(n.get('created_seq',0)) is not int or not 0<=n.get('created_seq',0)<=snap.observation_seq:
                return fallback('FUTURE_OR_INVALID_NODE_SEQUENCE')
            if 'iteration' in n and (type(n['iteration']) is not int or not 0<=n['iteration']<snap.iteration):
                return fallback('FUTURE_OR_INVALID_NODE')
            if n.get('kind') not in ('SOURCE','RESULT'):return fallback('INVALID_NODE_KIND')
            if n.get('kind')=='RESULT':
                if not nid.startswith('ev:'):return fallback('INVALID_RESULT_EVIDENCE_ID')
                if type(n.get('iteration')) is not int or n['iteration']>=snap.iteration:return fallback('FUTURE_OR_INVALID_NODE')
                if n.get('created_seq',0)>snap.observation_seq:return fallback('FUTURE_OBSERVATION_SEQUENCE')
            sc=sidecars.get(nid)
            if sc is not None:
                from .validation import validate_sidecar_binding
                try:validate_sidecar_binding(sc,nid,n,snap,SCHEMA_VERSION)
                except ProjectionInputError as exc:return fallback(str(exc))
                validated[nid]=sc;scope=sc.scope.to_dict()
            else:scope={'source_id':n.get('source_id'),'partition_id':None}
            if scope not in scopes:scopes.append(scope)
            si=scopes.index(scope)
            if n['kind']=='SOURCE':
                cards.append({'source_node_id':nid,'roles':roles[nid],'scope':si,'kind':'SOURCE'});continue
            card={'keep_id':nid,'result_ref':n.get('result_id'),'roles':roles[nid],'scope':si,
                  'tool':n.get('tool_name'),'iteration':n['iteration'],'status':n.get('status'),
                  'execution':[n.get('execution_started'),n.get('execution_completed')],
                  'L1':[],'L2':[],'L3':[]}
            if n.get('error_origin') or n.get('status') not in ('SUCCESS','SUCCESS_EMPTY'):
                card['error_origin']=n.get('error_origin') or 'UNKNOWN'
            if sc:
                card['keep_core']=sc.keep_core
                card['source_hashes']={k:v for k,v in [('RAW',sc.raw_sha256),('FIRST',sc.first_sha256 if sc.coverage['basis']=='FIRST_ORIGIN' else None)] if v}
                cov=sc.coverage
                card['coverage']={'facts_total':cov['extracted_facts'],'sidecar_omitted':cov['sidecar_omitted_facts'],
                                  'facts_inline':0,'facts_omitted':cov['extracted_facts'],
                                  # Keep source counts once, including inside the
                                  # stable core; do not pay twice for identical data.
                                  'counts':{k:v for k,v in cov['counts'].items() if k not in ('candidate_facts_attempted','facts_rejected_by_level') and sc.keep_core.get('counts',{}).get(k,object())!=v},
                                  'extraction_rejected_by_level':list(cov['counts']['facts_rejected_by_level'].values()),'extraction_limits':cov['extraction_limits'],
                                  'units_inline':[0,0,0],'units_total':[cov['units_by_level'][k] for k in ('L1','L2','L3')],'basis':cov['basis']}
                # Catalog counters distinguish a name directory from a complete
                # source-bound object or typed column. Every tuple is
                # [source total, inline, omitted], updated atomically with facts.
                counts=cov['counts'];catalog={}
                if sc.shape in ('file_index','directory'):
                    catalog['objects']=[counts['objects_returned'],0,counts['objects_returned']]
                elif sc.shape=='sqlite_schema':
                    catalog={'schema_objects':[counts['schema_object_count'],0,counts['schema_object_count']],
                             'column_names':[counts['schema_columns'],0,counts['schema_columns']],
                             'typed_columns':[counts['schema_columns'],0,counts['schema_columns']]}
                if catalog:card['coverage']['catalog']=catalog
            else:
                card['keep_core']={'available':False};card['coverage']={'facts_total':'UNKNOWN','facts_inline':0,
                    'facts_omitted':'UNKNOWN','reason':'MISSING_SIDECAR','partition_scope':'UNKNOWN'}
            cards.append(card)
        # Relationship tuples preserve each original edge's parent/relation/joint cardinality.
        calls=[]
        for child in dict.fromkeys(snap.new_ids):
            n=snap.nodes[child];es=[e for e in snap.edges if e['to_node_id']==child]
            reasons=Counter((r.get('argument_field','UNKNOWN'),r.get('reason','UNKNOWN'),r.get('candidate_count','UNKNOWN')) for r in n.get('unresolved_input_refs',[]))
            binding_count=sum(len(e.get('input_bindings',[])) for e in es)
            calls.append({'child':index[child],'tool':n.get('tool_name'),'status':n.get('status'),
                          'parents':[[index[e['from_node_id']],e.get('relation'),len(e.get('input_bindings',[])),e.get('joint_parent_count')] for e in es],
                          'multiple_parents':len({e['from_node_id'] for e in es})>1,
                          'unresolved':[[a,r,c,count] for (a,r,c),count in sorted(reasons.items(),key=lambda x:compact(x[0]))],
                          'unresolved_count':sum(reasons.values()),'bindings_omitted':binding_count,
                          'binding_ref':index[child]})
        metrics['bindings_omitted']=sum(c['bindings_omitted'] for c in calls)
        view={'task_evidence_view':{'presentation':SCHEMA_VERSION,'scopes':scopes,'cards':cards,'calls':calls,
             'boundary':BOUNDARY,'level_order':['L1','L2','L3'],'binding_ref_semantics':'cards[index].keep_id -> full graph bindings','main_consumption':'MAIN_CONSUMPTION_UNVERIFIED','time_units':'UNKNOWN_UNLESS_EXPLICIT_SOURCE_FACT',
             'coverage':{'frontier_nodes':len(ids),'nodes_inline':len(cards),'identities_omitted':0,'relations_omitted':0,
                         'bindings_omitted':metrics['bindings_omitted'],'binding_details':'FULL_GRAPH_ONLY','incomplete_content':True}},
              'minimal_tool_usage':json.loads(compact(snap.usage)),'context_limits':{'max_evidence_view_chars':settings.max_evidence_view_chars,'keep_core_chars':settings.keep_core_chars,'policy_version':POLICY_VERSION}}
        # Every stable core is mandatory. Failure returns legacy, never recuts it.
        size=encoded_chars(view);metrics['mandatory_encoded_chars']=size
        if size>settings.max_evidence_view_chars:return fallback('IDENTITIES_RELATIONS_OR_KEEP_CORE_EXCEED_BUDGET',view)
        initial=size;free=settings.max_evidence_view_chars-size;selected=set();score_audit=[]
        fact_total=sum(sc.coverage['extracted_facts'] for sc in validated.values())
        def fit_add(card,sc,f,cap):
            cov=card['coverage'];li=('L1','L2','L3').index(f.level);old_hashes=dict(card['source_hashes'])
            catalog=cov.get('catalog',{});catalog_key={'indexed_object':'objects','directory_object':'objects',
                'table_directory':'schema_objects','column_directory':'column_names','schema_field':'typed_columns'}.get(f.kind)
            counter=catalog.get(catalog_key)
            if counter is not None:counter[1]+=f.units;counter[2]-=f.units
            card['source_hashes'][f.origin.layer]=f.origin.source_sha256
            card[f.level].append(_wire_fact(f));cov['facts_inline']+=1;cov['facts_omitted']-=1;cov['units_inline'][li]+=f.units
            if encoded_chars(view)>cap:
                card[f.level].pop();cov['facts_inline']-=1;cov['facts_omitted']+=1;cov['units_inline'][li]-=f.units;card['source_hashes']=old_hashes
                if counter is not None:counter[1]-=f.units;counter[2]+=f.units
                return False
            selected.add((sc.evidence_id,f.fact_id))
            score_audit.append({'evidence_id':sc.evidence_id,'fact_id':f.fact_id,**score_fact(f,ctx,settings.ranking_enabled)})
            return True
        def fill(level,cap):
            queues=[]
            for nid,card in zip(ids,cards):
                sc=validated.get(nid)
                if sc is None:continue
                fs=[f for f in sc.facts if f.level==level and (nid,f.fact_id) not in selected]
                fs=order_facts(fs,ctx,settings)
                # Object/table directories precede details at L1; no task-specific field rules.
                if level=='L1':fs.sort(key=lambda f:0 if f.kind in ('table_directory','object_names','xml_field_directory','ordered_columns') else 1)
                queues.append((card,sc,fs))
            for j in range(max([len(q[2]) for q in queues]+[0])):
                for card,sc,fs in queues:
                    if j<len(fs):fit_add(card,sc,fs[j],cap)
        fill('L1',initial+int(free*.50));fill('L2',initial+int(free*.93));fill('L3',settings.max_evidence_view_chars)
        # Reuse otherwise stranded slack with the SAME priority and atomic accounting.
        for level in ('L1','L2','L3'):fill(level,settings.max_evidence_view_chars)
        measured=serialization_metrics(view,ctx);task=measured.pop('task')
        metrics.update(measured);metrics.update({'status':'CARDS_RENDERED','render_cpu_ns':time.process_time_ns()-started,
             'facts_total':fact_total,'facts_inline':len(selected),'facts_omitted':fact_total-len(selected),
             'source_scope_count':len(scopes),'keep_core_complete':True,'selected_scores':score_audit,
             'missing_sidecars':sum(nid not in validated and snap.nodes[nid]['kind']=='RESULT' for nid in ids)})
        return ProjectionResult(view,metrics,False,None,False,task)
    except (ProjectionInputError,ValueError,TypeError,KeyError,IndexError,AttributeError,RecursionError,UnicodeError):
        return fallback('INVALID_FRONTIER_OR_SERIALIZATION')
