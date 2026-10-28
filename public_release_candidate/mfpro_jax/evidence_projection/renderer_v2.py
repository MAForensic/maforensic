"""V2: lossless identity/edge directory, budgetable summaries, fair navigation.

Only presentation data is created. Source files, graph state, keep and cursor are
never modified. All fact values and locators are unchanged observed TypedFacts.
"""
from collections import Counter
import json,time
from .contracts import ProjectionInputError,ProjectionResult,versions_for,BOUNDARY,compact,digest,encoded_chars
from .renderer import validate_keep_update,_wire_fact
from .validation import validate_sidecar_binding
from .budget import serialization_metrics
from .ranking_v2 import order,score,score_v21


def render_v2(*,frontier_snapshot,sidecars,task_context=None,keep_ids=None,settings,selection_trace=None):
    versions=versions_for(settings.presentation);revised=settings.presentation=='evidence_cards_v2_1'
    scoring=score_v21 if revised else score
    t=time.process_time_ns();snap=frontier_snapshot;ctx=task_context or {};metrics={'schema_version':versions[0],'provider_tokens':'NOT_MEASURED'}
    def fail(reason,view=None):
        metrics.update(status='EVIDENCE_VIEW_INCOMPLETE',fallback_reason=reason,projection_not_delivered=True,
            render_cpu_ns=time.process_time_ns()-t,identities_omitted=metrics.get('identity_count','UNKNOWN'),relations_omitted=metrics.get('edge_count','UNKNOWN'))
        if view is not None:metrics['minimal_identity_relation_chars']=encoded_chars(view)
        return ProjectionResult(None,metrics,True,reason,True)
    try:
        keeps,rejections=validate_keep_update(snap.kept_ids,keep_ids,snap.nodes,settings.max_keep_ids)
        roles={}
        for role,ids in [('NEW',snap.new_ids),('PARENT',snap.parent_ids),('KEPT',keeps)]:
            for nid in ids:
                if nid not in snap.nodes:raise ProjectionInputError('MISSING_FRONTIER_NODE')
                if role not in roles.setdefault(nid,[]):roles[nid].append(role)
        ids=list(roles);idx={nid:i for i,nid in enumerate(ids)};validated={};scopes=[];tools=[]
        def tool_index(name):
            if name not in tools:tools.append(name)
            return tools.index(name)
        for field in ('question_uid','run_id','task_id'):
            values={snap.nodes[n].get(field,snap.nodes[n].get('provenance',{}).get(field)) for n in ids};values.discard(None)
            if len(values)>1:raise ProjectionInputError('FRONTIER_TASK_SCOPE_MISMATCH')
        if type(snap.iteration) is not int or type(snap.observation_seq) is not int or type(snap.cursor_end) is not int or snap.cursor_end<0:
            raise ProjectionInputError('INVALID_SNAPSHOT_SEQUENCE')
        identities=[];cards=[];cardbyid={};cores={}
        for nid in ids:
            n=snap.nodes[nid];kind=n.get('kind');seq=n.get('created_seq',0)
            if n.get('node_id')!=nid or kind not in ('SOURCE','RESULT'):raise ProjectionInputError('NODE_MAPPING_IDENTITY_MISMATCH')
            if type(seq) is not int or not 0<=seq<=snap.observation_seq:raise ProjectionInputError('FUTURE_OR_INVALID_NODE_SEQUENCE')
            if 'iteration' in n and (type(n['iteration']) is not int or not 0<=n['iteration']<snap.iteration):raise ProjectionInputError('FUTURE_OR_INVALID_NODE')
            if kind=='RESULT' and (not nid.startswith('ev:') or type(n.get('iteration')) is not int or not n.get('result_id')):
                raise ProjectionInputError('INVALID_RESULT_IDENTITY')
            sc=sidecars.get(nid)
            if kind=='RESULT':
                if sc is None:raise ProjectionInputError('MISSING_SIDECAR')
                validate_sidecar_binding(sc,nid,n,snap,versions[0]);validated[nid]=sc;scope=sc.scope.to_dict()
            else:scope={'source_id':n.get('source_id'),'partition_id':n.get('partition_id')}
            if scope not in scopes:scopes.append(scope)
            # Role and dictionary index are explicitly outside the immutable core.
            core=[nid,n.get('result_id'),kind,scope,n.get('tool_name'),n.get('status'),n.get('iteration'),seq,
                  [n.get('execution_started'),n.get('execution_completed')],n.get('error_origin') or None]
            cores[nid]=digest(compact(core))
            identities.append([nid,n.get('result_id'),kind,roles[nid],scopes.index(scope),
                tool_index(n.get('tool_name')) if kind=='RESULT' else None,n.get('status'),n.get('iteration'),seq,core[-2],core[-1]])
            if sc:
                cov=sc.coverage
                c={'node':idx[nid],'shape':sc.shape,'coverage':[cov['extracted_facts'],cov['sidecar_omitted_facts'],0,cov['extracted_facts']],
                   'basis':cov['basis'],'L1':[],'L2':[],'L3':[]}
                cards.append(c);cardbyid[nid]=c
        # All graph edge instances survive, including parallel edges and multi-parent
        # calls. Actual input binding values remain FULL_GRAPH_ONLY with exact counts.
        edges=[];unresolved=[]
        for e in snap.edges:
            a=e.get('from_node_id');b=e.get('to_node_id')
            if a not in idx or b not in idx or b not in snap.new_ids:raise ProjectionInputError('EDGE_PARTICIPANT_OUTSIDE_FRONTIER')
            edges.append([e.get('edge_id',e.get('id')),idx[a],idx[b],e.get('relation'),len(e.get('input_bindings',[])),e.get('joint_parent_count')])
        for nid in snap.new_ids:
            counts=Counter((x.get('argument_field','UNKNOWN'),x.get('reason','UNKNOWN'),x.get('candidate_count','UNKNOWN')) for x in snap.nodes[nid].get('unresolved_input_refs',[]))
            if counts:unresolved.append([idx[nid],[[*k,v] for k,v in sorted(counts.items(),key=lambda x:compact(x[0]))]])
        recent=[]
        for item in snap.usage.get('last_calls',[]):
            n=snap.nodes.get(item.get('node_id'))
            if (not n or n.get('kind')!='RESULT' or type(n.get('iteration')) is not int or n['iteration']>=snap.iteration
                or type(n.get('created_seq')) is not int or n['created_seq']>snap.observation_seq):raise ProjectionInputError('INVALID_OR_FUTURE_USAGE_REFERENCE')
            if item.get('tool_name',n['tool_name'])!=n['tool_name'] or item.get('status',n['status'])!=n['status']:raise ProjectionInputError('USAGE_NODE_IDENTITY_MISMATCH')
            recent.append([idx.get(n['node_id'],n['node_id']),n['tool_name'],n['status']])
        # Failure flags not in last_calls are still validated rather than silently dropped.
        for item in snap.usage.get('recent_failure_flags',[]):
            n=snap.nodes.get(item.get('node_id'))
            if not n or n.get('kind')!='RESULT' or type(n.get('iteration')) is not int or n['iteration']>=snap.iteration or type(n.get('created_seq')) is not int or n['created_seq']>snap.observation_seq or item.get('status',n['status'])!=n['status']:
                raise ProjectionInputError('INVALID_OR_FUTURE_USAGE_REFERENCE')
        observed={n.get('tool_name') for n in snap.nodes.values() if n.get('kind')=='RESULT' and type(n.get('iteration')) is int and n['iteration']<snap.iteration}
        used=snap.usage.get('used_tool_names',[])
        if any(x not in observed for x in used):raise ProjectionInputError('UNOBSERVED_USAGE_TOOL')
        usage={'used':list(used),'recent':recent,
            'failures':[[idx.get(x['node_id'],x['node_id']),x['status']] for x in snap.usage.get('recent_failure_flags',[]) if x['status'] not in ('SUCCESS','SUCCESS_EMPTY')]}
        binding_count=sum(e[4] for e in edges)
        view={'task_evidence_view':{'presentation':versions[0],'boundary':BOUNDARY,'scopes':scopes,'tools':tools,
            'identity_fields':['node_id','result_ref','kind','roles','scope','tool','status','iteration','seq','execution','error_origin'],
            'identities':identities,'edge_fields':['edge_id','parent','child','relation','binding_count','joint_parent_count'],
            'edges':edges,'unresolved_ref_count':sum(x[-1] for _,items in unresolved for x in items),'cards':cards,
            'fact_fields':['fact_id','kind','object_key','value','origin(layer,kind,pointers,transform,selector)'],
            'coverage_fields':['candidate_facts','sidecar_omitted','inline','total_omitted_including_sidecar'],
            'catalog_fields':['observed_source_total','inline','not_inline_including_sidecar_or_extraction_losses'],
            'visibility':'RAW/FIRST_ORIGIN; B_VISIBLE_UNVERIFIED; time units UNKNOWN unless source explicit',
            'binding_details':{'inline':False,'omitted':binding_count,'authority':'FULL_GRAPH_ONLY; identities[index].node_id'},
            'auxiliary_not_inline':['tool_usage','unresolved_details'],
            'missing_summary':'NOT_INLINE; keep identity retained; fields are budgeted, not permanently pinned',
            'missing_file_id':'UNAVAILABLE when absent from the same indexed/directory_object value; path is not a host handle'},
            'context_limits':{'max_evidence_view_chars':settings.max_evidence_view_chars,'policy_version':versions[2]}}
        size=encoded_chars(view);metrics.update(identity_count=len(ids),edge_count=len(edges),bindings_omitted=binding_count,
            identities_omitted=0,relations_omitted=0,mandatory_encoded_chars=size,minimal_identity_relation_chars=size,keep_rejections=list(rejections))
        if size>settings.max_evidence_view_chars:return fail('MINIMAL_IDENTITIES_RELATIONS_EXCEED_BUDGET',view)
        initial=size;free=settings.max_evidence_view_chars-size;selected=set();decisions=[];tiers={nid:'IDENTITY_ONLY' for nid in validated}
        current=size;allocation=Counter()
        def note(nid,f,cap,tier,reason,trial=None):
            if selection_trace is not None:
                selection_trace({'node_id':nid,'fact_id':f.fact_id,'kind':f.kind,'tier':tier,'reason':reason,'cap':cap,'current':current,'wire_cost':encoded_chars(_wire_fact(f)),'trial':trial})
        def add(nid,f,cap,tier):
            nonlocal current
            if (nid,f.fact_id) in selected:
                note(nid,f,cap,tier,'ALREADY_SELECTED');return False
            if f.kind=='sql_row' and not any((nid,x.fact_id) in selected and x.kind=='ordered_columns' for x in validated[nid].facts):
                note(nid,f,cap,tier,'MISSING_ORDERED_COLUMNS');return False
            wire=_wire_fact(f)
            if encoded_chars(wire)>cap-current:
                note(nid,f,cap,tier,'TIER_REMAINING_BELOW_ATOMIC_WIRE_COST');return False
            card=cardbyid[nid];oldhash=dict(card.get('source_hashes',{}))
            card.setdefault('source_hashes',{})[f.origin.layer]=f.origin.source_sha256
            card[f.level].append(wire);card['coverage'][2]+=1;card['coverage'][3]-=1
            ck={'indexed_object':'objects','directory_object':'objects','table_directory':'table_directory','column_directory':'column_names','schema_field':'typed_columns'}.get(f.kind)
            counter=card.get('catalog',{}).get(ck)
            if f.kind=='schema_field' and not (len(f.value['column'])>1 and isinstance(f.value['column'][1],str)):counter=None
            if counter is not None:counter[1]+=f.units;counter[2]-=f.units
            trial=encoded_chars(view)
            if trial>cap:
                note(nid,f,cap,tier,'EXACT_HASH_COUNTER_OVERHEAD_EXCEEDS_TIER',trial)
                card[f.level].pop();card['coverage'][2]-=1;card['coverage'][3]+=1
                if counter is not None:counter[1]-=f.units;counter[2]+=f.units
                if oldhash:card['source_hashes']=oldhash
                else:card.pop('source_hashes',None)
                return False
            note(nid,f,cap,tier,'SELECTED',trial);allocation[tier]+=trial-current
            selected.add((nid,f.fact_id));current=trial;tiers[nid]='DETAIL'
            decisions.append({'node_id':nid,'fact_id':f.fact_id,'tier':tier,**scoring(f,ctx,settings.ranking_enabled)})
            return True
        def summary(nid,cap):
            nonlocal current
            card=cardbyid[nid];sc=validated[nid]
            if 'summary' in card:return
            # Stable across roles/task/ranking. Optional as a whole, never recut.
            counts=sc.coverage['counts'];card['summary']={k:counts[k] for k in ('objects_returned','objects_valid_shape','schema_object_count','schema_columns','rows_returned','column_count','source_truncated','text_chars') if k in counts}
            card['extraction_limits']=sc.coverage['extraction_limits']
            catalog={}
            if sc.shape in ('file_index','directory'):catalog={'objects':[counts['objects_returned'],0,counts['objects_returned']]}
            if sc.shape=='sqlite_schema':catalog={'table_directory':[counts['schema_object_count'],0,counts['schema_object_count']], 'column_names':[counts['schema_columns'],0,counts['schema_columns']], 'typed_columns':[counts['schema_typed_columns'],0,counts['schema_typed_columns']]}
            for lev in ('L1','L2'):
                for ff in card[lev]:
                    key={'indexed_object':'objects','directory_object':'objects','table_directory':'table_directory','column_directory':'column_names','schema_field':'typed_columns'}.get(ff[1])
                    f=next(x for x in sc.facts if x.fact_id==ff[0])
                    if f.kind=='schema_field' and not (len(f.value['column'])>1 and isinstance(f.value['column'][1],str)):continue
                    if key in catalog:catalog[key][1]+=f.units;catalog[key][2]-=f.units
            if catalog:card['catalog']=catalog
            trial=encoded_chars(view)
            if trial>cap:card.pop('summary');card.pop('extraction_limits');card.pop('catalog',None)
            else:
                allocation['SUMMARY']+=trial-current
                current=trial
                if tiers[nid]=='IDENTITY_ONLY':tiers[nid]='SUMMARY'
        new=[n for n in ids if 'NEW' in roles[n] and n in validated]
        old=[n for n in ids if 'KEPT' in roles[n] and n in validated and n not in new]
        parent=[n for n in ids if n in validated and n not in new+old]
        def fair_fill(nids,predicate,cap,tier):
            queues=[(nid,order([f for f in validated[nid].facts if predicate(f) and (nid,f.fact_id) not in selected],ctx,settings)) for nid in nids]
            for j in range(max([len(fs) for _,fs in queues]+[0])):
                for nid,fs in queues:
                    if j<len(fs):add(nid,fs[j],cap,tier)
        if revised:
            from .selection_v21 import select
            select(new=new,old=old,parent=parent,validated=validated,initial=initial,free=free,
                limit=settings.max_evidence_view_chars,ctx=ctx,settings=settings,add=add,summary=summary,fair_fill=fair_fill)
        else:
            navcap=initial+int(free*.55)
            for nid in new:summary(nid,navcap)
            # One actionable object per NEW before name-only catalogs; no splicing.
            for nid in new:
                fs=order([f for f in validated[nid].facts if f.kind in ('indexed_object','directory_object','sql_row','xml_attributes')],ctx,settings)
                for f in fs:
                    if add(nid,f,navcap,'NEW_BOUND_NAVIGATION'):break
            fair_fill(new,lambda f:f.kind in ('table_directory','ordered_columns','result_state'),navcap,'NEW_DIRECTORY')
            # Directory distributions cannot preempt all bound object/column details.
            for nid in new:
                categories=set()
                for f in order([f for f in validated[nid].facts if f.kind=='path_distribution'],ctx,settings):
                    if f.group not in categories:
                        add(nid,f,navcap,'NEW_DISTRIBUTION');categories.add(f.group)
            fair_fill(new,lambda f:f.level!='L3' and f.kind not in ('object_names','path_distribution'),initial+int(free*.84),'NEW_DETAILS')
            for nid in old:summary(nid,initial+int(free*.92))
            fair_fill(old,lambda f:f.level!='L3',initial+int(free*.94),'KEPT_DETAILS')
            for nid in parent:summary(nid,initial+int(free*.96))
            fair_fill(parent,lambda f:f.level!='L3',initial+int(free*.97),'PARENT_DETAILS')
        # Usage and unresolved-candidate statistics are useful, but are not part
        # of the immutable identity/actual edge guarantee.
        for key,value in [('tool_usage',usage),('unresolved_details',unresolved)]:
            container=view['task_evidence_view'];container[key]=value;container['auxiliary_not_inline'].remove(key)
            trial=encoded_chars(view)
            if trial<=initial+int(free*.98):
                allocation['AUXILIARY']+=trial-current;current=trial
            else:container.pop(key);container['auxiliary_not_inline'].append(key)
        fair_fill(new+old+parent,lambda f:f.level=='L3',settings.max_evidence_view_chars,'LITERAL')
        # Reuse slack in the same order, giving NEW navigation another chance.
        if not revised:
            fair_fill(new,lambda f:f.level!='L3',settings.max_evidence_view_chars,'NEW_SLACK')
            fair_fill(old+parent,lambda f:f.level!='L3',settings.max_evidence_view_chars,'OLD_SLACK')
        for nid in validated:summary(nid,settings.max_evidence_view_chars)
        unseen=[]
        for nid in new:
            sc=validated[nid]
            if sc.shape in ('file_index','directory') and not any((nid,f.fact_id) in selected and f.kind in ('indexed_object','directory_object') and isinstance(f.value.get('path'),str) and isinstance(f.value.get('file_id'),str) and bool(f.value['file_id']) for f in sc.facts):
                has=any(f.kind in ('indexed_object','directory_object') and isinstance(f.value.get('path'),str) and isinstance(f.value.get('file_id'),str) and bool(f.value['file_id']) for f in sc.facts)
                unseen.append({'node_id':nid,'reason':'BUDGET_NO_ATOMIC_HANDLE_INLINE' if has else 'NO_ATOMIC_HANDLE_IN_RETAINED_FACTS'})
        measured=serialization_metrics(view,ctx);task=measured.pop('task')
        total=sum(sc.coverage['extracted_facts'] for sc in validated.values())
        metrics.update(measured);metrics.update(status='CARDS_RENDERED',render_cpu_ns=time.process_time_ns()-t,
            facts_total=total,facts_inline=len(selected),facts_omitted=total-len(selected),immutable_core_hashes=cores,
            summary_tiers=tiers,selected_scores=decisions,new_without_atomic_handle=unseen,
            keep_semantics='IDENTITY_IMMUTABLE; SUMMARY_OPTIONAL; DETAIL_BUDGETED',source_scope_count=len(scopes))
        if revised:metrics['allocation_encoded_chars']={'IDENTITY_RELATION_CORE':initial,**dict(allocation),'TOTAL':current}
        return ProjectionResult(view,metrics,False,None,False,task)
    except (ProjectionInputError,ValueError,TypeError,KeyError,AttributeError,IndexError,RecursionError,UnicodeError) as exc:
        return fail(str(exc) if isinstance(exc,ProjectionInputError) else 'INVALID_FRONTIER_OR_SERIALIZATION')
