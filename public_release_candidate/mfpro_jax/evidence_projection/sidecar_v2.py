"""Task-independent, byte-bounded fair retention over observed candidates only."""
from collections import Counter,defaultdict,deque
from dataclasses import replace
import time
from .contracts import versions_for,compact,digest,ProjectionInputError
from .sidecar import extraction_identity


def fair_order(facts):
    """Small global directories, then interleave kinds within every object group.

    Hash ordering makes no claim of random sampling. It avoids source-prefix bias
    without consulting tasks, filenames, probes or future results.
    """
    global_nav=[];distributions=defaultdict(list);groups=defaultdict(lambda:defaultdict(list))
    for f in facts:
        if f.kind=='path_distribution':distributions[f.group].append(f)
        elif f.kind in ('table_directory','result_state','ordered_columns'):
            global_nav.append(f)
        else:groups[f.group or f.object_key or f.kind][f.kind].append(f)
    out=sorted(global_nav,key=lambda f:(f.kind,f.fact_id))
    # One distribution slice per category, then object groups get a turn.
    # Thousands of long directory names must not consume the whole sidecar
    # before even a single actionable object can be considered.
    for fs in distributions.values():fs.sort(key=lambda f:f.fact_id)
    remainder=[]
    for category in sorted(distributions):
        fs=distributions[category];out.append(fs[0]);remainder.extend(fs[1:])
    queues=[]
    for group in sorted(groups,key=digest):
        kinds=groups[group];inter=[]
        for fs in kinds.values():fs.sort(key=lambda f:f.fact_id)
        ks=sorted(kinds,key=lambda k:(k not in ('column_directory','indexed_object','directory_object'),k))
        for i in range(max(len(v) for v in kinds.values())):
            for k in ks:
                if i<len(kinds[k]):inter.append(kinds[k][i])
        queues.append(deque(inter))
    while any(queues):
        for q in queues:
            if q:out.append(q.popleft())
    out.extend(remainder)
    return out


def retained_catalog(facts,counts):
    kinds=Counter(f.kind for f in facts)
    tables=set();columns=set();typed=set();objects=set();bound=set()
    for f in facts:
        v=f.value
        if f.kind=='table_directory':tables.update(x if isinstance(x,str) else x[1] for x in v if isinstance(x,str) or isinstance(x,list) and len(x)==2 and isinstance(x[1],str))
        if f.kind=='column_directory':
            columns.update((v['table'],x) for x in v['columns']);tables.add(v['table'])
        if f.kind=='schema_field':
            if len(v['column'])>1 and isinstance(v['column'][1],str):typed.add((v['table'],v['column'][0]))
            tables.add(v['table'])
        if f.kind in ('indexed_object','directory_object'):
            objects.add(f.object_key)
            if isinstance(v.get('path'),str) and isinstance(v.get('file_id'),str) and bool(v['file_id']):bound.add(f.object_key)
    return {'kinds':dict(kinds),'object_bindings':len(bound),'object_records':len(objects),
        'objects_source':counts.get('objects_returned'),'object_records_omitted':max(0,counts.get('objects_returned',0)-len(objects)),'tables_directory_or_field':len(tables),
        'tables_source':counts.get('schema_object_count'),'column_names':len(columns),'typed_columns':len(typed),
        'columns_source':counts.get('schema_columns'),'column_names_omitted':max(0,counts.get('schema_columns',0)-len(columns)),
        'typed_columns_source':counts.get('schema_typed_columns'),
        'typed_columns_omitted':max(0,counts.get('schema_typed_columns',counts.get('schema_columns',0))-len(typed)),
        'tables_omitted':max(0,counts.get('schema_object_count',0)-len(tables))}


def project_v2(**kwargs):
    started=time.process_time_ns();settings=kwargs['settings']
    versions=versions_for(settings.presentation)
    # Shared strict parser/transform/identity validation. Candidate memory remains
    # bounded by max_items/input/depth and this explicit 20k fact ceiling.
    candidate_settings=replace(settings,presentation='evidence_cards_v1',max_facts=20000,max_sidecar_bytes=1048576)
    # Use shared validation/parser before any byte retention. No intermediate
    # larger sidecar is serialized or persisted.
    from .sidecar import _project_candidates
    base,ex,coverage=_project_candidates(**{**kwargs,'settings':candidate_settings,'v2':True})
    facts=fair_order(ex.facts);counts=coverage['counts'];selected=[]
    core={'shape':ex.shape,'counts':{k:v for k,v in counts.items() if k not in ('candidate_facts_attempted','facts_rejected_by_level')},'partial':True}
    if len(compact(core))>settings.keep_core_chars:core={'shape':ex.shape,'partial':True,'counts_omitted':True}
    def build(fs):
        cov={**coverage,'policy_identity':extraction_identity(settings),'retained_facts':len(fs),
             'sidecar_omitted_facts':len(ex.facts)-len(fs),'retention_policy':'group_kind_round_robin_v2',
             'retention_catalog':retained_catalog(fs,counts),'candidate_fact_cap':20000}
        return replace(base,schema_version=versions[0],extractor_version=versions[1],policy_version=versions[2],
            facts=tuple(fs),coverage_json=compact(cov),keep_core_json=compact(core),cache_key='0'*64)
    # Reserve a bounded overhead for final coverage counters. A final exact byte
    # check handles digit growth; no single fact is cut or rewritten.
    empty=build([]);base_bytes=len(compact(empty.to_dict()).encode())
    if base_bytes>settings.max_sidecar_bytes:raise ProjectionInputError('SIDECAR_IDENTITY_EXCEEDS_LIMIT')
    used=base_bytes+2048
    for f in facts:
        cost=len(compact(f.to_dict()).encode())+1
        if len(selected)<settings.max_facts and used+cost<=settings.max_sidecar_bytes:selected.append(f);used+=cost
    result=build(selected)
    while len(compact(result.to_dict()).encode())>settings.max_sidecar_bytes and selected:
        selected.pop();result=build(selected)
    if len(compact(result.to_dict()).encode())>settings.max_sidecar_bytes:raise ProjectionInputError('SIDECAR_IDENTITY_EXCEEDS_LIMIT')
    result=replace(result,cache_key=result.content_key());result.validate()
    return replace(result,extraction_cpu_ns=time.process_time_ns()-started)
