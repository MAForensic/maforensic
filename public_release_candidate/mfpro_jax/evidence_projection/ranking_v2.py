"""Generic frozen-intent ranking, separate from V1 and from observe-time retention."""
from collections import defaultdict,deque
from .ranking import tokens
from .contracts import compact
from pathlib import PurePosixPath


def words(text):
    # Generic English plural normalization, plus CJK bigrams. No target vocabulary.
    import re
    out={x[:-1] if len(x)>3 and x.endswith('s') and not x.endswith('ss') else x for x in tokens(text)}
    for run in re.findall(r'[\u3400-\u9fff]{2,}',text):out.update(run[i:i+2] for i in range(len(run)-1))
    return out


def score(f,ctx,enabled=True):
    if not enabled:return {'task_overlap':0,'prior_overlap':0,'navigation':0,'total':0}
    value=words(compact(f.value));task=words(str(ctx.get('original_task','')))
    prior=words(str(ctx.get('prior_invocation_text','')))
    a=min(8,len(value&task));b=min(4,len(value&prior))
    nav=0
    if f.kind in ('indexed_object','directory_object'):
        v=f.value;path=str(v.get('path',v.get('name',''))).lower()
        nav=int(path.endswith(('.db','.sqlite','.sqlite3','.plist','.xml','.json')))
    return {'task_overlap':a,'prior_overlap':b,'navigation':nav,'total':a*4+b+nav}


def score_v21(f,ctx,enabled=True):
    """Score content identifiers, not JSON keys, source IDs or opaque handles.

    Basenames/columns are more discriminative than shared ancestors. Prior
    arguments are a weak tie-breaker: repetition must not outrank a usable
    container merely because a companion filename repeats query words. Generic
    temporal navigation is an observed-field hint, never a semantic assertion.
    """
    return _score_v21(f,words(str(ctx.get('original_task',''))),words(str(ctx.get('prior_invocation_text',''))),enabled)


def _score_v21(f,task,prior,enabled=True):
    empty={'task_overlap':0,'prior_overlap':0,'navigation':0,'temporal_hint':0,'total':0}
    if not enabled:return empty
    v=f.value
    focus=set();ancestors=set();nav=0;temporal=0
    if f.kind in ('indexed_object','directory_object'):
        path=str(v.get('path',v.get('name','')));p=PurePosixPath(path)
        focus=words(p.name);ancestors=words(str(p.parent))-focus
        nav=int(p.suffix.lower() in ('.db','.sqlite','.sqlite3','.plist','.xml','.json'))
    elif f.kind=='schema_field':
        focus=words(str(v['column'][0]));ancestors=words(v['table'])-focus
    elif f.kind=='column_directory':
        focus=words(compact(v['columns']));ancestors=words(v['table'])-focus
    elif f.kind in ('object_names','table_directory'):
        focus=words(compact([x[1] if isinstance(x,list) and len(x)==2 else x for x in v]))
    else:focus=words(compact(v))
    a=len(focus&task);b=len(focus&prior)
    parent_a=len(ancestors&task);parent_b=len(ancestors&prior)
    if f.kind in ('schema_field','column_directory'):
        temporal_intent=bool(task&{'time','date','when','first','last','earliest','latest','timestamp','时间','日期','何时','最早','最近','首次','最后'})
        temporal=int(temporal_intent and bool(focus&{'time','date','timestamp','stamp','created','updated','modified','epoch'}))
    return {'task_overlap':a,'prior_overlap':b,'ancestor_task_overlap':parent_a,'ancestor_prior_overlap':parent_b,
            'navigation':nav,'temporal_hint':temporal,'total':a*8+min(2,b)+min(2,parent_a)+int(parent_b>0)+nav*8+temporal*4}


def order(facts,ctx,settings):
    revised=settings.presentation=='evidence_cards_v2_1'
    # Render-local work only: no cross-task/cache state. Freeze intent once per
    # ordering operation instead of re-tokenizing a long history per candidate.
    if revised:
        task=words(str(ctx.get('original_task','')));prior=words(str(ctx.get('prior_invocation_text','')))
        scores={f.fact_id:_score_v21(f,task,prior,settings.ranking_enabled)['total'] for f in facts}
        ranked=sorted(facts,key=lambda f:(-scores[f.fact_id],f.fact_id))
    else:
        ranked=sorted(facts,key=lambda f:(-score(f,ctx,settings.ranking_enabled)['total'],f.fact_id))
    if not settings.diversity_enabled:return ranked
    if revised:
        # Diversity may break equal relevance ties; it cannot move an unrelated
        # directory ahead of an explicitly matching object in another group.
        out=[];buckets=defaultdict(list)
        for f in ranked:buckets[scores[f.fact_id]].append(f)
        for relevance in sorted(buckets,reverse=True):
            groups=defaultdict(deque)
            for f in buckets[relevance]:groups[f.group or f.object_key or f.kind].append(f)
            while groups:
                for g in list(groups):
                    out.append(groups[g].popleft())
                    if not groups[g]:del groups[g]
        return out
    groups=defaultdict(deque)
    for f in ranked:groups[f.group or f.object_key or f.kind].append(f)
    out=[]
    while groups:
        for g in list(groups):
            out.append(groups[g].popleft())
            if not groups[g]:del groups[g]
    return out
