"""Explainable, deterministic ordering from current task and already-seen intent."""
from __future__ import annotations
import re
from .contracts import compact


def tokens(text):
    # Split camelCase and punctuation, including multilingual continuous terms.
    text=re.sub(r'([a-z])([A-Z])',r'\1 \2',text)
    return set(x.lower() for x in re.findall(r'[A-Za-z][A-Za-z0-9]*|[\u3400-\u9fff]{2,}',text) if len(x)>1)


def score_fact(fact, task_context, enabled=True):
    if not enabled:return {'lexical':0,'type_hint':0,'path_structure':0,'total':0}
    # Only original task and passed, time-frozen invocation text. No catalog/RAW
    # instructions or future model messages are used as control/intent sources.
    text=str(task_context.get('original_task',''))+' '+str(task_context.get('prior_invocation_text',''))
    hints=tokens(text);ft=tokens(compact(fact.value));overlap=len(hints & ft)
    time_intent=bool(hints & {'date','time','when','earliest','first','timestamp','日期','时间','何时'})
    time_field=bool(ft & {'time','date','timestamp','stamp'})
    type_hint=2 if time_intent and time_field else 0
    if fact.kind in ('indexed_object','directory_object'):
        v=fact.value;path=str(v.get('path',v.get('name','')))
        path_hint=1 if path.lower().endswith(('.db','.sqlite','.sqlite3','.plist','.xml','.json')) else 0
    else:path_hint=0
    return {'lexical':overlap*4,'type_hint':type_hint,'path_structure':path_hint,'total':overlap*4+type_hint+path_hint}


def order_facts(facts, task_context, settings):
    """Rank within levels; diverse groups round-robin instead of prefix monopoly."""
    ranked=sorted(facts,key=lambda f:(-score_fact(f,task_context,settings.ranking_enabled)['total'],f.fact_id))
    if not settings.diversity_enabled:return ranked
    groups={}
    for f in ranked:groups.setdefault(f.group or f.kind,[]).append(f)
    out=[]
    while groups:
        for key in list(groups):
            out.append(groups[key].pop(0))
            if not groups[key]:del groups[key]
    return out
