"""Exact serialization accounting, including the existing nested Seed task wire path."""
from .contracts import compact, encoded_chars


def serialization_metrics(view, task_context=None):
    context=task_context or {}
    projection=compact(view)
    task=compact({'retrieval_state_schema':'task-evidence-graph-v1',
                  'original_task':context.get('original_task',''),
                  'initial_evidence_context':context.get('initial_evidence_context',{}),**view})
    template=context.get('seed_prompt_template')
    if isinstance(template,dict):
        import json
        seed=json.dumps({**template,'task':task},ensure_ascii=False,indent=2,allow_nan=False)
        seed_chars=len(seed)
    else:seed_chars=None
    return {'compact_chars':len(projection),'compact_bytes':len(projection.encode('utf8')),
            'encoded_chars':encoded_chars(view),'encoded_chars_scope':'evidence_projection_inside_Seed_task_string_only',
            'retrieval_task_chars':len(task),'retrieval_task_encoded_chars':len(compact(task))-2,
            'seed_user_chars':seed_chars,'provider_tokens':'NOT_MEASURED','task':task}
