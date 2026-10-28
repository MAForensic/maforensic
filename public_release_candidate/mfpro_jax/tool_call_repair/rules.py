"""Lossless proposals only. Rules never execute or mutate originals."""
from collections.abc import Mapping
from .contracts import Proposal, changed_fields, plain, strict_json, canonical


def propose(a, facts=()):
    facts = tuple(f for f in facts if f.trusted and f.authority_ref and f.scope == a.scope
                  and f.created_seq < a.batch_emitted_seq and f.created_seq < a.failure_seq)
    args = plain(a.arguments) if a.arguments is not None else None
    t = a.tool
    if args is None:
        if t.allow_leading_bom and a.raw_arguments.startswith('\ufeff'):
            try:
                candidate = strict_json(a.raw_arguments[1:])
                if isinstance(candidate, dict):
                    return Proposal('REPAIR_ONCE', t.name, candidate, tuple(sorted(candidate)),
                                    (), 'LOSSLESS_JSON', 'rules')
            except (ValueError, TypeError):
                pass
        # Only a prior exact canonical wire representation for this same call can
        # prove malformed JSON intent. No punctuation guessing or regex repair.
        matches = [f for f in facts if f.kind == 'JSON_CANONICAL'
                   and f.payload.get('raw_arguments') == a.raw_arguments
                   and f.payload.get('call_id') == a.sdk_tool_call_id]
        values = {canonical(f.payload.get('arguments')) for f in matches}
        if len(values) == 1:
            candidate = strict_json(next(iter(values)))
            if isinstance(candidate, dict):
                return Proposal('REPAIR_ONCE', t.name, candidate, tuple(sorted(candidate)),
                                tuple(f.ref for f in matches), 'LOSSLESS_JSON', 'rules')
        return None
    props = set(t.schema.get('properties', {}))
    if (t.semantic_kind == 'sqlite_select' and t.aliases.get('sql') == 'query'
        and t.alias_proof_ref and 'query' in t.schema.get('required', ())
        and 'sql' in args and 'query' not in args and isinstance(args['sql'], str)
        and set(args) - props == {'sql'}):
        candidate = dict(args)
        candidate['query'] = candidate.pop('sql')
        return Proposal('REPAIR_ONCE', t.name, candidate, changed_fields(args, candidate),
                        (), 'EXACT_ALIAS', 'rules')
    if (t.export_binding and t.literal_keyword and t.aliases.get('keyword') == 'pattern'
        and t.alias_proof_ref and 'file_path' in props and 'pattern' in props
        and args.get('source_id') == a.scope.source_id
        and args.get('file_id') == a.scope.target_id
        and isinstance(args.get('keyword'), str) and args.get('regex', False) is False
        and set(args) - props <= {'source_id', 'file_id', 'keyword'}):
        matches = [f for f in facts if f.kind == 'EXPORTED_HOST'
                   and f.payload.get('file_id') == args['file_id']
                   and f.payload.get('materialized') is True
                   and isinstance(f.payload.get('host_path'), str)]
        paths = {f.payload['host_path'] for f in matches}
        if any(f.kind == 'EXPORTED_HOST' and f.payload.get('host_path') in paths
               and f.payload.get('file_id') != args['file_id'] for f in facts):
            return None  # Conflicting object ownership of the same host path.
        if len(paths) == 1 and not ({'file_path', 'pattern'} & set(args)):
            candidate = {k: v for k, v in args.items() if k not in {'source_id', 'file_id', 'keyword'}}
            candidate.update(file_path=next(iter(paths)), pattern=args['keyword'])
            return Proposal('REPAIR_ONCE', t.name, candidate, changed_fields(args, candidate),
                            tuple(f.ref for f in matches), 'EXACT_EXPORTED_OBJECT', 'rules')
    return None
