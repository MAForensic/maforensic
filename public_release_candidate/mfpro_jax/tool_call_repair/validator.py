"""Independent fail-closed validation. No SQLite/file access, mandatory host guards."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import PurePosixPath
import re
from .contracts import (canonical, changed_fields, plain, Proposal, DECISIONS, REASONS, digest)
from .policy import REQUIRED_GUARDS, EffectsRegistry, SignatureLedger
from . import rules


@dataclass(frozen=True)
class Validation:
    allowed: bool
    reason: str


def schema_error(value, schema):
    try:
        return _schema_error(value, schema)
    except (TypeError, ValueError, KeyError, RecursionError, re.error):
        return 'INVALID_SCHEMA'


def _schema_error(value, schema):
    """Conservative recursive JSON Schema subset; unknown constraints deny.

    Covers all current central input schema constructs including numeric bounds.
    Full live tool/path/SQL authorization remains required through GuardReceipt.
    """
    known = {'type', 'properties', 'required', 'additionalProperties', 'enum',
             'const', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
             'minLength', 'maxLength', 'pattern', 'items', 'minItems', 'maxItems',
             'uniqueItems', 'minProperties', 'maxProperties', 'description',
             'title', 'default', 'examples', '$schema'}
    if not isinstance(schema, Mapping) or set(schema) - known:
        return 'UNSUPPORTED_SCHEMA'
    v = plain(value)
    kind = schema.get('type')
    checks = {'object': isinstance(v, dict), 'array': isinstance(v, list),
              'string': isinstance(v, str), 'integer': type(v) is int,
              'number': type(v) in (int, float), 'boolean': type(v) is bool,
              'null': v is None}
    if kind not in checks or not checks[kind]:
        return 'SCHEMA_TYPE'
    if 'enum' in schema and canonical(v) not in [canonical(x) for x in schema['enum']]:
        return 'SCHEMA_ENUM'
    if 'const' in schema and canonical(v) != canonical(schema['const']):
        return 'SCHEMA_CONST'
    if isinstance(v, dict):
        props = schema.get('properties', {})
        if set(schema.get('required', ())) - set(v):
            return 'SCHEMA_REQUIRED'
        for key, item in v.items():
            if key not in props:
                extra = schema.get('additionalProperties', False)
                if extra is True:
                    continue
                if not isinstance(extra, Mapping):
                    return 'SCHEMA_UNKNOWN_KEY'
                err = schema_error(item, extra)
            else:
                err = schema_error(item, props[key])
            if err:
                return err
        for key, check in [('minProperties', len(v) < schema.get('minProperties', 0)),
                           ('maxProperties', len(v) > schema.get('maxProperties', float('inf')))]:
            if check:
                return key
    if type(v) in (int, float):
        for key, bad in [('minimum', v < schema.get('minimum', -float('inf'))),
                         ('maximum', v > schema.get('maximum', float('inf'))),
                         ('exclusiveMinimum', 'exclusiveMinimum' in schema and v <= schema['exclusiveMinimum']),
                         ('exclusiveMaximum', 'exclusiveMaximum' in schema and v >= schema['exclusiveMaximum'])]:
            if bad:
                return 'SCHEMA_' + key
    if isinstance(v, (str, list)):
        prefix = 'Length' if isinstance(v, str) else 'Items'
        if len(v) < schema.get('min' + prefix, 0) or len(v) > schema.get('max' + prefix, float('inf')):
            return 'SCHEMA_SIZE'
    if isinstance(v, str) and 'pattern' in schema and not re.search(schema['pattern'], v):
        return 'SCHEMA_PATTERN'
    if isinstance(v, list):
        if schema.get('uniqueItems') and len({canonical(x) for x in v}) != len(v):
            return 'SCHEMA_UNIQUE'
        if 'items' in schema:
            for item in v:
                err = schema_error(item, schema['items'])
                if err:
                    return err
    return ''


SQL_TOKEN = re.compile(r"(?P<space>\s+)|(?P<comment>--[^\n]*(?:\n|$)|/\*[\s\S]*?\*/)|(?P<string>'(?:''|[^'])*')|(?P<quoted>\"(?:\"\"|[^\"])*\"|`[^`]*`|\[[^\]]*\])|(?P<word>[A-Za-z_][A-Za-z_0-9]*)|(?P<number>\d+(?:\.\d+)?)|(?P<symbol>[(),.*=<>!+/%;|&?-])")
BLOCKED_SQL = {'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'PRAGMA',
               'ATTACH', 'DETACH', 'REPLACE', 'VACUUM', 'REINDEX', 'ANALYZE',
               'BEGIN', 'COMMIT', 'ROLLBACK', 'SAVEPOINT', 'RELEASE', 'LOAD_EXTENSION'}


def sql_tokens(sql):
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 4096:
        raise ValueError('SQL_SIZE')
    pos, out = 0, []
    while pos < len(sql):
        m = SQL_TOKEN.match(sql, pos)
        if not m:
            raise ValueError('SQL_UNSUPPORTED_TOKEN')
        kind = m.lastgroup
        if kind not in {'space', 'comment'}:
            out.append((kind, m.group(), m.start(), m.end()))
        pos = m.end()
    return out


def sql_safety(sql):
    try:
        ts = sql_tokens(sql)
    except ValueError:
        return False
    if not ts or ts[0][0] != 'word' or ts[0][1].upper() not in {'SELECT', 'WITH'}:
        return False
    if any(t[0] == 'word' and t[1].upper() in BLOCKED_SQL for t in ts):
        return False
    # Even quoted function names must not allow extension loading.
    if any(t[1].strip('"`[]').upper() == 'LOAD_EXTENSION' for t in ts):
        return False
    semis = [i for i, t in enumerate(ts) if t[1] == ';']
    return not semis or semis == [len(ts) - 1]


def _sql_mapping(a, facts):
    if not a.tool.sql_identifier_repair or a.tool.semantic_kind != 'sqlite_select':
        return None
    args = plain(a.arguments or {})
    old = args.get('query')
    if not sql_safety(old):
        return None
    schemas = [f for f in facts if f.kind == 'SQL_SCHEMA'
               and f.payload.get('db_path') == args.get('db_path')
               and f.payload.get('complete') is True]
    schema_values = {canonical(f.payload.get('tables')) for f in schemas}
    if len(schema_values) != 1:
        return None
    tables = plain(schemas[0].payload.get('tables', {}))
    proposals = []
    ts = sql_tokens(old)
    for f in facts:
        p = f.payload
        if (f.kind != 'SQL_IDENTIFIER_MAPPING' or p.get('db_path') != args.get('db_path')
            or p.get('original_sql') != old or p.get('semantic_equivalence') is not True):
            continue
        ix = p.get('token_index')
        if type(ix) is not int or not 0 <= ix < len(ts):
            continue
        tok = ts[ix]
        before, after = p.get('from_identifier'), p.get('to_identifier')
        reserved = BLOCKED_SQL | {'SELECT', 'WITH', 'FROM', 'JOIN', 'WHERE', 'LIMIT', 'ORDER', 'BY', 'AS', 'ON', 'GROUP', 'HAVING', 'UNION', 'AND', 'OR', 'NOT', 'DISTINCT', 'OFFSET'}
        if not isinstance(before, str) or before.upper() in reserved or not isinstance(after, str) or after.upper() in reserved:
            continue
        if tok[0] != 'word' or tok[1] != before or not isinstance(after, str) or not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*', after):
            continue
        if p.get('identifier_kind') == 'table':
            if ix == 0 or ts[ix - 1][1].upper() not in {'FROM', 'JOIN'} or after not in tables:
                continue
        elif p.get('identifier_kind') == 'column':
            table = p.get('table')
            if after not in tables.get(table, []):
                continue
            from_tables = [ts[i + 1][1] for i, t in enumerate(ts[:-1]) if t[1].upper() in {'FROM', 'JOIN'}]
            if from_tables != [table]:
                continue
        else:
            continue
        new = old[:tok[2]] + after + old[tok[3]:]
        if not sql_safety(new):
            continue
        candidate = dict(args, query=new)
        proposals.append((candidate, (f.ref,) + tuple(s.ref for s in schemas)))
    values = {canonical(p[0]) for p in proposals}
    if len(values) == 1:
        return proposals[0]
    return None


def semantic_candidate(a, facts):
    """Closed local evidence transformations; no model-supplied equivalence proof."""
    facts = tuple(f for f in facts if f.scope == a.scope and f.trusted is True and f.authority_ref
                  and f.created_seq < a.batch_emitted_seq and f.created_seq < a.failure_seq)
    if len({f.ref for f in facts}) != len(facts):
        return None
    sql = _sql_mapping(a, facts)
    if sql:
        return sql
    args = plain(a.arguments or {})
    options = []
    protected = {'source_id', 'partition_id', 'index_generation', 'db_path', 'query', 'sql'}
    for f in facts:
        p = f.payload
        field = p.get('field')
        if (f.kind != 'ARGUMENT_BINDING' or field in protected
            or field not in a.tool.repairable_fields or field not in args
            or p.get('semantic_equivalence') is not True or not p.get('relationship')
            or canonical(p.get('original_value')) != canonical(args[field])
            or 'candidate_value' not in p):
            continue
        options.append((dict(args, **{field: plain(p['candidate_value'])}), (f.ref,)))
    values = {canonical(c) for c, _ in options}
    return options[0] if len(values) == 1 else None


def proposal_basis(a, facts):
    rule = rules.propose(a, facts)
    if rule:
        return rule.candidate_arguments, rule.evidence_refs
    return semantic_candidate(a, facts)


class CandidateValidator:
    def __init__(self, effects: EffectsRegistry, ledger: SignatureLedger):
        self.effects, self.ledger = effects, ledger

    def validate(self, a, p: Proposal, facts, current_scope, guard):
        from .classifier import classify
        if not classify(a).candidate:
            return Validation(False, 'NOT_A_PARAMETER_CANDIDATE')
        if not a.scope.verified or not a.scope.source_id or not a.scope.target_id:
            return Validation(False, 'MISSING_SCOPE_BINDING')
        if not a.tool.implementation_ref:
            return Validation(False, 'MISSING_IMPLEMENTATION_BINDING')
        if (not isinstance(p.decision, str) or not isinstance(p.reason_code, str)
            or p.decision not in DECISIONS or p.reason_code not in REASONS):
            return Validation(False, 'OUTPUT_INVALID')
        if p.decision != 'REPAIR_ONCE':
            return Validation(False, p.reason_code)
        if p.same_tool != a.tool.name or not a.tool.exposed or not a.tool.resolved:
            return Validation(False, 'SAME_TOOL_REQUIRED')
        if current_scope != a.scope:
            return Validation(False, 'SCOPE_CHANGED')
        if not self.effects.allows(a.tool) or a.execution_state == 'UNKNOWN':
            return Validation(False, 'UNSAFE_EFFECT')
        if not isinstance(p.candidate_arguments, Mapping):
            return Validation(False, 'CANDIDATE_NOT_OBJECT')
        candidate = plain(p.candidate_arguments)
        actual_diff = changed_fields(a.arguments or {}, candidate)
        if not actual_diff or tuple(sorted(p.changed_fields)) != actual_diff or len(set(p.changed_fields)) != len(p.changed_fields):
            return Validation(False, 'DIFF_MISMATCH')
        err = schema_error(candidate, a.tool.schema)
        if err:
            return Validation(False, err)
        # Scope facts must have passed the same cutoff even when validator is used alone.
        if any(f.scope != a.scope or not f.trusted or f.created_seq >= a.batch_emitted_seq or f.created_seq >= a.failure_seq for f in facts):
            return Validation(False, 'EVIDENCE_SCOPE_OR_TIME')
        by_ref = {f.ref: f for f in facts}
        if len(by_ref) != len(facts):
            return Validation(False, 'DUPLICATE_EVIDENCE_IDENTITY')
        if len(set(p.evidence_refs)) != len(p.evidence_refs) or set(p.evidence_refs) - set(by_ref):
            return Validation(False, 'FORGED_EVIDENCE_REF')
        basis = proposal_basis(a, facts)
        if not basis or canonical(candidate) != canonical(basis[0]) or set(p.evidence_refs) != set(basis[1]):
            return Validation(False, 'UNPROVEN_DIFF')
        original = a.arguments or {}
        for key in ('db_path', 'source_id', 'partition_id', 'index_generation'):
            if key in candidate and key in original and canonical(candidate[key]) != canonical(original[key]):
                return Validation(False, 'SOURCE_OR_DB_CHANGED')
        for key in ('file_path', 'db_path', 'root', 'archive_path'):
            if key not in candidate:
                continue
            path = candidate[key]
            if not isinstance(path, str) or '\x00' in path or '\\' in path or '..' in PurePosixPath(path).parts or not PurePosixPath(path).is_absolute():
                return Validation(False, 'PATH_DENIED')
            roots = [PurePosixPath(r) for r in a.tool.allowed_roots]
            q = PurePosixPath(path)
            if not roots or not any(q == r or r in q.parents for r in roots):
                return Validation(False, 'PATH_DENIED')
        if a.tool.semantic_kind == 'sqlite_select' and not sql_safety(candidate.get('query')):
            return Validation(False, 'SQL_POLICY')
        if self.ledger.is_blocked(a, candidate):
            return Validation(False, 'REPEAT_GUARD')
        try:
            receipt = guard(a, candidate)
        except Exception:
            return Validation(False, 'GUARD_UNAVAILABLE')
        if (receipt.allowed is not True or receipt.scope != a.scope or receipt.contract_version != a.tool.version
            or receipt.contract_fingerprint != digest(a.tool)
            or not REQUIRED_GUARDS <= receipt.checks):
            return Validation(False, 'GUARD_DENIED_OR_INCOMPLETE')
        return Validation(True, 'APPROVED')
