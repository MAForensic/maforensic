"""Immutable, dependency-free Repair V1 contracts. No production wiring."""
from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, '__dataclass_fields__'):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    return value


def canonical(value: Any) -> str:
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError('JSON keys must be strings')
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError('non-JSON or non-finite value')


def strict_json(text: str) -> Any:
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError('duplicate key')
            d[k] = v
        return d
    def number(x):
        n = float(x)
        if not math.isfinite(n):
            raise ValueError('non-finite number')
        return n
    def invalid(_):
        raise ValueError('non-finite constant')
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_float=number,
                          parse_constant=invalid)
    except RecursionError as exc:
        raise ValueError('JSON nesting exceeds parser limit') from exc


@dataclass(frozen=True)
class RepairConfig:
    mode: str = 'off'
    max_retries_per_call: int = 1
    require_safe_retry: bool = True
    max_llm_calls_per_failure: int = 1
    max_llm_calls_per_question: int = 2
    max_context_chars: int = 6000
    max_evidence_refs: int = 4
    max_chars_per_evidence: int = 1200
    max_error_chars: int = 800
    allow_same_batch_new_evidence: bool = False
    llm_timeout_seconds: float = 15
    max_total_repair_seconds_per_question: float = 90
    min_main_reserve_seconds: float = 30
    max_output_tokens: int = 1024
    provider_ref: str = 'agent'
    model_role: str = 'repair'

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode not in {'off', 'rules', 'agent', 'hybrid'}:
            raise ValueError('invalid repair mode')
        for name in ('max_retries_per_call', 'max_llm_calls_per_failure'):
            if type(getattr(self, name)) is not int or getattr(self, name) not in (0, 1):
                raise ValueError(name + ' must be 0 or 1')
        if self.require_safe_retry is not True:
            raise ValueError('safe retry cannot be disabled')
        # V1 intentionally does not enable newly produced sibling evidence.
        if self.allow_same_batch_new_evidence is not False:
            raise ValueError('same-batch new evidence is not supported in V1')
        for name in ('max_llm_calls_per_question', 'max_context_chars',
                     'max_evidence_refs', 'max_chars_per_evidence', 'max_error_chars',
                     'max_output_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) < (0 if name in {'max_llm_calls_per_question', 'max_evidence_refs'} else 1):
                raise ValueError('invalid ' + name)
        for name in ('llm_timeout_seconds', 'max_total_repair_seconds_per_question',
                     'min_main_reserve_seconds'):
            v = getattr(self, name)
            if type(v) not in (float, int) or not math.isfinite(v) or v < 0:
                raise ValueError('invalid ' + name)
        if not isinstance(self.provider_ref, str) or not self.provider_ref or self.model_role != 'repair':
            raise ValueError('repair provider role required')

    @classmethod
    def from_config(cls, config: Mapping | None = None) -> RepairConfig:
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise ValueError('config must be a mapping')
        data = config.get('tool_call_repair', {})
        if not isinstance(data, Mapping):
            raise ValueError('tool_call_repair must be a mapping')
        if set(data) - {f.name for f in fields(cls)}:
            raise ValueError('unknown repair setting')
        return cls(**data)


@dataclass(frozen=True)
class Scope:
    run_id: str
    question_id: str
    source_id: str
    partition_id: str
    index_generation: str
    target_id: str
    verified: bool = False

    def __post_init__(self):
        if type(self.verified) is not bool:
            raise ValueError('scope verification must be explicit')
        if any(not isinstance(getattr(self, f.name), str) for f in fields(self) if f.name != 'verified'):
            raise ValueError('scope fields must be strings')
        if self.verified and any(not getattr(self, f.name).strip() or
                                getattr(self, f.name).strip().upper() in {'UNKNOWN', 'UNVERIFIED'}
                                for f in fields(self) if f.name != 'verified'):
            raise ValueError('verified scope requires all identity bindings')


@dataclass(frozen=True)
class ToolContract:
    name: str
    version: str
    schema: Mapping
    exposed: bool = True
    resolved: bool = True
    semantic_kind: str = 'generic'
    aliases: Mapping = field(default_factory=dict)
    alias_proof_ref: str = ''
    export_binding: bool = False
    literal_keyword: bool = False
    repairable_fields: tuple[str, ...] = ()
    allow_leading_bom: bool = False
    sql_identifier_repair: bool = False
    timeout_seconds: float = 60
    allowed_roots: tuple[str, ...] = ()
    implementation_ref: str = ''  # Resolved executor identity, supplied by trusted runtime adapter.

    def __post_init__(self):
        object.__setattr__(self, 'schema', freeze(self.schema))
        object.__setattr__(self, 'aliases', freeze(self.aliases))
        object.__setattr__(self, 'allowed_roots', tuple(self.allowed_roots))
        object.__setattr__(self, 'repairable_fields', tuple(self.repairable_fields))
        for name in ('exposed', 'resolved', 'export_binding', 'literal_keyword', 'allow_leading_bom', 'sql_identifier_repair'):
            if type(getattr(self, name)) is not bool:
                raise ValueError('contract flags must be boolean')
        if (not self.name or not self.version or not isinstance(self.implementation_ref, str)
            or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError('contract identity and positive timeout required')


@dataclass(frozen=True)
class Failure:
    stage: str
    origin: str
    code: str
    message: str
    trusted: bool = False
    message_ref: str = ''

    def __post_init__(self):
        if type(self.trusted) is not bool or any(not isinstance(getattr(self, k), str)
                for k in ('stage', 'origin', 'code', 'message', 'message_ref')):
            raise ValueError('failure provenance must be typed, trust must be boolean')


@dataclass(frozen=True)
class OriginalAttempt:
    attempt_id: str
    scope: Scope
    batch_id: str
    batch_index: int
    sdk_tool_call_id: str
    tool: ToolContract
    raw_arguments: str
    arguments: Mapping | None
    status: str
    failure: Failure
    execution_state: Literal['NOT_STARTED', 'COMPLETED', 'UNKNOWN']
    original_result_ref: str
    batch_emitted_seq: int
    failure_seq: int
    deadline: float
    local_intent: str = ''
    origin: str = 'B'
    frozen_at: float = 0
    stored_arguments_hash: str = ''  # Optional authoritative Foundation sentinel hash (JSON rejection).

    def __post_init__(self):
        if self.arguments is not None:
            object.__setattr__(self, 'arguments', freeze(self.arguments))
            if not isinstance(self.arguments, Mapping):
                raise ValueError('arguments must be an object')
            if canonical(strict_json(self.raw_arguments)) != canonical(self.arguments):
                raise ValueError('raw/parsed arguments mismatch')
        if self.execution_state not in {'NOT_STARTED', 'COMPLETED', 'UNKNOWN'}:
            raise ValueError('invalid execution state')
        if not self.attempt_id or not self.sdk_tool_call_id or self.batch_index < 0:
            raise ValueError('call identity required')
        if not math.isfinite(self.deadline) or self.failure_seq < self.batch_emitted_seq:
            raise ValueError('invalid time boundary')

    @property
    def arguments_hash(self):
        return digest(self.arguments) if self.arguments is not None else digest(self.raw_arguments)

    @property
    def expected_stored_arguments_hash(self):
        return self.stored_arguments_hash or self.arguments_hash

    @property
    def key(self):
        return (self.scope.run_id, self.scope.question_id, self.attempt_id)


@dataclass(frozen=True)
class EvidenceFact:
    ref: str
    scope: Scope
    created_seq: int
    kind: str
    payload: Mapping
    authority_ref: str
    trusted: bool = True
    b_presented: str = 'UNVERIFIED'
    graph_node_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'payload', freeze(self.payload))
        if (not isinstance(self.payload, Mapping) or not isinstance(self.ref, str) or not self.ref
            or not isinstance(self.authority_ref, str) or not self.authority_ref
            or type(self.trusted) is not bool or type(self.created_seq) is not int or self.created_seq < 0):
            raise ValueError('fact provenance required')


DECISIONS = {'REPAIR_ONCE', 'ABSTAIN_NEEDS_MORE_EVIDENCE', 'ABSTAIN_NON_PARAMETER_FAILURE'}
REASONS = {'EXACT_ALIAS', 'EXACT_EXPORTED_OBJECT', 'GROUNDED_IDENTIFIER_MAPPING',
           'GROUNDED_SQL_IDENTIFIER', 'LOSSLESS_JSON', 'MISSING_BINDING', 'NOT_UNIQUE',
           'SCHEMA_SCOPE_MISMATCH', 'UNSAFE_EFFECT', 'EXECUTION_UNKNOWN', 'POLICY_DENIED',
           'REPEAT_GUARD', 'BUDGET_EXCEEDED', 'PROVIDER_UNAVAILABLE', 'OUTPUT_INVALID',
           'CAPABILITY_UNSUPPORTED', 'NON_PARAMETER_FAILURE', 'MODEL_ABSTAIN'}


@dataclass(frozen=True)
class Proposal:
    decision: str
    same_tool: str
    candidate_arguments: Mapping
    changed_fields: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    reason_code: str
    source: str = 'agent'

    def __post_init__(self):
        object.__setattr__(self, 'candidate_arguments', freeze(self.candidate_arguments))
        object.__setattr__(self, 'changed_fields', tuple(self.changed_fields))
        object.__setattr__(self, 'evidence_refs', tuple(self.evidence_refs))


def changed_fields(before: Mapping, after: Mapping) -> tuple[str, ...]:
    return tuple(sorted(k for k in set(before) | set(after)
                        if k not in before or k not in after or canonical(before[k]) != canonical(after[k])))


@dataclass(frozen=True)
class RetryRequest:
    attempt_id: str
    parent_original_attempt: str
    scope: Scope
    sdk_tool_call_id: str
    tool_name: str
    arguments: Mapping
    origin: str = 'repair_retry'

    def __post_init__(self):
        object.__setattr__(self, 'arguments', freeze(self.arguments))


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    raw: str
    execution_state: str = 'COMPLETED'

    def __post_init__(self):
        if self.status not in {'SUCCESS', 'SUCCESS_EMPTY', 'ERROR', 'TIMEOUT', 'CANCELLED'} or not isinstance(self.raw, str):
            raise ValueError('invalid execution result')
        if self.execution_state not in {'COMPLETED', 'UNKNOWN'}:
            raise ValueError('invalid retry completion state')
        if self.execution_state == 'UNKNOWN' and self.status in {'SUCCESS', 'SUCCESS_EMPTY'}:
            raise ValueError('unknown completion is not success')


@dataclass(frozen=True)
class ResultRef:
    result_id: str
    attempt_id: str
    status: str
    first_ref: str
    persistent_ref: str


@dataclass(frozen=True)
class RepairOutcome:
    decision: str
    reason: str
    delivery: Mapping
    retry: ResultRef | None = None
    repair_event_id: str | None = None
    audit_failed: bool = False
