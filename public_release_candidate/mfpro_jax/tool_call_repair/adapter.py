"""Gate 3 boundary protocols and immutable snapshot adapter, no core imports."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from .contracts import (OriginalAttempt, strict_json, canonical, Scope,
                        RetryRequest, ExecutionResult, ResultRef, digest, ToolContract)


@dataclass(frozen=True)
class FinalizationReceipt:
    result: ResultRef
    scope: Scope
    arguments_hash: str
    stored: bool
    views_saved: bool
    observation_complete: bool


class ExecutionAdapter(Protocol):
    def snapshot(self) -> Scope: ...
    def guard(self, original: OriginalAttempt, candidate) -> object: ...
    async def execute(self, request: RetryRequest) -> ExecutionResult: ...


class AttemptFinalizer(Protocol):
    def original_receipt(self, original: OriginalAttempt) -> FinalizationReceipt: ...
    def finalize_retry(self, request: RetryRequest, result: ExecutionResult) -> ResultRef: ...
    def retry_receipt(self, request: RetryRequest) -> FinalizationReceipt: ...
    def compose(self, original: OriginalAttempt, retry: ResultRef, repair_event_id: str,
                reason: str, mode: str = 'FIRST') -> dict: ...


def from_snapshot(**kwargs) -> OriginalAttempt:
    """Caller provides authoritative phase/error/scope facts; generic ERROR is not enough.

    Reject mismatched parsed/wire args; malformed wire data stays None for JSON_PARSE.
    This adapter never opens files, recovers tools or captures hidden reasoning.
    """
    raw = kwargs['raw_arguments']
    try:
        parsed = strict_json(raw)
    except (ValueError, TypeError):
        parsed = None
        if kwargs['failure'].stage != 'JSON_PARSE':
            raise ValueError('malformed JSON must be explicitly classified')
    if kwargs.get('arguments') is not None and canonical(parsed) != canonical(kwargs['arguments']):
        raise ValueError('wire/parsed arguments mismatch')
    if parsed is not None and not isinstance(parsed, dict):
        raise ValueError('arguments JSON must be object')
    kwargs['arguments'] = parsed
    return OriginalAttempt(**kwargs)


class Profile0GuardAdapter:
    """Calls injected CURRENT pure guards; never invokes a tool executor.

    Gate 3 supplies _validate_tool_arguments, _normalize_and_validate_paths (bound
    to RuntimeContext), _validate_select_query, authorization and duplicate guards.
    Missing guards fail closed. Gate 1 verifies these calls with local validators.
    """
    def __init__(self, *, snapshot, schema_guard, path_guard, sql_guard,
                 permission_guard, duplicate_guard, contract_snapshot=None):
        self.snapshot = snapshot
        self.schema_guard, self.path_guard, self.sql_guard = schema_guard, path_guard, sql_guard
        self.permission_guard, self.duplicate_guard = permission_guard, duplicate_guard
        self.contract_snapshot = contract_snapshot

    def __call__(self, a, candidate):
        from .policy import GuardReceipt
        from .contracts import plain
        checks = set()
        scope, fingerprint = a.scope, ''
        try:
            scope = self.snapshot()
            live = self.contract_snapshot() if self.contract_snapshot else None
            if not isinstance(live, ToolContract) or digest(live) != digest(a.tool):
                raise ValueError('current contract unavailable or changed')
            fingerprint = digest(live)
            if self.schema_guard(plain(candidate), plain(live.schema)) != '':
                raise ValueError('schema rejected')
            checks.add('schema')
            normalized = self.path_guard(plain(candidate))
            if canonical(normalized) != canonical(candidate):
                raise ValueError('path normalization changes proposed binding')
            checks.add('paths')
            if a.tool.semantic_kind == 'sqlite_select' and self.sql_guard(candidate.get('query')) != '':
                raise ValueError('SQL rejected')
            checks.add('sql')
            if self.permission_guard(a, candidate) is not True:
                raise ValueError('permission rejected')
            checks.add('permissions')
            if self.duplicate_guard(a, candidate) is not True:
                raise ValueError('duplicate rejected')
            checks.add('duplicate')
            allowed = (scope == a.scope and self.snapshot() == scope
                       and digest(self.contract_snapshot()) == fingerprint)
        except Exception:
            allowed = False
        return GuardReceipt(allowed, scope, a.tool.version, frozenset(checks),
                            contract_fingerprint=fingerprint)


def valid_delivery(delivery, original, retry, event_id, original_ref):
    """One SDK reply, two explicitly identified real components; never free text success."""
    try:
        if (set(delivery) != {'role', 'tool_call_id', 'content'} or delivery['role'] != 'tool'
            or delivery['tool_call_id'] != original.sdk_tool_call_id):
            return False
        body = strict_json(delivery['content'])
        if body['repair']['event_id'] != event_id:
            return False
        if body['active_result_ids'] != [original.original_result_ref, retry.result_id]:
            return False
        for component, rid, status in [('original', original.original_result_ref, original.status),
                                       ('retry', retry.result_id, retry.status)]:
            item = body[component]
            if (item['result_id'] != rid or item['status'] != status or
                not isinstance(item['view'], str) or not item.get('first_ref') or not item.get('persistent_ref')):
                return False
        return (body['original']['first_ref'] == original_ref.first_ref and
                body['original']['persistent_ref'] == original_ref.persistent_ref and
                body['retry']['first_ref'] == retry.first_ref and
                body['retry']['persistent_ref'] == retry.persistent_ref)
    except (KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ConfigResolution:
    config: object
    accepted: bool
    reason: str


def resolve_optional_config(config):
    """Reject invalid repair settings without making an existing off business fail."""
    from .contracts import RepairConfig
    try:
        return ConfigResolution(RepairConfig.from_config(config), True, 'ACCEPTED')
    except (ValueError, TypeError):
        return ConfigResolution(RepairConfig(), False, 'INVALID_REPAIR_CONFIG_DISABLED')
