"""Classify trusted phase/source pairs, never arbitrary Error text."""
from dataclasses import dataclass
from .contracts import OriginalAttempt, strict_json


@dataclass(frozen=True)
class Classification:
    decision: str
    reason: str
    candidate: bool = False


def classify(a: OriginalAttempt) -> Classification:
    f = a.failure
    if a.status == 'CANCELLED' or a.execution_state == 'UNKNOWN' or a.status in {'TIMEOUT', 'TIMED_OUT'}:
        return Classification('ABSTAIN_UNSAFE', 'ABSTAIN_EXECUTION_UNKNOWN')
    if f.stage == 'POLICY_OR_PATH_GUARD' or f.code in {'PERMISSION_DENIED', 'PATH_DENIED', 'SQL_POLICY', 'PRAGMA_DENIED', 'REPEAT_SUPPRESSED'}:
        return Classification('ABSTAIN_UNSAFE', f.code)
    if a.status in {'SUCCESS', 'SUCCESS_EMPTY'}:
        return Classification('NOT_TRIGGERED', 'NORMAL_RESULT')
    if a.origin != 'B':
        return Classification('NOT_TRIGGERED', 'NO_RECURSIVE_REPAIR')
    if f.origin in {'provider', 'network', 'processor', 'environment', 'internal'}:
        return Classification('NON_PARAMETER_FAILURE', 'NON_PARAMETER_FAILURE')
    if a.tool.name == 'recall_tool_result' or a.status == 'UNKNOWN_TOOL' or not a.tool.exposed or not a.tool.resolved:
        return Classification('NOT_TRIGGERED', 'EXISTING_JIT_OR_RECOVERY')
    if not f.trusted:
        return Classification('ABSTAIN_NEEDS_EVIDENCE', 'UNTRUSTED_ERROR_SOURCE')
    accepted = {
        ('JSON_PARSE', 'json_parser'): {'PARAM_JSON_INVALID'},
        ('SCHEMA_VALIDATE', 'schema_validator'): {'PARAM_MISSING_TYPE', 'PARAM_ENUM', 'PARAM_UNKNOWN_KEY'},
        ('EXECUTOR_TERMINAL', 'sqlite_executor'): {'SQL_NO_SUCH_TABLE', 'SQL_NO_SUCH_COLUMN', 'SQL_SYNTAX'},
        ('EXECUTOR_TERMINAL', 'tool_parameter_contract'): {'BAD_REFERENCE', 'PARAM_BINDING_ERROR', 'PARAM_MISSING_TYPE'},
    }
    if f.code not in accepted.get((f.stage, f.origin), set()):
        return Classification('NON_PARAMETER_FAILURE', 'UNVERIFIED_PARAMETER_CAUSE')
    if a.status not in {'ERROR', 'FAILED', 'INVALID_ARGUMENTS', 'INVALID_JSON'}:
        return Classification('ABSTAIN_NEEDS_EVIDENCE', 'UNVERIFIED_FAILURE_STATUS')
    if f.stage == 'JSON_PARSE':
        try:
            strict_json(a.raw_arguments)
        except (ValueError, TypeError):
            pass
        else:
            return Classification('ABSTAIN_NEEDS_EVIDENCE', 'JSON_FAILURE_WIRE_MISMATCH')
        if a.arguments is not None:
            return Classification('ABSTAIN_NEEDS_EVIDENCE', 'JSON_FAILURE_WIRE_MISMATCH')
    elif a.arguments is None:
        return Classification('ABSTAIN_NEEDS_EVIDENCE', 'PARSED_ARGUMENTS_MISSING')
    expected = 'COMPLETED' if f.stage == 'EXECUTOR_TERMINAL' else 'NOT_STARTED'
    if a.execution_state != expected:
        return Classification('ABSTAIN_UNSAFE', 'PHASE_EXECUTION_MISMATCH')
    return Classification('PARAMETER_CANDIDATE', f.code, True)
