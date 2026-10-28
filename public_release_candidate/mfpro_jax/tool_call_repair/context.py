"""Bounded C0/C1/C2 data serialization; no entire history or graph access."""
from dataclasses import dataclass
from .contracts import canonical, digest, plain
from .advisor import SYSTEM_RULES, ModelRequest


class ContextBudgetError(ValueError):
    pass


@dataclass(frozen=True)
class ContextPacket:
    text: str
    chars: int
    sha256: str
    evidence_refs: tuple[str, ...]
    level: str
    data_chars: int = 0


class ContextBuilder:
    def __init__(self, config):
        self.config = config

    def build(self, a, facts=(), level='C0'):
        cfg = self.config
        facts = tuple(facts)
        if len({f.ref for f in facts}) != len(facts):
            raise ContextBudgetError('duplicate evidence identity')
        if level not in {'C0', 'C1', 'C2'}:
            raise ValueError('invalid context level')
        data = {
            'protocol': 'tool-call-repair-v1',
            'error_event_id': a.attempt_id,
            'tool_name': a.tool.name,
            'frozen_scope': plain(a.scope),
            'tool_contract': {'version': a.tool.version, 'schema': plain(a.tool.schema),
                              'aliases': plain(a.tool.aliases), 'alias_proof_ref': a.tool.alias_proof_ref,
                              'implementation_ref': a.tool.implementation_ref,
                              'semantic_kind': a.tool.semantic_kind,
                              'repairable_fields': list(a.tool.repairable_fields)},
            'original_call': {'sdk_tool_call_id': a.sdk_tool_call_id, 'arguments': plain(a.arguments),
                              'raw_arguments': a.raw_arguments},
            'failure': {'stage': a.failure.stage, 'origin': a.failure.origin,
                        'code': a.failure.code, 'message': a.failure.message[:cfg.max_error_chars],
                        'message_hash': digest(a.failure.message), 'message_ref': a.failure.message_ref},
            'local_intent': a.local_intent,
            'constraints': {'same_tool_only': True, 'same_scope_target': True,
                            'no_new_tool': True, 'max_attempts': 1},
            'evidence': [],
        }
        if level == 'C0' and facts:
            raise ContextBudgetError('C0 cannot hide required evidence')
        if len(facts) > cfg.max_evidence_refs:
            raise ContextBudgetError('too many required references')
        for fact in facts:
            if (fact.scope != a.scope or fact.created_seq >= a.batch_emitted_seq
                or fact.created_seq >= a.failure_seq or fact.trusted is not True or not fact.authority_ref):
                raise ContextBudgetError('evidence outside frozen scope/time')
            item = {'ref': fact.ref, 'kind': fact.kind, 'payload': plain(fact.payload)}
            if level == 'C2':
                item.update(scope=plain(fact.scope), created_seq=fact.created_seq,
                            authority_ref=fact.authority_ref, b_presented=fact.b_presented,
                            graph_node_id=fact.graph_node_id, available_before_batch_emission=True,
                            field_proof_hash=digest(fact.payload))
            if len(canonical(item)) > cfg.max_chars_per_evidence:
                raise ContextBudgetError('required field proof too large')
            data['evidence'].append(item)
        text = canonical(data)
        cost = len(canonical(ModelRequest(SYSTEM_RULES, text, cfg.max_output_tokens,
                    cfg.llm_timeout_seconds, cfg.provider_ref, max_context_chars=cfg.max_context_chars)))
        if cost > cfg.max_context_chars:
            raise ContextBudgetError('serialized total context exceeds budget')
        return ContextPacket(text, cost, digest(data), tuple(f.ref for f in facts), level, len(text))
