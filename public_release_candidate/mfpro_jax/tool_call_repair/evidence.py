"""Read-only selection from caller-authorized stored facts; no I/O or Recovery."""
from .contracts import EvidenceFact, OriginalAttempt


class EvidenceReader:
    def __init__(self, facts=()):
        facts = tuple(facts)
        if len({f.ref for f in facts}) != len(facts):
            raise ValueError('duplicate evidence refs')
        self._facts = facts

    def select_prior(self, a: OriginalAttempt) -> tuple[EvidenceFact, ...]:
        return tuple(f for f in self._facts
                     if f.trusted and f.authority_ref and f.scope == a.scope
                     and f.created_seq < a.batch_emitted_seq
                     and f.created_seq < a.failure_seq
                     and f.kind in {'EXPORTED_HOST', 'ARGUMENT_BINDING', 'SQL_SCHEMA',
                                    'SQL_IDENTIFIER_MAPPING', 'JSON_CANONICAL'})
