"""Explicit effect contracts, budgets and shared execution guards."""
from dataclasses import dataclass
from collections.abc import Mapping
import math
from .contracts import Scope, OriginalAttempt, ToolContract, RepairConfig, digest


@dataclass(frozen=True)
class EffectContract:
    tool_name: str
    contract_version: str
    safety: str
    proof_ref: str
    effects: tuple[str, ...] = ()
    # An explicit safe branch can authorize a documented effect; never inferred.
    safe_branch_attested: bool = False
    contract_fingerprint: str = ''

    def __post_init__(self):
        if (type(self.safe_branch_attested) is not bool or not isinstance(self.proof_ref, str)
            or not isinstance(self.contract_fingerprint, str)):
            raise ValueError('effect branch attestation must be boolean')
        object.__setattr__(self, 'effects', tuple(self.effects))


class EffectsRegistry:
    def __init__(self, entries=()):
        self._entries = {}
        for e in entries:
            key = (e.tool_name, e.contract_version)
            if key in self._entries or e.safety not in {'SAFE_ONCE', 'UNSAFE', 'UNKNOWN'}:
                raise ValueError('invalid or duplicate effect contract')
            self._entries[key] = e

    def allows(self, tool: ToolContract) -> bool:
        e = self._entries.get((tool.name, tool.version))
        return bool(e and e.safety == 'SAFE_ONCE' and e.proof_ref and
                    (not e.contract_fingerprint or e.contract_fingerprint == digest(tool)) and
                    (not e.effects or e.safe_branch_attested))

    @classmethod
    def reviewed_host_reads(cls, tools):
        """Opt-in versioned entries; no Registry import or implicit registration.

        Current-source read review only; production guards still mandatory. Excludes
        indexed readers, exports, index creation, external tools and recursive scans.
        """
        reviewed = {'sqlite_query': 'tools/structured.py:_readonly_connection,sqlite_query',
                    'search_file_strings': 'tools/readers.py:search_file_strings',
                    'normalize_timestamp': 'tools/media_time.py:normalize_timestamp'}
        return cls(EffectContract(t.name, t.version, 'SAFE_ONCE', reviewed[t.name],
                                  contract_fingerprint=digest(t))
                   for t in tools if t.name in reviewed)


@dataclass(frozen=True)
class GuardReceipt:
    allowed: bool
    scope: Scope
    contract_version: str
    checks: frozenset[str]
    reason: str = ''
    contract_fingerprint: str = ''


REQUIRED_GUARDS = frozenset({'schema', 'paths', 'sql', 'permissions', 'duplicate'})


def signature(a: OriginalAttempt, arguments: Mapping) -> str:
    return digest({'tool': a.tool.name, 'scope': a.scope, 'arguments': arguments})


class SignatureLedger:
    """Mirrors upstream terminal state without erasing failed original signatures."""
    def __init__(self, terminal=()):
        self.blocked = set(terminal)

    def is_blocked(self, a, candidate):
        # The original failed signature remains blocked. A changed candidate may
        # execute once; an identical or previously failed candidate cannot.
        return signature(a, candidate) == signature(a, a.arguments or {}) or signature(a, candidate) in self.blocked

    def mark(self, a, candidate):
        self.blocked.add(signature(a, candidate))


def budget_reason(a, cfg: RepairConfig, now, spent, llm_seconds=0):
    if not math.isfinite(now) or not math.isfinite(spent) or spent < 0:
        return 'INVALID_CLOCK'
    required = a.tool.timeout_seconds + llm_seconds
    if a.deadline - now < cfg.min_main_reserve_seconds + required:
        return 'DEADLINE_RESERVE'
    if cfg.max_total_repair_seconds_per_question - spent < required:
        return 'REPAIR_TIME_BUDGET'
    return ''
