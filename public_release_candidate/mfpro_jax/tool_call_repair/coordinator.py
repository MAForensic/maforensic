"""Local serial coordination with one shared retry and immutable parent identity."""
from __future__ import annotations
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from .contracts import RepairOutcome, RetryRequest, ExecutionResult, Proposal, changed_fields, canonical, digest
from .adapter import valid_delivery
from .classifier import classify
from .policy import budget_reason, SignatureLedger
from .validator import CandidateValidator, proposal_basis
from .context import ContextBuilder, ContextBudgetError
from .audit import RepairEvent, RepairMetrics
from . import rules


@dataclass
class _QuestionState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    spent: float = 0
    llm_calls: int = 0
    handled: set = field(default_factory=set)
    outcomes: dict = field(default_factory=dict)
    sdk_originals: dict = field(default_factory=dict)


class _DispatchRejected(Exception):
    pass


class RepairCoordinator:
    """Own one instance per run. All dependencies are injected, no construction I/O.

    Gate 1 fake finalizer/execution implement the exact contracts Gate 3 must bind.
    Unknown execution/cancel never schedules a second attempt. Off returns the
    identical supplied delivery without reading evidence, finalizer or advisor.
    """
    def __init__(self, config, *, effects, evidence, execution, finalizer, audit,
                 advisor_factory=None, ledger=None, clock=time.monotonic):
        self.config, self.effects, self.evidence = config, effects, evidence
        self.execution, self.finalizer, self.audit = execution, finalizer, audit
        self.advisor_factory, self.clock = advisor_factory, clock
        self.ledger = ledger if ledger is not None else SignatureLedger()
        self.metrics = RepairMetrics()
        self._questions = {}

    def _emit(self, a, eid, kind, reason, **details):
        try:
            self.audit.emit(RepairEvent(eid, kind, a.attempt_id, reason, details))
            return True
        except Exception:
            self.metrics.counts['audit_failures'] += 1
            return False

    async def handle(self, a, original_delivery):
        cfg = self.config
        if cfg.mode == 'off':
            return RepairOutcome('NOT_TRIGGERED', 'OFF', original_delivery)
        if a.status == 'CANCELLED':
            self.metrics.counts['cancelled'] += 1
            self._emit(a, 'cancel:' + a.attempt_id, 'ABSTAIN', 'ABSTAIN_EXECUTION_UNKNOWN', usage_status='CANCELLED')
            raise asyncio.CancelledError
        key = (a.scope.run_id, a.scope.question_id)
        state = self._questions.setdefault(key, _QuestionState())
        async with state.lock:
            sdk_key = (a.batch_id, a.sdk_tool_call_id)
            identity = (a.attempt_id, a.arguments_hash, a.original_result_ref, a.scope)
            if sdk_key in state.sdk_originals and state.sdk_originals[sdk_key] != identity:
                self.metrics.counts['reason:ORIGINAL_IDENTITY_CONFLICT'] += 1
                self._emit(a, 'identity:' + a.attempt_id, 'ABSTAIN', 'ORIGINAL_IDENTITY_CONFLICT')
                return RepairOutcome('ABSTAIN_UNSAFE', 'ORIGINAL_IDENTITY_CONFLICT', original_delivery)
            state.sdk_originals[sdk_key] = identity
            if a.key in state.outcomes:
                return state.outcomes[a.key]
            if a.key in state.handled:
                return RepairOutcome('ABSTAIN_UNSAFE', 'ALREADY_HANDLED', original_delivery)
            state.handled.add(a.key)
            start, cpu = self.clock(), time.process_time()
            eid = 'repair:' + uuid.uuid4().hex
            audit_ok = True
            advisor, advisor_requests_before, provider_attempts_before = None, 0, 0
            advisor_completed = False
            retry_started = None
            def stop(reason, decision='ABSTAIN_NEEDS_EVIDENCE', retry=None):
                nonlocal audit_ok
                self.metrics.counts[decision] += 1
                self.metrics.counts['reason:' + reason] += 1
                audit_ok = self._emit(a, eid, 'DECISION', reason, decision=decision) and audit_ok
                delivery = original_delivery
                if retry_started is not None:
                    delivery = {'role': 'tool', 'tool_call_id': a.sdk_tool_call_id,
                                'content': canonical({'original_response': dict(original_delivery),
                                  'repair_notice': {'event_id': eid, 'attempt_id': retry_started,
                                                    'reason': reason, 'completion': 'SEE_RECORDED_RESULT_OR_UNKNOWN',
                                                    'retry_result_ref': retry.result_id if retry else None}})}
                result = RepairOutcome(decision, reason, delivery, retry, eid, not audit_ok)
                state.outcomes[a.key] = result
                return result
            try:
                c = classify(a)
                self.metrics.counts['classification:' + c.decision] += 1
                if not c.candidate:
                    return stop(c.reason, c.decision)
                if not self._emit(a, eid, 'CLASSIFIED', c.reason):
                    audit_ok = False
                    return stop('AUDIT_UNAVAILABLE', 'ABSTAIN_UNSAFE')
                if cfg.max_retries_per_call == 0:
                    return stop('RETRY_DISABLED', 'NOT_TRIGGERED')
                if not a.scope.verified:
                    return stop('UNVERIFIED_SCOPE', 'ABSTAIN_UNSAFE')
                if not self.effects.allows(a.tool):
                    return stop('UNSAFE_EFFECT', 'ABSTAIN_UNSAFE')
                try:
                    r = self.finalizer.original_receipt(a)
                    ready = (r.stored is True and r.views_saved is True and r.observation_complete is True
                             and r.result.result_id == a.original_result_ref
                             and r.result.attempt_id == a.attempt_id and r.result.status == a.status
                             and r.scope == a.scope and r.arguments_hash == a.expected_stored_arguments_hash
                             and bool(r.result.first_ref) and bool(r.result.persistent_ref))
                except Exception:
                    ready = False
                if not ready:
                    return stop('ORIGINAL_NOT_FINALIZED', 'ABSTAIN_UNSAFE')
                if self.execution.snapshot() != a.scope:
                    return stop('SCOPE_CHANGED', 'ABSTAIN_UNSAFE')
                reason = budget_reason(a, cfg, self.clock(), state.spent + self.clock() - start)
                if reason:
                    return stop(reason, 'ABSTAIN_UNSAFE')
                facts = self.evidence.select_prior(a)
                rule = rules.propose(a, facts)
                basis = proposal_basis(a, facts)
                if not basis:
                    return stop('NO_UNIQUE_GROUNDED_CANDIDATE')
                proposal = None
                if cfg.mode in {'rules', 'hybrid'} and rule:
                    proposal = rule
                    self.metrics.counts['rule_candidates'] += 1
                    self.metrics.counts['candidate_proposals'] += 1
                elif cfg.mode == 'rules':
                    return stop('NO_RULE')
                else:
                    if cfg.max_llm_calls_per_failure == 0 or state.llm_calls >= cfg.max_llm_calls_per_question:
                        return stop('LLM_CALL_BUDGET', 'ABSTAIN_UNSAFE')
                    if self.advisor_factory is None:
                        return stop('ADVISOR_UNAVAILABLE')
                    reason = budget_reason(a, cfg, self.clock(), state.spent + self.clock() - start,
                                           cfg.llm_timeout_seconds)
                    if reason:
                        return stop(reason, 'ABSTAIN_UNSAFE')
                    preflight = rule or Proposal('REPAIR_ONCE', a.tool.name, basis[0],
                        changed_fields(a.arguments or {}, basis[0]), tuple(basis[1]),
                        'GROUNDED_IDENTIFIER_MAPPING', 'preflight')
                    precheck = CandidateValidator(self.effects, self.ledger).validate(
                        a, preflight, facts, self.execution.snapshot(), self.execution.guard)
                    if not precheck.allowed:
                        self.metrics.counts['validation_rejects'] += 1
                        return stop(precheck.reason, 'ABSTAIN_UNSAFE')
                    refs = set(basis[1])
                    selected = tuple(f for f in facts if f.ref in refs)
                    level = 'C0' if not selected else 'C2' if not rule else 'C1'
                    try:
                        packet = ContextBuilder(cfg).build(a, selected, level)
                    except ContextBudgetError:
                        return stop('CONTEXT_BUDGET_EXCEEDED')
                    if not self._emit(a, eid, 'LLM_INTENT', 'GROUNDED_CANDIDATE', context_hash=packet.sha256,
                                      context_chars=packet.chars, evidence_refs=list(packet.evidence_refs)):
                        audit_ok = False
                        return stop('AUDIT_UNAVAILABLE', 'ABSTAIN_UNSAFE')
                    state.llm_calls += 1  # Reserve before factory/request; errors do not refund.
                    self.metrics.counts['advisor_calls'] += 1
                    self.metrics.context_chars.append(packet.chars)
                    advisor = self.advisor_factory()
                    advisor_requests_before = advisor.request_count
                    provider_attempts_before = advisor.provider_attempt_count
                    d = min(a.deadline - cfg.min_main_reserve_seconds - a.tool.timeout_seconds,
                            self.clock() + cfg.llm_timeout_seconds,
                            start + cfg.max_total_repair_seconds_per_question - state.spent - a.tool.timeout_seconds)
                    ans = await advisor.propose(packet, d)
                    advisor_completed = True
                    self.metrics.usage.append({'role': 'repair', 'usage': dict(ans.usage), 'status': ans.usage_status})
                    if not self._emit(a, eid, 'LLM_RESULT', ans.reason, usage=dict(ans.usage), usage_status=ans.usage_status,
                                      request_sent=ans.request_sent, wall_seconds=ans.wall_seconds):
                        audit_ok = False
                        return stop('AUDIT_UNAVAILABLE', 'ABSTAIN_UNSAFE')
                    if ans.proposal is None:
                        return stop(ans.reason)
                    proposal = ans.proposal
                    if proposal.decision != 'REPAIR_ONCE':
                        return stop(proposal.reason_code, proposal.decision)
                    self.metrics.counts['candidate_proposals'] += 1
                v = CandidateValidator(self.effects, self.ledger).validate(
                    a, proposal, facts, self.execution.snapshot(), self.execution.guard)
                if not v.allowed:
                    self.metrics.counts['validation_rejects'] += 1
                    return stop(v.reason, 'ABSTAIN_UNSAFE')
                reason = budget_reason(a, cfg, self.clock(), state.spent + self.clock() - start)
                if reason:
                    return stop(reason, 'ABSTAIN_UNSAFE')
                # All live guards run again immediately before the sole execution.
                v = CandidateValidator(self.effects, self.ledger).validate(
                    a, proposal, facts, self.execution.snapshot(), self.execution.guard)
                if not v.allowed:
                    self.metrics.counts['validation_rejects'] += 1
                    return stop(v.reason, 'ABSTAIN_UNSAFE')
                request = RetryRequest('retry:' + uuid.uuid4().hex, a.attempt_id, a.scope,
                                       a.sdk_tool_call_id, a.tool.name, proposal.candidate_arguments)
                if not self._emit(a, eid, 'RETRY_INTENT', proposal.reason_code, origin='repair_retry',
                                  attempt_id=request.attempt_id, parent_original_attempt=a.attempt_id,
                                  changed_fields=list(proposal.changed_fields), evidence_refs=list(proposal.evidence_refs)):
                    audit_ok = False
                    return stop('AUDIT_UNAVAILABLE', 'ABSTAIN_UNSAFE')
                # Synchronous guards/audit can consume time or expose changed state.
                # This final gate follows the durable intent and immediately precedes dispatch.
                v = CandidateValidator(self.effects, self.ledger).validate(
                    a, proposal, facts, self.execution.snapshot(), self.execution.guard)
                if not v.allowed:
                    self.metrics.counts['validation_rejects'] += 1
                    return stop(v.reason, 'ABSTAIN_UNSAFE')
                if self.execution.snapshot() != a.scope:
                    return stop('SCOPE_CHANGED', 'ABSTAIN_UNSAFE')
                reason = budget_reason(a, cfg, self.clock(), state.spent + self.clock() - start)
                if reason:
                    return stop(reason, 'ABSTAIN_UNSAFE')
                async def dispatch():
                    nonlocal retry_started
                    # wait_for schedules a child task: recheck AFTER that scheduling
                    # boundary, not only before it. The injected adapter must keep
                    # its own validation-to-Registry execution boundary atomic too.
                    v = CandidateValidator(self.effects, self.ledger).validate(
                        a, proposal, facts, self.execution.snapshot(), self.execution.guard)
                    if not v.allowed:
                        raise _DispatchRejected(v.reason)
                    reason = budget_reason(a, cfg, self.clock(), state.spent + self.clock() - start)
                    if reason:
                        raise _DispatchRejected(reason)
                    if self.execution.snapshot() != a.scope:
                        raise _DispatchRejected('SCOPE_CHANGED')
                    self.ledger.mark(a, proposal.candidate_arguments)  # Never refund claims.
                    self.metrics.counts['safe_once_authorized'] += 1
                    self.metrics.counts['retry_executor_calls'] += 1
                    retry_started = request.attempt_id
                    return await self.execution.execute(request)
                try:
                    # Cooperative timeout only: synchronous SQLite/file sections
                    # cannot be promised an immediate asyncio interruption.
                    executed = await asyncio.wait_for(dispatch(), a.tool.timeout_seconds)
                except _DispatchRejected as exc:
                    self.metrics.counts['validation_rejects'] += 1
                    return stop(str(exc), 'ABSTAIN_UNSAFE')
                except asyncio.CancelledError:
                    if retry_started is None:
                        self.metrics.counts['cancelled_before_dispatch'] += 1
                        self._emit(a, eid, 'RETRY_NOT_DISPATCHED', 'CANCELLED_BEFORE_DISPATCH')
                    else:
                        self.metrics.counts['retry_cancelled'] += 1
                        self._emit(a, eid, 'RETRY_UNKNOWN', 'ABSTAIN_EXECUTION_UNKNOWN', attempt_id=request.attempt_id)
                        try:
                            self.finalizer.finalize_retry(request, ExecutionResult('CANCELLED', 'Retry cancelled; completion unknown', 'UNKNOWN'))
                        except Exception:
                            self.metrics.counts['retry_store_failures'] += 1
                    raise
                except (TimeoutError, asyncio.TimeoutError):
                    executed = ExecutionResult('TIMEOUT', 'Retry completion unknown', 'UNKNOWN')
                except Exception:
                    executed = ExecutionResult('ERROR', 'Retry completion unknown', 'UNKNOWN')
                if retry_started is None:
                    return stop('RETRY_NOT_DISPATCHED', 'ABSTAIN_UNSAFE')
                try:
                    retry = self.finalizer.finalize_retry(request, executed)
                    receipt = self.finalizer.retry_receipt(request)
                    if (not retry.result_id or retry.result_id == a.original_result_ref
                        or retry.attempt_id != request.attempt_id or retry.status != executed.status
                        or not retry.first_ref or not retry.persistent_ref):
                        raise ValueError('retry identity collision')
                    if (receipt.stored is not True or receipt.views_saved is not True
                        or receipt.observation_complete is not True or receipt.result != retry
                        or receipt.scope != request.scope or receipt.arguments_hash != digest(request.arguments)):
                        raise ValueError('retry not fully finalized')
                except Exception:
                    self.metrics.counts['retry_store_failures'] += 1
                    return stop('RETRY_STORE_FAILED_NO_REEXECUTION', 'ABSTAIN_UNSAFE')
                self.metrics.counts['retry_status:' + executed.status] += 1
                audit_ok = self._emit(a, eid, 'RETRY_RESULT', executed.status, attempt_id=request.attempt_id,
                                      result_id=retry.result_id, execution_state=executed.execution_state) and audit_ok
                try:
                    delivery = self.finalizer.compose(a, retry, eid, proposal.reason_code)
                    if not valid_delivery(delivery, a, retry, eid, r.result):
                        raise ValueError('SDK delivery identity')
                except Exception:
                    return stop('DELIVERY_FAILED_NO_REEXECUTION', 'ABSTAIN_UNSAFE', retry)
                outcome = RepairOutcome('RETRY_RECORDED', 'RETRY_' + executed.status, delivery, retry, eid, not audit_ok)
                state.outcomes[a.key] = outcome
                self.metrics.counts['RETRY_RECORDED'] += 1
                return outcome
            except asyncio.CancelledError:
                self.metrics.counts['cancelled'] += 1
                if advisor is not None and not advisor_completed and advisor.request_count > advisor_requests_before:
                    self.metrics.usage.append({'role': 'repair', 'usage': None, 'status': 'CANCELLED'})
                self._emit(a, eid, 'CANCELLED', 'ABSTAIN_EXECUTION_UNKNOWN', usage_status='CANCELLED')
                raise
            except Exception:
                return stop('REPAIR_INTERNAL_FAILURE', 'ABSTAIN_UNSAFE')
            finally:
                if advisor is not None:
                    self.metrics.counts['llm_requests'] += advisor.request_count - advisor_requests_before
                    self.metrics.counts['provider_dispatch_attempts'] += advisor.provider_attempt_count - provider_attempts_before
                elapsed = max(0, self.clock() - start)
                state.spent += elapsed
                self.metrics.wall_seconds += elapsed
                self.metrics.cpu_seconds += max(0, time.process_time() - cpu)

    async def handle_batch(self, items):
        items = tuple(items)
        if len({a.sdk_tool_call_id for a, _ in items}) != len(items):
            raise ValueError('duplicate SDK call ID in batch')
        if items and (len({(a.scope.run_id, a.scope.question_id, a.batch_id) for a, _ in items}) != 1
                      or [a.batch_index for a, _ in items] != sorted({a.batch_index for a, _ in items})):
            raise ValueError('batch identity/order mismatch')
        return tuple([await self.handle(a, delivery) for a, delivery in items])
