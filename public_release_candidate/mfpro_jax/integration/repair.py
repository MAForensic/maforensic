"""Gate 3 native-loop bridge; unsupported runtime permissions fail closed.

Only a lossless sqlite sql->query alias on an absolute active TREE host object is
admitted. Image exports and Hpost identifier mappings remain unverified here.
"""
from __future__ import annotations
import asyncio
import inspect
import json
import math
import os
from pathlib import Path
import time
from ..tool_call_repair import (RepairCoordinator, EffectsRegistry, EvidenceReader,
    Profile0GuardAdapter, from_snapshot, resolve_optional_config, Scope, ToolContract, Failure)
from ..tool_call_repair.contracts import ResultRef, ExecutionResult, digest, canonical, plain
from ..tool_call_repair.adapter import FinalizationReceipt
from ..tool_call_repair.audit import RepairEvent
from ..tool_call_repair.validator import sql_safety
from ..representation_runtime import RepresentationMode
from .retry_boundary import retry_boundary_guard


def resolve_repair_runtime(ctx, registry, llm, question_id, started):
    # No bridge, evidence, advisor, cache or extra kwargs in the OFF path.
    raw = ctx.config.get('tool_call_repair', {})
    if isinstance(raw, dict) and raw.get('mode', 'off') == 'off' and not raw:
        return {}
    resolution = resolve_optional_config(ctx.config)
    if not resolution.accepted:
        import warnings
        warnings.warn(resolution.reason, RuntimeWarning, stacklevel=2)
    if resolution.config.mode == 'off':
        return {}
    try:
        seconds = float(ctx.config.get('exam', {}).get('per_question_timeout_sec', 1800))
        deadline = getattr(asyncio.current_task(), '_profile0_question_deadline',
                           (started if started is not None else time.monotonic()) + seconds)
        if not math.isfinite(deadline):
            raise ValueError('unverified question deadline')
        return {'repair_runtime': RepairRuntime(ctx, registry, llm, question_id,
                                                deadline, resolution.config)}
    except Exception as exc:
        import warnings
        warnings.warn('Optional repair unavailable: ' + type(exc).__name__, RuntimeWarning, stacklevel=2)
        return {}


class _Audit:
    def __init__(self, runtime):
        self.runtime = runtime
    def emit(self, event):
        # Dedicated auxiliary log; no private args/error bodies or new Trace types.
        item = {'event_id': event.event_id, 'kind': event.kind,
                'original_attempt': event.original_attempt, 'reason': event.reason,
                'details': plain(event.details)}
        with self.runtime.audit_path.open('a', encoding='utf-8') as stream:
            stream.write(canonical(item) + '\n')
        self.runtime.events.append(item)


class RepairRuntime:
    def __init__(self, ctx, registry, llm, question_id, deadline, config):
        self.ctx, self.registry, self.llm = ctx, registry, llm
        self.question_id, self.deadline, self.config = question_id, deadline, config
        self.events, self.composites, self.receipts, self.components = [], {}, {}, {}
        self.skip_counts = {}
        self.original_ids = set()
        self._handle_lock = asyncio.Lock()
        self._sdk_seen = set()
        self._dispatches = set()
        self._active = None
        self._reserved = None
        self._retry_entered = {}
        self._retry_inputs = {}
        self._retry_provenance = {}
        # Capture reviewed implementation dependencies, not merely a tool name.
        from ..tools import structured
        self._sqlite_functions = tuple(getattr(structured, key) for key in
            ('sqlite_query', '_readonly_connection', '_sqlite_readonly_authorizer', '_execute_readonly_sql'))
        root = ctx.run_dir / 'tool_call_repair'
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        from hashlib import sha256
        scope_key = sha256((ctx.trace.run_id + '\0' + question_id).encode()).hexdigest()[:24]
        self.audit_path = root / (scope_key + '.jsonl')
        self.audit_path.touch(mode=0o600, exist_ok=True)

    def bind(self, *, lifecycle, store, terminal_actions, observe, interactions,
             result_observer, trace):
        self.lifecycle, self.store = lifecycle, store
        self.terminal_actions, self.observe_hook = terminal_actions, observe
        self.interactions, self.result_observer, self.trace = interactions, result_observer, trace
        factory = getattr(self.llm, 'make_repair_advisor_factory', lambda cfg: None)(self.config)
        tool = self.contract('sqlite_query')
        effects = EffectsRegistry.reviewed_host_reads([tool] if tool is not None and tool.semantic_kind == 'sqlite_select' else [])
        self.coordinator = RepairCoordinator(self.config, effects=effects, evidence=EvidenceReader(),
            execution=self, finalizer=self, audit=_Audit(self), advisor_factory=factory)

    def begin_batch(self, iteration, calls):
        self.iteration = iteration
        self.batch_id = f'{self.ctx.trace.run_id}:{self.question_id}:i{iteration}'
        self.batch_seq = len(self.interactions) + 1
        ids = [str(c.get('id', '')) for c in calls]
        self.batch_conflicts = {x for x in ids if not x or ids.count(x) > 1 or x in self._sdk_seen}
        self._sdk_seen.update(ids)

    def contract(self, name):
        d = self.registry.tools.get(name)
        if d is None:
            return None
        from ..tools import structured
        current = tuple(getattr(structured, key) for key in
            ('sqlite_query', '_readonly_connection', '_sqlite_readonly_authorizer', '_execute_readonly_sql'))
        # The Registry wrapper must be the actual bound Registry instrumenter.
        closure = inspect.getclosurevars(d.executor).nonlocals
        registry_owned = (closure.get('definition') is d and closure.get('self') is self.registry
                          and getattr(d.executor, '_profile0_context', None) is self.ctx)
        sqlite = (name == 'sqlite_query' and registry_owned
                  and closure.get('original') is structured.sqlite_query
                  and current == self._sqlite_functions
                  and d.read_only is True and d.deterministic is True)
        implementation = digest({'definition_source': d.source_sha256,
            'executor_identity': id(d.executor), 'sqlite_dependencies': [id(f) for f in current]})
        roots = (str(Path(self.ctx.active_source.path).resolve()),) if self.ctx.active_source else ()
        return ToolContract(name, 'runtime-contract-v1:' + d.source_sha256, d.input_schema,
            semantic_kind='sqlite_select' if sqlite else 'generic',
            aliases={'sql': 'query'} if sqlite else {},
            alias_proof_ref='gate3:lossless-sql-key-alias-v1' if sqlite else '',
            timeout_seconds=d.timeout_sec, allowed_roots=roots,
            implementation_ref=implementation)

    def scope(self, arguments):
        source = self.ctx.active_source
        sid = source.id if source is not None else ''
        state = dict(self.ctx.source_access.get(sid, {}))
        # HOST_TREE is an explicit not-partitioned namespace, not a disk partition ID.
        partition = str(state.get('partition_id') or ('HOST_TREE' if source and source.access_mode == 'tree' else ''))
        metadata = self.ctx.index_metadata.get(sid)
        entries = self.ctx.indexes.get(sid)
        generation = digest({'metadata': metadata, 'entries': entries,
                             'partition': partition})
        path = arguments.get('db_path', '') if isinstance(arguments, dict) else ''
        verified = bool(source and source.access_mode == 'tree' and self.ctx.case.get_source(sid) is source
                        and path and Path(path).is_absolute())
        # No reading Source, stat or access claims here. Actual authorization below.
        return Scope(self.ctx.trace.run_id, self.question_id, sid, partition, generation,
                     str(path), verified=verified)

    def capture(self, call, index, executor, schema):
        try:
            arguments = json.loads(call['arguments'] or '{}')
            scope = self.scope(arguments)
            contract = self.contract(call['name'])
            d = self.registry.tools.get(call['name'])
            if d is None or d.executor is not executor or canonical(schema) != canonical(d.input_schema):
                contract = None
            return (scope, contract)
        except Exception:
            return None

    def observation_complete(self, event):
        graph = getattr(self.observe_hook, '__self__', None)
        if graph is None:
            # TEG OFF still has the bridge's own committed observation receipt.
            return self.observe_hook is None
        try:
            return graph.available and any(n.get('result_id') == event['envelope'].result_id
                and n.get('tool_call_id') == event['tool_call_id'] for n in graph.nodes.values())
        except Exception:
            return False

    def receipt(self, event, attempt_id, scope):
        env = event['envelope']
        first = self.lifecycle.view_store.get(env.result_id, RepresentationMode.FIRST)
        persistent = self.lifecycle.view_store.get(env.result_id, RepresentationMode.PERSISTENT)
        ref = ResultRef(env.result_id, attempt_id, env.status,
                        env.result_id + ':FIRST' if first else '',
                        env.result_id + ':PERSISTENT' if persistent else '')
        self.components[env.result_id] = (env, first, persistent)
        return FinalizationReceipt(ref, scope, env.arguments_sha256,
            self.store.has_result(env.result_id), first is not None and persistent is not None,
            self.observation_complete(event))

    async def handle_result(self, **kwargs):
        async with self._handle_lock:
            return await self._handle_result(**kwargs)

    async def _handle_result(self, *, event, call, index, capture, schema_error,
                             original_delivery):
        if event.get('envelope') is not None:
            self.original_ids.add(event['envelope'].result_id)
        if event.get('status') in {'SUCCESS', 'SUCCESS_EMPTY'}:
            self.skip_counts['NORMAL_RESULT'] = self.skip_counts.get('NORMAL_RESULT', 0) + 1
            return original_delivery
        if capture is None or event.get('envelope') is None:
            reason = ('UNVERIFIED_EXECUTOR_PARAMETER_CAUSE' if event.get('execution_completed')
                      else 'UNVERIFIED_ORIGINAL')
            self.skip_counts[reason] = self.skip_counts.get(reason, 0) + 1
            return original_delivery
        scope, tool = capture
        if tool is None:
            self.skip_counts['UNVERIFIED_CONTRACT'] = self.skip_counts.get('UNVERIFIED_CONTRACT', 0) + 1
            return original_delivery
        if call['id'] in self.batch_conflicts:
            self.events.append({'kind': 'ABSTAIN', 'reason': 'SDK_ID_CONFLICT',
                                'original_result_id': event['envelope'].result_id})
            return original_delivery
        failure = Failure('UNVERIFIED', 'executor', '', '', False)
        if schema_error and not event['execution_started']:
            failure = Failure('SCHEMA_VALIDATE', 'schema_validator', 'PARAM_MISSING_TYPE',
                              schema_error, True, event['envelope'].result_id)
        # Hpost currently supplies text only; do NOT promote generic Error strings.
        attempt_id = 'original:' + event['envelope'].result_id
        state = ('COMPLETED' if event['execution_completed'] else
                 'UNKNOWN' if event['execution_started'] else 'NOT_STARTED')
        try:
            a = from_snapshot(attempt_id=attempt_id, scope=scope, batch_id=self.batch_id,
                batch_index=index, sdk_tool_call_id=call['id'], tool=tool,
                raw_arguments=call['arguments'] or '{}', arguments=event['arguments'],
                status=event['status'], failure=failure, execution_state=state,
                original_result_ref=event['envelope'].result_id,
                batch_emitted_seq=self.batch_seq, failure_seq=len(self.interactions) + 1,
                deadline=self.deadline)
            self.receipts[attempt_id] = self.receipt(event, attempt_id, scope)
            self._active = a
            _Audit(self).emit(RepairEvent(attempt_id, 'ORIGINAL_FINALIZED', attempt_id,
                event['status'], {'result_id': a.original_result_ref,
                                 'ready': self.receipts[attempt_id].observation_complete}))
            outcome = await self.coordinator.handle(a, original_delivery)
            if outcome.retry is not None:
                self.composites[call['id']] = (a, outcome.retry, outcome.repair_event_id, outcome.reason)
            return dict(outcome.delivery)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.events.append({'kind': 'ABSTAIN', 'reason': 'BRIDGE_' + type(exc).__name__,
                                'original_result_id': event['envelope'].result_id})
            return original_delivery
        finally:
            self._active = None

    def snapshot(self):
        return self.scope(plain(self._active.arguments or {}))

    def permission(self, a, candidate):
        # A stricter subset of the actual Registry root policy: same active tree
        # object, exact normalized absolute db_path, no cross-Source/run guessing.
        source = self.ctx.active_source
        if (not source or self.ctx.case.get_source(source.id) is not source or source.access_mode != 'tree'
            or a.scope != self.snapshot() or candidate.get('db_path') != a.scope.target_id):
            return False
        from ..tool_registry import _normalize_and_validate_paths, _within
        path, root = Path(candidate['db_path']), Path(source.path).resolve()
        normalized = _normalize_and_validate_paths(plain(candidate), self.ctx)
        return (_within(path.resolve(), root) and path.is_absolute() and path.is_file() and os.access(path, os.R_OK)
                and str(path.resolve()) == str(path) and canonical(normalized) == canonical(candidate))

    def guard(self, a, candidate):
        from ..llm_client import _validate_tool_arguments, _action_signature
        from ..tool_registry import _normalize_and_validate_paths
        from ..tools.structured import _validate_select_query
        def duplicate(_a, args):
            sig = _action_signature(a.tool.name, plain(args), a.scope.source_id,
                                    self.ctx.source_access.get(a.scope.source_id, {}))
            return sig not in self.terminal_actions or (self._reserved is not None and
                self._reserved == (sig, self.terminal_actions.get(sig)))
        def sql(query):
            return '' if _validate_select_query(query) == '' and sql_safety(query) else 'SQL_POLICY'
        return Profile0GuardAdapter(snapshot=self.snapshot, schema_guard=_validate_tool_arguments,
            path_guard=lambda args: _normalize_and_validate_paths(args, self.ctx), sql_guard=sql,
            permission_guard=self.permission, duplicate_guard=duplicate,
            contract_snapshot=lambda: self.contract(a.tool.name))(a, candidate)

    async def execute(self, request):
        from ..llm_client import (_action_signature, _tool_interaction_status,
            _evidence_runtime_snapshot, _result_provenance_snapshot)
        a = self._active
        if a is None or request.scope != self.snapshot() or request.parent_original_attempt in self._dispatches:
            raise RuntimeError('RETRY_BOUNDARY_REJECTED')
        if self.guard(a, request.arguments).allowed is not True:
            raise RuntimeError('ABSTAIN_UNVERIFIED_GUARD')
        if self.deadline - time.monotonic() < self.config.min_main_reserve_seconds + a.tool.timeout_seconds:
            raise RuntimeError('DEADLINE_RESERVE')
        sig = _action_signature(a.tool.name, plain(request.arguments), a.scope.source_id,
                                self.ctx.source_access.get(a.scope.source_id, {}))
        if sig in self.terminal_actions:
            raise RuntimeError('NATIVE_DUPLICATE')
        reservation = ('repair_reserved', request.attempt_id)
        self.terminal_actions[sig] = reservation
        self._reserved = (sig, reservation)
        self._dispatches.add(request.parent_original_attempt)
        self._retry_entered[request.attempt_id] = False
        self._retry_inputs[request.attempt_id] = _evidence_runtime_snapshot(self.ctx)
        self._retry_provenance[request.attempt_id] = _result_provenance_snapshot(
            self.registry.tools[a.tool.name].executor, plain(request.arguments))
        def boundary(definition, args):
            if (definition is not self.registry.tools.get(a.tool.name)
                or request.scope != self.snapshot() or self.guard(a, args).allowed is not True
                or self.deadline - time.monotonic() < self.config.min_main_reserve_seconds + a.tool.timeout_seconds):
                raise RuntimeError('RETRY_LAST_BOUNDARY_REJECTED')
            self._retry_entered[request.attempt_id] = True
        token = retry_boundary_guard.set(boundary)
        before = len(self.trace.events)
        try:
            result = await self.registry.tools[a.tool.name].executor(**plain(request.arguments))
            terminal = next((e for e in reversed(self.trace.events[before:])
                             if e.tool_name == a.tool.name and e.event_type != 'tool_call_started'), None)
            status = _tool_interaction_status(str(result), terminal.status if terminal else '')
            self.terminal_actions[sig] = (terminal.status if terminal else 'failed', str(result))
            completion = 'COMPLETED' if self._retry_entered[request.attempt_id] and status != 'TIMEOUT' else 'UNKNOWN'
            return ExecutionResult(status, str(result), completion)
        finally:
            retry_boundary_guard.reset(token)
            self._reserved = None
            # Never free a reservation after timeout, cancellation or unknown execution.

    def original_receipt(self, original):
        return self.receipts[original.attempt_id]

    def finalize_retry(self, request, result):
        from ..llm_client import ToolInteraction, _notify_tool_result, _evidence_runtime_snapshot, _result_provenance_snapshot
        # Independent attempt/Result identity; only the outer SDK reply uses original ID.
        env = self.store.store_raw_result(question_id=self.question_id, iteration=self.iteration,
            call_index=self._active.batch_index, tool_call_id=request.attempt_id,
            tool_name=request.tool_name, arguments=plain(request.arguments), status=result.status,
            raw_result=result.raw, provenance={**self._retry_provenance.get(request.attempt_id, {}),
                'origin': 'repair_retry', 'parent_original_attempt': request.parent_original_attempt,
                'parent_original_result_id': self._active.original_result_ref,
                'sdk_tool_call_id': request.sdk_tool_call_id})
        first, persistent = self.lifecycle.materialize_first_persistent(env)
        event = {'iteration': self.iteration, 'call_index': self._active.batch_index,
            'tool_call_id': request.attempt_id, 'tool_name': request.tool_name,
            'arguments': plain(request.arguments), 'status': result.status,
            'execution_started': self._retry_entered.get(request.attempt_id, False),
            'execution_completed': result.execution_state == 'COMPLETED',
            'input_context': self._retry_inputs.get(request.attempt_id, {}),
            'output_context': _evidence_runtime_snapshot(self.ctx),
            'envelope': env, 'first_view': first, 'raw_result': result.raw,
            'ingress_content': first.text, 'error_origin': 'repair_retry', 'trace_event_id': None}
        _notify_tool_result(self.observe_hook, event)
        captured = self.result_observer.capture_raw(trace=self.trace,
            question_id=self.question_id, iteration=self.iteration,
            tool_call_id=request.attempt_id, call_index=self._active.batch_index,
            tool_name=request.tool_name, arguments=plain(request.arguments), status=result.status,
            execution_ms=None, raw_result=result.raw, result_id=env.result_id)
        self.result_observer.capture_ingress(captured=captured, tool_message_content=first.text,
            truncation_limit=len(first.text), raw_result=result.raw)
        self.interactions.append(ToolInteraction(self.iteration, request.tool_name,
            plain(request.arguments), result.status, first.text, request.attempt_id, env.result_id))
        receipt = self.receipt(event, request.attempt_id, request.scope)
        self.receipts[request.attempt_id] = receipt
        return receipt.result

    def retry_receipt(self, request):
        return self.receipts[request.attempt_id]

    def compose(self, original, retry, repair_event_id, reason, mode='FIRST'):
        def component(rid):
            env, first, persistent = self.components[rid]
            view = first if mode == 'FIRST' else persistent
            return {'result_id': rid, 'status': env.status, 'view': view.text,
                    'first_ref': rid + ':FIRST', 'persistent_ref': rid + ':PERSISTENT'}
        return {'role': 'tool', 'tool_call_id': original.sdk_tool_call_id, 'content': canonical({
            'repair': {'event_id': repair_event_id, 'reason': reason},
            'active_result_ids': [original.original_result_ref, retry.result_id],
            'original': component(original.original_result_ref), 'retry': component(retry.result_id)})}

    def active_messages(self, messages, iteration):
        # Foundation has already selected modes. Overlay the two separately stored
        # representations into ONE original SDK reply without changing Foundation.
        out = []
        for message in messages:
            item = self.composites.get(message.get('tool_call_id')) if message.get('role') == 'tool' else None
            if item:
                a, retry, eid, reason = item
                env = self.components[a.original_result_ref][0]
                mode = 'FIRST' if env.iteration == iteration - 1 else 'PERSISTENT'
                message = self.compose(a, retry, eid, reason, mode)
            out.append(message)
        return out

    def finish(self):
        try:
            payload = {'mode': self.config.mode, 'original_results': len(self.original_ids),
                'skip_counts': self.skip_counts,
                'bridge_notices': [e for e in self.events if 'event_id' not in e],
                'retry_real_executor_entries': sum(self._retry_entered.values()),
                'b_single_reply_composites': len(self.composites),
                'metrics': self.coordinator.metrics.snapshot() if hasattr(self, 'coordinator') else None}
            self.audit_path.with_suffix('.metrics.json').write_text(canonical(payload), encoding='utf-8')
        except Exception:
            pass  # observational failure never changes B or creates a retry
