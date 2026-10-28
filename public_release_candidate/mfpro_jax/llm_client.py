"""Minimal OpenAI-compatible native tool-calling client."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Awaitable, Callable
import warnings
import httpx
from openai import AsyncOpenAI, APIConnectionError, APIStatusError
from .context_lifecycle import CONTROL_TOOL_NAME, RecoveryStatus, RepresentationViewStore, SharedContextLifecycle, is_final_answer_text
from .processor_mapping import RunProcessorRouter
from .result_store import ResultEnvelope, ResultStore
from .representation_runtime import RepresentationMode, RepresentationRuntime
from .trace import TraceRecorder

@dataclass(frozen=True)
class ToolInteraction:
    iteration: int
    tool_name: str
    arguments: dict[str, Any]
    status: str
    observation: str
    tool_call_id: str = ''
    result_id: str | None = None

@dataclass(frozen=True)
class IterationPreparedContext:
    tools: list[dict]
    executors: dict[str, Any]
    ephemeral_messages: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ToolRecoveryResult:
    recovered: bool
    tool_name: str
    tool_definition: dict[str, Any] | None = None
    executor: Any = None
    registry_exists: bool = False
    graph_node_exists: bool = False
    decision_reason: str = ''
IterationPreparationHook = Callable[[int, list[dict], tuple[ToolInteraction, ...]], IterationPreparedContext | Awaitable[IterationPreparedContext]]
ToolRecoveryHook = Callable[[str, dict[str, Any], dict[str, Any]], ToolRecoveryResult | Awaitable[ToolRecoveryResult]]
ToolResultObservationHook = Callable[[dict[str, Any]], None]

def _notify_tool_result(hook: ToolResultObservationHook | None, event: dict[str, Any]) -> None:
    if hook is None:
        return
    try:
        hook(event)
    except Exception as exc:
        warnings.warn(f'Tool result observer failed open: {type(exc).__name__}', RuntimeWarning, stacklevel=2)

def _evidence_runtime_snapshot(ctx: Any, access_state: dict[str, Any] | None=None) -> dict[str, Any]:
    """Read already available state; never call access_state/activate an index."""
    try:
        source = getattr(ctx, 'active_source', None)
        source_id = getattr(source, 'id', '')
        state = access_state if access_state is not None else getattr(ctx, 'source_access', {}).get(source_id, {})
        return {'source_id': source_id, 'partition_id': str(state.get('partition_id', '')), 'index_present': source_id in getattr(ctx, 'indexes', {})}
    except Exception:
        return {'snapshot_unavailable': True}

def _openai_tools(tools: list[dict]) -> list[dict]:
    return [{'type': 'function', 'function': {'name': item['name'], 'description': item['description'], 'parameters': item['input_schema']}} for item in tools]

def _usage_value(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        result = value.get(name)
    else:
        result = getattr(value, name, None)
    return int(result) if result is not None else None

def _normalize_usage(usage: Any) -> dict[str, int | None] | None:
    if usage is None:
        return None
    prompt_details = usage.get('prompt_tokens_details') if isinstance(usage, dict) else getattr(usage, 'prompt_tokens_details', None)
    completion_details = usage.get('completion_tokens_details') if isinstance(usage, dict) else getattr(usage, 'completion_tokens_details', None)
    result = {'prompt_tokens': _usage_value(usage, 'prompt_tokens'), 'completion_tokens': _usage_value(usage, 'completion_tokens'), 'reasoning_tokens': _usage_value(completion_details, 'reasoning_tokens'), 'cached_tokens': _usage_value(prompt_details, 'cached_tokens'), 'total_tokens': _usage_value(usage, 'total_tokens')}
    return result

def _schema_keys(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _schema_keys(item) for (key, item) in sorted(value.items())}
    if hasattr(value, 'model_dump'):
        try:
            return _schema_keys(value.model_dump())
        except Exception:
            pass
    if hasattr(value, '__dict__'):
        return {str(key): _schema_keys(item) for (key, item) in sorted(vars(value).items()) if not str(key).startswith('_')}
    if isinstance(value, (list, tuple)):
        return [_schema_keys(value[0])] if value else []
    return type(value).__name__

def _content_chars(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))

def _message_chars(message: dict[str, Any]) -> int:
    total = _content_chars(message.get('content'))
    total += _content_chars(message.get('reasoning_content'))
    total += _content_chars(message.get('tool_calls'))
    return total

def _context_composition(persistent_messages: list[dict], ephemeral_messages: list[dict]) -> dict[str, int]:
    system_chars = 0
    original_task_chars = 0
    assistant_history_chars = 0
    tool_call_history_chars = 0
    tool_result_history_chars = 0
    for message in persistent_messages:
        role = message.get('role')
        if role == 'system':
            system_chars += _content_chars(message.get('content'))
        elif role == 'user':
            original_task_chars += _content_chars(message.get('content'))
        elif role == 'assistant':
            assistant_history_chars += _content_chars(message.get('content'))
            assistant_history_chars += _content_chars(message.get('reasoning_content'))
            tool_call_history_chars += _content_chars(message.get('tool_calls'))
        elif role == 'tool':
            tool_result_history_chars += _content_chars(message.get('content'))
    return {'system_message_chars': system_chars, 'original_task_chars': original_task_chars, 'assistant_history_chars': assistant_history_chars, 'tool_call_history_chars': tool_call_history_chars, 'tool_result_history_chars': tool_result_history_chars, 'main_graph_context_chars': sum((_message_chars(message) for message in ephemeral_messages))}

def _tool_definitions_chars(tools: list[dict]) -> tuple[int, int]:
    api_tools = _openai_tools(tools)
    total = len(json.dumps(api_tools, ensure_ascii=False, sort_keys=True, default=str))
    schema_total = sum((len(json.dumps(item.get('input_schema', {}), ensure_ascii=False, sort_keys=True, default=str)) for item in tools))
    return (total, schema_total)

def _safe_path_component(value: str) -> str:
    safe = ''.join((ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in str(value)))
    return safe or 'unknown'

def _write_main_llm_observability_artifact(*, run_dir: Path, question_id: str, iteration: int, payload: dict[str, Any]) -> str:
    root = run_dir / 'main_llm_observability' / _safe_path_component(question_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'iteration_{iteration:03d}.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    return str(path.relative_to(run_dir))

def _result_provenance_snapshot(executor: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    ctx = getattr(executor, '_profile0_context', None) if executor else None
    active_source = getattr(ctx, 'active_source', None) if ctx is not None else None
    snapshot: dict[str, Any] = {'input_argument_keys': sorted((str(key) for key in (arguments or {}).keys())), 'input_paths': [str(value) for (key, value) in (arguments or {}).items() if key in {'path', 'file_path', 'db_path', 'archive_path', 'root'} and isinstance(value, str)]}
    if active_source is not None:
        snapshot.update({'source_id': active_source.id, 'source_path': active_source.path, 'source_type': active_source.type, 'access_mode': active_source.access_mode, 'access_state': ctx.access_state() if ctx is not None else {}})
    for key in ('source_id', 'file_id', 'path', 'file_path', 'db_path', 'archive_path', 'root', 'offset', 'length', 'max_bytes'):
        if key in (arguments or {}):
            snapshot[f'argument_{key}'] = arguments[key]
    return snapshot

def _llm_error_code(exc: Exception) -> str:
    if isinstance(exc, APIConnectionError):
        return 'network_error'
    if isinstance(exc, APIStatusError):
        status = getattr(exc, 'status_code', None)
        text = type(exc).__name__.casefold()
        if status == 402 or 'insufficient balance' in text or 'insufficient_balance' in text:
            return 'insufficient_balance'
        return 'provider_error'
    return 'llm_api_error'

def _action_signature(name: str, args: dict, source_id: str, access_state: dict) -> str:
    payload = {'source_id': source_id, 'access_state': access_state, 'tool': name, 'args': args}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

class LLMClient:

    def __init__(self, *, api_key: str, base_url: str, model: str, max_tokens: int=8192, temperature: float | None=None, seed: int | None=None, reasoning_effort: str | None=None, thinking_enabled: bool=False, proxy: str | None='auto', thinking_budget: int | None=None, processor_mode: str='baseline') -> None:
        if not api_key or not base_url or (not model):
            raise ValueError('Provider is not configured: set MODEL_API_KEY, MODEL_BASE_URL and MODEL_NAME')
        self._processor_router = RunProcessorRouter(processor_mode)
        proxy_url = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY') if proxy == 'auto' else proxy
        http_client = httpx.AsyncClient(proxy=proxy_url, timeout=300.0) if proxy_url else None
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url.rstrip('/'), timeout=300.0, http_client=http_client)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.thinking_enabled = thinking_enabled
        self.thinking_budget = thinking_budget

    @property
    def processor_mode(self) -> str:
        return self._processor_router.processor_mode

    def preflight_processor_run(self, run_dir: Path) -> dict:
        """Read-only mode/history check; call before creating a run Trace."""
        return self._processor_router.preflight_run(run_dir)

    def processor_selection_receipt(self, run_dir: Path) -> dict:
        return self._processor_router.bind_run(run_dir)

    def make_repair_advisor_factory(self, config):
        """Lazy dedicated transport. Deployment capability attestation is mandatory.

        An embedding application may inject repair_transport_factory(config), or
        repair_capabilities plus repair_client_factory. Default is safe abstention.
        No client/credential/network access occurs in the OFF or rules paths.
        """
        if config.mode not in {'agent', 'hybrid'}:
            return None
        from .tool_call_repair import RepairAdvisor, OpenAIChatTransport, ProviderCapabilities
        injected = getattr(self, 'repair_transport_factory', None)
        if injected is not None:
            return lambda : RepairAdvisor(injected(config), config)
        capabilities = getattr(self, 'repair_capabilities', None)
        if not isinstance(capabilities, ProviderCapabilities):
            return None

        def create():
            factory = getattr(self, 'repair_client_factory', None)
            client = factory() if factory is not None else self.client.with_options(max_retries=0)
            return RepairAdvisor(OpenAIChatTransport(client, self.model, capabilities), config)
        return create

    async def close(self) -> None:
        await self.client.close()

    async def _call(self, messages: list[dict], tools: list[dict], *, thinking_enabled: bool | None=None) -> tuple[str, str | None, list[dict], dict[str, int | None] | None, dict[str, Any]]:
        kwargs: dict[str, Any] = {'model': self.model, 'messages': messages, 'stream': True, 'stream_options': {'include_usage': True}}
        glm_thinking_model = self.model.lower() == 'glm-5.3'
        if not glm_thinking_model:
            kwargs['max_tokens'] = self.max_tokens
        if tools:
            kwargs['tools'] = _openai_tools(tools)
        if self.reasoning_effort:
            kwargs['reasoning_effort'] = self.reasoning_effort
        if self.temperature is not None:
            kwargs['temperature'] = float(self.temperature)
        if self.seed is not None:
            kwargs['seed'] = int(self.seed)
        effective_thinking_enabled = True if glm_thinking_model else self.thinking_enabled if thinking_enabled is None else thinking_enabled
        if effective_thinking_enabled:
            kwargs['extra_body'] = {'thinking': {'type': 'enabled'}}
        elif thinking_enabled is False:
            kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}
        if glm_thinking_model and self.thinking_budget is not None:
            kwargs.setdefault('extra_body', {})['thinking_budget'] = self.thinking_budget
        stream = await self.client.chat.completions.create(**kwargs)
        (text_parts, reasoning_parts, calls, usage) = ([], [], {}, None)
        usage_schema = None
        finish_reason = None
        (response_model, system_fingerprint) = ('', '')
        async for chunk in stream:
            if getattr(chunk, 'model', ''):
                response_model = chunk.model
            if getattr(chunk, 'system_fingerprint', ''):
                system_fingerprint = chunk.system_fingerprint
            if chunk.usage is not None:
                usage_schema = _schema_keys(chunk.usage)
                usage = _normalize_usage(chunk.usage)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            status = getattr(choice, 'finish_reason', None)
            if status in {'stop', 'length', 'tool_calls', 'content_filter', 'function_call'}:
                finish_reason = status
            delta = choice.delta
            if delta.content:
                text_parts.append(delta.content)
            reasoning = getattr(delta, 'reasoning_content', None)
            if reasoning:
                reasoning_parts.append(reasoning)
            for call in delta.tool_calls or []:
                item = calls.setdefault(call.index, {'id': None, 'name': '', 'arguments': ''})
                if call.id:
                    item['id'] = call.id
                if call.function:
                    if call.function.name:
                        item['name'] = call.function.name
                    if call.function.arguments:
                        item['arguments'] += call.function.arguments
        metadata = {'requested_model': self.model, 'response_model': response_model, 'system_fingerprint': system_fingerprint, 'temperature': self.temperature, 'seed': self.seed, 'thinking_enabled': effective_thinking_enabled, 'reasoning_effort': self.reasoning_effort, 'dedicated_reasoning_budget_parameter': 'thinking_budget' if glm_thinking_model and self.thinking_budget is not None else None, 'dedicated_reasoning_budget_value': self.thinking_budget if glm_thinking_model else None, 'max_tokens': kwargs.get('max_tokens'), 'max_completion_tokens': kwargs.get('max_completion_tokens'), 'usage_schema': usage_schema, 'finish_reason': finish_reason, 'finish_reason_source': 'provider_choice.finish_reason' if finish_reason else None}
        return (''.join(text_parts), ''.join(reasoning_parts) or None, [calls[key] for key in sorted(calls)], usage, metadata)

    async def tool_call_loop(self, *, messages: list[dict], tools: list[dict], executors: dict[str, Any], system: str, trace: TraceRecorder, max_iterations: int=40, prepare_iteration: IterationPreparationHook | None=None, recover_tool: ToolRecoveryHook | None=None, context_lifecycle: SharedContextLifecycle | None=None, llm_role: str='main', observe_tool_result: ToolResultObservationHook | None=None, repair_runtime: Any=None) -> tuple[str, list[dict]]:
        return await self._tool_call_loop(messages=messages, tools=tools, executors=executors, system=system, trace=trace, max_iterations=max_iterations, prepare_iteration=prepare_iteration, recover_tool=recover_tool, context_lifecycle=context_lifecycle, llm_role=llm_role, observe_tool_result=observe_tool_result, repair_runtime=repair_runtime)

    async def _tool_call_loop(self, *, messages: list[dict], tools: list[dict], executors: dict[str, Any], system: str, trace: TraceRecorder, max_iterations: int=40, prepare_iteration: IterationPreparationHook | None=None, recover_tool: ToolRecoveryHook | None=None, context_lifecycle: SharedContextLifecycle | None=None, llm_role: str='main', observe_tool_result: ToolResultObservationHook | None=None, repair_runtime: Any=None) -> tuple[str, list[dict]]:
        router = getattr(self, '_processor_router', None)
        if router is None:
            router = RunProcessorRouter()
            self._processor_router = router
        router.bind_run(trace.run_dir)
        if context_lifecycle is not None:
            provider = context_lifecycle.processor_provider
            owns_provider = getattr(provider, '__self__', None) is router and getattr(provider, '__func__', None) is RunProcessorRouter.provider
            baseline_identity = router.processor_mode == 'baseline' and provider is None
            if not (owns_provider or baseline_identity):
                raise RuntimeError("Formal supplied lifecycle must use this client's run processor provider (or baseline identity)")
        history = [{'role': 'system', 'content': system}, *messages]
        terminal_actions: dict[str, tuple[str, str]] = {}
        repeated_failures = 0
        tool_interactions: list[ToolInteraction] = []
        result_store: ResultStore | None = context_lifecycle.runtime.result_store if context_lifecycle is not None else None
        if result_store is None:
            try:
                result_store = ResultStore(trace.run_dir / 'result_store', run_id=trace.run_id)
            except Exception as exc:
                warnings.warn(f'ResultStore unavailable; continuing without formal raw result storage: {type(exc).__name__}', RuntimeWarning, stacklevel=2)
        if context_lifecycle is None and result_store is not None:
            try:
                context_lifecycle = SharedContextLifecycle(runtime=RepresentationRuntime(result_store), view_store=RepresentationViewStore(trace.run_dir / 'representation_views'), processor_provider=router.provider)
            except Exception as exc:
                warnings.warn(f'SharedContextLifecycle unavailable; continuing with legacy-compatible tool result context: {type(exc).__name__}', RuntimeWarning, stacklevel=2)
                context_lifecycle = None
        if context_lifecycle is None:
            raise RuntimeError('Formal modes require Foundation storage and lifecycle; refusing untracked result views')
        result_observer = None
        if repair_runtime is not None:
            try:
                repair_runtime.bind(lifecycle=context_lifecycle, store=result_store, terminal_actions=terminal_actions, observe=observe_tool_result, interactions=tool_interactions, result_observer=result_observer, trace=trace)
            except Exception as exc:
                warnings.warn('Optional repair binding disabled: ' + type(exc).__name__, RuntimeWarning, stacklevel=2)
                repair_runtime = None
        for iteration in range(max_iterations):
            current_tools = tools
            current_executors = executors
            current_messages = history
            ephemeral_messages: list[dict] = []
            prepared_metadata: dict[str, Any] = {}
            recovery_followup_active = bool(context_lifecycle is not None and context_lifecycle.recovery_request is not None and (context_lifecycle.recovery_request.status is RecoveryStatus.REQUESTED))
            shared_snapshot = None
            if context_lifecycle is not None:
                shared_snapshot = context_lifecycle.build_snapshot(iteration=iteration + 1, interactions=tuple(tool_interactions))
                current_messages = context_lifecycle.active_messages(history, shared_snapshot)
                if repair_runtime is not None:
                    current_messages = repair_runtime.active_messages(current_messages, iteration + 1)
            if prepare_iteration is not None:
                if recovery_followup_active:
                    current_tools = []
                    current_executors = {}
                    prepared_metadata = {'retrieval_skipped_for_recovery': True, 'final_exposed_tools': [], 'graph_exposed_tools': [], 'ttl1_retained_tools': []}
                else:
                    prepared_interactions = tuple(tool_interactions)
                    if context_lifecycle is not None and shared_snapshot is not None:
                        prepared_interactions = context_lifecycle.active_interactions(prepared_interactions, shared_snapshot)
                    prepared = prepare_iteration(iteration + 1, history, prepared_interactions)
                    if inspect.isawaitable(prepared):
                        prepared = await prepared
                    current_tools = prepared.tools
                    current_executors = prepared.executors
                    prepared_metadata = dict(prepared.metadata or {})
                    if prepared.ephemeral_messages:
                        ephemeral_messages = list(prepared.ephemeral_messages)
                        current_messages = [*current_messages, *ephemeral_messages]
            if context_lifecycle is not None and shared_snapshot is not None:
                current_tools = context_lifecycle.maybe_add_control_tool(current_tools, shared_snapshot)
            schemas = {item['name']: item.get('input_schema', {}) for item in current_tools}
            composition = _context_composition(history, ephemeral_messages)
            (tool_definitions_chars, tool_schema_chars) = _tool_definitions_chars(current_tools)
            messages_chars = sum((_message_chars(message) for message in current_messages))
            started = time.monotonic()
            started_event = trace.emit('llm_call_started', status='started', details={'iteration': iteration + 1, 'llm_role': llm_role})
            question_id_for_capture = started_event.question_id
            try:
                call_result = await self._call(current_messages, current_tools)
            except asyncio.CancelledError as exc:
                cancel_reason = getattr(asyncio.current_task(), '_profile0_cancel_reason', 'run_interrupted')
                trace.emit('llm_call_failed', status='cancelled', reason_code=cancel_reason, duration_ms=(time.monotonic() - started) * 1000, error=type(exc).__name__, details={'iteration': iteration + 1, 'llm_role': llm_role, 'exception_type': type(exc).__name__})
                raise
            except Exception as exc:
                trace.emit('llm_call_failed', status='failed', reason_code=_llm_error_code(exc), duration_ms=(time.monotonic() - started) * 1000, error=type(exc).__name__, details={'iteration': iteration + 1, 'llm_role': llm_role, 'exception_type': type(exc).__name__})
                raise
            if len(call_result) == 3:
                (text, reasoning, raw_calls) = call_result
                usage = None
                metadata = {}
            elif len(call_result) == 4:
                (text, reasoning, raw_calls, usage) = call_result
                metadata = {}
            else:
                (text, reasoning, raw_calls, usage, metadata) = call_result
            call_duration_ms = (time.monotonic() - started) * 1000
            usage_source = 'API_REPORTED' if usage is not None else 'NOT_AVAILABLE'
            prompt_tokens = usage.get('prompt_tokens') if usage else None
            completion_tokens = usage.get('completion_tokens') if usage else None
            reasoning_tokens = usage.get('reasoning_tokens') if usage else None
            cached_tokens = usage.get('cached_tokens') if usage else None
            total_tokens = usage.get('total_tokens') if usage else None
            observability = {'question_id': '', 'iteration': iteration + 1, 'main_model': metadata.get('requested_model', getattr(self, 'model', '')), 'main_thinking_enabled': metadata.get('thinking_enabled', getattr(self, 'thinking_enabled', None)), 'main_max_tokens': getattr(self, 'max_tokens', None), 'main_message_count': len(current_messages), 'main_messages_chars': messages_chars, **composition, 'main_visible_tool_count': len(current_tools), 'main_tool_definitions_chars': tool_definitions_chars, 'main_tool_schema_chars': tool_schema_chars, 'main_prompt_tokens': prompt_tokens, 'main_completion_tokens': completion_tokens, 'main_reasoning_tokens': reasoning_tokens, 'main_reasoning_token_status': 'API_REPORTED' if reasoning_tokens is not None else 'NOT_AVAILABLE', 'main_cached_tokens': cached_tokens, 'main_cached_token_status': 'API_REPORTED' if cached_tokens is not None else 'NOT_AVAILABLE', 'main_total_tokens': total_tokens, 'usage_source': usage_source, 'usage_schema': metadata.get('usage_schema'), 'main_llm_call_ms': call_duration_ms, 'messages_sha256': hashlib.sha256(json.dumps(current_messages, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest(), 'tool_definitions_sha256': hashlib.sha256(json.dumps(_openai_tools(current_tools), ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()}
            trace.emit('llm_call_completed', status='completed', duration_ms=call_duration_ms, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, reasoning_tokens=reasoning_tokens, cached_tokens=cached_tokens, total_tokens=total_tokens, details={'iteration': iteration + 1, 'llm_role': llm_role, 'requested_model': metadata.get('requested_model', getattr(self, 'model', '')), 'response_model': metadata.get('response_model', ''), 'system_fingerprint': metadata.get('system_fingerprint', ''), 'temperature': metadata.get('temperature', getattr(self, 'temperature', None)), 'seed': metadata.get('seed', getattr(self, 'seed', None)), 'usage_available': usage is not None, 'usage_source': usage_source, 'main_llm_observability': observability, 'tool_calls_requested': len(raw_calls)})
            question_id = trace.events[-1].question_id if trace.events else ''
            observability['question_id'] = question_id
            trace.events[-1].details['main_llm_observability']['question_id'] = question_id
            artifact_path = _write_main_llm_observability_artifact(run_dir=trace.run_dir, question_id=question_id, iteration=iteration + 1, payload=observability)
            trace.events[-1].details['main_llm_observability_path'] = artifact_path
            with trace.path.open('r+', encoding='utf-8') as stream:
                lines = stream.readlines()
                if lines:
                    lines[-1] = json.dumps(trace.events[-1].to_dict(), ensure_ascii=False) + '\n'
                    stream.seek(0)
                    stream.writelines(lines)
                    stream.truncate()
            if not raw_calls:
                response = {'role': 'assistant', 'content': text}
                if reasoning:
                    response['reasoning_content'] = reasoning
                history.append(response)
                if recovery_followup_active and context_lifecycle is not None:
                    context_lifecycle.clear_recovery()
                    if not is_final_answer_text(text):
                        continue
                return (text, history)
            assistant_message: dict[str, Any] = {'role': 'assistant', 'content': text or None, 'tool_calls': [{'id': call['id'], 'type': 'function', 'function': {'name': call['name'], 'arguments': call['arguments']}} for call in raw_calls]}
            if reasoning:
                assistant_message['reasoning_content'] = reasoning
            history.append(assistant_message)
            if recovery_followup_active and context_lifecycle is not None:
                context_lifecycle.clear_recovery()
            jit_recovered_tools: list[str] = []
            if repair_runtime is not None:
                repair_runtime.begin_batch(iteration + 1, raw_calls)
            for (call_index, call) in enumerate(raw_calls, start=1):
                name = call['name']
                status = 'ERROR'
                execution_ms_for_capture: float | None = None
                executor = None
                evidence_input_context: dict[str, Any] = {}
                evidence_execution_started = False
                evidence_execution_completed = False
                evidence_terminal = None
                repair_capture = None
                repair_schema_error = ''
                try:
                    args = json.loads(call['arguments'] or '{}')
                except json.JSONDecodeError as exc:
                    if context_lifecycle is not None and name == CONTROL_TOOL_NAME:
                        recovery_request = context_lifecycle.request_recovery(result_id='', tool_call_id=str(call.get('id') or ''))
                        history.append({'role': 'tool', 'tool_call_id': call['id'], 'content': recovery_request.message})
                        continue
                    args = {'_invalid_json': call['arguments']}
                    status = 'INVALID_ARGUMENTS'
                    signature = _action_signature(name, args, '', {})
                    prior = terminal_actions.get(signature)
                    if prior:
                        repeated_failures += 1
                        result = f'Error: repeated unchanged invalid tool arguments suppressed.'
                        _emit_repeated_failure(trace, name, args, signature, 'invalid')
                    else:
                        trace.emit('invalid_tool_arguments', tool_name=name, status='invalid', reason_code='invalid_tool_arguments', error=type(exc).__name__, output_preview=call['arguments'][:2000])
                        result = f'Error: invalid JSON tool arguments: {type(exc).__name__}'
                        terminal_actions[signature] = ('failed', result)
                else:
                    if context_lifecycle is not None and name == CONTROL_TOOL_NAME:
                        requested_result_id = str(args.get('result_id', '')) if isinstance(args, dict) else ''
                        recovery_request = context_lifecycle.request_recovery(result_id=requested_result_id, tool_call_id=str(call.get('id') or ''))
                        history.append({'role': 'tool', 'tool_call_id': call['id'], 'content': recovery_request.message})
                        continue
                    executor = current_executors.get(name)
                    schema = schemas.get(name, {})
                    jit_recovered = False
                    jit_details: dict[str, Any] | None = None
                    if executor is None and recover_tool is not None:
                        attempt_details = {'iteration': iteration + 1, 'tool_call_index': call_index, 'tool_call_id': call.get('id'), 'requested_tool': name, 'arguments_sha256': hashlib.sha256(json.dumps(args, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest(), 'was_in_graph_exposed': name in set(prepared_metadata.get('graph_exposed_tools', [])), 'was_in_ttl_retained': name in set(prepared_metadata.get('ttl1_retained_tools', [])), 'was_in_final_exposed': name in set(prepared_metadata.get('final_exposed_tools', [tool['name'] for tool in current_tools])), 'jit_attempted': True}
                        trace.emit('jit_tool_recovery_attempt', tool_name=name, args=args, status='started', details=attempt_details)
                        recovery = recover_tool(name, args, {'iteration': iteration + 1, 'tool_call_index': call_index, 'tool_call_id': call.get('id'), 'prepared_metadata': prepared_metadata})
                        if inspect.isawaitable(recovery):
                            recovery = await recovery
                        jit_details = {**attempt_details, 'registry_exists': recovery.registry_exists, 'graph_node_exists': recovery.graph_node_exists, 'jit_decision': 'RECOVERED' if recovery.recovered else 'REJECTED', 'decision_reason': recovery.decision_reason, 'jit_executor_resolved': recovery.executor is not None, 'execution_attempted': bool(recovery.recovered and recovery.executor is not None)}
                        if recovery.recovered and recovery.executor is not None and (recovery.tool_definition is not None):
                            executor = recovery.executor
                            schema = recovery.tool_definition.get('input_schema', {})
                            schemas[name] = schema
                            jit_recovered = True
                            jit_recovered_tools.append(name)
                        else:
                            trace.emit('jit_tool_recovery_completed', tool_name=name, args=args, status='rejected', reason_code=recovery.decision_reason, details={**jit_details, 'final_status': 'UNKNOWN_TOOL', 'ttl_eligible_next_iteration': False})
                    ctx = getattr(executor, '_profile0_context', None) if executor else None
                    access_state = ctx.access_state() if ctx is not None else {}
                    source_id = ctx.active_source.id if ctx is not None and ctx.active_source else ''
                    if observe_tool_result is not None or repair_runtime is not None:
                        evidence_input_context = _evidence_runtime_snapshot(ctx, access_state)
                    signature = _action_signature(name, args, source_id, access_state)
                    prior = terminal_actions.get(signature)
                    if prior and prior[0] in {'failed', 'empty', 'timed_out'}:
                        repeated_failures += 1
                        result = f'Error: repeated unchanged {prior[0]} tool action suppressed.'
                        _emit_repeated_failure(trace, name, args, signature, prior[0])
                    elif executor is None:
                        trace.emit('unknown_tool_call', tool_name=name, args=args, status='unknown', reason_code='unknown_tool')
                        result = f"Error: unknown tool '{name}'"
                        status = 'UNKNOWN_TOOL'
                        terminal_actions[signature] = ('failed', result)
                    else:
                        error = _validate_tool_arguments(args, schema)
                        if error:
                            repair_schema_error = error
                            if repair_runtime is not None:
                                repair_capture = repair_runtime.capture(call, call_index, executor, schema)
                            trace.emit('invalid_tool_arguments', tool_name=name, args=args, status='invalid', reason_code='invalid_tool_arguments', error=error)
                            result = f'Error: invalid arguments for {name}: {error}'
                            status = 'INVALID_ARGUMENTS'
                            terminal_actions[signature] = ('failed', result)
                            if jit_details is not None:
                                trace.emit('jit_tool_recovery_completed', tool_name=name, args=args, status='completed', reason_code='RECOVERED_REGISTRY_AND_GRAPH_VALID', details={**jit_details, 'final_status': status, 'ttl_eligible_next_iteration': True})
                        else:
                            before = len(trace.events)
                            exec_started = time.monotonic()
                            evidence_execution_started = True
                            try:
                                result = await executor(**args)
                            except (Exception, asyncio.CancelledError) as exc:
                                if observe_tool_result is not None:
                                    _notify_tool_result(observe_tool_result, {'iteration': iteration + 1, 'call_index': call_index, 'tool_call_id': str(call.get('id') or ''), 'tool_name': name, 'arguments': args, 'status': 'CANCELLED' if isinstance(exc, asyncio.CancelledError) else 'TIMEOUT' if isinstance(exc, TimeoutError) else 'ERROR', 'execution_started': True, 'execution_completed': False, 'error_origin': type(exc).__name__, 'input_context': evidence_input_context, 'raw_result': None, 'trace_event_id': None})
                                raise
                            evidence_execution_completed = True
                            exec_duration_ms = (time.monotonic() - exec_started) * 1000
                            execution_ms_for_capture = exec_duration_ms
                            terminal = next((event for event in reversed(trace.events[before:]) if event.tool_name == name and event.event_type.startswith('tool_call_') and (event.event_type != 'tool_call_started')), None)
                            if terminal:
                                terminal_actions[signature] = (terminal.status, str(result))
                            status = _tool_interaction_status(str(result), terminal.status if terminal else '')
                            evidence_terminal = terminal
                            if jit_details is not None:
                                trace.emit('jit_tool_recovery_completed', tool_name=name, args=args, status='completed', reason_code='RECOVERED_REGISTRY_AND_GRAPH_VALID', duration_ms=exec_duration_ms, details={**jit_details, 'final_status': status, 'execution_duration_ms': exec_duration_ms, 'ttl_eligible_next_iteration': True})
                        if jit_recovered and (not any((event.event_type == 'jit_tool_recovery_completed' and event.tool_name == name and (event.details.get('tool_call_id') == call.get('id')) for event in trace.events))):
                            trace.emit('jit_tool_recovery_completed', tool_name=name, args=args, status='completed', reason_code='RECOVERED_REGISTRY_AND_GRAPH_VALID', details={**(jit_details or {}), 'final_status': status, 'ttl_eligible_next_iteration': status != 'UNKNOWN_TOOL'})
                formal_result: ResultEnvelope | None = None
                formal_arguments = args if isinstance(args, dict) else {'_arguments': args}
                if result_store is not None:
                    try:
                        formal_result = result_store.store_raw_result(question_id=question_id_for_capture, iteration=iteration + 1, call_index=call_index, tool_call_id=str(call.get('id') or ''), tool_name=name, arguments=formal_arguments, status=status, raw_result=result, provenance=_result_provenance_snapshot(executor, formal_arguments))
                    except Exception as exc:
                        warnings.warn(f'ResultStore failed open for tool result {name}: {type(exc).__name__}', RuntimeWarning, stacklevel=2)
                if context_lifecycle is not None and formal_result is not None:
                    try:
                        context_lifecycle.materialize_first_persistent(formal_result)
                    except Exception as exc:
                        warnings.warn(f'SharedContextLifecycle failed open for tool result {name}: {type(exc).__name__}', RuntimeWarning, stacklevel=2)
                ingress_content = str(result)[:16000]
                history_content = ingress_content
                history.append({'role': 'tool', 'tool_call_id': call['id'], 'content': history_content})
                tool_interactions.append(ToolInteraction(iteration=iteration + 1, tool_name=name, arguments=args if isinstance(args, dict) else {'_arguments': args}, status=status, observation=history_content if formal_result is not None else ingress_content, tool_call_id=str(call.get('id') or ''), result_id=formal_result.result_id if formal_result is not None else None))
                if observe_tool_result is not None or repair_runtime is not None:
                    first_view = None
                    try:
                        if context_lifecycle is not None and formal_result is not None:
                            first_view = context_lifecycle.view_store.get(formal_result.result_id, RepresentationMode.FIRST)
                    except Exception:
                        pass
                    observation_event = {'iteration': iteration + 1, 'call_index': call_index, 'tool_call_id': str(call.get('id') or ''), 'tool_name': name, 'arguments': formal_arguments, 'status': status, 'execution_started': evidence_execution_started, 'execution_completed': evidence_execution_completed, 'input_context': evidence_input_context, 'output_context': _evidence_runtime_snapshot(getattr(executor, '_profile0_context', None)), 'error_origin': getattr(evidence_terminal, 'reason_code', '') or ('LOOP_PRE_EXECUTION' if not evidence_execution_started else ''), 'trace_event_id': getattr(evidence_terminal, 'event_id', None), 'envelope': formal_result, 'first_view': first_view, 'raw_result': result, 'ingress_content': history_content}
                    _notify_tool_result(observe_tool_result, observation_event)
                    if repair_runtime is not None:
                        original_message_index = len(history) - 1
                        history[original_message_index] = await repair_runtime.handle_result(event=observation_event, call=call, index=call_index, capture=repair_capture, schema_error=repair_schema_error, original_delivery=history[original_message_index])
            if repeated_failures >= 3:
                trace.emit('iteration_exhausted', status='exhausted', reason_code='tool_repeated_failure', details={'repeated_failures': repeated_failures})
                return ('[Repeated failed tool actions suppressed; answer not formed]', history)
        trace.emit('iteration_exhausted', status='exhausted', reason_code='iteration_exhausted', details={'max_iterations': max_iterations})
        return ('[Max tool call iterations reached]', history)

def _emit_repeated_failure(trace: TraceRecorder, name: str, args: dict, signature: str, prior_status: str) -> None:
    trace.emit('tool_call_started', tool_name=name, args=args, status='started', details={'suppressed_repeat': True, 'action_signature': signature[:1000]})
    trace.emit('tool_call_failed', tool_name=name, args=args, status='failed', reason_code='tool_repeated_failure', output_preview=f'Error: repeated unchanged {prior_status} tool action suppressed.', details={'prior_status': prior_status, 'action_signature': signature[:1000]})

def _tool_interaction_status(result: str, terminal_status: str) -> str:
    if terminal_status == 'timed_out':
        return 'TIMEOUT'
    if terminal_status == 'failed' or result.startswith('Error:'):
        return 'ERROR'
    if terminal_status == 'empty' or _looks_empty_observation(result):
        return 'SUCCESS_EMPTY'
    return 'SUCCESS'

def _looks_empty_observation(result: str) -> bool:
    stripped = result.strip()
    if not stripped:
        return True
    lowered = stripped.casefold()
    if any((marker in lowered for marker in ('"matches": 0', '"rows": []', '"results": []', '"entries": []', 'no matches', 'no partitions'))):
        return True
    try:
        value = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return False
    if value in ([], {}, None, ''):
        return True
    if isinstance(value, dict):
        if value.get('truncated') is True:
            return False
        count_keys = ('matches', 'returned_rows', 'returned_matches', 'entries', 'total_objects')
        return any((value.get(key) == 0 for key in count_keys)) and any((value.get(key) == [] for key in ('results', 'rows', 'entries', 'objects')))
    return False

def _validate_tool_arguments(args: Any, schema: dict) -> str:
    if not isinstance(args, dict):
        return 'arguments must be a JSON object'
    required = set(schema.get('required', []))
    missing = sorted(required - set(args))
    if missing:
        return f"missing required arguments: {', '.join(missing)}"
    allowed = set(schema.get('properties', {}))
    unknown = sorted(set(args) - allowed)
    if unknown:
        return f"unknown arguments: {', '.join(unknown)}"
    properties = schema.get('properties', {})
    for (name, value) in args.items():
        definition = properties.get(name, {})
        expected = definition.get('type')
        valid = {'string': isinstance(value, str), 'integer': isinstance(value, int) and (not isinstance(value, bool)), 'number': isinstance(value, (int, float)) and (not isinstance(value, bool)), 'boolean': isinstance(value, bool), 'array': isinstance(value, list), 'object': isinstance(value, dict)}.get(expected, True)
        if not valid:
            return f"argument '{name}' must be {expected}"
        if 'enum' in definition and value not in definition['enum']:
            return f"argument '{name}' must be one of: {', '.join(map(str, definition['enum']))}"
        item_type = definition.get('items', {}).get('type')
        if expected == 'array' and item_type:
            for (index, item) in enumerate(value):
                item_valid = {'string': isinstance(item, str), 'integer': isinstance(item, int) and (not isinstance(item, bool)), 'number': isinstance(item, (int, float)) and (not isinstance(item, bool)), 'boolean': isinstance(item, bool), 'object': isinstance(item, dict)}.get(item_type, True)
                if not item_valid:
                    return f"argument '{name}[{index}]' must be {item_type}"
    return ''
