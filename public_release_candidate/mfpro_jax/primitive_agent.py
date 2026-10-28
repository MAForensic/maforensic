"""Single Profile 0 agent with no expert routing or artifact semantics."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time
from typing import Any
import uuid

from .capsules.context_v2 import CapsuleRetrievalV2Provider
from .llm_client import IterationPreparedContext, LLMClient, ToolInteraction, ToolRecoveryResult
from .runtime import RuntimeContext
from .tool_registry import ToolRegistry
from .task_evidence_graph import TaskEvidenceGraph, VERSION as TASK_EVIDENCE_GRAPH_VERSION

RETRIEVAL_STATE_SCHEMA_VERSION = "profile0-iteration-retrieval-state-v1"
RETRIEVAL_ARGUMENT_CHAR_LIMIT = 2000
RETRIEVAL_OBSERVATION_CHAR_LIMIT = 1200
RETRIEVAL_HISTORY_CHAR_LIMIT = 8000

SYSTEM_PROMPT = """You are Profile 0, a Primitive Evidence Agent.

Your only responsibilities are generic static evidence access:
- probe evidence sources and partitions;
- enumerate and index files;
- read bounded text and binary ranges;
- compute hashes and identify generic file types;
- run generic keyword, regular-expression, and strings searches;
- inspect generic SQLite, plist, JSON, XML, CSV, INI, archive, image-metadata, and timestamp structures;
- report concrete evidence paths and tool outputs.

Paths returned by an image file index are evidence-internal labels, not host paths. Never pass them to host/tree path tools. Use the indexed-member tools with source_id and file_id; export_indexed_file first when a generic parser needs a host path.

Do not infer semantics that are not directly exposed by the available generic tools. Do not guess or diagnose missing capabilities. Tool failures and limitations are recorded automatically as objective raw failure events. A truncated search does not prove that a value is absent from the whole file. End with `ANSWER: <value>` only when evidence supports an answer; otherwise end with `NO ANSWER: <reason>`.
"""

ANSWER_VERIFICATION_PROMPT = """You are verifying only the final answer format and consistency of a forensic answer.

Use only the evidence analysis already present in the supplied response. Do not introduce new evidence, use tools, or guess.
Respect the declared question type:
- single_choice: select exactly one supported option;
- multi_choice: select one or more supported options;
- judgment: answer with a clear true/false, correct/incorrect, yes/no, or equivalent judgment;
- freetext: answer with a non-empty value supported by the supplied analysis.

Return only `ANSWER: <value>` in the format required by the question type when the supplied analysis supports an answer.
Return only `NO ANSWER: <reason>` when it does not.
"""


class PrimitiveEvidenceAgent:
    name = "primitive_evidence"

    def __init__(
        self,
        llm: LLMClient,
        ctx: RuntimeContext,
        registry: ToolRegistry,
        max_iterations: int = 40,
        capsule_context_provider: CapsuleRetrievalV2Provider | None = None,
    ) -> None:
        self.llm, self.ctx, self.registry = llm, ctx, registry
        self.max_iterations = max_iterations
        self.capsule_context_provider = capsule_context_provider

    async def run(self, task: str, question_id: str) -> tuple[str, list[dict]]:
        task_id = f"task-{uuid.uuid4().hex[:10]}"
        source_id = self.ctx.active_source.id if self.ctx.active_source else ""
        self.ctx.trace.set_context(question_id=question_id, task_id=task_id, source_id=source_id)
        question_started = time.monotonic()
        evidence_graph = TaskEvidenceGraph.create(ctx=self.ctx, task_id=task_id, question_id=question_id)
        outcome = "aborted"
        try:
            result, history = await self._run_question(task, question_id, evidence_graph, question_started)
            outcome = "completed"
            return result, history
        finally:
            if evidence_graph is not None:
                try:
                    evidence_graph.seal(outcome)
                except Exception as exc:
                    evidence_graph.fail("seal", exc)

    async def _run_question(
        self, task: str, question_id: str, evidence_graph: TaskEvidenceGraph | None,
        question_started: float | None = None,
    ) -> tuple[str, list[dict]]:
        effective_task = task
        tool_definitions = self.registry.model_visible_definitions()
        executors = self.registry.executors()
        prepare_iteration = None
        recover_tool = None
        if self.capsule_context_provider is not None:
            source = self.ctx.active_source
            source_metadata = dict(source.meta) if source is not None else {}
            if source is not None:
                source_metadata.update({
                    "label": source.label,
                    "type": source.type,
                    "access_mode": source.access_mode,
                    "owner": source.owner,
                })
            prepare_iteration = self._build_dynamic_retrieval_hook(
                task=task,
                question_id=question_id,
                source_id=source.id if source is not None else "",
                source_metadata=source_metadata,
                base_tool_definitions=tool_definitions,
                base_executors=executors,
                evidence_graph=evidence_graph,
            )
            recover_tool = self._build_jit_tool_recovery_hook(
                base_tool_definitions=tool_definitions,
                base_executors=executors,
            )
        observation_options = {"observe_tool_result": evidence_graph.observe} if evidence_graph is not None else {}
        repair_options = {}
        repair_config = self.ctx.config.get("tool_call_repair", {})
        if not isinstance(repair_config, dict) or repair_config.get("mode", "off") != "off" or set(repair_config) - {"mode"}:
            from .integration.repair import resolve_repair_runtime
            repair_options = resolve_repair_runtime(self.ctx, self.registry, self.llm,
                                                    question_id, question_started)
        try:
            result, history = await self.llm.tool_call_loop(
                messages=[{"role": "user", "content": effective_task}],
                tools=tool_definitions, executors=executors,
                system=SYSTEM_PROMPT, trace=self.ctx.trace, max_iterations=self.max_iterations,
                prepare_iteration=prepare_iteration, recover_tool=recover_tool,
                **observation_options, **repair_options,
            )
        finally:
            runtime = repair_options.get("repair_runtime")
            if runtime is not None:
                runtime.finish()
        self.ctx.trace.emit("agent_decision_summary", status="completed", output_preview=result[:4000])
        return result, history

    def _build_dynamic_retrieval_hook(
        self,
        *,
        task: str,
        question_id: str,
        source_id: str,
        source_metadata: dict[str, Any],
        base_tool_definitions: list[dict],
        base_executors: dict[str, Any],
        evidence_graph: TaskEvidenceGraph | None = None,
    ):
        previous_tools: set[str] = set()

        async def prepare_iteration(
            iteration: int,
            history: list[dict],
            tool_interactions: tuple[ToolInteraction, ...],
        ) -> IterationPreparedContext:
            nonlocal previous_tools
            prepare_started = time.monotonic()
            state_started = time.monotonic()
            evidence_options = None
            evidence_cursor = None
            evidence_metrics: dict[str, Any] = {}
            graph_state = None
            if evidence_graph is not None:
                try:
                    projection = getattr(evidence_graph, "projection", None)
                    if iteration > 1 and projection is not None:
                        graph_state, evidence_metrics, evidence_cursor = projection.build(
                            iteration=iteration, task_context={"original_task": task,
                                "initial_source_id": source_id,
                                "prior_invocation_text": json.dumps([i.arguments for i in tool_interactions], ensure_ascii=False)})
                    else:
                        graph_state, evidence_metrics, evidence_cursor = evidence_graph.build_view()
                    if (iteration > 1 and evidence_graph.settings.mode == "active"
                            and getattr(self.capsule_context_provider, "supports_task_evidence_graph", False)):
                        evidence_options = evidence_graph.seed_options()
                    elif iteration > 1 and evidence_graph.settings.mode == "active":
                        evidence_metrics["fallback_reason"] = "provider_does_not_support_evidence_view"
                except Exception as exc:
                    # Oversized frontier is a per-retrieval fallback; a broken
                    # graph is never presented as a genuinely empty state.
                    evidence_metrics = {"mode": evidence_graph.settings.mode,
                                        "fallback_reason": str(exc) if type(exc).__name__ == "EvidenceViewUnavailable" else type(exc).__name__}
                    evidence_graph.record("view_fallback", **evidence_metrics)
                    # The legacy retrieval will still include these results;
                    # checkpoint only after that actual Seed call completes.
                    if evidence_graph.available:
                        evidence_cursor = len(evidence_graph.result_order)
            if iteration == 1:
                retrieval_task = task
                retrieval_input_mode = "legacy_raw_task"
                retrieval_history_count = 0
                retrieval_history_chars = 0
            elif evidence_options is not None and graph_state is not None:
                retrieval_task = json.dumps({
                    "retrieval_state_schema": TASK_EVIDENCE_GRAPH_VERSION,
                    "original_task": task,
                    "initial_evidence_context": {"source_id": source_id, "source_metadata": source_metadata},
                    **graph_state,
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                retrieval_input_mode = (evidence_metrics["presentation"] if evidence_metrics.get("presentation") in {"evidence_cards_v1", "evidence_cards_v2", "evidence_cards_v2_1"}
                                        else "task_evidence_graph_v1")
                retrieval_history_count = 0
                retrieval_history_chars = 0
            else:
                retrieval_task, retrieval_history_count, retrieval_history_chars = _build_iteration_retrieval_state(
                    original_task=task,
                    source_id=source_id,
                    source_metadata=source_metadata,
                    tool_interactions=tool_interactions,
                )
                retrieval_input_mode = "dynamic_retrieval_state"
            retrieval_state_build_ms = (time.monotonic() - state_started) * 1000
            retrieval_input_hash = hashlib.sha256(retrieval_task.encode()).hexdigest()
            retrieval_input_chars = len(retrieval_task)
            retrieval_input_path = _write_retrieval_input_artifact(
                run_dir=self.ctx.run_dir,
                question_id=question_id,
                iteration=iteration,
                mode=retrieval_input_mode,
                actual_retrieval_input=retrieval_task,
                sha256=retrieval_input_hash,
                chars=retrieval_input_chars,
                history_count=retrieval_history_count,
                history_chars=retrieval_history_chars,
            )
            select_started = time.monotonic()
            seed_options = {"evidence_context": evidence_options} if evidence_options is not None else {}
            selection = self.capsule_context_provider.select(
                task=retrieval_task,
                source_id=source_id,
                source_metadata=source_metadata,
                **seed_options,
            )
            if inspect.isawaitable(selection):
                selection = await selection
            if evidence_graph is not None and evidence_options is not None:
                projection = getattr(evidence_graph, "projection", None)
                if projection is not None and retrieval_input_mode in {"evidence_cards_v1", "evidence_cards_v2", "evidence_cards_v2_1"}:
                    projection.counts["card_seed_ingress"] += 1
                    evidence_graph.record("projection_seed_ingress", iteration=iteration,
                        retrieval_task_chars=len(retrieval_task),
                        full_seed_user_message_chars=getattr(selection, "seed_prompt_chars", None),
                        metrics=evidence_metrics)
            if evidence_graph is not None and evidence_cursor is not None:
                try:
                    # SHADOW persists the same bounded view but supplies the
                    # unchanged legacy task to Seed. First iteration stays raw.
                    evidence_graph.record("view_prepared", iteration=iteration, view=graph_state,
                                          delivered_to_seed=evidence_options is not None,
                                          metrics=evidence_metrics)
                    evidence_graph.finish_retrieval(cursor=evidence_cursor, selection=selection,
                                                    active=evidence_options is not None, metrics=evidence_metrics)
                except Exception as exc:
                    evidence_graph.fail("retrieval_observation", exc)
            select_total_ms = (time.monotonic() - select_started) * 1000
            filter_started = time.monotonic()
            graph_allowed = set(getattr(selection, "exposed_tool_names", ()))
            if not graph_allowed:
                raise RuntimeError("Typed retrieval produced no callable tools.")
            retained_tool_reasons = _called_only_ttl1_retention_reasons(
                iteration=iteration,
                tool_interactions=tool_interactions,
                valid_tool_names=set(base_executors),
            )
            ttl1_retained = {item["tool_name"] for item in retained_tool_reasons}
            final_allowed = graph_allowed | ttl1_retained
            current_tools = [definition for definition in base_tool_definitions if definition["name"] in final_allowed]
            current_executors = {name: executor for name, executor in base_executors.items() if name in final_allowed}
            current_names = {definition["name"] for definition in current_tools}
            if not current_tools or set(current_executors) != current_names:
                raise RuntimeError("Typed retrieved tool exposure is inconsistent with ToolRegistry.")
            tool_definitions_hash = _sha256_json(current_tools)
            tool_filter_ms = (time.monotonic() - filter_started) * 1000
            graph_context_hash = hashlib.sha256(selection.rendered_context.encode()).hexdigest()
            added = sorted(current_names - previous_tools)
            removed = sorted(previous_tools - current_names)
            retained = sorted(previous_tools & current_names)
            previous_tools = set(current_names)
            usage = dict(getattr(selection, "seed_usage", {}) or {})
            stage_timings = dict(getattr(selection, "stage_timings", {}) or {})
            duration_ms = (time.monotonic() - prepare_started) * 1000
            details = {
                "iteration": iteration,
                "retrieval_input_mode": retrieval_input_mode,
                "retrieval_input_hash": retrieval_input_hash,
                "retrieval_input_chars": retrieval_input_chars,
                "retrieval_input_path": retrieval_input_path,
                "retrieval_history_count": retrieval_history_count,
                "retrieval_history_chars": retrieval_history_chars,
                "seed_tools": [item.tool_name for item in getattr(selection, "selected_seed_tools", ())],
                "graph_exposed_tools": sorted(graph_allowed),
                "graph_tool_count": len(graph_allowed),
                "ttl1_retained_tools": sorted(ttl1_retained),
                "retained_tool_count": len(ttl1_retained),
                "ttl1_retention_reasons": retained_tool_reasons,
                "final_exposed_tools": sorted(final_allowed),
                "final_tool_count": len(final_allowed),
                "exposed_tool_names": sorted(final_allowed),
                "visible_tool_count": len(current_tools),
                "graph_context_hash": graph_context_hash,
                "tool_definitions_hash": tool_definitions_hash,
                "executor_tool_names": sorted(current_executors),
                "retrieval_duration_ms": duration_ms,
                "retrieval_duration": duration_ms,
                "prepare_iteration_total_ms": duration_ms,
                "retrieval_state_build_ms": retrieval_state_build_ms,
                "seed_prompt_build_ms": stage_timings.get("seed_prompt_build_ms"),
                "seed_llm_call_ms": stage_timings.get("seed_llm_call_ms"),
                "seed_response_parse_ms": stage_timings.get("seed_response_parse_ms"),
                "graph_expansion_ms": stage_timings.get("graph_expansion_ms"),
                "graph_context_render_ms": stage_timings.get("graph_context_render_ms"),
                "tool_filter_ms": tool_filter_ms,
                "provider_select_total_ms": select_total_ms,
                "seed_prompt_chars": getattr(selection, "seed_prompt_chars", len(getattr(selection, "seed_prompt", ""))),
                "seed_catalog_tool_count": len(getattr(selection, "catalog", ()) or ()),
                "seed_prompt_tokens": usage.get("prompt_tokens"),
                "seed_completion_tokens": usage.get("completion_tokens"),
                "seed_total_tokens": usage.get("total_tokens"),
                "seed_cached_tokens": usage.get("cached_tokens"),
                "seed_reasoning_tokens": usage.get("reasoning_tokens"),
                "seed_usage_source": getattr(selection, "seed_usage_source", "API_REPORTED" if usage else "NOT_AVAILABLE"),
                "seed_call_used": bool(getattr(selection, "seed_call_used", True)),
                "seed_llm_call_success": not bool(getattr(selection, "fallback_used", False)),
                "fallback_used": bool(getattr(selection, "fallback_used", False)),
                "fallback_reason": getattr(selection, "fallback_reason", ""),
                "added_tools": added,
                "removed_tools": removed,
                "retained_tools": retained,
                "rendered_context_chars": getattr(selection, "rendered_chars", 0),
            }
            if evidence_graph is not None:
                details["task_evidence_graph"] = evidence_metrics
            if getattr(selection, "retrieval_policy_version", "") == "typed-can-feed-v1":
                details["retrieval_policy_version"] = selection.retrieval_policy_version
                details["retrieval_diagnostics"] = dict(selection.retrieval_diagnostics)
            self.ctx.trace.emit("iteration_retrieval_completed", status="completed", details=details)
            recorder = getattr(self.capsule_context_provider, "record_iteration", None)
            if recorder is not None:
                recorder(
                    question_id=question_id,
                    iteration=iteration,
                    retrieval_task=retrieval_task,
                    selection=selection,
                    details=details,
                )
            return IterationPreparedContext(
                tools=current_tools,
                executors=current_executors,
                ephemeral_messages=[{"role": "user", "content": selection.rendered_context}],
                metadata=details,
            )

        return prepare_iteration

    def _build_jit_tool_recovery_hook(
        self,
        *,
        base_tool_definitions: list[dict],
        base_executors: dict[str, Any],
    ):
        definitions_by_name = {item["name"]: item for item in base_tool_definitions}
        graph = getattr(self.capsule_context_provider, "graph", None)
        graph_by_tool = getattr(graph, "by_tool_name", {}) if graph is not None else {}

        def recover_tool(tool_name: str, arguments: dict[str, Any], context: dict[str, Any]) -> ToolRecoveryResult:
            registry_exists = tool_name in definitions_by_name and tool_name in base_executors
            graph_node_exists = tool_name in graph_by_tool
            if not registry_exists:
                return ToolRecoveryResult(
                    recovered=False,
                    tool_name=tool_name,
                    registry_exists=False,
                    graph_node_exists=graph_node_exists,
                    decision_reason="REJECT_REGISTRY_MISSING",
                )
            if not graph_node_exists:
                return ToolRecoveryResult(
                    recovered=False,
                    tool_name=tool_name,
                    registry_exists=True,
                    graph_node_exists=False,
                    decision_reason="REJECT_GRAPH_MISSING",
                )
            return ToolRecoveryResult(
                recovered=True,
                tool_name=tool_name,
                tool_definition=definitions_by_name[tool_name],
                executor=base_executors[tool_name],
                registry_exists=True,
                graph_node_exists=True,
                decision_reason="RECOVERED_REGISTRY_AND_GRAPH_VALID",
            )

        return recover_tool

    async def verify_answer_format(self, question: dict, response: str, history: list[dict]) -> str:
        options = "\n".join(
            f"{item.get('letter', '')}. {item.get('text', '')}" for item in question.get("options", [])
        )
        reasoning = next(
            (
                str(item.get("reasoning_content", ""))
                for item in reversed(history)
                if item.get("role") == "assistant" and item.get("reasoning_content")
            ),
            "",
        )
        task = (
            f"Question type: {question.get('question_type')}\n"
            f"Question: {question.get('question_text', '')}\n"
            f"Options:\n{options or '(none)'}\n\n"
            f"Original final-step reasoning, if available:\n{reasoning[-8000:] or '(not available)'}\n\n"
            f"Original response to verify:\n{response}"
        )
        result, _history = await self.llm.tool_call_loop(
            messages=[{"role": "user", "content": task}],
            tools=[], executors={}, system=ANSWER_VERIFICATION_PROMPT,
            trace=self.ctx.trace, max_iterations=1, llm_role="verifier",
        )
        return result


def _build_iteration_retrieval_state(
    *,
    original_task: str,
    source_id: str,
    source_metadata: dict[str, Any],
    tool_interactions: tuple[ToolInteraction, ...],
) -> tuple[str, int, int]:
    interactions = []
    total_chars = 0
    for interaction in reversed(tool_interactions):
        observation = str(interaction.observation)
        item = {
            "iteration": interaction.iteration,
            "tool_name": interaction.tool_name,
            "arguments": _bounded_json_value(interaction.arguments, RETRIEVAL_ARGUMENT_CHAR_LIMIT),
            "status": interaction.status,
            "compact_observation": observation,
        }
        if getattr(interaction, "result_id", None):
            item["result_id"] = getattr(interaction, "result_id", None)
        item_chars = len(json.dumps(item, ensure_ascii=False, sort_keys=True))
        interactions.append(item)
        total_chars += item_chars
    interactions.reverse()
    state = {
        "retrieval_state_schema": RETRIEVAL_STATE_SCHEMA_VERSION,
        "original_task": original_task,
        "initial_evidence_context": {
            "source_id": source_id,
            "source_metadata": source_metadata,
        },
        "previous_tool_interactions": interactions,
        "history_limits": {
            "argument_char_limit": RETRIEVAL_ARGUMENT_CHAR_LIMIT,
            "observation_char_limit": None,
            "total_history_char_limit": None,
            "retention": "most_recent_interactions",
            "tool_result_payload_budget": "shared_active_representation_no_retrieval_clipping",
            "legacy_observation_char_limit": RETRIEVAL_OBSERVATION_CHAR_LIMIT,
            "legacy_total_history_char_limit": RETRIEVAL_HISTORY_CHAR_LIMIT,
        },
    }
    return json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), len(interactions), total_chars


def _called_only_ttl1_retention_reasons(
    *,
    iteration: int,
    tool_interactions: tuple[ToolInteraction, ...],
    valid_tool_names: set[str],
) -> list[dict[str, Any]]:
    if iteration <= 1:
        return []
    retained: dict[str, dict[str, Any]] = {}
    for interaction in tool_interactions:
        if interaction.iteration != iteration - 1:
            continue
        if interaction.status == "UNKNOWN_TOOL":
            continue
        if interaction.tool_name not in valid_tool_names:
            continue
        retained[interaction.tool_name] = {
            "tool_name": interaction.tool_name,
            "source_iteration": interaction.iteration,
            "previous_status": interaction.status,
            "previous_arguments": _bounded_json_value(interaction.arguments, RETRIEVAL_ARGUMENT_CHAR_LIMIT),
        }
    return [retained[name] for name in sorted(retained)]


def _write_retrieval_input_artifact(
    *,
    run_dir: Path,
    question_id: str,
    iteration: int,
    mode: str,
    actual_retrieval_input: str,
    sha256: str,
    chars: int,
    history_count: int,
    history_chars: int,
) -> str:
    root = run_dir / "retrieval_inputs" / _safe_path_component(question_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"iteration_{iteration:03d}.json"
    payload = {
        "question_id": question_id,
        "iteration": iteration,
        "mode": mode,
        "actual_retrieval_input": actual_retrieval_input,
        "sha256": sha256,
        "chars": chars,
        "history_count": history_count,
        "history_chars": history_chars,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path.relative_to(run_dir))


def _safe_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "unknown"


def _compact_observation(value: str) -> str:
    return _truncate_text(value, RETRIEVAL_OBSERVATION_CHAR_LIMIT)


def _bounded_json_value(value: Any, limit: int) -> Any:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded) <= limit:
        return value
    return {"_truncated_json": _truncate_text(encoded, limit)}


def _truncate_text(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    suffix = "\n[truncated]"
    return value[: max(0, limit - len(suffix))] + suffix


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
