"""Atomic audit manifest for enabled capsule-aware prompt context."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..processor_mapping import CENTRAL_TOOL_IDS

SCHEMA_VERSION = "capsule-runtime-manifest-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()
    return _sha256_bytes(encoded)


def _profile0_document_hash(filename: str) -> str:
    path = Path(__file__).with_name("profile0") / filename
    return _sha256_bytes(path.read_bytes())


def _question_uid(task: str, selection: Any) -> str:
    for label in ("Question UID:", "Question ID:"):
        for line in task.splitlines():
            if line.startswith(label):
                value = line.partition(":")[2].strip()
                if value:
                    return value
    query_hash = getattr(selection, "query_sha256", "") or getattr(selection, "seed_prompt_sha256", "")
    return f"query-{query_hash[:16] or 'unknown'}"






class CapsuleRetrievalV2RuntimeManifestRecorder:
    def __init__(self, run_dir: Path, provider: Any) -> None:
        self.path = run_dir / "capsule_runtime_manifest.json"
        settings = provider.settings
        self.document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "profile_id": "profile0",
            "enabled": True,
            "mode": settings.mode,
            "provider_version": provider.provider_version,
            "retrieval_policy_version": provider.retrieval_policy_version,
            "seed_selection_prompt_version": getattr(provider, "seed_prompt_version", "seed-selection-typed-can-feed-v1"),
            "seed_output_schema_version": getattr(provider, "seed_schema_version", "seed-direction-selection-v1"),
            "config": {
                "max_seed_tools": settings.max_seed_tools,
                "max_nodes": settings.max_nodes,
                "max_edges": settings.max_edges,
                "max_context_chars": settings.max_context_chars,
                "annotate_trace": settings.annotate_trace,
                "seed_failure_fallback": settings.seed_failure_fallback,
                "source_evidence_types": dict(settings.source_evidence_types),
            },
            "retrieval_graph_sha256": _profile0_document_hash(getattr(provider, "graph_asset_name", "typed_can_feed_graph.yaml")),
            "registry_compatibility": {
                "expected_tools": len(CENTRAL_TOOL_IDS),
                "mapped_tools": len(provider.graph.by_tool_name),
                "passed": set(provider.graph.by_tool_name) == CENTRAL_TOOL_IDS,
            },
            "allowed_graph_hints": list(getattr(provider, "graph_hints", ("can_feed",))),
            "tool_filtering": True,
            "graph_execution": False,
            "automatic_requires": False,
            "automatic_fallback": False,
            "question_contexts": {},
        }

    def write_static(self) -> Path:
        self._atomic_write()
        return self.path

    def record_question(self, task: str, effective_task: str, selection: Any) -> None:
        uid = _question_uid(task, selection)
        usage = dict(selection.seed_usage)
        record = {
            "retrieval_version": selection.retrieval_policy_version,
            "source_id": selection.source_id,
            "evidence_type": selection.evidence_type,
            "evidence_compatible_catalog_sha256": selection.evidence_compatible_catalog_sha256,
            "llm_seed_prompt_sha256": selection.seed_prompt_sha256,
            "llm_seed_response_sha256": selection.seed_response_sha256,
            "selected_seed_tools": [item.to_manifest_dict() for item in selection.selected_seed_tools],
            "invalid_seed_attempts": list(selection.invalid_seed_attempts),
            "graph_query_types": list(selection.graph_query_types),
            "retrieved_node_ids": list(selection.retrieved_node_ids),
            "retrieved_edge_ids": list(selection.retrieved_edge_ids),
            "graph_exposed_tools": list(selection.exposed_tool_names),
            "graph_tool_count": len(selection.exposed_tool_names),
            "ttl1_retained_tools": [],
            "retained_tool_count": 0,
            "ttl1_retention_reasons": [],
            "final_exposed_tools": list(selection.exposed_tool_names),
            "final_tool_count": len(selection.exposed_tool_names),
            "exposed_tool_names": list(selection.exposed_tool_names),
            "exposed_tool_count": len(selection.exposed_tool_names),
            "rendered_context_sha256": selection.context_sha256,
            "prompt_context_sha256": selection.context_sha256,
            "effective_task_sha256": _sha256_bytes(effective_task.encode()),
            "fallback_used": selection.fallback_used,
            "fallback_reason": selection.fallback_reason,
            "context_truncated": selection.truncated,
            "rendered_context_chars": selection.rendered_chars,
            "seed_call_used": selection.seed_call_used,
            "seed_model": selection.seed_model,
            "seed_response_model": selection.seed_response_model,
            "seed_usage_source": getattr(selection, "seed_usage_source", "API_REPORTED" if usage else "NOT_AVAILABLE"),
            "retrieval_diagnostics": dict(getattr(selection, "retrieval_diagnostics", {})),
        }
        if usage:
            record.update({
                "seed_prompt_tokens": usage.get("prompt_tokens", 0),
                "seed_completion_tokens": usage.get("completion_tokens", 0),
                "seed_total_tokens": usage.get("total_tokens", 0),
                "seed_cached_tokens": usage.get("cached_tokens", 0),
                "seed_reasoning_tokens": usage.get("reasoning_tokens", 0),
            })
        self.document["question_contexts"][_question_uid(task, selection)] = record
        self._atomic_write()

    def record_iteration(
        self,
        *,
        question_id: str,
        iteration: int,
        retrieval_task: str,
        selection: Any,
        details: Mapping[str, Any],
    ) -> None:
        usage = dict(selection.seed_usage)
        question = self.document["question_contexts"].setdefault(question_id, {
            "retrieval_version": selection.retrieval_policy_version,
            "source_id": selection.source_id,
            "evidence_type": selection.evidence_type,
            "iterations": [],
        })
        question.setdefault("iterations", []).append({
            "iteration": iteration,
            "retrieval_input_mode": details.get("retrieval_input_mode", ""),
            "retrieval_input_sha256": _sha256_bytes(retrieval_task.encode()),
            "retrieval_input_hash": details.get("retrieval_input_hash", ""),
            "retrieval_input_chars": details.get("retrieval_input_chars", len(retrieval_task)),
            "retrieval_input_path": details.get("retrieval_input_path", ""),
            "retrieval_history_count": details.get("retrieval_history_count", 0),
            "retrieval_history_chars": details.get("retrieval_history_chars", 0),
            "selected_seed_tools": [item.to_manifest_dict() for item in selection.selected_seed_tools],
            "graph_query_types": list(selection.graph_query_types),
            "retrieval_diagnostics": dict(getattr(selection, "retrieval_diagnostics", {})),
            "retrieved_node_ids": list(selection.retrieved_node_ids),
            "retrieved_edge_ids": list(selection.retrieved_edge_ids),
            "graph_exposed_tools": details.get("graph_exposed_tools", list(selection.exposed_tool_names)),
            "graph_tool_count": details.get("graph_tool_count", len(selection.exposed_tool_names)),
            "ttl1_retained_tools": details.get("ttl1_retained_tools", []),
            "retained_tool_count": details.get("retained_tool_count", 0),
            "ttl1_retention_reasons": details.get("ttl1_retention_reasons", []),
            "final_exposed_tools": details.get("final_exposed_tools", details.get("exposed_tool_names", list(selection.exposed_tool_names))),
            "final_tool_count": details.get("final_tool_count", details.get("visible_tool_count", len(selection.exposed_tool_names))),
            "exposed_tool_names": details.get("exposed_tool_names", list(selection.exposed_tool_names)),
            "exposed_tool_count": details.get("visible_tool_count", len(selection.exposed_tool_names)),
            "rendered_context_sha256": selection.context_sha256,
            "graph_context_hash": details.get("graph_context_hash", selection.context_sha256),
            "tool_definitions_sha256": details.get("tool_definitions_hash", ""),
            "tool_definitions_hash": details.get("tool_definitions_hash", ""),
            "executor_tool_names": details.get("executor_tool_names", []),
            "retrieval_duration_ms": details.get("retrieval_duration_ms", 0),
            "retrieval_duration": details.get("retrieval_duration", details.get("retrieval_duration_ms", 0)),
            "prepare_iteration_total_ms": details.get("prepare_iteration_total_ms", details.get("retrieval_duration_ms", 0)),
            "retrieval_state_build_ms": details.get("retrieval_state_build_ms"),
            "seed_prompt_build_ms": details.get("seed_prompt_build_ms"),
            "seed_llm_call_ms": details.get("seed_llm_call_ms"),
            "seed_response_parse_ms": details.get("seed_response_parse_ms"),
            "graph_expansion_ms": details.get("graph_expansion_ms"),
            "graph_context_render_ms": details.get("graph_context_render_ms"),
            "tool_filter_ms": details.get("tool_filter_ms"),
            "provider_select_total_ms": details.get("provider_select_total_ms"),
            "added_tools": details.get("added_tools", []),
            "removed_tools": details.get("removed_tools", []),
            "retained_tools": details.get("retained_tools", []),
            "fallback_used": selection.fallback_used,
            "fallback_reason": selection.fallback_reason,
            "context_truncated": selection.truncated,
            "rendered_context_chars": selection.rendered_chars,
            "seed_call_used": selection.seed_call_used,
            "seed_model": selection.seed_model,
            "seed_response_model": selection.seed_response_model,
            "seed_prompt_chars": details.get("seed_prompt_chars", getattr(selection, "seed_prompt_chars", 0)),
            "seed_catalog_tool_count": details.get("seed_catalog_tool_count", 0),
            "seed_usage_source": details.get("seed_usage_source", getattr(selection, "seed_usage_source", "API_REPORTED" if usage else "NOT_AVAILABLE")),
            "seed_prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "seed_completion_tokens": usage.get("completion_tokens") if usage else None,
            "seed_total_tokens": usage.get("total_tokens") if usage else None,
            "seed_cached_tokens": usage.get("cached_tokens") if usage else None,
            "seed_reasoning_tokens": usage.get("reasoning_tokens") if usage else None,
        })
        self._atomic_write()

    def _atomic_write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        payload = json.dumps(self.document, ensure_ascii=False, indent=2) + "\n"
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


class ManifestRecordingCapsuleRetrievalV2Provider:
    """Delegate Typed selection unchanged and record exact context identities."""

    provider_version = "capsule-retrieval-typed-provider"
    retrieval_policy_version = "typed-can-feed-v1"

    def __init__(self, provider: Any, recorder: CapsuleRetrievalV2RuntimeManifestRecorder) -> None:
        self._provider = provider
        self._recorder = recorder
        self.settings = provider.settings
        self.graph = provider.graph
        self.llm = provider.llm
        self.retrieval_policy_version = provider.retrieval_policy_version
        self.provider_version = provider.provider_version

    @property
    def supports_task_evidence_graph(self) -> bool:
        return bool(getattr(self._provider, "supports_task_evidence_graph", False))

    def resolve_evidence_type(self, **kwargs: Any) -> str:
        return self._provider.resolve_evidence_type(**kwargs)

    def build_evidence_compatible_catalog(self, *args: Any, **kwargs: Any) -> Any:
        return self._provider.build_evidence_compatible_catalog(*args, **kwargs)

    async def select(self, **kwargs: Any) -> Any:
        return await self._provider.select(**kwargs)

    def append_to_task(self, task: str, selection: Any) -> str:
        effective_task = self._provider.append_to_task(task, selection)
        self._recorder.record_question(task, effective_task, selection)
        return effective_task

    def record_iteration(
        self,
        *,
        question_id: str,
        iteration: int,
        retrieval_task: str,
        selection: Any,
        details: Mapping[str, Any],
    ) -> None:
        self._recorder.record_iteration(
            question_id=question_id,
            iteration=iteration,
            retrieval_task=retrieval_task,
            selection=selection,
            details=details,
        )

    def tool_annotations(self) -> Mapping[str, Mapping[str, str]]:
        return self._provider.tool_annotations()
