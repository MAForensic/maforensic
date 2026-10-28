"""B3 shared context lifecycle and recovery coordination.

This module coordinates already-materialized B2 active representations. It
does not implement profiling, family logic, consumption tracking, budgeting,
or cache behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .representation_runtime import (
    ActiveRepresentation,
    CanonicalFirstRecord,
    ContextProcessor,
    RepresentationDecision,
    RepresentationMode,
    RepresentationRuntime,
)
from .result_store import ResultEnvelope, _atomic_write_text


CONTROL_TOOL_NAME = "recall_tool_result"
VIEW_STORE_SCHEMA_VERSION = "b3-representation-view-store-v1"


class RecoveryStatus(str, Enum):
    REQUESTED = "REQUESTED"
    UNKNOWN_RESULT = "UNKNOWN_RESULT"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    RENDER_FAILED = "RENDER_FAILED"


@dataclass(frozen=True)
class StoredRepresentationView:
    result_id: str
    mode: RepresentationMode
    text: str
    payload_chars: int
    final_chars: int
    truncated: bool
    processor_applied: bool
    decision_reason: str
    processor_id: str = ""
    raw_chars: int = 0
    candidate_chars: int = 0
    structural_boundary_used: bool = False
    benefit_observation: str = "NOT_APPLICABLE"
    recovery_footer_added: bool = False
    fallback_reason: str = ""
    schema_version: str = VIEW_STORE_SCHEMA_VERSION

    @classmethod
    def from_active(cls, active: ActiveRepresentation) -> "StoredRepresentationView":
        return cls(
            result_id=active.result_id,
            mode=active.mode,
            text=active.text,
            payload_chars=active.payload_chars,
            final_chars=active.final_chars,
            truncated=active.truncated,
            processor_applied=active.processor_applied,
            decision_reason=active.decision_reason.value,
            processor_id=active.processor_id,
            raw_chars=active.raw_chars,
            candidate_chars=active.candidate_chars,
            structural_boundary_used=active.structural_boundary_used,
            benefit_observation=active.benefit_observation,
            recovery_footer_added=active.recovery_footer_added,
            fallback_reason=active.fallback_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StoredRepresentationView":
        return cls(
            schema_version=str(data.get("schema_version", VIEW_STORE_SCHEMA_VERSION)),
            result_id=str(data["result_id"]),
            mode=RepresentationMode(str(data["mode"])),
            text=str(data.get("text", "")),
            payload_chars=int(data.get("payload_chars", 0)),
            final_chars=int(data.get("final_chars", 0)),
            truncated=bool(data.get("truncated", False)),
            processor_applied=bool(data.get("processor_applied", False)),
            decision_reason=str(data.get("decision_reason", "")),
            processor_id=str(data.get("processor_id", "")),
            raw_chars=int(data.get("raw_chars", 0)),
            candidate_chars=int(data.get("candidate_chars", 0)),
            structural_boundary_used=bool(data.get("structural_boundary_used", False)),
            benefit_observation=str(data.get("benefit_observation", "NOT_APPLICABLE")),
            recovery_footer_added=bool(data.get("recovery_footer_added", False)),
            fallback_reason=str(data.get("fallback_reason", "")),
        )


class RepresentationViewStore:
    """Run-local store for materialized FIRST/PERSISTENT views."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[tuple[str, RepresentationMode], Path] = {}
        self._canonical_index: dict[str, Path] = {}
        self._load_existing()

    def materialize_canonical_first(
        self,
        *,
        runtime: RepresentationRuntime,
        result_id: str,
        processor: ContextProcessor | None = None,
        processor_inputs: Mapping[str, Any] | None = None,
    ) -> CanonicalFirstRecord:
        existing = self.get_canonical_first(result_id)
        if existing is not None:
            return existing
        canonical = runtime.generate_canonical_first(
            result_id,
            processor=processor,
            processor_inputs=processor_inputs,
        )
        path = self._canonical_path(result_id)
        _atomic_write_text(path, json.dumps(self._canonical_to_dict(canonical), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self._canonical_index[result_id] = path
        return canonical

    def store_canonical_first(self, canonical: CanonicalFirstRecord) -> CanonicalFirstRecord:
        existing = self.get_canonical_first(canonical.result_id)
        if existing is not None:
            if existing != canonical:
                raise RuntimeError(f"Conflicting Canonical FIRST for result_id: {canonical.result_id}")
            return existing
        path = self._canonical_path(canonical.result_id)
        _atomic_write_text(path, json.dumps(self._canonical_to_dict(canonical), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self._canonical_index[canonical.result_id] = path
        return canonical

    def get_canonical_first(self, result_id: str) -> CanonicalFirstRecord | None:
        path = self._canonical_index.get(result_id)
        if path is None:
            candidate = self._canonical_path(result_id)
            if not candidate.exists():
                return None
            path = candidate
            self._canonical_index[result_id] = path
        return self._canonical_from_dict(json.loads(path.read_text(encoding="utf-8")))

    def materialize(
        self,
        *,
        runtime: RepresentationRuntime,
        result_id: str,
        mode: RepresentationMode,
        processor: ContextProcessor | None = None,
        processor_inputs: Mapping[str, Any] | None = None,
        canonical_first: CanonicalFirstRecord | None = None,
    ) -> StoredRepresentationView:
        if mode is RepresentationMode.RECOVERED_ONCE:
            raise ValueError("RECOVERED_ONCE views are rendered on demand and not pre-materialized")
        if canonical_first is not None and canonical_first.result_id != result_id:
            raise ValueError("canonical_first.result_id does not match materialized result_id")
        if mode is RepresentationMode.PERSISTENT and canonical_first is None:
            canonical_first = self.get_canonical_first(result_id)
            if canonical_first is None:
                raise RuntimeError("PERSISTENT materialization requires an existing Canonical FIRST")
        existing = self.get(result_id, mode)
        if existing is not None:
            return existing
        if canonical_first is None:
            if mode is RepresentationMode.FIRST:
                canonical_first = self.materialize_canonical_first(
                    runtime=runtime,
                    result_id=result_id,
                    processor=processor,
                    processor_inputs=processor_inputs,
                )
        if canonical_first is not None and mode is RepresentationMode.FIRST:
            active = runtime.render_canonical_first(canonical_first, mode)
        elif canonical_first is not None and mode is RepresentationMode.PERSISTENT:
            active = runtime.render_persistent_from_canonical(
                canonical_first,
                processor=processor,
                processor_inputs=processor_inputs,
            )
        else:
            active = runtime.render(
                result_id,
                mode,
                processor=processor,
                processor_inputs=processor_inputs,
            )
        view = StoredRepresentationView.from_active(active)
        path = self._path(result_id, mode)
        _atomic_write_text(path, json.dumps(view.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self._index[(result_id, mode)] = path
        return view

    def get(self, result_id: str, mode: RepresentationMode) -> StoredRepresentationView | None:
        path = self._index.get((result_id, mode))
        if path is None or not path.exists():
            return None
        return StoredRepresentationView.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def has(self, result_id: str, mode: RepresentationMode) -> bool:
        return self.get(result_id, mode) is not None

    def _path(self, result_id: str, mode: RepresentationMode) -> Path:
        return self.root / result_id / f"{mode.value.lower()}.json"

    def _canonical_path(self, result_id: str) -> Path:
        return self.root / result_id / "canonical_first" / "record.json"

    def _load_existing(self) -> None:
        if not self.root.exists():
            return
        for path in self.root.glob("*/*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                result_id = str(data["result_id"])
                mode = RepresentationMode(str(data["mode"]))
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                continue
            self._index[(result_id, mode)] = path
        for path in self.root.glob("*/canonical_first/record.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                result_id = str(data["result_id"])
            except (OSError, KeyError, json.JSONDecodeError):
                continue
            self._canonical_index[result_id] = path

    @staticmethod
    def _canonical_to_dict(canonical: CanonicalFirstRecord) -> dict[str, Any]:
        return {
            "schema_version": canonical.schema_version,
            "result_id": canonical.result_id,
            "text": canonical.text,
            "preferred_cut_points": list(canonical.preferred_cut_points),
            "processor_id": canonical.processor_id,
            "requires_recovery": canonical.requires_recovery,
            "processor_applied": canonical.processor_applied,
            "decision_reason": canonical.decision_reason.value,
            "raw_chars": canonical.raw_chars,
            "candidate_chars": canonical.candidate_chars,
            "structural_boundary_used": canonical.structural_boundary_used,
            "benefit_observation": canonical.benefit_observation,
            "fallback_reason": canonical.fallback_reason,
        }

    @staticmethod
    def _canonical_from_dict(data: Mapping[str, Any]) -> CanonicalFirstRecord:
        return CanonicalFirstRecord(
            schema_version=str(data.get("schema_version", "b2-canonical-first-v1")),
            result_id=str(data["result_id"]),
            text=str(data.get("text", "")),
            preferred_cut_points=tuple(
                int(item)
                for item in data.get("preferred_cut_points", [])
                if isinstance(item, int) and not isinstance(item, bool)
            ),
            processor_id=str(data.get("processor_id", "")),
            requires_recovery=bool(data.get("requires_recovery", False)),
            processor_applied=bool(data.get("processor_applied", False)),
            decision_reason=RepresentationDecision(str(data.get("decision_reason", RepresentationDecision.RAW_NO_PROCESSOR.value))),
            raw_chars=int(data.get("raw_chars", 0)),
            candidate_chars=int(data.get("candidate_chars", 0)),
            structural_boundary_used=bool(data.get("structural_boundary_used", False)),
            benefit_observation=str(data.get("benefit_observation", "NOT_APPLICABLE")),
            fallback_reason=str(data.get("fallback_reason", "")),
        )


@dataclass(frozen=True)
class SnapshotEntry:
    tool_call_id: str
    result_id: str | None
    mode: RepresentationMode | str
    text: str
    status: str
    tool_name: str
    recovery_footer_added: bool = False
    fallback: bool = False


@dataclass(frozen=True)
class SharedContextSnapshot:
    iteration: int
    recovery_followup: bool
    entries: tuple[SnapshotEntry, ...]

    @property
    def has_recoverable_view(self) -> bool:
        return any(entry.recovery_footer_added for entry in self.entries)

    def by_tool_call_id(self) -> dict[str, SnapshotEntry]:
        return {entry.tool_call_id: entry for entry in self.entries if entry.tool_call_id}

    def by_result_id(self) -> dict[str, SnapshotEntry]:
        return {entry.result_id: entry for entry in self.entries if entry.result_id}


@dataclass
class RecoveryRequest:
    result_id: str
    control_tool_call_id: str
    status: RecoveryStatus
    message: str
    recovered_text: str = ""
    recovery_footer_added: bool = False


def recall_tool_definition() -> dict[str, Any]:
    return {
        "name": CONTROL_TOOL_NAME,
        "description": (
            "Recover a previously shortened tool result by result_id when the active context "
            "shows full_result_available=true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": "The result_id shown in a shortened tool result footer.",
                }
            },
            "required": ["result_id"],
        },
    }


def recovery_placeholder(result_id: str, status: RecoveryStatus) -> str:
    if status is RecoveryStatus.REQUESTED:
        return f"[recall_tool_result status=success result_id={result_id}]"
    if status is RecoveryStatus.UNKNOWN_RESULT:
        return f"[recall_tool_result status=unknown_result_id result_id={result_id}]"
    if status is RecoveryStatus.RENDER_FAILED:
        return f"[recall_tool_result status=render_failed result_id={result_id}]"
    return f"[recall_tool_result status=invalid_arguments result_id={result_id}]"


def is_final_answer_text(text: str) -> bool:
    stripped = (text or "").lstrip()
    return stripped.startswith("ANSWER:") or stripped.startswith("NO ANSWER:")


class SharedContextLifecycle:
    """B3 coordinator for view materialization, snapshots, and recall control."""

    def __init__(
        self,
        *,
        runtime: RepresentationRuntime,
        view_store: RepresentationViewStore,
        processor_provider: Callable[[ResultEnvelope, RepresentationMode], ContextProcessor | None] | None = None,
        processor_inputs_provider: Callable[[ResultEnvelope, RepresentationMode], Mapping[str, Any] | None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.view_store = view_store
        self.processor_provider = processor_provider
        self.processor_inputs_provider = processor_inputs_provider
        self.recovery_request: RecoveryRequest | None = None
        self.last_snapshot: SharedContextSnapshot | None = None
        self.snapshot_build_count = 0
        self.materialize_calls = 0

    def materialize_first_persistent(self, envelope: ResultEnvelope) -> tuple[StoredRepresentationView, StoredRepresentationView]:
        views = []
        canonical = self.view_store.materialize_canonical_first(
            runtime=self.runtime,
            result_id=envelope.result_id,
            processor=self._processor(envelope, RepresentationMode.FIRST),
            processor_inputs=self._processor_inputs(envelope, RepresentationMode.FIRST),
        )
        for mode in (RepresentationMode.FIRST, RepresentationMode.PERSISTENT):
            processor = None if mode is RepresentationMode.FIRST else self._processor(envelope, mode)
            processor_inputs = None if mode is RepresentationMode.FIRST else self._processor_inputs(envelope, mode)
            before = self.view_store.has(envelope.result_id, mode)
            view = self.view_store.materialize(
                runtime=self.runtime,
                result_id=envelope.result_id,
                mode=mode,
                processor=processor,
                processor_inputs=processor_inputs,
                canonical_first=canonical,
            )
            if not before:
                self.materialize_calls += 1
            views.append(view)
        return views[0], views[1]

    def build_snapshot(
        self,
        *,
        iteration: int,
        interactions: tuple[Any, ...],
    ) -> SharedContextSnapshot:
        recovery = self.recovery_request
        entries = []
        for interaction in interactions:
            result_id = getattr(interaction, "result_id", None)
            if result_id:
                # New results keep FIRST on their first consuming request, including recovery followups.
                mode = (
                    RepresentationMode.FIRST
                    if int(getattr(interaction, "iteration", 0)) == iteration - 1
                    else RepresentationMode.PERSISTENT
                    if recovery is not None and recovery.status is RecoveryStatus.REQUESTED
                    else RepresentationMode.PERSISTENT
                )
                view = self.view_store.get(result_id, mode)
                if view is None:
                    mode = RepresentationMode.PERSISTENT
                    view = self.view_store.get(result_id, mode)
                if view is not None:
                    entries.append(SnapshotEntry(
                        tool_call_id=str(getattr(interaction, "tool_call_id", "")),
                        result_id=result_id,
                        mode=mode,
                        text=view.text,
                        status=str(getattr(interaction, "status", "")),
                        tool_name=str(getattr(interaction, "tool_name", "")),
                        recovery_footer_added=view.recovery_footer_added,
                    ))
                    continue
            entries.append(SnapshotEntry(
                tool_call_id=str(getattr(interaction, "tool_call_id", "")),
                result_id=result_id,
                mode="FALLBACK",
                text=str(getattr(interaction, "observation", "")),
                status=str(getattr(interaction, "status", "")),
                tool_name=str(getattr(interaction, "tool_name", "")),
                fallback=True,
            ))
        snapshot = SharedContextSnapshot(
            iteration=iteration,
            recovery_followup=recovery is not None and recovery.status is RecoveryStatus.REQUESTED,
            entries=tuple(entries),
        )
        self.last_snapshot = snapshot
        self.snapshot_build_count += 1
        return snapshot

    def active_messages(self, history: list[dict[str, Any]], snapshot: SharedContextSnapshot) -> list[dict[str, Any]]:
        by_tool_call_id = snapshot.by_tool_call_id()
        active = []
        recovery = self.recovery_request
        for message in history:
            copied = dict(message)
            if copied.get("role") == "tool":
                tool_call_id = str(copied.get("tool_call_id") or "")
                entry = by_tool_call_id.get(tool_call_id)
                if entry is not None:
                    copied["content"] = entry.text
                elif recovery is not None and tool_call_id == recovery.control_tool_call_id and recovery.status is RecoveryStatus.REQUESTED:
                    copied["content"] = recovery.recovered_text
            active.append(copied)
        return active

    def active_interactions(self, interactions: tuple[Any, ...], snapshot: SharedContextSnapshot) -> tuple[Any, ...]:
        by_tool_call_id = snapshot.by_tool_call_id()
        updated = []
        for interaction in interactions:
            entry = by_tool_call_id.get(str(getattr(interaction, "tool_call_id", "")))
            if entry is None:
                updated.append(interaction)
                continue
            replace = getattr(interaction, "__dataclass_fields__", None)
            if replace and "observation" in replace:
                from dataclasses import replace as dataclass_replace

                updated.append(dataclass_replace(interaction, observation=entry.text))
            else:
                updated.append(interaction)
        return tuple(updated)

    def maybe_add_control_tool(self, tools: list[dict[str, Any]], snapshot: SharedContextSnapshot) -> list[dict[str, Any]]:
        if not snapshot.has_recoverable_view:
            return tools
        if any(tool.get("name") == CONTROL_TOOL_NAME for tool in tools):
            return tools
        return [*tools, recall_tool_definition()]

    def request_recovery(self, *, result_id: str, tool_call_id: str) -> RecoveryRequest:
        if not result_id:
            request = RecoveryRequest(
                result_id="",
                control_tool_call_id=tool_call_id,
                status=RecoveryStatus.INVALID_ARGUMENTS,
                message=recovery_placeholder("", RecoveryStatus.INVALID_ARGUMENTS),
            )
            return request
        if not self.runtime.result_store.has_result(result_id):
            request = RecoveryRequest(
                result_id=result_id,
                control_tool_call_id=tool_call_id,
                status=RecoveryStatus.UNKNOWN_RESULT,
                message=recovery_placeholder(result_id, RecoveryStatus.UNKNOWN_RESULT),
            )
            return request
        try:
            envelope = self.runtime.result_store.get_result(result_id)
            canonical = self.view_store.materialize_canonical_first(
                runtime=self.runtime,
                result_id=result_id,
                processor=self._processor(envelope, RepresentationMode.FIRST),
                processor_inputs=self._processor_inputs(envelope, RepresentationMode.FIRST),
            )
            active = self.runtime.render_canonical_first(canonical, RepresentationMode.RECOVERED_ONCE)
        except Exception:
            return RecoveryRequest(
                result_id=result_id,
                control_tool_call_id=tool_call_id,
                status=RecoveryStatus.RENDER_FAILED,
                message=recovery_placeholder(result_id, RecoveryStatus.RENDER_FAILED),
            )
        request = RecoveryRequest(
            result_id=result_id,
            control_tool_call_id=tool_call_id,
            status=RecoveryStatus.REQUESTED,
            message=recovery_placeholder(result_id, RecoveryStatus.REQUESTED),
            recovered_text=active.text,
            recovery_footer_added=active.recovery_footer_added,
        )
        self.recovery_request = request
        return request

    def clear_recovery(self) -> None:
        self.recovery_request = None

    def _processor(self, envelope: ResultEnvelope, mode: RepresentationMode) -> ContextProcessor | None:
        if self.processor_provider is None:
            return None
        return self.processor_provider(envelope, mode)

    def _processor_inputs(self, envelope: ResultEnvelope, mode: RepresentationMode) -> Mapping[str, Any] | None:
        if self.processor_inputs_provider is None:
            return None
        return self.processor_inputs_provider(envelope, mode)
