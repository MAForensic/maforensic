"""Structured append-only run trace for Profile 0."""

from __future__ import annotations

import contextvars
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
import uuid
from typing import Any, Mapping

EVENT_TYPES = {
    "tool_call_started", "tool_call_succeeded", "tool_call_empty",
    "tool_call_failed", "tool_call_timed_out", "unknown_tool_call",
    "invalid_tool_arguments", "question_timed_out", "iteration_exhausted",
    "agent_decision_summary", "answer_formed", "answer_not_formed",
    "answer_verification_started", "answer_verification_completed", "answer_verification_failed",
    "capability_gap_observation",  # Legacy trace compatibility only.
    "iteration_retrieval_completed",
    "jit_tool_recovery_attempt", "jit_tool_recovery_completed",
    "llm_call_started", "llm_call_completed", "llm_call_failed",
    "question_started", "question_completed", "question_failed",
    "run_started", "run_completed", "run_failed", "run_interrupted",
}

_question_id = contextvars.ContextVar("profile0_question_id", default="")
_task_id = contextvars.ContextVar("profile0_task_id", default="")
_source_id = contextvars.ContextVar("profile0_source_id", default="")

CAPSULE_ANNOTATED_EVENT_TYPES = {
    "tool_call_started", "tool_call_succeeded", "tool_call_empty",
    "tool_call_failed", "tool_call_timed_out", "invalid_tool_arguments",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _seconds_between(start: str, end: str) -> float | None:
    started = _parse_utc(start)
    finished = _parse_utc(end)
    if started is None or finished is None:
        return None
    return max(0.0, (finished - started).total_seconds())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


@dataclass
class TraceEvent:
    event_type: str
    event_id: str = field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}")
    run_id: str = ""
    question_id: str = ""
    task_id: str = ""
    source_id: str = ""
    tool_name: str = ""
    capability_category: str = ""
    args: dict = field(default_factory=dict)
    status: str = ""
    reason_code: str = ""
    input_paths: list[str] = field(default_factory=list)
    output_preview: str = ""
    output_path: str = ""
    output_sha256: str = ""
    evidence_paths: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    duration_ms: float | None = None
    error: str = ""
    gap_type: str = ""
    evidence_path: str = ""
    observation: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class TraceRecorder:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.path = run_dir / "trace_events.jsonl"
        self.segment_path = run_dir / "execution_segments.json"
        self.events: list[TraceEvent] = []
        self._capsule_annotations: Mapping[str, Mapping[str, str]] = MappingProxyType({})
        self._segment_started_monotonic = time.monotonic()
        self._current_segment_id = f"segment-{uuid.uuid4().hex[:12]}"
        self._segments: list[dict[str, Any]] = []
        self._load_existing_events()
        self._start_execution_segment()

    def _load_existing_events(self) -> None:
        if not self.path.exists():
            return
        event_fields = {item.name for item in fields(TraceEvent)}
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict) or raw.get("event_type") not in EVENT_TYPES:
                    continue
                filtered = {key: value for key, value in raw.items() if key in event_fields}
                self.events.append(TraceEvent(**filtered))

    def _load_segments(self) -> list[dict[str, Any]]:
        if not self.segment_path.exists():
            return []
        try:
            data = json.loads(self.segment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        segments = data.get("execution_segments", data) if isinstance(data, dict) else data
        if not isinstance(segments, list):
            return []
        return [dict(item) for item in segments if isinstance(item, dict)]

    def _write_segments(self) -> None:
        totals = self.segment_totals(update_current=False)
        _atomic_json(
            self.segment_path,
            {
                "schema_version": "baseline-execution-segments-v1",
                "run_id": self.run_id,
                "execution_segments": self._segments,
                **totals,
            },
        )

    def _start_execution_segment(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._segments = self._load_segments()
        for segment in self._segments:
            if segment.get("status") == "running":
                last = str(segment.get("last_heartbeat_at") or segment.get("started_at") or "")
                if not segment.get("active_wall_clock_seconds"):
                    recovered = _seconds_between(str(segment.get("started_at", "")), last)
                    if recovered is not None:
                        segment["active_wall_clock_seconds"] = recovered
                segment["finished_at"] = last
                segment["status"] = "recovered_after_interruption"
                segment["exit_status"] = "recovered_after_interruption"
        now = utc_now()
        self._segments.append({
            "segment_id": self._current_segment_id,
            "pid": os.getpid(),
            "started_at": now,
            "last_heartbeat_at": now,
            "finished_at": "",
            "active_wall_clock_seconds": 0.0,
            "exit_status": "",
            "status": "running",
        })
        self._write_segments()

    def _current_segment(self) -> dict[str, Any] | None:
        for segment in reversed(self._segments):
            if segment.get("segment_id") == self._current_segment_id:
                return segment
        return None

    def heartbeat_segment(self) -> None:
        segment = self._current_segment()
        if segment is None or segment.get("status") != "running":
            return
        segment["last_heartbeat_at"] = utc_now()
        segment["active_wall_clock_seconds"] = max(0.0, time.monotonic() - self._segment_started_monotonic)
        self._write_segments()

    def finalize_segment(self, exit_status: str = "completed") -> None:
        segment = self._current_segment()
        if segment is None:
            return
        self.heartbeat_segment()
        segment["finished_at"] = segment.get("last_heartbeat_at") or utc_now()
        segment["exit_status"] = exit_status
        segment["status"] = "finalized"
        self._write_segments()

    def segment_totals(self, *, update_current: bool = True) -> dict[str, Any]:
        if update_current:
            self.heartbeat_segment()
        first_started = ""
        final_finished = ""
        active = 0.0
        for segment in self._segments:
            started = str(segment.get("started_at", ""))
            finished = str(
                segment.get("finished_at")
                or segment.get("last_heartbeat_at")
                or segment.get("started_at")
                or ""
            )
            if started and (not first_started or started < first_started):
                first_started = started
            if finished and (not final_finished or finished > final_finished):
                final_finished = finished
            active += float(segment.get("active_wall_clock_seconds") or 0.0)
        calendar = _seconds_between(first_started, final_finished)
        return {
            "execution_segment_count": len(self._segments),
            "experiment_first_started_at": first_started,
            "experiment_final_finished_at": final_finished,
            "experiment_elapsed_calendar_seconds": calendar,
            "total_active_wall_clock_seconds": active,
            "total_active_wall_clock_minutes": active / 60,
        }

    def configure_capsule_annotations(
        self, annotations: Mapping[str, Mapping[str, str]] | None,
    ) -> None:
        self._capsule_annotations = MappingProxyType({
            str(tool_name): MappingProxyType(dict(annotation))
            for tool_name, annotation in (annotations or {}).items()
        })

    def set_context(
        self, *, question_id: str | None = None, task_id: str | None = None,
        source_id: str | None = None,
    ) -> None:
        if question_id is not None:
            _question_id.set(str(question_id))
        if task_id is not None:
            _task_id.set(task_id)
        if source_id is not None:
            _source_id.set(source_id)

    def emit(self, event_type: str, **kwargs: Any) -> TraceEvent:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported trace event type: {event_type}")
        tool_name = str(kwargs.get("tool_name", ""))
        annotation = (
            self._capsule_annotations.get(tool_name)
            if event_type in CAPSULE_ANNOTATED_EVENT_TYPES and tool_name
            else None
        )
        if annotation is not None:
            details = dict(kwargs.get("details") or {})
            existing = details.get("capsule")
            expected = dict(annotation)
            if existing is not None and existing != expected:
                raise ValueError(f"Conflicting capsule trace annotation for tool: {tool_name}")
            details["capsule"] = expected
            kwargs["details"] = details
        event = TraceEvent(
            event_type=event_type,
            run_id=self.run_id,
            question_id=kwargs.pop("question_id", _question_id.get()),
            task_id=kwargs.pop("task_id", _task_id.get()),
            source_id=kwargs.pop("source_id", _source_id.get()),
            **kwargs,
        )
        self.events.append(event)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self.heartbeat_segment()
        return event

    def observe_gap(self, gap_type: str, evidence_path: str, observation: str) -> TraceEvent:
        """Read/write compatibility helper for legacy traces; not used by Profile 0."""
        return self.emit(
            "capability_gap_observation",
            status="observed",
            gap_type=gap_type,
            evidence_path=evidence_path,
            evidence_paths=[evidence_path] if evidence_path else [],
            observation=observation,
        )

    def flush(self) -> None:
        """Events are append-flushed on emit; fsync the trace for graceful shutdown."""
        if not self.path.exists():
            return
        with self.path.open("a", encoding="utf-8") as stream:
            stream.flush()
            __import__("os").fsync(stream.fileno())

    def token_usage(self, question_id: str | None = None) -> dict[str, int]:
        events = [
            event for event in self.events
            if event.event_type == "llm_call_completed"
            and (question_id is None or event.question_id == str(question_id))
        ]
        reported = [event for event in events if event.total_tokens is not None]
        main_events = [event for event in reported if event.details.get("llm_role", "main") == "main"]
        verifier_events = [event for event in reported if event.details.get("llm_role") == "verifier"]
        seed_events = [
            event for event in self.events
            if event.event_type == "iteration_retrieval_completed"
            and (question_id is None or event.question_id == str(question_id))
        ]

        def sum_events(items: list[TraceEvent]) -> dict[str, int]:
            return {
                "input_tokens": sum(event.prompt_tokens or 0 for event in items),
                "output_tokens": sum(event.completion_tokens or 0 for event in items),
                "reasoning_tokens": sum(event.reasoning_tokens or 0 for event in items),
                "cached_tokens": sum(event.cached_tokens or 0 for event in items),
                "total_tokens": sum(event.total_tokens or 0 for event in items),
            }

        def sum_seed(name: str) -> int:
            return sum(int(event.details.get(name) or 0) for event in seed_events)

        main = sum_events(main_events)
        verifier = sum_events(verifier_events)
        seed = {
            "input_tokens": sum_seed("seed_prompt_tokens"),
            "output_tokens": sum_seed("seed_completion_tokens"),
            "reasoning_tokens": sum_seed("seed_reasoning_tokens"),
            "cached_tokens": sum_seed("seed_cached_tokens"),
            "total_tokens": sum_seed("seed_total_tokens"),
        }
        total = {
            key: main[key] + verifier[key] + seed[key]
            for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "total_tokens")
        }
        return {
            "llm_calls": len(events),
            "main_llm_calls": len(main_events),
            "seed_llm_calls": sum(
                1 for event in seed_events
                if event.details.get("seed_call_used") or event.details.get("seed_usage_source") in {"API_REPORTED", "NOT_AVAILABLE"}
            ),
            "verifier_llm_calls": len(verifier_events),
            "usage_reported_calls": len(reported),
            "usage_missing_calls": len(events) - len(reported),
            "seed_usage_reported_calls": sum(1 for event in seed_events if event.details.get("seed_total_tokens") is not None),
            "seed_usage_missing_calls": sum(
                1 for event in seed_events
                if (event.details.get("seed_call_used") or event.details.get("seed_usage_source") in {"API_REPORTED", "NOT_AVAILABLE"})
                and event.details.get("seed_total_tokens") is None
            ),
            "main_input_tokens": main["input_tokens"],
            "main_output_tokens": main["output_tokens"],
            "main_reasoning_tokens": main["reasoning_tokens"],
            "main_cached_tokens": main["cached_tokens"],
            "main_total_tokens": main["total_tokens"],
            "seed_input_tokens": seed["input_tokens"],
            "seed_output_tokens": seed["output_tokens"],
            "seed_reasoning_tokens": seed["reasoning_tokens"],
            "seed_cached_tokens": seed["cached_tokens"],
            "seed_total_tokens": seed["total_tokens"],
            "verifier_input_tokens": verifier["input_tokens"],
            "verifier_output_tokens": verifier["output_tokens"],
            "verifier_reasoning_tokens": verifier["reasoning_tokens"],
            "verifier_cached_tokens": verifier["cached_tokens"],
            "verifier_total_tokens": verifier["total_tokens"],
            "total_input_tokens": total["input_tokens"],
            "total_output_tokens": total["output_tokens"],
            "reasoning_tokens": total["reasoning_tokens"],
            "cached_tokens": total["cached_tokens"],
            "total_tokens": total["total_tokens"],
            "prompt_tokens": total["input_tokens"],
            "completion_tokens": total["output_tokens"],
        }
