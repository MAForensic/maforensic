"""Formal raw Tool Interaction Result store.

The store is the B1 source of truth for the complete string representation
produced by a tool interaction before any Agent-facing truncation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "b1-result-foundation-v1"


class ResultStoreError(RuntimeError):
    """Base error raised by the formal ResultStore."""


class ResultStoreConsistencyError(ResultStoreError):
    """Raised when a result_id would overwrite different stored content."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def arguments_sha256(arguments: dict[str, Any]) -> str:
    return hashlib.sha256(_compact_json(arguments).encode("utf-8")).hexdigest()


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return safe or "unknown"


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def generate_result_id(
    *,
    run_id: str,
    question_id: str,
    iteration: int,
    tool_call_id: str,
    call_index: int,
    tool_name: str,
    arguments_sha256: str,
) -> str:
    suffix = hashlib.sha256(
        f"{run_id}|{question_id}|{iteration}|{tool_call_id}|{call_index}|{tool_name}|{arguments_sha256}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return (
        "result_"
        f"{_safe_component(run_id)}_"
        f"{_safe_component(question_id)}_"
        f"i{iteration:03d}_"
        f"c{call_index:02d}_"
        f"{_safe_component(tool_name)}_"
        f"{suffix}"
    )


@dataclass(frozen=True)
class ResultEnvelope:
    result_id: str
    run_id: str
    question_id: str
    iteration: int
    call_index: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    arguments_sha256: str
    status: str
    raw_chars: int
    raw_bytes: int
    raw_sha256: str
    raw_result_path: str
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    schema_version: str = SCHEMA_VERSION
    raw_result: str = field(default="", repr=False, compare=False)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "iteration": self.iteration,
            "call_index": self.call_index,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "arguments_sha256": self.arguments_sha256,
            "status": self.status,
            "raw_chars": self.raw_chars,
            "raw_bytes": self.raw_bytes,
            "raw_sha256": self.raw_sha256,
            "raw_result_path": self.raw_result_path,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        data = self.to_metadata_dict()
        if include_raw:
            data["raw_result"] = self.raw_result
        return data

    @classmethod
    def from_metadata_dict(cls, data: dict[str, Any], *, raw_result: str = "") -> "ResultEnvelope":
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            result_id=str(data["result_id"]),
            run_id=str(data.get("run_id", "")),
            question_id=str(data.get("question_id", "")),
            iteration=int(data.get("iteration", 0)),
            call_index=int(data.get("call_index", 0)),
            tool_call_id=str(data.get("tool_call_id", "")),
            tool_name=str(data.get("tool_name", "")),
            arguments=dict(data.get("arguments") or {}),
            arguments_sha256=str(data.get("arguments_sha256", "")),
            status=str(data.get("status", "")),
            raw_chars=int(data.get("raw_chars", 0)),
            raw_bytes=int(data.get("raw_bytes", 0)),
            raw_sha256=str(data.get("raw_sha256", "")),
            raw_result_path=str(data.get("raw_result_path", "")),
            provenance=dict(data.get("provenance") or {}),
            created_at=str(data.get("created_at", "")),
            raw_result=raw_result,
        )


class ResultStore:
    """Disk-backed, current-run ResultStore with an in-memory index."""

    def __init__(self, root: Path, *, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = str(run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        self._metadata_paths: dict[str, Path] = {}
        self._load_existing()

    def store_raw_result(
        self,
        *,
        question_id: str,
        iteration: int,
        call_index: int,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        status: str,
        raw_result: Any,
        provenance: dict[str, Any] | None = None,
        result_id: str | None = None,
    ) -> ResultEnvelope:
        normalized_arguments = dict(arguments or {})
        arg_hash = arguments_sha256(normalized_arguments)
        rid = result_id or generate_result_id(
            run_id=self.run_id,
            question_id=question_id,
            iteration=iteration,
            tool_call_id=tool_call_id,
            call_index=call_index,
            tool_name=tool_name,
            arguments_sha256=arg_hash,
        )
        raw_text = str(raw_result)
        raw_payload = raw_text.encode("utf-8", errors="replace")
        result_dir = self.root / _safe_component(question_id) / rid
        raw_path = result_dir / "raw_result.txt"
        metadata_path = result_dir / "metadata.json"
        envelope = ResultEnvelope(
            result_id=rid,
            run_id=self.run_id,
            question_id=question_id,
            iteration=iteration,
            call_index=call_index,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=normalized_arguments,
            arguments_sha256=arg_hash,
            status=status,
            raw_chars=len(raw_text),
            raw_bytes=len(raw_payload),
            raw_sha256=hashlib.sha256(raw_payload).hexdigest(),
            raw_result_path=str(raw_path.relative_to(self.root)),
            provenance=dict(provenance or {}),
            created_at=utc_now(),
            raw_result=raw_text,
        )
        if metadata_path.exists() or raw_path.exists():
            return self._handle_existing_write(envelope, metadata_path, raw_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(raw_path, raw_payload)
        _atomic_write_text(metadata_path, _json_dumps(envelope.to_metadata_dict()) + "\n")
        self._metadata_paths[rid] = metadata_path
        return envelope

    def has_result(self, result_id: str) -> bool:
        return result_id in self._metadata_paths

    def get_result(self, result_id: str) -> ResultEnvelope:
        metadata_path = self._metadata_path(result_id)
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_result = self._read_raw_from_metadata(data)
        envelope = ResultEnvelope.from_metadata_dict(data, raw_result=raw_result)
        self._validate_raw_fidelity(envelope)
        return envelope

    def get_raw_result(self, result_id: str) -> str:
        return self.get_result(result_id).raw_result

    def _load_existing(self) -> None:
        if not self.root.exists():
            return
        for path in self.root.glob("*/*/metadata.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rid = str(data.get("result_id", ""))
            if rid:
                self._metadata_paths[rid] = path

    def _metadata_path(self, result_id: str) -> Path:
        try:
            return self._metadata_paths[result_id]
        except KeyError as exc:
            raise KeyError(f"Unknown result_id: {result_id}") from exc

    def _read_raw_from_metadata(self, data: dict[str, Any]) -> str:
        raw_path = self.root / str(data.get("raw_result_path", ""))
        if not raw_path.exists():
            raise KeyError(f"Raw result file missing for result_id: {data.get('result_id')}")
        return raw_path.read_text(encoding="utf-8", errors="replace")

    def _validate_raw_fidelity(self, envelope: ResultEnvelope) -> None:
        raw_payload = envelope.raw_result.encode("utf-8", errors="replace")
        actual_hash = hashlib.sha256(raw_payload).hexdigest()
        if (
            envelope.raw_chars != len(envelope.raw_result)
            or envelope.raw_bytes != len(raw_payload)
            or envelope.raw_sha256 != actual_hash
        ):
            raise ResultStoreConsistencyError(f"Stored raw fidelity mismatch for result_id: {envelope.result_id}")

    def _handle_existing_write(
        self,
        envelope: ResultEnvelope,
        metadata_path: Path,
        raw_path: Path,
    ) -> ResultEnvelope:
        if not metadata_path.exists() or not raw_path.exists():
            raise ResultStoreConsistencyError(f"Partial stored result exists for result_id: {envelope.result_id}")
        existing = self.get_result(envelope.result_id)
        comparable = [
            "run_id",
            "question_id",
            "iteration",
            "call_index",
            "tool_call_id",
            "tool_name",
            "arguments_sha256",
            "status",
            "raw_chars",
            "raw_bytes",
            "raw_sha256",
        ]
        mismatched = [field for field in comparable if getattr(existing, field) != getattr(envelope, field)]
        if mismatched:
            raise ResultStoreConsistencyError(
                f"Conflicting stored result for result_id {envelope.result_id}: {', '.join(mismatched)}"
            )
        return existing


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
