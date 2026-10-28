"""Mutable runtime context shared by primitive tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .case import Case, EvidenceSource
from .trace import TraceRecorder


@dataclass
class RuntimeContext:
    case: Case
    run_dir: Path
    trace: TraceRecorder
    config: dict
    active_source: EvidenceSource | None = None
    evidence_roots: dict[str, Path] = field(default_factory=dict)
    indexes: dict[str, list[dict]] = field(default_factory=dict)
    index_metadata: dict[str, dict] = field(default_factory=dict)
    index_cache: dict[str, dict] = field(default_factory=dict)
    indexed_sources: set[str] = field(default_factory=set)
    file_hashes: dict[str, str] = field(default_factory=dict)
    source_access: dict[str, dict] = field(default_factory=dict)
    # Borrowed from the normal Agent; its owner retains client lifecycle.
    vision_client: Any = None

    def set_active_source(self, source: EvidenceSource) -> None:
        self.active_source = source
        self.trace.set_context(source_id=source.id)

    def clear_active_source(self) -> None:
        self.active_source = None
        self.trace.set_context(source_id="")

    def reset_source_access(self, source: EvidenceSource | None = None) -> None:
        source = source or self.active_source
        if source is None:
            return
        previous = self.source_access.get(source.id)
        self.source_access[source.id] = {
            "partition_id": source.default_partition_id,
            "sector_size": source.default_sector_size,
            "start_sector": source.default_partition_offset,
            "byte_offset": source.default_partition_offset * source.default_sector_size,
        }
        if previous and previous != self.source_access[source.id]:
            self.activate_cached_index(source)

    def cache_active_index(self, source: EvidenceSource, entries: list[dict], metadata: dict) -> None:
        self.indexes[source.id] = entries
        self.index_metadata[source.id] = metadata
        self.index_cache[source.id] = {
            "partition_id": self.access_state(source).get("partition_id", ""),
            "entries": entries,
            "metadata": metadata,
        }

    def activate_cached_index(self, source: EvidenceSource) -> None:
        cached = self.index_cache.get(source.id)
        if cached and cached.get("partition_id", "") == self.access_state(source).get("partition_id", ""):
            self.indexes[source.id] = cached["entries"]
            self.index_metadata[source.id] = cached["metadata"]
            return
        self.indexes.pop(source.id, None)
        self.index_metadata.pop(source.id, None)

    def access_state(self, source: EvidenceSource | None = None) -> dict:
        source = source or self.active_source
        if source is None:
            return {}
        if source.id not in self.source_access:
            self.reset_source_access(source)
        return self.source_access[source.id]

    @property
    def source_root(self) -> Path:
        if self.active_source is None:
            raise RuntimeError("No active evidence source.")
        return self.evidence_roots.get(self.active_source.id, Path(self.active_source.path))
