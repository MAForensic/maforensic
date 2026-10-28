"""Minimal multi-source case model migrated from MASForensics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re

import yaml

SOURCE_TYPES = {"disk_image", "directory", "archive", "mobile_extraction", "media_collection"}
ACCESS_MODES = {"image", "tree", "archive"}
DISK_IMAGE_EXTS = {".001", ".dd", ".raw", ".img", ".bin", ".e01", ".iso", ".vmdk", ".vhd"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "source"


@dataclass
class EvidenceSource:
    id: str
    label: str
    type: str
    path: str
    access_mode: str
    owner: str = ""
    partition_offset: int = 0
    sector_size: int = 512
    default_partition_id: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def default_partition_offset(self) -> int:
        return self.partition_offset

    @property
    def default_sector_size(self) -> int:
        return self.sector_size


@dataclass
class Case:
    case_id: str
    name: str
    sources: list[EvidenceSource] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def get_source(self, source_id: str) -> EvidenceSource | None:
        return next((source for source in self.sources if source.id == source_id), None)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "name": self.name,
            "sources": [source.to_dict() for source in self.sources],
            "meta": self.meta,
        }


def _source_from_dict(raw: dict, base_dir: Path, index: int) -> EvidenceSource:
    label = str(raw.get("label") or raw.get("id") or f"source-{index}")
    path = Path(str(raw.get("path", ""))).expanduser()
    if path and not path.is_absolute():
        path = base_dir / path
    source_type = str(raw.get("type") or "directory")
    suffix = path.suffix.lower()
    if source_type not in SOURCE_TYPES:
        source_type = "disk_image" if suffix in DISK_IMAGE_EXTS else "archive" if suffix in ARCHIVE_EXTS else "directory"
    default_mode = "image" if source_type == "disk_image" else "archive" if source_type == "archive" else "tree"
    access_mode = str(raw.get("access_mode") or default_mode)
    if access_mode not in ACCESS_MODES:
        access_mode = default_mode
    source_id = str(raw.get("id") or f"src-{slugify(label)}")
    if not source_id.startswith("src-"):
        source_id = f"src-{slugify(source_id)}"
    return EvidenceSource(
        id=source_id,
        label=label,
        type=source_type,
        path=str(path),
        access_mode=access_mode,
        owner=str(raw.get("owner", "")),
        partition_offset=int(raw.get("partition_offset", 0) or 0),
        sector_size=int(raw.get("sector_size", 512) or 512),
        default_partition_id=str(raw.get("partition_id", "")),
        meta=dict(raw.get("meta", {})),
    )


def load_case(path: str | Path) -> Case:
    case_path = Path(path)
    data = yaml.safe_load(case_path.read_text(encoding="utf-8")) or {}
    sources = [_source_from_dict(raw, case_path.resolve().parent, i) for i, raw in enumerate(data.get("sources", []))]
    return Case(
        case_id=str(data.get("case_id", "case")),
        name=str(data.get("name", "Untitled case")),
        sources=sources,
        meta=dict(data.get("meta", {})),
    )
