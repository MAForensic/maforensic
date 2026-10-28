"""Read-only loader and indexes for capsule metadata."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import yaml

from .models import CapsuleDescriptor, FailureModes, PortDescriptor
from .validation import validate_capsule_document


class CapsuleRegistry:
    def __init__(self, capsules: tuple[CapsuleDescriptor, ...]) -> None:
        self._capsules = capsules
        self._by_id = MappingProxyType({item.capsule_id: item for item in capsules})
        self._by_tool = MappingProxyType({item.tool_name: item for item in capsules})

    def __iter__(self) -> Iterator[CapsuleDescriptor]:
        return iter(self._capsules)

    def __len__(self) -> int:
        return len(self._capsules)

    @property
    def by_id(self) -> Mapping[str, CapsuleDescriptor]:
        return self._by_id

    @property
    def by_tool_name(self) -> Mapping[str, CapsuleDescriptor]:
        return self._by_tool

    def get_by_id(self, capsule_id: str) -> CapsuleDescriptor:
        return self._by_id[capsule_id]

    def get_by_tool_name(self, tool_name: str) -> CapsuleDescriptor:
        return self._by_tool[tool_name]


def _ports(items: list[dict[str, Any]]) -> tuple[PortDescriptor, ...]:
    return tuple(PortDescriptor(item["port"], item["type"], item.get("required", False)) for item in items)


def _descriptor(item: Mapping[str, Any]) -> CapsuleDescriptor:
    modes = item["failure_modes"]
    return CapsuleDescriptor(
        capsule_id=item["capsule_id"], tool_name=item["tool_name"], name=item["name"],
        version=item["version"], origin_profile=item["origin_profile"], status=item["status"],
        maturity=item["maturity"], capsule_type=item["capsule_type"],
        capability_layer=item["capability_layer"], short_description=item["short_description"],
        executor_ref=item["executor_ref"], consumes=_ports(item["consumes"]),
        produces=_ports(item["produces"]), applicable_to=tuple(item["applicable_to"]),
        not_applicable_to=tuple(item["not_applicable_to"]),
        underlying_dependencies=tuple(item["underlying_dependencies"]),
        failure_modes=FailureModes(tuple(modes["reason_codes"]), tuple(modes["output_conditions"])),
        failure_semantics=MappingProxyType(dict(item["failure_semantics"])),
        provenance_policy=MappingProxyType(dict(item["provenance_policy"])),
        runtime_limits=MappingProxyType(dict(item["runtime_limits"])),
        evolution=MappingProxyType(dict(item["evolution"])),
        composition=MappingProxyType(dict(item["composition"])),
    )


def load_capsules(path: Path) -> CapsuleRegistry:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = validate_capsule_document(document)
    return CapsuleRegistry(tuple(_descriptor(item) for item in entries))


def load_profile0_capsules(path: Path | None = None) -> CapsuleRegistry:
    path = path or Path(__file__).with_name("profile0") / "capsules.yaml"
    return load_capsules(path)
