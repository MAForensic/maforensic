"""Immutable models for read-only capsule metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PortDescriptor:
    port: str
    type: str
    required: bool = False


@dataclass(frozen=True)
class FailureModes:
    reason_codes: tuple[str, ...] = ()
    output_conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapsuleDescriptor:
    capsule_id: str
    tool_name: str
    name: str
    version: str
    origin_profile: str
    status: str
    maturity: str
    capsule_type: str
    capability_layer: str
    short_description: str
    executor_ref: str
    consumes: tuple[PortDescriptor, ...]
    produces: tuple[PortDescriptor, ...]
    applicable_to: tuple[str, ...]
    not_applicable_to: tuple[str, ...]
    underlying_dependencies: tuple[str, ...]
    failure_modes: FailureModes
    failure_semantics: Mapping[str, str]
    provenance_policy: Mapping[str, Any]
    runtime_limits: Mapping[str, Any]
    evolution: Mapping[str, Any]
    composition: Mapping[str, Any]

    def input_port(self, name: str) -> PortDescriptor | None:
        return next((port for port in self.consumes if port.port == name), None)

    def output_port(self, name: str) -> PortDescriptor | None:
        return next((port for port in self.produces if port.port == name), None)

    def lightweight_node(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "tool_name": self.tool_name,
            "name": self.name,
            "version": self.version,
            "origin_profile": self.origin_profile,
            "status": self.status,
            "capsule_type": self.capsule_type,
            "capability_layer": self.capability_layer,
            "short_description": self.short_description,
            "applicable_to": list(self.applicable_to),
            "not_applicable_to": list(self.not_applicable_to),
            "consumes": [port.__dict__ for port in self.consumes],
            "produces": [port.__dict__ for port in self.produces],
            "failure_modes": {
                "reason_codes": list(self.failure_modes.reason_codes),
                "output_conditions": list(self.failure_modes.output_conditions),
            },
        }
