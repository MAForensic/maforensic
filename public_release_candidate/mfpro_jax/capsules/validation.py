"""Strict validation for inert capsule metadata."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .models import CapsuleDescriptor

CAPSULE_TYPES = {
    "primitive_capsule", "artifact_capsule", "workflow_capsule",
    "application_capsule", "analysis_capsule", "growth_capsule",
}
PROFILE0_LAYERS = {"primitive", "generic_parser"}
FORBIDDEN_NODE_NAMES = {
    "RuntimeContext", "ToolRegistry", "TraceRecorder", "ExamRunner",
    "LLMClient", "scorer", "evaluator", "path_guard", "timeout",
    "cancellation", "prepare_source",
}
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_CAPSULE_ID = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$")


class CapsuleValidationError(ValueError):
    """Raised when capsule metadata or graph contracts are invalid."""


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapsuleValidationError(f"{label} must be a mapping")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapsuleValidationError(f"{label} must be a non-empty string")
    return value


def _require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise CapsuleValidationError(f"{label} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise CapsuleValidationError(f"{label} contains duplicates")
    return value


def _validate_ports(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise CapsuleValidationError(f"{label} must be a list")
    names = []
    for index, item in enumerate(value):
        port = _require_mapping(item, f"{label}[{index}]")
        names.append(_require_string(port.get("port"), f"{label}[{index}].port"))
        _require_string(port.get("type"), f"{label}[{index}].type")
        if "required" in port and not isinstance(port["required"], bool):
            raise CapsuleValidationError(f"{label}[{index}].required must be boolean")
    if len(names) != len(set(names)):
        raise CapsuleValidationError(f"{label} contains duplicate port names")


def validate_capsule_document(document: Any) -> list[Mapping[str, Any]]:
    root = _require_mapping(document, "capsule document")
    if root.get("schema_version") != "capsule-v1":
        raise CapsuleValidationError("schema_version must be capsule-v1")
    entries = root.get("capsules")
    if not isinstance(entries, list) or not entries:
        raise CapsuleValidationError("capsules must be a non-empty list")
    ids, tools = set(), set()
    required = {
        "capsule_id", "tool_name", "name", "version", "origin_profile",
        "status", "maturity", "capsule_type", "capability_layer",
        "short_description", "executor_ref", "consumes", "produces",
        "applicable_to", "not_applicable_to", "underlying_dependencies",
        "failure_modes", "failure_semantics", "provenance_policy",
        "runtime_limits", "evolution", "composition",
    }
    for index, raw in enumerate(entries):
        item = _require_mapping(raw, f"capsules[{index}]")
        missing = sorted(required - set(item))
        if missing:
            raise CapsuleValidationError(f"capsules[{index}] missing fields: {', '.join(missing)}")
        capsule_id = _require_string(item["capsule_id"], f"capsules[{index}].capsule_id")
        tool_name = _require_string(item["tool_name"], f"capsules[{index}].tool_name")
        if not _CAPSULE_ID.fullmatch(capsule_id):
            raise CapsuleValidationError(f"invalid capsule_id: {capsule_id}")
        if capsule_id in ids or tool_name in tools:
            raise CapsuleValidationError(f"duplicate capsule or tool mapping: {capsule_id}/{tool_name}")
        ids.add(capsule_id)
        tools.add(tool_name)
        if tool_name in FORBIDDEN_NODE_NAMES:
            raise CapsuleValidationError(f"infrastructure cannot be a capsule: {tool_name}")
        _require_string(item["name"], f"{capsule_id}.name")
        if not _VERSION.fullmatch(_require_string(item["version"], f"{capsule_id}.version")):
            raise CapsuleValidationError(f"invalid semantic version for {capsule_id}")
        if item["origin_profile"] != "profile0":
            raise CapsuleValidationError(f"Profile 0 capsule has invalid origin: {capsule_id}")
        if item["status"] != "active" or item["maturity"] != "stable":
            raise CapsuleValidationError(f"Profile 0 capsule must be active/stable: {capsule_id}")
        if item["capsule_type"] not in CAPSULE_TYPES:
            raise CapsuleValidationError(f"unsupported capsule_type for {capsule_id}")
        if item["capsule_type"] != "primitive_capsule":
            raise CapsuleValidationError(f"Profile 0 capsule must be primitive_capsule: {capsule_id}")
        if item["capability_layer"] not in PROFILE0_LAYERS:
            raise CapsuleValidationError(f"invalid Profile 0 capability_layer for {capsule_id}")
        _require_string(item["short_description"], f"{capsule_id}.short_description")
        expected_ref = f"existing_tool_registry:{tool_name}"
        if item["executor_ref"] != expected_ref:
            raise CapsuleValidationError(f"executor_ref must be {expected_ref}")
        _validate_ports(item["consumes"], f"{capsule_id}.consumes")
        _validate_ports(item["produces"], f"{capsule_id}.produces")
        _require_string_list(item["applicable_to"], f"{capsule_id}.applicable_to")
        _require_string_list(item["not_applicable_to"], f"{capsule_id}.not_applicable_to")
        _require_string_list(item["underlying_dependencies"], f"{capsule_id}.underlying_dependencies")
        modes = _require_mapping(item["failure_modes"], f"{capsule_id}.failure_modes")
        _require_string_list(modes.get("reason_codes", []), f"{capsule_id}.failure_modes.reason_codes")
        _require_string_list(modes.get("output_conditions", []), f"{capsule_id}.failure_modes.output_conditions")
        _require_mapping(item["failure_semantics"], f"{capsule_id}.failure_semantics")
        _require_mapping(item["provenance_policy"], f"{capsule_id}.provenance_policy")
        _require_mapping(item["runtime_limits"], f"{capsule_id}.runtime_limits")
        evolution = _require_mapping(item["evolution"], f"{capsule_id}.evolution")
        if evolution.get("verification_state") != "accepted":
            raise CapsuleValidationError(f"Profile 0 capsule must be accepted: {capsule_id}")
        composition = _require_mapping(item["composition"], f"{capsule_id}.composition")
        if composition.get("enabled") is not False:
            raise CapsuleValidationError(f"composition must be disabled in Profile 0 V1: {capsule_id}")
    return entries


def validate_registry_compatibility(
    capsules: Iterable[CapsuleDescriptor], tool_registry: Any,
) -> None:
    capsules = tuple(capsules)
    by_tool = {capsule.tool_name: capsule for capsule in capsules}
    registry_tools = tool_registry.tools
    if set(by_tool) != set(registry_tools):
        missing = sorted(set(registry_tools) - set(by_tool))
        extra = sorted(set(by_tool) - set(registry_tools))
        raise CapsuleValidationError(f"ToolRegistry mapping mismatch; missing={missing}, extra={extra}")
    for tool_name, definition in registry_tools.items():
        capsule = by_tool[tool_name]
        if capsule.capability_layer != definition.category:
            raise CapsuleValidationError(f"capability layer mismatch for {tool_name}")
        if capsule.executor_ref != f"existing_tool_registry:{tool_name}":
            raise CapsuleValidationError(f"executor reference mismatch for {tool_name}")
        if set(capsule.underlying_dependencies) != set(definition.dependencies):
            raise CapsuleValidationError(f"external dependency mismatch for {tool_name}")
