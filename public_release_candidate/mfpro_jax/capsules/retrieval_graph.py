"""Shared inert catalog/Seed/node values for Typed capability retrieval.

Historical module path retained for internal consumers; no Legacy traversal or asset loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


from .validation import CapsuleValidationError

EVIDENCE_TYPES = {
    "smartphone",
    "android_phone",
    "ios_phone",
    "windows_pc",
    "mac_pc",
    "linux_server",
    "usb_storage",
    "generic",
    "*",
}


@dataclass(frozen=True)
class ToolCatalogEntry:
    tool_name: str
    brief: str
    capability_family: str
    workflow_stage: str

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            "tool_name": self.tool_name,
            "brief": self.brief,
            "capability_family": self.capability_family,
            "workflow_stage": self.workflow_stage,
        }


@dataclass(frozen=True)
class SeedToolSelection:
    tool_name: str
    reason: str
    query_type: Literal["forward", "backward", "both"] = "both"

    def to_manifest_dict(self) -> dict[str, str]:
        return {
            "tool_name": self.tool_name,
            "reason": self.reason,
            "query_type": self.query_type,
        }


@dataclass(frozen=True)
class RetrievalGraphNode:
    node_id: str
    tool_name: str
    brief: str
    capability_family: str
    workflow_stage: str
    evidence_types: tuple[str, ...]

    def catalog_entry(self) -> ToolCatalogEntry:
        return ToolCatalogEntry(
            tool_name=self.tool_name,
            brief=self.brief,
            capability_family=self.capability_family,
            workflow_stage=self.workflow_stage,
        )

    def to_context_dict(self) -> dict[str, str]:
        return {
            "tool_name": self.tool_name,
            "brief": self.brief,
            "family": self.capability_family,
            "stage": self.workflow_stage,
        }




@dataclass(frozen=True)
class RetrievalSubgraph:
    seed_tools: tuple[SeedToolSelection, ...]
    nodes: tuple[RetrievalGraphNode, ...]
    edges: tuple[Any, ...]
    truncated: bool

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    @property
    def edge_ids(self) -> tuple[str, ...]:
        return tuple(edge.edge_id for edge in self.edges)


def compatible_evidence_type(node: RetrievalGraphNode, evidence_type: str) -> bool:
    types = set(node.evidence_types)
    if "*" in types:
        return True
    if evidence_type == "smartphone":
        return bool({"smartphone", "android_phone", "ios_phone"} & types)
    return evidence_type in types




def _required_text(item: Mapping[str, Any], field: str, label: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CapsuleValidationError(f"{label}.{field} must be a non-empty string")
    return value.strip()


def _node(item: Mapping[str, Any], index: int) -> RetrievalGraphNode:
    label = f"nodes[{index}]"
    evidence_types = item.get("evidence_types")
    if (
        not isinstance(evidence_types, list)
        or not evidence_types
        or any(not isinstance(value, str) or value not in EVIDENCE_TYPES for value in evidence_types)
    ):
        raise CapsuleValidationError(f"{label}.evidence_types must use supported evidence type strings")
    if len(evidence_types) != len(set(evidence_types)):
        raise CapsuleValidationError(f"{label}.evidence_types contains duplicates")
    return RetrievalGraphNode(
        node_id=_required_text(item, "node_id", label),
        tool_name=_required_text(item, "tool_name", label),
        brief=_required_text(item, "brief", label),
        capability_family=_required_text(item, "capability_family", label),
        workflow_stage=_required_text(item, "workflow_stage", label),
        evidence_types=tuple(evidence_types),
    )
