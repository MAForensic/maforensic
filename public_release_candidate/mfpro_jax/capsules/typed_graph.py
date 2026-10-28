"""Potential typed capability connections, independent of Legacy adjacency.

This is static contract navigation, never an executable input binding. Metadata
paths/pointers are provenance only; this module never opens evidence or proofs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .retrieval_graph import (
    EVIDENCE_TYPES, RetrievalGraphNode, RetrievalSubgraph, SeedToolSelection,
    _node, _required_text, compatible_evidence_type,
)
from .validation import CapsuleValidationError

TYPED_GRAPH_VERSION = "typed-can-feed-v1"
TYPED_GRAPH_SCHEMA = "typed-can-feed-graph-v1"
DIRECTIONS = frozenset({"forward", "backward", "both"})
TYPED_GRAPH_PATH = Path(__file__).with_name("profile0") / "typed_can_feed_graph.yaml"


@dataclass(frozen=True)
class TypedEdge:
    edge_id: str
    producer: str
    consumer: str
    evidence_type: str
    # JSON-owned metadata retains the reviewed binding/constraints/proof exactly.
    metadata: Mapping[str, Any]

    @property
    def from_tool(self) -> str:
        return self.producer

    @property
    def to_tool(self) -> str:
        return self.consumer

    @property
    def edge_type(self) -> str:
        return "can_feed"

    def to_context_dict(self) -> dict[str, Any]:
        return {"id": self.edge_id, "producer": self.producer, "consumer": self.consumer,
                "type": "can_feed", "evidence_type": self.evidence_type,
                "binding": dict(self.metadata["binding_fields"]),
                "conditions": list(self.metadata["preconditions"])}


@dataclass(frozen=True)
class TypedSubgraph(RetrievalSubgraph):
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def validate_typed_document(document: Any) -> None:
    """Reusable static admission check, not dynamic graph generation.

    It validates structure, catalog names, binding parameter names and provenance.
    Semantic truth of a newly proposed port/edge still requires human review.
    """
    if not isinstance(document, dict) or document.get("schema_version") != TYPED_GRAPH_SCHEMA:
        raise CapsuleValidationError("Unsupported typed graph schema")
    if document.get("profile") != "profile0" or document.get("graph_version") != TYPED_GRAPH_VERSION:
        raise CapsuleValidationError("Unsupported typed graph profile/version")
    if not isinstance(document.get("provenance"), dict):
        raise CapsuleValidationError("Typed graph requires provenance")
    raw_nodes, raw_edges = document.get("nodes"), document.get("can_feed")
    if not isinstance(raw_nodes, list) or not raw_nodes or not isinstance(raw_edges, list):
        raise CapsuleValidationError("Typed nodes/can_feed must be lists")
    tools, ids, schemas = set(), set(), {}
    for i, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            raise CapsuleValidationError("Typed node must be a mapping")
        node = _node(item, i)
        if node.tool_name in tools or node.node_id in ids:
            raise CapsuleValidationError("Duplicate typed node identity")
        tools.add(node.tool_name); ids.add(node.node_id)
        ports = item.get("ports")
        if not isinstance(ports, dict) or not isinstance(ports.get("input_schema"), dict):
            raise CapsuleValidationError("Typed node requires input port schema")
        for key in ("consumes", "produces"):
            if not isinstance(ports.get(key), list):
                raise CapsuleValidationError("Typed node requires consumes/produces ports")
        schemas[node.tool_name] = ports["input_schema"]
    edge_ids, semantic_keys = set(), set()
    context_bindings = {"active_source", "active_source_partition"}
    for i, edge in enumerate(raw_edges):
        if not isinstance(edge, dict):
            raise CapsuleValidationError("Typed edge must be a mapping")
        for name in ("edge_id", "producer", "consumer", "evidence_type", "source_pointer", "confidence"):
            _required_text(edge, name, f"can_feed[{i}]")
        if edge.get("relation") != "can_feed" or edge.get("status") != "CONFIRMED":
            raise CapsuleValidationError("Only CONFIRMED can_feed edges are admissible")
        if edge["producer"] not in tools or edge["consumer"] not in tools:
            raise CapsuleValidationError("Typed edge references unknown tool")
        bindings = edge.get("binding_fields")
        if not isinstance(bindings, dict) or not bindings or any(
            not isinstance(k, str) or not isinstance(v, str) or not v for k, v in bindings.items()
        ):
            raise CapsuleValidationError("Typed edge requires named binding fields")
        parameters = schemas[edge["consumer"]].get("properties", {})
        if set(bindings) - set(parameters) - context_bindings:
            raise CapsuleValidationError("Typed binding is not a consumer parameter/context binding")
        for key in ("preconditions", "proof"):
            if not isinstance(edge.get(key), list) or not edge[key]:
                raise CapsuleValidationError("Typed edge requires preconditions/proof")
        if any(not isinstance(p, str) or not p for p in edge["preconditions"]):
            raise CapsuleValidationError("Typed preconditions must be nonempty text")
        semantic = (edge["producer"], edge["consumer"], edge["evidence_type"],
                    json.dumps(bindings, sort_keys=True), json.dumps(edge["preconditions"]))
        if edge["edge_id"] in edge_ids or semantic in semantic_keys:
            raise CapsuleValidationError("Duplicate typed edge")
        edge_ids.add(edge["edge_id"]); semantic_keys.add(semantic)


class TypedGraphProvider:
    """One physical directed edge set; separate outgoing/incoming indexes."""
    graph_version = TYPED_GRAPH_VERSION

    def __init__(self, document: dict[str, Any], *, max_seed_tools: int = 8,
                 max_nodes: int = 16, max_edges: int = 16) -> None:
        validate_typed_document(document)
        if not 1 <= max_seed_tools <= max_nodes or max_edges < 0:
            raise CapsuleValidationError("Typed node budget must accommodate all legal seeds")
        self.nodes = tuple(_node(n, i) for i, n in enumerate(document["nodes"]))
        self.edges = tuple(TypedEdge(e["edge_id"], e["producer"], e["consumer"],
                                     e["evidence_type"], e) for e in document["can_feed"])
        self._by_tool = {n.tool_name: n for n in self.nodes}
        self.ports = {n["tool_name"]: n["ports"] for n in document["nodes"]}
        self.provenance = document["provenance"]
        self.max_seed_tools, self.max_nodes, self.max_edges = max_seed_tools, max_nodes, max_edges
        self.outgoing = {n.tool_name: tuple(e for e in self.edges if e.producer == n.tool_name) for n in self.nodes}
        self.incoming = {n.tool_name: tuple(e for e in self.edges if e.consumer == n.tool_name) for n in self.nodes}
        self.asset_sha256 = ""

    def __iter__(self):
        return iter(self.nodes)

    @property
    def by_tool_name(self):
        return self._by_tool

    def build_evidence_compatible_catalog(self, evidence_type: str):
        normalized = evidence_type if evidence_type in EVIDENCE_TYPES else "generic"
        return [n.catalog_entry() for n in self.nodes if compatible_evidence_type(n, normalized)]

    def query(self, tool_name: str, direction: str) -> tuple[TypedEdge, ...]:
        if tool_name not in self._by_tool or direction not in DIRECTIONS:
            raise CapsuleValidationError("Unknown Typed tool/direction")
        edges = (() if direction == "backward" else self.outgoing[tool_name]) + (
            () if direction == "forward" else self.incoming[tool_name])
        return tuple(sorted({e.edge_id: e for e in edges}.values(), key=lambda e: e.edge_id))

    def expand_subgraph(self, selected_tools: list[SeedToolSelection], *, evidence_type: str = "generic") -> TypedSubgraph:
        # Stable first selection wins, including its direction. Unknowns fail closed.
        seeds_by_name = {}
        for s in selected_tools:
            self.query(s.tool_name, s.query_type)
            seeds_by_name.setdefault(s.tool_name, s)
        seeds = tuple(seeds_by_name.values())[:self.max_seed_tools]
        allowed = {c.tool_name for c in self.build_evidence_compatible_catalog(evidence_type)}
        if any(s.tool_name not in allowed for s in seeds):
            raise CapsuleValidationError("Typed seed outside evidence-compatible catalog")
        traces, edge_by_id = [], {}
        for seed in seeds:
            edges = self.query(seed.tool_name, seed.query_type)
            compatible = [e for e in edges if e.producer in allowed and e.consumer in allowed]
            candidates = sorted({e.consumer if e.producer == seed.tool_name else e.producer for e in compatible})
            traces.append({"seed": seed.tool_name, "direction": seed.query_type,
                           "role": "upstream supplier" if seed.query_type == "backward" else "potential capability navigation",
                           "candidate_tools": candidates, "candidate_count": len(candidates),
                           "forward_count": sum(e.producer == seed.tool_name for e in compatible),
                           "backward_count": sum(e.consumer == seed.tool_name for e in compatible),
                           "support_edges": [e.edge_id for e in compatible],
                           "evidence_filtered_edges": [e.edge_id for e in edges if e not in compatible]})
            edge_by_id.update((e.edge_id, e) for e in compatible)
        # All legal seeds first in original order, then round-robin over each
        # seed's alphabetic candidate queue. No old/family priority or padding.
        ordered = list(seeds_by_name)[:self.max_seed_tools]
        for i in range(max((len(t["candidate_tools"]) for t in traces), default=0)):
            for t in traces:
                if i < len(t["candidate_tools"]) and t["candidate_tools"][i] not in ordered:
                    ordered.append(t["candidate_tools"][i])
        kept = ordered[:self.max_nodes]
        edges = tuple(e for e in sorted(edge_by_id.values(), key=lambda e: e.edge_id)
                      if e.producer in kept and e.consumer in kept)
        for t in traces:
            t["kept_tools"] = [n for n in t["candidate_tools"] if n in kept]
            t["omitted_tools"] = [n for n in t["candidate_tools"] if n not in kept]
            t["truncated"] = bool(t["omitted_tools"])
        diagnostics = {"graph_mode": TYPED_GRAPH_VERSION, "graph_sha256": self.asset_sha256,
                       "candidate_policy": "seeds-first; per-seed alphabetic round-robin",
                       "seeds": traces, "candidate_tools": ordered,
                       "node_limit_omitted": ordered[self.max_nodes:],
                       "edge_limit_omitted": [e.edge_id for e in edges[self.max_edges:]],
                       "source_scope": "navigation only; runtime Source/Partition/bindings still mandatory"}
        return TypedSubgraph(seeds, tuple(self._by_tool[n] for n in kept), edges[:self.max_edges],
                             len(ordered) > len(kept) or len(edges) > self.max_edges or len(seeds_by_name) > len(seeds), diagnostics)


def load_typed_graph(path: Path | None = None, **budgets: int) -> TypedGraphProvider:
    path = path or TYPED_GRAPH_PATH
    data = path.read_bytes()
    graph = TypedGraphProvider(yaml.safe_load(data), **budgets)
    graph.asset_sha256 = hashlib.sha256(data).hexdigest()
    return graph
