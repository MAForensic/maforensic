"""Typed-only Seed protocol and budgeted capability navigation rendering."""
from dataclasses import replace
import json

from .context_v2 import CapsuleRetrievalV2Provider, ANSWER_FORMATION_REMINDER
from .llm_retrieval import (
    build_typed_seed_selection_prompt, parse_typed_seed_selection_response,
    TYPED_SEED_PROMPT_VERSION, TYPED_SEED_SCHEMA_VERSION,
)
from .retrieval_graph import SeedToolSelection
from .typed_graph import TYPED_GRAPH_VERSION
from .validation import CapsuleValidationError


class TypedCapsuleRetrievalProvider(CapsuleRetrievalV2Provider):
    retrieval_policy_version = TYPED_GRAPH_VERSION
    seed_prompt_version = TYPED_SEED_PROMPT_VERSION
    seed_schema_version = TYPED_SEED_SCHEMA_VERSION
    graph_asset_name = "typed_can_feed_graph.yaml"
    graph_hints = ("can_feed",)
    build_seed_prompt = staticmethod(build_typed_seed_selection_prompt)
    parse_seed_response = staticmethod(parse_typed_seed_selection_response)

    def _fallback_seed_tools(self, catalog):
        # Reuse existing safe/basic catalog choice, but NEVER Legacy traversal.
        return tuple(SeedToolSelection(s.tool_name, s.reason, "both")
                     for s in super()._fallback_seed_tools(catalog))

    def expand_subgraph(self, seeds, evidence_type):
        return self.graph.expand_subgraph(list(seeds), evidence_type=evidence_type)

    def selection_diagnostics(self, response, subgraph):
        result = dict(subgraph.diagnostics)
        try:
            items = json.loads(response).get("selected_tools", [])
            result["defaulted_direction_tools"] = [s["tool_name"] for s in items
                if isinstance(s, dict) and "query_type" not in s and s.get("tool_name") in self.graph.by_tool_name]
        except (ValueError, TypeError, AttributeError):
            result["defaulted_direction_tools"] = []
        result["legacy_fallback"] = False
        return result

    def prepare_context(self, *, source_id, evidence_type, selected_seed_tools, subgraph):
        # Compact bounded metadata; full binding/proof stays in the asset and
        # diagnostics. Tool descriptions/Seed reasons are not repeated verbatim.
        nodes, edges = list(subgraph.nodes), list(subgraph.edges)
        seed_names = {s.tool_name for s in selected_seed_tools}
        omitted_nodes, omitted_edges = [], []
        diagnostics = dict(subgraph.diagnostics)

        def render():
            payload = {"retrieval_graph_context": {
                "mode": TYPED_GRAPH_VERSION, "source_id": source_id, "evidence_type": evidence_type,
                "selected_seed_tools": [{"tool_name": s.tool_name, "query_type": s.query_type,
                    "lookup": "upstream supplier" if s.query_type == "backward" else s.query_type}
                    for s in selected_seed_tools],
                "nodes": [{"tool_name": n.tool_name, "brief": n.brief[:100]} for n in nodes],
                "edges": [{"producer": e.producer, "consumer": e.consumer,
                    "type": "can_feed", "evidence_type": e.evidence_type,
                    "binding_fields": list(e.metadata["binding_fields"]),
                    "condition_summary": "; ".join(e.metadata["preconditions"])[-180:]}
                    for e in edges],
                "reminder": "Candidate capability navigation, not an execution plan or proof of satisfied inputs. "
                    "Same Source/Partition, successful nonempty objects, actual format/existence, parameter schema "
                    "and resource checks still apply. Backward finds upstream suppliers; all edges retain producer -> consumer. "
                    "Only node tools plus separately logged TTL1/JIT tools are callable by original tool_name.",
                "answer_formation_reminder": ANSWER_FORMATION_REMINDER,
            }}
            # Same marker as C-v2: shared dynamic interface, explicit typed mode.
            from .context_v2 import CONTEXT_START, CONTEXT_END
            return CONTEXT_START + "\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n" + CONTEXT_END

        rendered = render()
        while len(rendered) > self.settings.max_context_chars and edges:
            omitted_edges.append(edges.pop().edge_id)
            rendered = render()
        while len(rendered) > self.settings.max_context_chars:
            removable = next((i for i in range(len(nodes)-1, -1, -1) if nodes[i].tool_name not in seed_names), None)
            if removable is None:
                raise CapsuleValidationError("Typed retrieval context cannot fit legal seeds within max_context_chars")
            omitted_nodes.append(nodes.pop(removable).tool_name)
            names = {n.tool_name for n in nodes}
            edges = [e for e in edges if e.producer in names and e.consumer in names]
            rendered = render()
        diagnostics.update(character_limit_omitted_tools=omitted_nodes, character_limit_omitted_edges=omitted_edges,
                           final_tools=[n.tool_name for n in nodes], final_edge_ids=[e.edge_id for e in edges],
                           rendered_chars=len(rendered))
        # Do not mutate expansion records shared with the pre-render graph.
        diagnostics["seeds"] = [dict(t, final_kept_tools=[n for n in t["candidate_tools"] if n in diagnostics["final_tools"]],
                                     final_omitted_tools=[n for n in t["candidate_tools"] if n not in diagnostics["final_tools"]])
                                for t in diagnostics["seeds"]]
        truncated = bool(omitted_nodes or omitted_edges)
        final = replace(subgraph, nodes=tuple(nodes), edges=tuple(edges),
                        truncated=subgraph.truncated or truncated, diagnostics=diagnostics)
        return final, rendered, truncated
