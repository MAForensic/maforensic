"""Observe-only, in-memory projection state scoped to one authoritative graph."""
from ..evidence_projection import (ProjectionSettings, project_from_observation,
    frontier_from_graph_snapshot, build_view_adapter)
from ..task_evidence_graph import EvidenceViewUnavailable


class ProjectionSession:
    def __init__(self, graph, settings):
        self.graph, self.settings = graph, settings
        self.sidecars, self.content_keys = {}, {}
        self.counts = {'observed': 0, 'sidecar_generated': 0, 'sidecar_rejected': 0,
                       'card_seed_ingress': 0, 'fallback': 0}

    def observe(self, event, node):
        self.counts['observed'] += 1
        try:
            sidecar, error = project_from_observation(event=event, node=node, settings=self.settings)
            if sidecar is None:
                raise ValueError(error or 'SIDECAR_UNAVAILABLE')
            self.sidecars[node['node_id']] = sidecar
            self.content_keys[node['node_id']] = sidecar.content_key()
            self.counts['sidecar_generated'] += 1
        except Exception as exc:
            self.counts['sidecar_rejected'] += 1
            self.graph.record('projection_rejected', node_id=node['node_id'],
                              reason=str(exc) if type(exc) is ValueError else type(exc).__name__)

    def build(self, *, iteration, task_context):
        graph = self.graph
        def legacy_history():
            # This exception reaches the Agent's existing Retrieval State seam.
            # The legacy graph preview is NOT the legacy Agent history.
            raise EvidenceViewUnavailable('projection_legacy_retrieval_state')
        def record(result):
            graph.record('projection_render', metrics=result.metrics,
                         fallback_reason=result.fallback_reason)
            if result.use_legacy:
                self.counts['fallback'] += 1
        try:
            if not graph.available:
                raise EvidenceViewUnavailable('graph_unavailable')
            snapshot = frontier_from_graph_snapshot(nodes=graph.nodes, result_order=graph.result_order,
                cursor=graph.cursor, edges=[e for edges in graph.edges_by_child.values() for e in edges],
                keep_ids=graph.keep_ids, iteration=iteration, used_tools=graph.used_tools)
            # Only keys registered at observation time are accepted. No disk/RAW reads.
            sidecars = {nid: s for nid, s in self.sidecars.items()
                        if self.content_keys.get(nid) == s.content_key()}
            required = set(snapshot.new_ids + snapshot.parent_ids + snapshot.kept_ids)
            if any(graph.nodes[nid].get('kind') == 'RESULT' and nid not in sidecars for nid in required):
                self.counts['fallback'] += 1
                graph.record('projection_fallback', reason='SIDECAR_UNAVAILABLE_OR_UNTRUSTED_KEY')
                raise EvidenceViewUnavailable('projection_sidecar_unavailable')
            view, metrics, cursor = build_view_adapter(frontier_snapshot=snapshot, sidecars=sidecars,
                task_context=task_context, keep_ids=graph.keep_ids, settings=self.settings,
                legacy_builder=legacy_history, record_projection=record)
            return view, {**metrics, 'mode': graph.settings.mode,
                          'presentation': self.settings.presentation}, cursor
        except EvidenceViewUnavailable:
            raise
        except Exception as exc:
            self.counts['fallback'] += 1
            graph.record('projection_fallback', reason=type(exc).__name__)
            raise EvidenceViewUnavailable('projection_' + type(exc).__name__) from exc


def create_projection_session(graph, config):
    # OFF never constructs a graph; SHADOW and LEGACY never extract sidecars.
    if graph.settings.mode != 'active':
        return None
    try:
        settings = ProjectionSettings.from_config(config)
        return ProjectionSession(graph, settings) if settings.enabled else None
    except Exception as exc:
        graph.record('projection_config_disabled', reason=type(exc).__name__)
        return None
