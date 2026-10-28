"""Opt-in Evidence Projection V1/V2. No production wiring on import."""
from .contracts import (ProjectionSettings, ProjectionSidecar, ProjectionResult, EvidenceCard,
                        SourceScope, SourceLocation, TypedFact, FrontierSnapshot, ProjectionInputError)
from .sidecar import project_observed_result, SidecarCache
from .renderer import render_evidence_view, validate_keep_update
from .adapter import project_from_observation, frontier_from_graph_snapshot, build_view_adapter
from .parsing import verify_fact
from .budget import serialization_metrics

__all__=['ProjectionSettings','ProjectionSidecar','ProjectionResult','EvidenceCard','SourceScope',
         'SourceLocation','TypedFact','FrontierSnapshot','ProjectionInputError','project_observed_result',
         'SidecarCache','render_evidence_view','validate_keep_update','project_from_observation',
         'frontier_from_graph_snapshot','build_view_adapter','verify_fact','serialization_metrics']
