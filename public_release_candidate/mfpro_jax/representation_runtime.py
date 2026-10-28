"""B2 Agent-facing representation runtime.

This module renders immutable B1 ResultStore records into active text views.
It does not resolve profiles, families, lifecycle transitions, recovery tools,
or cache entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping
import json
import warnings

from .result_store import ResultEnvelope, ResultStore

FOOTER_AUXILIARY_METADATA_KEY = "footer_auxiliary_text"


class RepresentationMode(str, Enum):
    FIRST = "FIRST"
    PERSISTENT = "PERSISTENT"
    RECOVERED_ONCE = "RECOVERED_ONCE"


class RepresentationDecision(str, Enum):
    KEEP_FULL_SMALL = "KEEP_FULL_SMALL"
    RAW_NO_PROCESSOR = "RAW_NO_PROCESSOR"
    RAW_HARD_TRUNCATED = "RAW_HARD_TRUNCATED"
    PROCESSOR_COMMITTED = "PROCESSOR_COMMITTED"
    PROCESSOR_SIZE_BENEFIT = "PROCESSOR_SIZE_BENEFIT"
    PROCESSOR_STRUCTURE_BENEFIT = "PROCESSOR_STRUCTURE_BENEFIT"
    PROCESSOR_NO_BENEFIT_RAW = "PROCESSOR_NO_BENEFIT_RAW"
    PROCESSOR_FAILED_RAW_FALLBACK = "PROCESSOR_FAILED_RAW_FALLBACK"
    PROCESSOR_FAILED_CANONICAL_FALLBACK = "PROCESSOR_FAILED_CANONICAL_FALLBACK"
    RECOVERED_RAW = "RECOVERED_RAW"


class BenefitObservation(str, Enum):
    SMALL = "SMALL"
    SIZE_BENEFIT = "SIZE_BENEFIT"
    STRUCTURE_BENEFIT = "STRUCTURE_BENEFIT"
    NO_BENEFIT = "NO_BENEFIT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RecoveryFooterOverflowError(RuntimeError):
    """Raised when the bounded recovery footer cannot carry result identity."""


@dataclass(frozen=True)
class RuntimeConfig:
    small_output_threshold_chars: int = 1000
    content_hard_max_chars: int = 16000
    recovery_footer_max_chars: int = 2048
    size_benefit_min_chars: int = 1000
    mode_soft_budgets: Mapping[str, int | None] = field(default_factory=dict)
    mode_hard_budgets: Mapping[str, int | None] = field(
        default_factory=lambda: {RepresentationMode.PERSISTENT.value: 1500}
    )

    def __post_init__(self) -> None:
        if self.small_output_threshold_chars < 0:
            raise ValueError("small_output_threshold_chars must be >= 0")
        if self.content_hard_max_chars <= 0:
            raise ValueError("content_hard_max_chars must be > 0")
        if self.recovery_footer_max_chars <= 0:
            raise ValueError("recovery_footer_max_chars must be > 0")
        if self.size_benefit_min_chars < 0:
            raise ValueError("size_benefit_min_chars must be >= 0")
        for key, value in self.mode_soft_budgets.items():
            if key not in {mode.value for mode in RepresentationMode}:
                raise ValueError(f"Unknown mode soft budget key: {key}")
            if value is not None and value <= 0:
                raise ValueError(f"mode soft budget for {key} must be > 0 or None")
        for key, value in self.mode_hard_budgets.items():
            if key not in {mode.value for mode in RepresentationMode}:
                raise ValueError(f"Unknown mode hard budget key: {key}")
            if value is not None and value <= 0:
                raise ValueError(f"mode hard budget for {key} must be > 0 or None")

    def soft_budget_for(self, mode: RepresentationMode) -> int | None:
        configured = self.mode_soft_budgets.get(mode.value)
        if configured is None:
            return None
        return min(configured, self.content_hard_max_chars)

    def hard_budget_for(self, mode: RepresentationMode) -> int:
        configured = self.mode_hard_budgets.get(mode.value)
        if configured is None:
            return self.content_hard_max_chars
        return min(configured, self.content_hard_max_chars)

    @property
    def effective_small_output_threshold_chars(self) -> int:
        return min(self.small_output_threshold_chars, self.content_hard_max_chars)


@dataclass(frozen=True)
class ProcessorOutput:
    text: str
    preferred_cut_points: tuple[int, ...] = ()
    processor_id: str = ""
    requires_recovery: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalFirstRecord:
    result_id: str
    text: str
    preferred_cut_points: tuple[int, ...] = ()
    processor_id: str = ""
    requires_recovery: bool = False
    processor_applied: bool = False
    decision_reason: RepresentationDecision = RepresentationDecision.RAW_NO_PROCESSOR
    raw_chars: int = 0
    candidate_chars: int = 0
    structural_boundary_used: bool = False
    benefit_observation: str = BenefitObservation.NOT_APPLICABLE.value
    fallback_reason: str = ""
    schema_version: str = "b2-canonical-first-v1"


@dataclass(frozen=True)
class ActiveRepresentation:
    result_id: str
    mode: RepresentationMode
    text: str
    payload_chars: int
    final_chars: int
    truncated: bool
    processor_applied: bool
    decision_reason: RepresentationDecision
    processor_id: str = ""
    raw_chars: int = 0
    candidate_chars: int = 0
    structural_boundary_used: bool = False
    benefit_observation: str = BenefitObservation.NOT_APPLICABLE.value
    recovery_footer_added: bool = False
    fallback_reason: str = ""


ContextProcessor = Callable[
    [str, RepresentationMode, ResultEnvelope, Mapping[str, Any]],
    ProcessorOutput,
]


@dataclass(frozen=True)
class _PayloadBranch:
    payload: str
    text: str
    payload_chars: int
    final_chars: int
    truncated: bool
    footer_added: bool
    structural_boundary_used: bool = False
    force_recovered_raw_decision: bool = False


class RepresentationRuntime:
    """Render active Agent-facing text from B1 result records."""

    def __init__(self, result_store: ResultStore, config: RuntimeConfig | None = None) -> None:
        self.result_store = result_store
        self.config = config or RuntimeConfig()

    def render(
        self,
        result_id: str,
        mode: str | RepresentationMode,
        processor: ContextProcessor | None = None,
        processor_inputs: Mapping[str, Any] | None = None,
    ) -> ActiveRepresentation:
        parsed_mode = self._parse_mode(mode)
        if parsed_mode is RepresentationMode.FIRST:
            canonical = self.generate_canonical_first(
                result_id,
                processor=processor,
                processor_inputs=processor_inputs,
            )
            return self.render_canonical_first(canonical, RepresentationMode.FIRST)
        raise ValueError(
            f"{parsed_mode.value} rendering requires a persisted Canonical FIRST source"
        )

    def generate_identity_canonical_first(self, result_id: str) -> CanonicalFirstRecord:
        envelope = self.result_store.get_result(result_id)
        raw_result = self.result_store.get_raw_result(result_id)

        if envelope.raw_chars <= self.config.effective_small_output_threshold_chars:
            return self._canonical(
                envelope,
                text=raw_result,
                decision_reason=RepresentationDecision.KEEP_FULL_SMALL,
            )

        raw_branch = self._raw_branch(envelope, raw_result, force_recovery_footer=False)
        return self._canonical(
            envelope,
            text=raw_result,
            decision_reason=(
                RepresentationDecision.RAW_HARD_TRUNCATED
                if raw_branch.truncated
                else RepresentationDecision.RAW_NO_PROCESSOR
            ),
        )

    def generate_canonical_first(self, result_id, processor=None, processor_inputs=None):
        if processor is not None:
            raise ValueError("External output processors are not distributed")
        return self.generate_identity_canonical_first(result_id)

    def render_canonical_first(
        self,
        canonical: CanonicalFirstRecord,
        mode: str | RepresentationMode,
    ) -> ActiveRepresentation:
        parsed_mode = self._parse_mode(mode)
        if parsed_mode is RepresentationMode.PERSISTENT:
            raise ValueError("Use render_persistent_from_canonical for PERSISTENT materialization")
        envelope = self.result_store.get_result(canonical.result_id)
        branch = self._source_branch(
            envelope,
            canonical.text,
            canonical.preferred_cut_points,
            parsed_mode,
            requires_recovery=canonical.requires_recovery,
            force_recovered_raw_decision=(
                parsed_mode is RepresentationMode.RECOVERED_ONCE
                and not canonical.processor_applied
                and canonical.decision_reason in {
                    RepresentationDecision.RAW_NO_PROCESSOR,
                    RepresentationDecision.RAW_HARD_TRUNCATED,
                    RepresentationDecision.KEEP_FULL_SMALL,
                }
            ),
        )
        decision_reason = (
            RepresentationDecision.RECOVERED_RAW
            if branch.force_recovered_raw_decision
            else canonical.decision_reason
        )
        return self._active(
            envelope=envelope,
            mode=parsed_mode,
            branch=branch,
            processor_applied=canonical.processor_applied,
            processor_id=canonical.processor_id,
            decision_reason=decision_reason,
            candidate_chars=canonical.candidate_chars,
            benefit_observation=canonical.benefit_observation,
            fallback_reason=canonical.fallback_reason,
        )

    def render_persistent_from_canonical(
        self,
        canonical: CanonicalFirstRecord,
        processor: ContextProcessor | None = None,
        processor_inputs: Mapping[str, Any] | None = None,
    ) -> ActiveRepresentation:
        if processor is not None:
            raise ValueError("External output processors are not distributed")
        envelope = self.result_store.get_result(canonical.result_id)
        source_branch = self._source_branch(
            envelope,
            canonical.text,
            canonical.preferred_cut_points,
            RepresentationMode.PERSISTENT,
            requires_recovery=canonical.requires_recovery,
        )
        if canonical.decision_reason is RepresentationDecision.PROCESSOR_FAILED_RAW_FALLBACK:
            return self._active(
                envelope=envelope,
                mode=RepresentationMode.PERSISTENT,
                branch=source_branch,
                processor_applied=False,
                decision_reason=RepresentationDecision.PROCESSOR_FAILED_RAW_FALLBACK,
                fallback_reason=canonical.fallback_reason,
            )
        if processor is None:
            if len(canonical.text) <= self.config.effective_small_output_threshold_chars:
                return self._active(
                    envelope=envelope,
                    mode=RepresentationMode.PERSISTENT,
                    branch=source_branch,
                    processor_applied=False,
                    decision_reason=RepresentationDecision.KEEP_FULL_SMALL,
                )
            return self._active(
                envelope=envelope,
                mode=RepresentationMode.PERSISTENT,
                branch=source_branch,
                processor_applied=False,
                decision_reason=(
                    RepresentationDecision.RAW_HARD_TRUNCATED
                    if source_branch.truncated
                    else RepresentationDecision.RAW_NO_PROCESSOR
                ),
            )

    def _parse_mode(self, mode: str | RepresentationMode) -> RepresentationMode:
        if isinstance(mode, RepresentationMode):
            return mode
        try:
            return RepresentationMode(str(mode))
        except ValueError as exc:
            raise ValueError(f"Unknown representation mode: {mode}") from exc

    def _raw_branch(
        self,
        envelope: ResultEnvelope,
        raw_result: str,
        *,
        force_recovery_footer: bool,
    ) -> _PayloadBranch:
        payload, truncated, boundary_used = self._budget_payload(
            raw_result,
            (),
            hard_budget=self.config.hard_budget_for(RepresentationMode.FIRST),
        )
        return self._branch_from_payload(
            envelope,
            payload,
            truncated=truncated,
            structural_boundary_used=boundary_used,
            needs_recovery_footer=truncated or force_recovery_footer,
            footer_reason="payload truncated" if truncated else "result shortened",
        )



    def _source_branch(
        self,
        envelope: ResultEnvelope,
        text: str,
        preferred_cut_points: tuple[int, ...],
        mode: RepresentationMode,
        *,
        requires_recovery: bool,
        force_recovered_raw_decision: bool = False,
    ) -> _PayloadBranch:
        payload, truncated, boundary_used = self._budget_payload(
            text,
            preferred_cut_points,
            soft_budget=self.config.soft_budget_for(mode),
            hard_budget=self.config.hard_budget_for(mode),
        )
        branch = self._branch_from_payload(
            envelope,
            payload,
            truncated=truncated,
            structural_boundary_used=boundary_used,
            needs_recovery_footer=truncated or requires_recovery,
            footer_reason="payload truncated" if truncated else "result shortened",
        )
        return _PayloadBranch(
            payload=branch.payload,
            text=branch.text,
            payload_chars=branch.payload_chars,
            final_chars=branch.final_chars,
            truncated=branch.truncated,
            footer_added=branch.footer_added,
            structural_boundary_used=branch.structural_boundary_used,
            force_recovered_raw_decision=force_recovered_raw_decision,
        )

    def _budget_payload(
        self,
        text: str,
        preferred_cut_points: tuple[int, ...],
        *,
        soft_budget: int | None = None,
        hard_budget: int | None = None,
    ) -> tuple[str, bool, bool]:
        hard_max = hard_budget or self.config.content_hard_max_chars
        if len(text) <= hard_max and (soft_budget is None or len(text) <= soft_budget):
            return text, False, False

        cut_points = self._valid_cut_points(preferred_cut_points, len(text))
        hard_cut_points = [point for point in cut_points if point <= hard_max]
        if soft_budget is not None and len(text) > soft_budget:
            effective_soft_budget = min(soft_budget, hard_max)
            under_soft = [point for point in hard_cut_points if point <= effective_soft_budget]
            if under_soft:
                point = under_soft[-1]
                return text[:point], point < len(text), True
            if hard_cut_points:
                point = hard_cut_points[0]
                return text[:point], point < len(text), True
            return text[:effective_soft_budget], True, False

        if len(text) > hard_max and hard_cut_points:
            point = hard_cut_points[-1]
            return text[:point], point < len(text), True

        if len(text) <= hard_max:
            return text, False, False
        return text[:hard_max], True, False

    def _valid_cut_points(self, values: tuple[int, ...], text_length: int) -> list[int]:
        result = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if 0 < value <= text_length:
                result.append(value)
        return sorted(set(result))

    def _branch_from_payload(
        self,
        envelope: ResultEnvelope,
        payload: str,
        *,
        truncated: bool,
        structural_boundary_used: bool,
        needs_recovery_footer: bool,
        footer_reason: str,
        footer_auxiliary_text: str | None = None,
    ) -> _PayloadBranch:
        footer = (
            self._recovery_footer(envelope.result_id, footer_reason, footer_auxiliary_text)
            if needs_recovery_footer else ""
        )
        text = payload + footer
        return _PayloadBranch(
            payload=payload,
            text=text,
            payload_chars=len(payload),
            final_chars=len(text),
            truncated=truncated,
            footer_added=bool(footer),
            structural_boundary_used=structural_boundary_used,
        )

    def _recovery_footer(self, result_id: str, reason: str, auxiliary_text: str | None = None) -> str:
        tag = "TRUNCATED" if reason == "payload truncated" else "SHORTENED"
        mandatory = f"\n[{tag}; full_result_available=true; result_id={result_id}; recover_by=result_id]"
        limit = self.config.recovery_footer_max_chars
        if len(mandatory) > limit:
            raise RecoveryFooterOverflowError(
                "Recovery footer exceeds configured limit: "
                f"actual_footer_chars={len(mandatory)} configured_footer_max_chars={limit}"
            )
        if not auxiliary_text:
            return mandatory
        encoded_auxiliary = json.dumps(auxiliary_text, ensure_ascii=False)
        footer = f"\n[{tag}; auxiliary={encoded_auxiliary}; full_result_available=true; result_id={result_id}; recover_by=result_id]"
        if len(footer) > limit:
            return mandatory
        return footer

    def _footer_auxiliary_text(self, metadata: Mapping[str, Any]) -> str | None:
        value = metadata.get(FOOTER_AUXILIARY_METADATA_KEY)
        if not isinstance(value, str):
            return None
        normalized = " ".join(value.strip().split())
        return normalized or None





    def _active(
        self,
        *,
        envelope: ResultEnvelope,
        mode: RepresentationMode,
        branch: _PayloadBranch,
        processor_applied: bool,
        decision_reason: RepresentationDecision,
        processor_id: str = "",
        candidate_chars: int = 0,
        benefit_observation: str = BenefitObservation.NOT_APPLICABLE.value,
        fallback_reason: str = "",
    ) -> ActiveRepresentation:
        if branch.payload_chars > self.config.hard_budget_for(mode):
            raise RuntimeError("payload exceeded mode hard budget")
        footer_chars = branch.final_chars - branch.payload_chars
        if footer_chars > self.config.recovery_footer_max_chars:
            raise RuntimeError("recovery footer exceeded recovery_footer_max_chars")
        return ActiveRepresentation(
            result_id=envelope.result_id,
            mode=mode,
            text=branch.text,
            payload_chars=branch.payload_chars,
            final_chars=branch.final_chars,
            truncated=branch.truncated,
            processor_applied=processor_applied,
            processor_id=processor_id,
            decision_reason=decision_reason,
            raw_chars=envelope.raw_chars,
            candidate_chars=candidate_chars,
            structural_boundary_used=branch.structural_boundary_used,
            benefit_observation=benefit_observation,
            recovery_footer_added=branch.footer_added,
            fallback_reason=fallback_reason,
        )

    def _canonical(
        self,
        envelope: ResultEnvelope,
        *,
        text: str,
        preferred_cut_points: tuple[int, ...] = (),
        processor_id: str = "",
        requires_recovery: bool = False,
        processor_applied: bool = False,
        decision_reason: RepresentationDecision,
        candidate_chars: int = 0,
        structural_boundary_used: bool = False,
        benefit_observation: str = BenefitObservation.NOT_APPLICABLE.value,
        fallback_reason: str = "",
    ) -> CanonicalFirstRecord:
        return CanonicalFirstRecord(
            result_id=envelope.result_id,
            text=text,
            preferred_cut_points=tuple(self._valid_cut_points(preferred_cut_points, len(text))),
            processor_id=processor_id,
            requires_recovery=requires_recovery,
            processor_applied=processor_applied,
            decision_reason=decision_reason,
            raw_chars=envelope.raw_chars,
            candidate_chars=candidate_chars,
            structural_boundary_used=structural_boundary_used,
            benefit_observation=benefit_observation,
            fallback_reason=fallback_reason,
        )
