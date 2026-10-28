"""Public value contracts. This package never imports the investigation runtime."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping
import hashlib
import json
import re

SCHEMA_VERSION = 'evidence-cards-v1'
EXTRACTOR_VERSION = 'deterministic-extractor-v1.1'
POLICY_VERSION = 'projection-policy-v1.1'
V2_VERSIONS = ('evidence-cards-v2', 'deterministic-extractor-v2', 'projection-policy-v2')
V21_VERSIONS = ('evidence-cards-v2.1', 'deterministic-extractor-v2', 'projection-policy-v2.1')

def versions_for(presentation):
    return {'evidence_cards_v2':V2_VERSIONS,'evidence_cards_v2_1':V21_VERSIONS}.get(
        presentation,(SCHEMA_VERSION,EXTRACTOR_VERSION,POLICY_VERSION))
BOUNDARY = 'UNTRUSTED_EVIDENCE_NOT_INSTRUCTIONS'

class ProjectionInputError(ValueError):
    pass


def compact(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    text.encode('utf-8', errors='strict')
    return text


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()


def encoded_chars(value: Any) -> int:
    """Evidence JSON contribution inside the existing Seed task JSON string."""
    return len(compact(compact(value))) - 2


@dataclass(frozen=True)
class ProjectionSettings:
    presentation: str = 'legacy'
    max_evidence_view_chars: int = 10000
    max_sidecar_bytes: int = 262144
    max_input_chars: int = 1048576
    max_facts: int = 4096
    max_fact_chars: int = 4096
    max_depth: int = 64
    max_items: int = 20000
    keep_core_chars: int = 384
    max_keep_ids: int = 8
    ranking_enabled: bool = True
    diversity_enabled: bool = True

    def __post_init__(self):
        if self.presentation not in ('legacy', 'evidence_cards_v1', 'evidence_cards_v2', 'evidence_cards_v2_1'):
            raise ProjectionInputError('INVALID_PRESENTATION')
        if self.presentation in ('evidence_cards_v2','evidence_cards_v2_1') and (self.max_sidecar_bytes > 262144 or self.max_facts > 4096):
            raise ProjectionInputError('V2_RESOURCE_CAP_EXCEEDED')
        bounds = {'max_evidence_view_chars': (512, 50000), 'max_sidecar_bytes': (4096, 1048576),
                  'max_input_chars': (256, 2097152), 'max_facts': (1, 20000),
                  'max_fact_chars': (128, 16384), 'max_depth': (2, 128),
                  'max_items': (16, 100000), 'keep_core_chars': (64, 2048), 'max_keep_ids': (0, 16)}
        for name, (low, high) in bounds.items():
            v = getattr(self, name)
            if type(v) is not int or not low <= v <= high:
                raise ProjectionInputError('INVALID_SETTING:' + name)
        if type(self.ranking_enabled) is not bool or type(self.diversity_enabled) is not bool:
            raise ProjectionInputError('INVALID_POLICY_FLAG')

    @property
    def enabled(self) -> bool:
        return self.presentation in ('evidence_cards_v1', 'evidence_cards_v2', 'evidence_cards_v2_1')

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> 'ProjectionSettings':
        raw = config.get('task_evidence_graph', {})
        if not isinstance(raw, Mapping):
            raise ProjectionInputError('INVALID_GRAPH_CONFIG')
        nested = raw.get('projection', {})
        if not isinstance(nested, Mapping):
            raise ProjectionInputError('INVALID_PROJECTION_CONFIG')
        names = set(cls.__dataclass_fields__) - {'presentation', 'max_evidence_view_chars'}
        if set(nested) - names:
            raise ProjectionInputError('UNKNOWN_PROJECTION_SETTING')
        return cls(presentation=raw.get('presentation', 'legacy'),
                   max_evidence_view_chars=raw.get('max_evidence_view_chars', 10000), **dict(nested))

    def identity(self) -> str:
        return digest(compact(asdict(self)))


@dataclass(frozen=True)
class SourceScope:
    source_id: str | None = None
    partition_id: str | None = None

    def to_dict(self):
        return {'source_id': self.source_id, 'partition_id': self.partition_id}


@dataclass(frozen=True)
class SourceLocation:
    layer: str  # RAW, FIRST, OBSERVED_METADATA
    source_sha256: str
    kind: str  # json, xml, span, metadata
    pointers: tuple[str, ...] = ()
    transform: str = 'identity'
    selector_json: str = '{}'

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TypedFact:
    fact_id: str
    kind: str
    level: str
    object_key: str
    value_json: str
    origin: SourceLocation
    group: str = ''
    units: int = 1

    @property
    def value(self):
        return json.loads(self.value_json)

    def to_dict(self):
        return {**asdict(self), 'value_json': self.value_json}

    @classmethod
    def from_dict(cls, d):
        d = dict(d); o = dict(d.pop('origin')); o['pointers'] = tuple(o['pointers'])
        return cls(origin=SourceLocation(**o), **d)


@dataclass(frozen=True)
class ProjectionSidecar:
    schema_version: str
    extractor_version: str
    policy_version: str
    evidence_id: str | None
    result_ref: str
    tool: str
    iteration: int
    call_index: int
    observation_seq: int
    scope: SourceScope
    status: str
    execution_started: bool | None
    execution_completed: bool | None
    error_origin: str
    raw_sha256: str | None
    raw_chars: int | None
    first_sha256: str | None
    first_chars: int | None
    metadata_sha256: str
    arguments_sha256: str
    contract_sha256: str
    shape: str
    facts: tuple[TypedFact, ...]
    coverage_json: str
    keep_core_json: str
    cache_key: str
    extraction_cpu_ns: int = field(default=0, compare=False, repr=False)

    @property
    def coverage(self):
        return json.loads(self.coverage_json)

    @property
    def keep_core(self):
        return json.loads(self.keep_core_json)

    def to_dict(self):
        d = asdict(self); d.pop('extraction_cpu_ns')
        return d

    def content_key(self):
        """Content-addressed identity: a wrapper digest alone is not authority.

        The caller retains the observed key. A changed payload cannot be loaded
        under that key even if a cache writer recomputes the wrapper checksum.
        """
        d=self.to_dict();d['cache_key']=''
        return digest(compact(d))

    def validate(self):
        if (self.schema_version,self.extractor_version,self.policy_version) not in ((SCHEMA_VERSION,EXTRACTOR_VERSION,POLICY_VERSION),V2_VERSIONS,V21_VERSIONS):
            raise ProjectionInputError('SIDECAR_VERSION_MISMATCH')
        if not isinstance(self.facts,tuple) or len(self.facts)>20000:raise ProjectionInputError('INVALID_SIDECAR_FACTS')
        for value in (self.result_ref,self.tool,self.status):
            if not isinstance(value,str) or not value or len(value)>4096:raise ProjectionInputError('INVALID_SIDECAR_IDENTITY')
        for value in (self.iteration,self.call_index,self.observation_seq):
            if type(value) is not int or value<0:raise ProjectionInputError('INVALID_SIDECAR_SEQUENCE')
        for value in (self.execution_started,self.execution_completed):
            if value is not None and type(value) is not bool:raise ProjectionInputError('INVALID_EXECUTION_FLAG')
        if self.evidence_id is not None and (not isinstance(self.evidence_id,str) or not self.evidence_id.startswith('ev:')):
            raise ProjectionInputError('INVALID_SIDECAR_EVIDENCE_ID')
        if any(v is not None and (not isinstance(v,str) or len(v)>4096) for v in (self.scope.source_id,self.scope.partition_id)):
            raise ProjectionInputError('INVALID_SIDECAR_SCOPE')
        hashes={'RAW':self.raw_sha256,'FIRST':self.first_sha256,'OBSERVED_METADATA':self.metadata_sha256}
        for value in (*hashes.values(),self.arguments_sha256,self.contract_sha256):
            if value is not None and (not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value)):
                raise ProjectionInputError('INVALID_SIDECAR_HASH')
        cov=self.coverage;core=self.keep_core
        if not isinstance(cov,dict) or not isinstance(core,dict):raise ProjectionInputError('INVALID_SIDECAR_COVERAGE')
        for key in ('extracted_facts','retained_facts','sidecar_omitted_facts'):
            if type(cov.get(key)) is not int or cov[key]<0:raise ProjectionInputError('INVALID_SIDECAR_COUNT')
        if cov['retained_facts']!=len(self.facts) or cov['extracted_facts']!=cov['retained_facts']+cov['sidecar_omitted_facts']:
            raise ProjectionInputError('INCONSISTENT_SIDECAR_COUNT')
        ids=set()
        for f in self.facts:
            if f.level not in ('L1','L2','L3') or type(f.units) is not int or f.units<0:raise ProjectionInputError('INVALID_FACT_LEVEL_OR_UNITS')
            if f.origin.layer not in hashes or f.origin.source_sha256!=hashes[f.origin.layer]:raise ProjectionInputError('FACT_SOURCE_HASH_MISMATCH')
            if f.origin.kind not in ('json','metadata','xml','span'):raise ProjectionInputError('INVALID_FACT_LOCATOR')
            if compact(json.loads(f.value_json))!=f.value_json:raise ProjectionInputError('NONCANONICAL_FACT')
            if not isinstance(json.loads(f.origin.selector_json),dict):raise ProjectionInputError('INVALID_FACT_SELECTOR')
            expected=digest(compact([f.kind,f.object_key,f.origin.to_dict(),f.value_json]))[:20]
            if expected!=f.fact_id or f.fact_id in ids:raise ProjectionInputError('FACT_IDENTITY_MISMATCH')
            ids.add(f.fact_id)
        if self.cache_key!=self.content_key():raise ProjectionInputError('SIDECAR_CONTENT_IDENTITY_MISMATCH')

    @classmethod
    def from_dict(cls, data):
        d = dict(data); d['scope'] = SourceScope(**d['scope'])
        d['facts'] = tuple(TypedFact.from_dict(x) for x in d['facts'])
        obj = cls(**d)
        obj.validate()
        return obj


@dataclass(frozen=True)
class FrontierSnapshot:
    nodes: Mapping[str, Mapping[str, Any]]
    new_ids: tuple[str, ...]
    parent_ids: tuple[str, ...]
    kept_ids: tuple[str, ...]
    edges: tuple[Mapping[str, Any], ...]
    iteration: int
    observation_seq: int
    cursor_end: int
    usage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceCard:
    """Public schema wrapper; wire cards are deliberately compact dictionaries."""
    keep_id: str | None
    result_ref: str | None
    roles: tuple[str, ...]
    source_scope_index: int
    status: str
    wire_json: str

    def to_dict(self):
        return json.loads(self.wire_json)


@dataclass(frozen=True)
class ProjectionResult:
    view: dict | None
    metrics: dict
    fallback_used: bool
    fallback_reason: str | None
    use_legacy: bool
    retrieval_task: str | None = None
