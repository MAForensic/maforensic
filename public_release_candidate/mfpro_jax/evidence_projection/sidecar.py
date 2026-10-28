"""Observe-time sidecars and optional, explicitly rooted, fail-isolated cache."""
from __future__ import annotations
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import re
import time
import stat
from .contracts import (ProjectionSettings, ProjectionSidecar, SourceScope, ProjectionInputError,
                        SCHEMA_VERSION, EXTRACTOR_VERSION, POLICY_VERSION, compact, digest, encoded_chars, versions_for)
from .extractors import extract
from .parsing import strict_json

_FIELDS = ('result_id','tool_name','iteration','call_index','tool_call_id','arguments','arguments_sha256',
           'status','raw_sha256','raw_chars','execution_started','execution_completed','error_origin')


def metadata_dict(envelope):
    """Accept existing ResultEnvelope or its metadata dict; never read raw_result_path."""
    m = {k: envelope.get(k) if isinstance(envelope, dict) else getattr(envelope, k, None) for k in _FIELDS}
    for name in ('result_id','tool_name','status'):
        if not isinstance(m[name],str) or not m[name] or len(m[name])>4096:
            raise ProjectionInputError('INVALID_ENVELOPE_IDENTITY:' + name)
    for name in ('iteration','call_index'):
        if type(m[name]) is not int or m[name]<0:raise ProjectionInputError('INVALID_ENVELOPE_SEQUENCE')
    if not isinstance(m['arguments'],dict):raise ProjectionInputError('INVALID_ARGUMENT_METADATA')
    # Canonical clone rejects NaN/surrogates and prevents retained mutable aliases.
    return json.loads(compact(m))


def extraction_identity(settings):
    if settings.presentation in ('evidence_cards_v2','evidence_cards_v2_1'):
        return digest(compact({'version':versions_for(settings.presentation),'base':extraction_identity(replace(settings,presentation='evidence_cards_v1'))}))
    return digest(compact({k:getattr(settings,k) for k in ('max_input_chars','max_sidecar_bytes','max_facts','max_fact_chars','max_depth','max_items','keep_core_chars')}))


def project_observed_result(*, raw_text, first_text, envelope, tool_contract=None,
                            source_scope=None, observation_seq, settings=None, evidence_id=None):
    """Pure extraction. Legacy returns None without inspecting any supplied result."""
    settings = settings or ProjectionSettings()
    if not settings.enabled:return None
    if settings.presentation in ('evidence_cards_v2','evidence_cards_v2_1'):
        from .sidecar_v2 import project_v2
        return project_v2(raw_text=raw_text,first_text=first_text,envelope=envelope,tool_contract=tool_contract,
            source_scope=source_scope,observation_seq=observation_seq,settings=settings,evidence_id=evidence_id)
    started=time.process_time_ns()
    base,ex,coverage=_project_candidates(raw_text=raw_text,first_text=first_text,envelope=envelope,
        tool_contract=tool_contract,source_scope=source_scope,observation_seq=observation_seq,
        settings=settings,evidence_id=evidence_id)
    facts=sorted(ex.facts,key=lambda f:({'L1':0,'L2':1,'L3':2}[f.level],f.fact_id))
    def build(n):
        cov={**coverage,'retained_facts':n,'sidecar_omitted_facts':len(facts)-n}
        return replace(base,facts=tuple(facts[:n]),coverage_json=compact(cov))
    # Bounded prefix of stable level/fact order. Count EVERY omitted fact.
    lo=0;hi=len(facts)
    if len(compact(build(0).to_dict()).encode())>settings.max_sidecar_bytes:
        raise ProjectionInputError('SIDECAR_IDENTITY_EXCEEDS_LIMIT')
    while lo<hi:
        mid=(lo+hi+1)//2
        if len(compact(build(mid).to_dict()).encode())<=settings.max_sidecar_bytes:lo=mid
        else:hi=mid-1
    result=build(lo)
    result=replace(result,cache_key=result.content_key())
    result.validate()
    return replace(result,extraction_cpu_ns=time.process_time_ns()-started)


def _project_candidates(*,raw_text,first_text,envelope,tool_contract=None,source_scope=None,
                        observation_seq,settings,evidence_id=None,v2=False):
    """Shared validated bounded parsing, before either version's retention."""
    started=time.process_time_ns();m=metadata_dict(envelope)
    if type(observation_seq) is not int or observation_seq<0:raise ProjectionInputError('INVALID_OBSERVATION_SEQUENCE')
    if evidence_id is not None and (not isinstance(evidence_id,str) or not evidence_id.startswith('ev:') or len(evidence_id)>4096):
        raise ProjectionInputError('INVALID_EVIDENCE_ID')
    scope=source_scope if isinstance(source_scope,SourceScope) else SourceScope(**(source_scope or {}))
    if any(v is not None and (not isinstance(v,str) or len(v)>4096) for v in (scope.source_id,scope.partition_id)):
        raise ProjectionInputError('INVALID_SCOPE')
    mt=compact(m)
    if len(mt)>settings.max_input_chars:raise ProjectionInputError('METADATA_CHAR_LIMIT')
    contract=tool_contract or {};ct=compact(contract)
    if len(ct)>settings.max_input_chars:raise ProjectionInputError('CONTRACT_CHAR_LIMIT')
    raw_hash=digest(raw_text) if isinstance(raw_text,str) else None
    first_hash=digest(first_text) if isinstance(first_text,str) else None
    expected=m.get('raw_sha256');limits=[]
    raw_valid=raw_text is not None and isinstance(raw_text,str) and (not expected or expected==raw_hash)
    if raw_text is not None and not raw_valid:limits.append('RAW_HASH_MISMATCH_OR_TYPE')
    layer='RAW' if raw_valid else 'FIRST'
    body=raw_text if raw_valid else first_text if isinstance(first_text,str) else None
    ex=extract(text=body,layer=layer,metadata=m,metadata_text=mt,contract=contract,settings=settings,v2=v2)
    limits.extend(ex.limits)
    counts=dict(ex.counts)
    total_by_level={lev:sum(f.units for f in ex.facts if f.level==lev) for lev in ('L1','L2','L3')}
    # Stable core is independent of task/ranking/role. It is not an all-fields guarantee.
    core_counts={k:v for k,v in counts.items() if k not in ('candidate_facts_attempted','facts_rejected_by_level')}
    core={'shape':ex.shape,'counts':core_counts,'directory':[],'partial':True}
    # Large count metadata is never allowed to grow an unbounded core.
    if encoded_chars(core)>settings.keep_core_chars:core={'shape':ex.shape,'partial':True,'counts_omitted':True,'directory':[]}
    for f in ex.facts:
        if f.level!='L1':continue
        item=[f.kind,f.object_key,f.value]
        trial={**core,'directory':core['directory']+[item]}
        if encoded_chars(trial)<=settings.keep_core_chars:core=trial
    core['partial']=True
    # Include an explicit directory omission count without ever recutting a core at render time.
    core['omitted_directory_facts']=sum(f.level=='L1' for f in ex.facts)-len(core['directory'])
    while encoded_chars(core)>settings.keep_core_chars and core['directory']:core['directory'].pop();core['omitted_directory_facts']+=1
    if encoded_chars(core)>settings.keep_core_chars:
        core={'partial':True}
    coverage={'basis':layer+'_ORIGIN' if body is not None else 'NO_BODY','main_consumption':'MAIN_CONSUMPTION_UNVERIFIED',
              'units_semantics':'UNKNOWN_UNLESS_SOURCE_EXPLICIT','consumability':'UNVERIFIED',
              'counts':counts,'extracted_facts':len(ex.facts),'units_by_level':total_by_level,
              'extraction_limits':sorted(set(limits)),'policy_identity':extraction_identity(settings),
              'raw_integrity':'MATCH' if raw_valid else 'NOT_AVAILABLE_OR_MISMATCH','retained_facts':len(ex.facts),'sidecar_omitted_facts':0}
    common=dict(schema_version=SCHEMA_VERSION,extractor_version=EXTRACTOR_VERSION,policy_version=POLICY_VERSION,
                evidence_id=evidence_id,result_ref=m['result_id'],tool=m['tool_name'],iteration=m['iteration'],call_index=m['call_index'],
                observation_seq=observation_seq,scope=scope,status=m['status'],execution_started=m.get('execution_started'),
                execution_completed=m.get('execution_completed'),error_origin=m.get('error_origin') or '',raw_sha256=expected or raw_hash,
                raw_chars=len(raw_text) if isinstance(raw_text,str) else m.get('raw_chars'),first_sha256=first_hash,
                first_chars=len(first_text) if isinstance(first_text,str) else None,metadata_sha256=digest(mt),
                arguments_sha256=m.get('arguments_sha256') or digest(compact(m['arguments'])),contract_sha256=digest(ct),shape=ex.shape,
                keep_core_json=compact(core),cache_key='0'*64)
    base=ProjectionSidecar(**common,facts=(),coverage_json=compact(coverage))
    return base,ex,coverage




class SidecarCache:
    """Only this explicitly supplied root is accessed; no storage_ref or Source I/O.

    Root is a trusted Gate-3 run-local path. Existing symlinks in its ancestry are
    rejected. dir_fd + O_NOFOLLOW prevent member path traversal/race substitution.
    """
    def __init__(self, root, settings=None):
        self.settings=settings or ProjectionSettings();self.available=False;self.root=None
        self.last_error=None
        if not self.settings.enabled:
            self.last_error='LEGACY_DISABLED'
            return
        try:
            self.root=Path(root).absolute()
            fd=self._walk(create=True)
            try:os.fchmod(fd,0o700)
            finally:os.close(fd)
            self.available=True
        except OSError:self.last_error='CACHE_ROOT_UNAVAILABLE'

    def _walk(self,create=False):
        """Walk every component with NOFOLLOW; a replaced ancestor cannot redirect I/O."""
        if any(part in ('.','..') for part in self.root.parts[1:]):raise OSError('UNSAFE_ROOT_COMPONENT')
        fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
        try:
            for part in self.root.parts[1:]:
                if part in ('.','..'):raise OSError('UNSAFE_ROOT_COMPONENT')
                try:child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                except FileNotFoundError:
                    if not create:raise
                    try:os.mkdir(part,0o700,dir_fd=fd)
                    except FileExistsError:pass
                    child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                os.close(fd);fd=child
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _fd(self):return self._walk()

    def put(self, sidecar):
        if not self.available:return False
        temp=None;dfd=None
        try:
            sidecar.validate()
            if sidecar.schema_version!=versions_for(self.settings.presentation)[0]:raise ValueError('CACHE_VERSION_MISMATCH')
            data=compact(sidecar.to_dict());b=compact({'sidecar':sidecar.to_dict(),'sha256':digest(data)}).encode()
            if len(data.encode())>self.settings.max_sidecar_bytes:raise ValueError('CACHE_SIZE_LIMIT')
            if not re.fullmatch('[0-9a-f]{64}',sidecar.cache_key):raise ValueError('INVALID_CACHE_KEY')
            dfd=self._fd();temp='.'+sidecar.cache_key+'.'+str(time.monotonic_ns())+'.tmp'
            fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=dfd)
            with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
            # Never follow an existing target symlink. replace replaces the link itself.
            os.replace(temp,sidecar.cache_key+'.json',src_dir_fd=dfd,dst_dir_fd=dfd);temp=None
            return True
        except (OSError,ValueError,TypeError,UnicodeError,KeyError,AttributeError,RecursionError):self.last_error='CACHE_WRITE_FAILED';return False
        finally:
            if temp and dfd is not None:
                try:os.unlink(temp,dir_fd=dfd)
                except OSError:pass
            if dfd is not None:os.close(dfd)

    def get(self, key):
        if not self.available or not isinstance(key,str) or not re.fullmatch('[0-9a-f]{64}',key):return None
        dfd=None
        try:
            dfd=self._fd();fd=os.open(key+'.json',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=dfd)
            with os.fdopen(fd,'rb') as f:
                info=os.fstat(f.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_mode&0o077:raise ValueError('UNSAFE_CACHE_MEMBER')
                b=f.read(self.settings.max_sidecar_bytes+257)
            if len(b)>self.settings.max_sidecar_bytes+256:raise ValueError('CACHE_SIZE_LIMIT')
            parse_settings=replace(self.settings,max_input_chars=min(2097152,self.settings.max_sidecar_bytes+256),max_items=100000,max_depth=128)
            obj=strict_json(b.decode('utf8'),parse_settings);data=obj['sidecar']
            if digest(compact(data))!=obj['sha256']:raise ValueError('CACHE_DIGEST_MISMATCH')
            result=ProjectionSidecar.from_dict(data)
            if result.schema_version!=versions_for(self.settings.presentation)[0]:raise ValueError('CACHE_VERSION_MISMATCH')
            if result.cache_key!=key:raise ValueError('CACHE_IDENTITY_MISMATCH')
            if result.coverage['policy_identity']!=extraction_identity(self.settings):raise ValueError('CACHE_POLICY_MISMATCH')
            return result
        except (OSError,ValueError,TypeError,KeyError,UnicodeError,AttributeError,RecursionError):self.last_error='CACHE_READ_FAILED';return None
        finally:
            if dfd is not None:os.close(dfd)
