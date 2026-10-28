"""Shared authoritative sidecar/node binding checks for both renderers."""
from .contracts import ProjectionInputError


def validate_sidecar_binding(sc,nid,n,snap,expected_version):
    if sc.schema_version!=expected_version:raise ProjectionInputError('SIDECAR_PRESENTATION_VERSION_MISMATCH')
    if sc.evidence_id!=nid or sc.result_ref!=n.get('result_id') or sc.tool!=n.get('tool_name') or sc.status!=n.get('status'):
        raise ProjectionInputError('SIDECAR_NODE_IDENTITY_MISMATCH')
    if sc.iteration!=n.get('iteration') or sc.observation_seq!=n.get('created_seq') or sc.observation_seq>snap.observation_seq:
        raise ProjectionInputError('SIDECAR_TIME_MISMATCH')
    nh=n.get('payload',{}).get('content_sha256')
    if nh and sc.raw_sha256!=nh:raise ProjectionInputError('SIDECAR_RAW_IDENTITY_MISMATCH')
    if sc.scope.source_id!=n.get('source_id'):raise ProjectionInputError('SIDECAR_SOURCE_MISMATCH')
    expected_partition=n.get('partition_id',n.get('input_context',{}).get('partition_id'))
    if expected_partition is not None and sc.scope.partition_id!=str(expected_partition):raise ProjectionInputError('SIDECAR_PARTITION_MISMATCH')
    expected_args=n.get('provenance',{}).get('arguments_sha256')
    if expected_args and sc.arguments_sha256!=expected_args:raise ProjectionInputError('SIDECAR_ARGUMENTS_MISMATCH')
    if (sc.execution_started,sc.execution_completed)!=(n.get('execution_started'),n.get('execution_completed')):
        raise ProjectionInputError('SIDECAR_EXECUTION_MISMATCH')
    try:sc.validate()
    except ProjectionInputError as exc:raise ProjectionInputError('SIDECAR_INTEGRITY_REJECTED:'+str(exc))
