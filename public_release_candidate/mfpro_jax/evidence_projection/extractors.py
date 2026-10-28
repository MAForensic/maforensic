"""Type/shape adapters. Never use task text or historical evaluation probes here."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import PurePosixPath
from .contracts import TypedFact, SourceLocation, ProjectionInputError, compact, digest
from .parsing import strict_json, xml_elements, transform_value, leaves, pointer

HOST_PRODUCERS = frozenset({'export_indexed_file', 'export_file', 'extract_image_file'})

@dataclass
class Extraction:
    facts: list
    shape: str
    counts: dict
    limits: list


def extract(*, text, layer, metadata, metadata_text, contract, settings, v2=False):
    facts = []; counts = {'candidate_facts_attempted':0,'facts_rejected_by_level':{'L1':0,'L2':0,'L3':0}}; limits = []; parse_cache = {}
    source_hash = digest(text) if text is not None else None
    tool = metadata['tool_name']; status = metadata['status']
    def add(kind, level, pointers=(), transform='identity', selector=None, object_key='', group='', units=1, source_layer=None):
        origin = SourceLocation(source_layer or layer, digest(metadata_text) if source_layer == 'OBSERVED_METADATA' else source_hash,
                                'metadata' if source_layer == 'OBSERVED_METADATA' else 'json', tuple(pointers), transform, compact(selector or {}))
        return add_origin(kind, level, origin, object_key, group, units)
    def add_origin(kind, level, origin, object_key='', group='', units=1):
        counts['candidate_facts_attempted']+=1
        if len(facts) >= settings.max_facts:
            counts['facts_rejected_by_level'][level]+=1
            if 'FACT_COUNT_LIMIT' not in limits: limits.append('FACT_COUNT_LIMIT')
            return
        try:
            value = transform_value(origin, text or '', metadata_text, settings, parse_cache)
            val = compact(value)
            if len(val) > settings.max_fact_chars:
                counts['facts_rejected_by_level'][level]+=1
                if 'ATOMIC_FACT_SIZE_LIMIT' not in limits: limits.append('ATOMIC_FACT_SIZE_LIMIT')
                return
            if kind in ('object_names','table_directory','path_distribution','xml_field_directory','ordered_columns'):units=len(value)
            elif kind=='column_directory':units=len(value['columns'])
            fid = digest(compact([kind, object_key, origin.to_dict(), val]))[:20]
            facts.append(TypedFact(fid, kind, level, object_key, val, origin, group, units))
        except (ProjectionInputError, TypeError, ValueError, KeyError, IndexError, UnicodeError):
            counts['facts_rejected_by_level'][level]+=1
            if 'FACT_SHAPE_OR_SERIALIZATION_LIMIT' not in limits: limits.append('FACT_SHAPE_OR_SERIALIZATION_LIMIT')
    # Input objects and operation are prior observed arguments, not extracted RAW output.
    args = metadata.get('arguments', {})
    for key in ('source_id', 'file_id', 'partition_id', 'db_path', 'file_path', 'archive_path'):
        if key in args:
            add('input_object', 'L2', ('/arguments/' + key,), source_layer='OBSERVED_METADATA', object_key='input:' + key)
    for key in ('query', 'sql', 'path_pattern', 'name_pattern', 'pattern', 'format'):
        if key in args and len(compact(args[key])) <= 512:
            add('operation', 'L3', ('/arguments/' + key,), source_layer='OBSERVED_METADATA', object_key='input:' + key)
    if text is None:
        limits.append('NO_RESULT_BODY'); return Extraction(facts, 'missing', counts, limits)
    if len(text) > settings.max_input_chars:
        limits.append('INPUT_CHAR_LIMIT'); return Extraction(facts, 'oversize', counts, limits)
    try:
        obj = strict_json(text, settings)
    except ProjectionInputError as exc:
        limits.append(str(exc))
        # Reject malformed structured values as structure, preserve only a bounded literal span.
        if str(exc) not in ('NON_JSON_OR_INVALID_UNICODE',):
            return Extraction(facts, 'rejected_structured', counts, limits)
        # Some contracts return literal XML, others wrap XML inside JSON.text.
        # Both carry exact element ordinals; XML never becomes control text.
        try:
            els=xml_elements(text,settings)
        except ProjectionInputError as xml_error:
            if str(xml_error)!='NON_XML':limits.append(str(xml_error))
        else:
            counts['xml_elements']=len(els)
            add_origin('xml_field_directory','L1',SourceLocation(layer,source_hash,'xml',(),'xml_names','{}'),group='fields')
            for i,el in enumerate(els):
                if el.attrib:
                    add_origin('xml_attributes','L2',SourceLocation(layer,source_hash,'xml',(),'xml_attributes',compact({'element':i})),f'/xml/{i}',el.tag)
                if el.text and el.text.strip():
                    add_origin('xml_literal','L3',SourceLocation(layer,source_hash,'xml',(),'xml_text',compact({'element':i})),f'/xml/{i}',el.tag)
                if el.tail and el.tail.strip():
                    add_origin('xml_literal','L3',SourceLocation(layer,source_hash,'xml',(),'xml_tail',compact({'element':i})),f'/xml/{i}/tail',el.tag)
            return Extraction(facts,'xml',counts,limits)
        origin = SourceLocation(layer, source_hash, 'span', (), 'identity', compact({'start': 0, 'end': min(240, len(text))}))
        add_origin('error_text' if status not in ('SUCCESS', 'SUCCESS_EMPTY') else 'text_span', 'L3', origin)
        return Extraction(facts, 'error_text' if status not in ('SUCCESS', 'SUCCESS_EMPTY') else 'text', counts, limits)
    if status not in ('SUCCESS', 'SUCCESS_EMPTY'):
        # Structured failure output stays a failure; generic fields are not success objects.
        shape = 'structured_error'
    else: shape = 'json_' + type(obj).__name__
    if status in ('SUCCESS', 'SUCCESS_EMPTY') and isinstance(obj, dict) and tool in ('query_file_index', 'list_directory') and isinstance(obj.get('results', obj.get('entries')), list):
        key = 'results' if 'results' in obj else 'entries'; rows = obj[key]
        counts['objects_returned'] = len(rows); counts['objects_valid_shape'] = sum(isinstance(r, dict) for r in rows)
        counts['source_truncated'] = obj.get('truncated', 'UNKNOWN')
        for start in range(0, len(rows), 16):
            add('object_names', 'L1', ('/' + key,), 'index_names', {'start': start, 'end': start + 16}, group='names', units=min(16, len(rows)-start))
        from .parsing import index_distribution
        for category, entries in index_distribution(rows,include_types=v2).items():
            for start in range(0, len(entries), 8):
                add('path_distribution', 'L1', ('/' + key,), 'distribution', {'category': category, 'start': start, 'end': start + 8}, group=category, units=min(8,len(entries)-start))
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                limits.append('INVALID_INDEX_ROW'); continue
            keys = [k for k in ('source_id','partition_id','file_id','path','name','kind','type','size','inode') if k in row]
            path = row.get('path', row.get('name',''))
            group = str(PurePosixPath(path).parent) if isinstance(path,str) else 'unknown'
            add('indexed_object' if tool=='query_file_index' else 'directory_object', 'L2', (f'/{key}/{i}',), 'subset', {'keys':keys}, f'/{key}/{i}', group)
        shape = 'file_index' if tool=='query_file_index' else 'directory'
    elif status in ('SUCCESS','SUCCESS_EMPTY') and isinstance(obj,dict) and tool=='sqlite_schema' and isinstance(obj.get('objects'),list):
        objects = obj['objects']; counts['schema_object_count'] = len(objects)
        counts['schema_table_count'] = sum(isinstance(x,dict) and x.get('type')=='table' for x in objects)
        counts['schema_type_unreported'] = sum(not isinstance(x,dict) or 'type' not in x for x in objects)
        counts['schema_columns'] = sum(len(x.get('columns',[])) for x in objects if isinstance(x,dict) and isinstance(x.get('columns'),list))
        if v2:
            counts['schema_typed_columns'] = sum(isinstance(c,list) and len(c)>1 and isinstance(c[1],str) for x in objects if isinstance(x,dict) and isinstance(x.get('columns'),list) for c in x['columns'])
        counts['source_truncated'] = obj.get('truncated','UNKNOWN')
        for start in range(0,len(objects),24):
            add('table_directory','L1',('/objects',),'object_names',{'start':start,'end':start+24},group='tables',units=min(24,len(objects)-start))
        for i, ob in enumerate(objects):
            if not isinstance(ob,dict) or not isinstance(ob.get('name'),str) or not isinstance(ob.get('columns'),list):
                limits.append('INVALID_SCHEMA_OBJECT');continue
            cols = ob['columns']; valid = all(isinstance(c,list) and c and isinstance(c[0],str) for c in cols)
            if not valid: limits.append('INVALID_SCHEMA_COLUMNS');continue
            for start in range(0,len(cols),12):
                add('column_directory','L1',(f'/objects/{i}/name',f'/objects/{i}/columns'),'schema_columns',{'start':start,'end':start+12},f'/objects/{i}',ob['name'],min(12,len(cols)-start))
            for ci, col in enumerate(cols):
                add('schema_field','L2',(f'/objects/{i}/name',f'/objects/{i}/columns/{ci}'),'schema_field',object_key=f'/objects/{i}',group=ob['name'])
        shape = 'sqlite_schema'
    elif status in ('SUCCESS','SUCCESS_EMPTY') and isinstance(obj,dict) and tool=='sqlite_query' and isinstance(obj.get('columns'),list) and isinstance(obj.get('rows'),list):
        cols = obj['columns']; rows = obj['rows'];counts.update(rows_returned=len(rows),column_count=len(cols),source_truncated=obj.get('truncated','UNKNOWN'))
        add('ordered_columns','L1',('/columns',),group='columns',units=len(cols))
        for i,row in enumerate(rows):
            if not isinstance(row,list) or len(row)!=len(cols):limits.append('INVALID_SQL_ROW_WIDTH');continue
            add('sql_row','L2',(f'/rows/{i}',),object_key=f'/rows/{i}',group='rows')
        shape='sqlite_rows'
    elif status=='SUCCESS' and isinstance(obj,dict) and tool in HOST_PRODUCERS and isinstance(obj.get('output_path'),str):
        add('materialized_path','L2',('/output_path',),object_key='output_host_path')
        counts['materialization_claim']='SUCCESS_OUTPUT_ONLY_NOT_PERMISSION_PROOF';shape='materialization'
    elif isinstance(obj,dict) and isinstance(obj.get('text'),str) and status in ('SUCCESS','SUCCESS_EMPTY'):
        body = obj['text']; shape='text_range';counts['text_chars']=len(body)
        for k in ('offset','bytes_read','encoding','decode_lossy','truncated','next_offset'):
            if k in obj:add('range_metadata','L2',('/'+k,),object_key='text')
        try:
            inner=strict_json(body,settings)
            for p,v in leaves(inner):
                add('embedded_field','L2',(p,),selector={'embedded':'/text'},object_key='/text'+p)
        except ProjectionInputError:
            try:
                els=xml_elements(body,settings)
                xmlnames=SourceLocation(layer,source_hash,'xml',('/text',),'xml_names','{}')
                add_origin('xml_field_directory','L1',xmlnames,group='fields',units=sum('name' in el.attrib for el in els))
                for i,el in enumerate(els):
                    if el.attrib:
                        origin=SourceLocation(layer,source_hash,'xml',('/text',),'xml_attributes',compact({'element':i}))
                        add_origin('xml_attributes','L2',origin,f'/text/xml/{i}',el.tag)
                    if el.text and el.text.strip():
                        origin=SourceLocation(layer,source_hash,'xml',('/text',),'xml_text',compact({'element':i}))
                        add_origin('xml_literal','L3',origin,f'/text/xml/{i}',el.tag)
                    if el.tail and el.tail.strip():
                        origin=SourceLocation(layer,source_hash,'xml',('/text',),'xml_tail',compact({'element':i}))
                        add_origin('xml_literal','L3',origin,f'/text/xml/{i}/tail',el.tag)
            except ProjectionInputError as exc:limits.append(str(exc))
        for start in range(0,min(len(body),960),240):
            origin=SourceLocation(layer,source_hash,'span',('/text',),'identity',compact({'start':start,'end':min(start+240,len(body))}))
            add_origin('text_span','L3',origin,object_key='/text')
    else:
        # Includes unseen tools and structured errors: preserve actual JSON scalar positions.
        for p,v in leaves(obj):
            if isinstance(v,str) and len(v)>240:
                origin=SourceLocation(layer,source_hash,'span',(p,),'identity',compact({'start':0,'end':240}))
                add_origin('literal_span','L3',origin,object_key=p)
            else:add('generic_scalar','L2',(p,),object_key=p,group=p.rsplit('/',1)[0])
        if isinstance(obj,(list,dict)):counts['top_level_items']=len(obj)
    # Count/status/truncation facts are retained separately when present, without filling defaults.
    if isinstance(obj,dict):
        for k in ('truncated','matches','returned_rows','returned_matches','total_objects','entries'):
            if k in obj and not isinstance(obj[k],(list,dict)):add('result_state','L1',('/'+k,),group='state')
    return Extraction(facts,shape,counts,sorted(set(limits)))
