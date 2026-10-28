"""Bounded, strict data parsing and independently checkable source transforms."""
from __future__ import annotations
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath
from .contracts import ProjectionInputError, compact, digest


def strict_json(text, settings):
    if not isinstance(text, str) or len(text) > settings.max_input_chars:
        raise ProjectionInputError('INPUT_CHAR_LIMIT')
    try:
        text.encode('utf-8', errors='strict')
        depth = 0; quoted = False; escaped = False
        for c in text:
            if quoted:
                if escaped: escaped = False
                elif c == '\\': escaped = True
                elif c == '"': quoted = False
            elif c == '"': quoted = True
            elif c in '[{':
                depth += 1
                if depth > settings.max_depth: raise ProjectionInputError('DEPTH_LIMIT')
            elif c in ']}': depth -= 1
        def pairs(ps):
            d = {}
            for k, v in ps:
                if k in d: raise ProjectionInputError('DUPLICATE_JSON_KEY')
                d[k] = v
            return d
        def bad(_): raise ProjectionInputError('NONFINITE_JSON')
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=bad)
        stack = [value]; count = 0
        while stack:
            v = stack.pop(); count += 1
            if count > settings.max_items: raise ProjectionInputError('ITEM_LIMIT')
            if isinstance(v, dict):
                for k in v: k.encode('utf-8', errors='strict')
                stack.extend(v.values())
            elif isinstance(v, list): stack.extend(v)
            elif isinstance(v, float) and not math.isfinite(v): raise ProjectionInputError('NONFINITE_JSON')
            elif isinstance(v, str): v.encode('utf-8', errors='strict')
        return value
    except ProjectionInputError: raise
    except UnicodeError as exc:
        raise ProjectionInputError('INVALID_UNICODE') from exc
    except (ValueError, RecursionError, OverflowError) as exc:
        raise ProjectionInputError('NON_JSON_OR_INVALID_UNICODE') from exc


def xml_elements(text, settings):
    if len(text) > settings.max_input_chars: raise ProjectionInputError('XML_CHAR_LIMIT')
    if re.search(r'<!\s*(DOCTYPE|ENTITY)\b', text, re.I):
        raise ProjectionInputError('XML_DECLARATION_REJECTED')
    try:
        # Bound construction while parsing, rather than after a potentially
        # enormous tree has already been allocated. One feed can overshoot by
        # at most 4096 input characters before events are checked.
        parser = ET.XMLPullParser(events=('start', 'end'))
        root = None; n = 0; depth = 0
        for start in range(0,len(text),4096):
            parser.feed(text[start:start+4096])
            for event,el in parser.read_events():
                if event=='start':
                    if root is None:root=el
                    n+=1;depth+=1
                    if depth>settings.max_depth or n>settings.max_items:
                        raise ProjectionInputError('XML_STRUCTURE_LIMIT')
                else:depth-=1
        parser.close()
        if root is None:raise ProjectionInputError('NON_XML')
        return list(root.iter())
    except (ET.ParseError,UnicodeError) as exc: raise ProjectionInputError('NON_XML') from exc


def pointer(value, p):
    if p and not p.startswith('/'):
        raise ProjectionInputError('INVALID_JSON_POINTER')
    for k in p[1:].split('/') if p else []:
        if re.search(r'~(?![01])',k):raise ProjectionInputError('INVALID_JSON_POINTER')
        k = k.replace('~1', '/').replace('~0', '~')
        if isinstance(value,list):
            if not re.fullmatch(r'0|[1-9][0-9]*',k):raise ProjectionInputError('INVALID_ARRAY_POINTER')
            value=value[int(k)]
        else:value=value[k]
    return value


def escape_pointer(s):
    return str(s).replace('~', '~0').replace('/', '~1')


def leaves(obj, p=''):
    if isinstance(obj,(dict,list)) and not obj:
        yield p,obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items(): yield from leaves(v, p + '/' + escape_pointer(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj): yield from leaves(v, p + '/' + str(i))
    else: yield p, obj


def index_distribution(rows, *, include_types=False):
    from collections import Counter
    ext = Counter(); parent = Counter(); kinds = Counter(); types = Counter()
    for row in rows:
        if not isinstance(row, dict): continue
        path = row.get('path', row.get('name'))
        if isinstance(path, str):
            p = PurePosixPath(path); ext[p.suffix] += 1; parent[str(p.parent)] += 1
        if isinstance(row.get('kind'), str): kinds[row['kind']] += 1
        if include_types and isinstance(row.get('type'), str): types[row['type']] += 1
    result = {'extensions': sorted(ext.items()), 'parents': sorted(parent.items()), 'kinds': sorted(kinds.items())}
    if include_types: result['types'] = sorted(types.items())
    return result


def transform_value(origin, text, metadata_text, settings, _cache=None):
    """Recompute a fact; used by extraction AND by offline provenance verification."""
    cache = {} if _cache is None else _cache
    sel = json.loads(origin.selector_json)
    def parsed(t, key):
        if key not in cache: cache[key] = strict_json(t, settings)
        return cache[key]
    if origin.kind == 'span':
        base = text if not origin.pointers else pointer(parsed(text, (origin.source_sha256, 'json')), origin.pointers[0])
        return base[sel['start']:sel['end']]
    if origin.kind == 'xml':
        base = pointer(parsed(text, (origin.source_sha256, 'json')), origin.pointers[0]) if origin.pointers else text
        xmlkey=(origin.source_sha256,'xml',origin.pointers[0] if origin.pointers else '')
        if xmlkey not in cache: cache[xmlkey]=xml_elements(base, settings)
        els=cache[xmlkey]
        if origin.transform == 'xml_names':
            return [[i, k, v] for i, el in enumerate(els) for k, v in el.attrib.items() if k == 'name']
        el = els[sel['element']]
        if origin.transform=='xml_attributes':return dict(el.attrib)
        if origin.transform=='xml_text':return el.text
        if origin.transform=='xml_tail':return el.tail
        raise ProjectionInputError('UNKNOWN_XML_TRANSFORM')
    if origin.kind == 'metadata':
        obj = json.loads(metadata_text)
    else:
        obj = parsed(text, (origin.source_sha256, 'json'))
        if sel.get('embedded'):
            obj = parsed(pointer(obj, sel['embedded']), (origin.source_sha256, 'embedded', sel['embedded']))
    vals = [pointer(obj, p) for p in origin.pointers]
    v = vals[0] if vals else obj
    trans = origin.transform
    if trans == 'identity': return v
    if trans == 'subset': return {k: v[k] for k in sel['keys']}
    if trans == 'schema_field': return {'table': vals[0], 'column': vals[1]}
    if trans == 'schema_columns': return {'table': vals[0], 'columns': [c[0] for c in vals[1][sel['start']:sel['end']]]}
    if trans == 'index_names': return [[i, row['name']] for i, row in enumerate(v) if isinstance(row, dict) and isinstance(row.get('name'), str)][sel['start']:sel['end']]
    if trans == 'distribution':
        dk=(origin.source_sha256,'distribution',origin.pointers,sel['category']=='types')
        if dk not in cache:cache[dk]=index_distribution(v,include_types=sel['category']=='types')
        return cache[dk][sel['category']][sel['start']:sel['end']]
    if trans == 'object_names': return [[i, row['name']] for i, row in enumerate(v) if isinstance(row, dict) and isinstance(row.get('name'), str)][sel['start']:sel['end']]
    raise ProjectionInputError('UNKNOWN_SOURCE_TRANSFORM')


def verify_fact(fact, *, raw_text=None, first_text=None, metadata_text='{}', settings):
    text = raw_text if fact.origin.layer == 'RAW' else first_text if fact.origin.layer == 'FIRST' else metadata_text
    if not isinstance(text, str) or digest(text) != fact.origin.source_sha256: return False
    try: return compact(transform_value(fact.origin, text, metadata_text, settings)) == fact.value_json
    except (ProjectionInputError, TypeError, ValueError, KeyError, IndexError): return False
