def read_windows_registry_value(hive_path: str, key_path: str, value_name: str) -> dict:
    """
    Read a named value from an offline Windows Registry hive using a hive-relative
    registry key path and value name, and return the decoded value together with
    its Windows Registry value type.

    All three arguments are required (none of them has a default value) and the
    function is fully generic: it works on any hive file, any key path relative to
    the root of that hive, and any value name supplied by the caller at runtime.

    Args:
        hive_path: Required path to an offline Windows Registry hive file (no default value).
        key_path: Required registry key path relative to the root of the supplied hive (no default value).
        value_name: Required name of the Registry value to read (no default value).

    Returns:
        dict with the following structure:
        {
          'result': dict  # Structured Registry value information containing at
                          # least the decoded value and the Windows Registry
                          # value type ('value', 'type', 'type_name', ...).
        }
    """
    import json
    import os
    import struct
    import sys
    import traceback
    from datetime import datetime, timedelta

    # ---------------------------------------------------------------------------
    # Registry format constants (format-level knowledge, not example values)
    # ---------------------------------------------------------------------------
    HIVE_DATA_BASE = 0x1000          # first hbin starts right after the 4 KiB regf header
    MAX_NAME_LENGTH = 1024           # sanity cap for key/value name lengths
    MAX_SUBKEYS = 200000             # sanity cap while enumerating subkeys
    MAX_HEX_BYTES = 65536            # cap on binary payloads rendered as hex strings
    REG_TYPE_NAMES = {
        0: 'REG_NONE',
        1: 'REG_SZ',
        2: 'REG_EXPAND_SZ',
        3: 'REG_BINARY',
        4: 'REG_DWORD',
        5: 'REG_DWORD_BIG_ENDIAN',
        6: 'REG_LINK',
        7: 'REG_MULTI_SZ',
        8: 'REG_RESOURCE_LIST',
        9: 'REG_FULL_RESOURCE_DESCRIPTOR',
        10: 'REG_RESOURCE_REQUIREMENTS_LIST',
        11: 'REG_QWORD',
    }
    STRING_VALUE_TYPES = (1, 2, 6, 7)
    LIST_SIGNATURES = (b'lf', b'lh', b'li', b'ri', b'il')

    def log(message):
        pass  # Console diagnostics suppressed: evidence stays in the result contract.

    def make_error(message, **extra):
        result = {
            'found': False,
            'error': message,
            'hive_path': hive_path if isinstance(hive_path, str) else None,
            'key_path': key_path if isinstance(key_path, str) else None,
            'value_name': value_name if isinstance(value_name, str) else None,
        }
        result.update(extra)
        return {'result': result}

    def to_hex(raw):
        """Hex string for a binary payload (kept JSON-serializable and bounded)."""
        if raw is None:
            return None
        if len(raw) > MAX_HEX_BYTES:
            return raw[:MAX_HEX_BYTES].hex() + ('...<truncated %d bytes total>' % len(raw))
        return raw.hex()

    # ---------------------------------------------------------------------------
    # Argument validation (all arguments are required; no defaults are used)
    # ---------------------------------------------------------------------------
    if hive_path is None or key_path is None or value_name is None:
        return make_error('all three arguments are required: hive_path, key_path, value_name '
                          '(none of them may be None)')
    if not isinstance(hive_path, str) or not isinstance(key_path, str) or not isinstance(value_name, str):
        return make_error('hive_path, key_path and value_name must all be strings')
    if not hive_path.strip():
        return make_error('hive_path must not be an empty string')
    if not os.path.isfile(hive_path):
        return make_error('registry hive file does not exist or is not a regular file: %r' % (hive_path,))

    try:
        with open(hive_path, 'rb') as fh:
            buf = fh.read()
        file_size = len(buf)
        log('opened hive %r (%d bytes)' % (hive_path, file_size))

        if file_size < HIVE_DATA_BASE + 8 or bytes(buf[0:4]) != b'regf':
            return make_error('not a Windows Registry hive file (missing "regf" signature): %r' % (hive_path,))
        root_cell_offset = struct.unpack_from('<I', buf, 0x24)[0]
        hbin_area_size = struct.unpack_from('<I', buf, 0x28)[0]
        log('regf header OK: root cell offset=%#x, hbin area size=%#x' % (root_cell_offset, hbin_area_size))

        # ------------------------------------------------------------------
        # Low level accessors
        # ------------------------------------------------------------------
        def u16(off):
            return struct.unpack_from('<H', buf, off)[0]

        def u32(off):
            return struct.unpack_from('<I', buf, off)[0]

        def i32(off):
            return struct.unpack_from('<i', buf, off)[0]

        def u64(off):
            return struct.unpack_from('<Q', buf, off)[0]

        def sig_at(off):
            if off + 6 > file_size:
                return b''
            return bytes(buf[off + 4:off + 6])

        def cell_at(hive_offset):
            """Return (file_offset, cell_size) for the cell at a hive-relative offset, else None."""
            if hive_offset is None:
                return None
            if hive_offset in (0, 0xFFFFFFFF, 0xFFFFFFFE):
                return None
            off = hive_offset & 0x7FFFFFFF
            if off <= 0:
                return None
            fo = HIVE_DATA_BASE + off
            if fo + 8 > file_size:
                return None
            size = i32(fo)
            if size == 0:
                return None
            cell_size = -size if size < 0 else size
            if cell_size < 8 or fo + cell_size > file_size:
                return None
            return fo, cell_size

        def decode_name_bytes(raw, ascii_name):
            if not raw:
                return None
            if ascii_name:
                try:
                    return raw.decode('ascii')
                except UnicodeDecodeError:
                    return raw.decode('latin-1')
            if len(raw) % 2:
                return None
            try:
                return raw.decode('utf-16-le')
            except UnicodeDecodeError:
                return None

        def valid_list_cell(hive_offset):
            if hive_offset in (0, 0xFFFFFFFF):
                return False
            cell = cell_at(hive_offset)
            if cell is None:
                return False
            return sig_at(cell[0]) in LIST_SIGNATURES

        def valid_value_list(hive_offset, count):
            if count <= 0:
                return hive_offset in (0, 0xFFFFFFFF)
            if hive_offset in (0, 0xFFFFFFFF):
                return False
            cell = cell_at(hive_offset)
            if cell is None:
                return False
            vfo, _vcsz = cell
            if vfo + 8 > file_size:
                return False
            first = u32(vfo + 4) & 0x7FFFFFFF
            if first in (0, 0xFFFFFFFF):
                return False
            first_cell = cell_at(first)
            return first_cell is not None and sig_at(first_cell[0]) == b'vk'

        # ------------------------------------------------------------------
        # nk (key node) parsing.  Two field layouts are supported and the
        # winner is chosen per key by validating the resulting pointers:
        #   classic (Windows 2000/XP/2003/Win7): subkey-list@0x1C, value-count@0x20,
        #       value-list@0x24, security@0x28, class@0x2C, name-length@0x44, name@0x48
        #   win10 (Windows 8.1/10/11): subkey-list@0x20, value-count@0x28,
        #       value-list@0x2C, security@0x30, class@0x34, name-length@0x4C, name@0x50
        # ------------------------------------------------------------------
        def parse_nk(hive_offset, preferred_layout=None):
            cell = cell_at(hive_offset)
            if cell is None:
                return None
            fo, cell_size = cell
            if sig_at(fo) != b'nk':
                return None
            flags = u16(fo + 0x06)
            timestamp = u64(fo + 0x08)
            parent = u32(fo + 0x14)
            nsub = u32(fo + 0x18)
            if nsub >= 0xFFFFFFF0:
                nsub = 0
            ascii_name = bool(flags & 0x20)

            name_candidates = []
            name_by_layout = {}
            for layout_name, len_off, name_off in (('classic', 0x44, 0x48),
                                                   ('win10', 0x4C, 0x50),
                                                   ('alt', 0x48, 0x4C)):
                name_by_layout[layout_name] = None
                if len_off + 2 > cell_size or name_off >= cell_size:
                    continue
                name_len = u16(fo + len_off)
                if name_len == 0:
                    continue
                if name_off + name_len > cell_size or name_len > MAX_NAME_LENGTH:
                    continue
                decoded = decode_name_bytes(bytes(buf[fo + name_off:fo + name_off + name_len]), ascii_name)
                if decoded is not None:
                    name_candidates.append(decoded)
                    name_by_layout[layout_name] = decoded

            best = None
            for layout_name, sk_off, nval_off, vlist_off, sec_off in (
                    ('classic', 0x1C, 0x20, 0x24, 0x28),
                    ('win10', 0x20, 0x28, 0x2C, 0x30)):
                sklist = u32(fo + sk_off)
                nval = u32(fo + nval_off)
                if nval >= 0xFFFFFFF0:
                    nval = 0
                vlist = u32(fo + vlist_off)
                sec = u32(fo + sec_off)
                score = 0
                if nsub == 0:
                    if sklist in (0, 0xFFFFFFFF):
                        score += 3
                elif valid_list_cell(sklist):
                    score += 3
                if valid_value_list(vlist, nval):
                    score += 3
                if sec not in (0, 0xFFFFFFFF):
                    sec_cell = cell_at(sec)
                    if sec_cell is not None and sig_at(sec_cell[0]) == b'sk':
                        score += 1
                if name_by_layout.get(layout_name):
                    score += 1
                candidate = {
                    'layout': layout_name,
                    'sklist': sklist,
                    'nval': nval,
                    'vlist': vlist,
                    'score': score,
                }
                if (best is None
                        or score > best['score']
                        or (score == best['score'] and layout_name == preferred_layout)):
                    best = candidate

            chosen_name = name_by_layout.get(best['layout'])
            if chosen_name is None:
                chosen_name = name_candidates[0] if name_candidates else ''
            return {
                'offset': hive_offset,
                'name': chosen_name,
                'names': name_candidates,
                'flags': flags,
                'timestamp': timestamp,
                'parent': parent,
                'nsub': nsub,
                'sklist': best['sklist'],
                'nval': best['nval'],
                'vlist': best['vlist'],
                'layout': best['layout'],
                'cell_size': cell_size,
            }

        # ------------------------------------------------------------------
        # Subkey enumeration (handles lf / lh / li / ri / il lists and
        # indirect "ri" lists pointing at further lists)
        # ------------------------------------------------------------------
        def enumerate_subkey_offsets(nk):
            sklist = nk['sklist']
            if sklist in (0, 0xFFFFFFFF) or nk['nsub'] == 0:
                return []
            collected = []
            stack = [sklist]
            seen = set()
            hops = 0
            while stack and len(collected) < MAX_SUBKEYS:
                lo = stack.pop() & 0x7FFFFFFF
                if lo == 0 or lo in seen:
                    continue
                seen.add(lo)
                hops += 1
                if hops > 256:
                    break
                cell = cell_at(lo)
                if cell is None:
                    continue
                lfo, lcsz = cell
                lsig = sig_at(lfo)
                if lsig == b'nk':
                    collected.append(lo)
                    continue
                if lsig not in LIST_SIGNATURES:
                    continue
                stride = 8 if lsig in (b'lf', b'lh') else 4
                if lfo + 8 > file_size:
                    continue
                count = u16(lfo + 6)
                max_count = (lcsz - 8) // stride
                if max_count < 0:
                    max_count = 0
                if count > max_count:
                    count = max_count
                for i in range(count):
                    p = lfo + 8 + i * stride
                    if p + 4 > file_size:
                        break
                    entry = u32(p) & 0x7FFFFFFF
                    if entry in (0, 0xFFFFFFFF):
                        continue
                    entry_cell = cell_at(entry)
                    if entry_cell is None:
                        continue
                    entry_sig = sig_at(entry_cell[0])
                    if entry_sig == b'nk':
                        collected.append(entry)
                    elif entry_sig in LIST_SIGNATURES:
                        stack.append(entry)
            return collected

        def enumerate_value_offsets(nk):
            if nk['nval'] <= 0:
                return []
            vlist = nk['vlist']
            if vlist in (0, 0xFFFFFFFF):
                return []
            cell = cell_at(vlist)
            if cell is None:
                return []
            vfo, vcsz = cell
            count = nk['nval']
            entries = []
            for i in range(min(count, 100000)):
                p = vfo + 4 + i * 4
                if p + 4 > file_size:
                    break
                if p + 4 > vfo + max(vcsz, 4 + count * 4):
                    break
                entry = u32(p) & 0x7FFFFFFF
                if entry in (0, 0xFFFFFFFF):
                    continue
                entries.append(entry)
            return entries

        # ------------------------------------------------------------------
        # vk (value) record parsing
        # ------------------------------------------------------------------
        def parse_vk(hive_offset):
            cell = cell_at(hive_offset)
            if cell is None:
                return None
            fo, cell_size = cell
            if sig_at(fo) != b'vk':
                return None
            name_len = u16(fo + 0x06)
            data_size_flags = u32(fo + 0x08)
            data_offset = u32(fo + 0x0C)
            vtype = u32(fo + 0x10)
            vflags = u16(fo + 0x14)
            name = ''
            if name_len:
                available = cell_size - 0x18
                if available > 0:
                    take = min(name_len, available, MAX_NAME_LENGTH)
                    raw = bytes(buf[fo + 0x18:fo + 0x18 + take])
                    if vflags & 1:
                        try:
                            name = raw.decode('ascii')
                        except UnicodeDecodeError:
                            name = raw.decode('latin-1')
                    elif len(raw) % 2 == 0:
                        try:
                            name = raw.decode('utf-16-le')
                        except UnicodeDecodeError:
                            name = raw.decode('latin-1')
                    else:
                        name = raw.decode('latin-1')
            return {
                'offset': hive_offset,
                'name': name,
                'data_size_flags': data_size_flags,
                'data_offset': data_offset,
                'type': vtype,
                'flags': vflags,
                'cell_size': cell_size,
            }

        # ------------------------------------------------------------------
        # Value data retrieval.  Small data lives inline in the vk record's
        # data-offset field; larger data lives in a data cell (the offset
        # points at the cell, whose payload follows the 4-byte cell size).
        # ------------------------------------------------------------------
        def string_quality(raw, vtype):
            if not raw:
                return -1.0
            text = None
            if len(raw) % 2 == 0:
                try:
                    text = raw.decode('utf-16-le')
                except UnicodeDecodeError:
                    text = None
            if text is None:
                try:
                    text = raw.decode('utf-8')
                except UnicodeDecodeError:
                    text = raw.decode('latin-1')
            text = text.rstrip('\x00')
            if not text:
                return 0.0
            good = sum(1 for ch in text
                       if ch.isprintable() or ch in '\t\r\n' or (vtype == 7 and ch == '\x00'))
            return good / len(text)

        def read_value_data(vk):
            data_size_flags = vk['data_size_flags']
            data_offset = vk['data_offset']
            vtype = vk['type']
            if data_size_flags & 0x80000000:
                # data is stored inline in the data-offset field itself
                dlen = data_size_flags & 0x7FFFFFFF
                if dlen > 4:
                    dlen = 4
                raw = struct.pack('<I', data_offset & 0xFFFFFFFF)[:dlen] if dlen else b''
                return raw, dlen, True
            dlen = data_size_flags & 0x7FFFFFFF
            if dlen == 0:
                return b'', 0, False
            if dlen > file_size:
                return None, dlen, False
            fo = HIVE_DATA_BASE + (data_offset & 0x7FFFFFFF)
            at_plus4 = fo + 4 + dlen <= file_size
            at_plus0 = fo + dlen <= file_size
            if not at_plus4 and not at_plus0:
                return None, dlen, False
            use_plus4 = False
            if at_plus4 and fo + 4 <= file_size:
                size_field = i32(fo)
                cell_size = -size_field if size_field < 0 else size_field
                if (cell_size >= dlen + 4 and cell_size % 8 == 0
                        and cell_size <= dlen + 4 + 0x1000):
                    use_plus4 = True
            if use_plus4:
                return bytes(buf[fo + 4:fo + 4 + dlen]), dlen, False
            if vtype in STRING_VALUE_TYPES and at_plus4 and at_plus0:
                variant_a = bytes(buf[fo + 4:fo + 4 + dlen])
                variant_b = bytes(buf[fo:fo + dlen])
                if string_quality(variant_b, vtype) > string_quality(variant_a, vtype):
                    return variant_b, dlen, False
                return variant_a, dlen, False
            if at_plus4:
                return bytes(buf[fo + 4:fo + 4 + dlen]), dlen, False
            return bytes(buf[fo:fo + dlen]), dlen, False

        def decode_string_value(raw, keep_internal_nuls=False):
            if not raw:
                return ''
            text = None
            if len(raw) % 2 == 0:
                try:
                    text = raw.decode('utf-16-le')
                except UnicodeDecodeError:
                    text = None
            if text is None:
                try:
                    text = raw.decode('utf-8')
                except UnicodeDecodeError:
                    text = raw.decode('latin-1')
            if not keep_internal_nuls:
                text = text.rstrip('\x00')
                # fall back to a single-byte interpretation if the UTF-16 guess
                # produced mostly unprintable characters
                printable = sum(1 for ch in text if ch.isprintable() or ch in '\t\r\n')
                if text and printable / len(text) < 0.7:
                    try:
                        alt = raw.rstrip(b'\x00').decode('utf-8')
                    except UnicodeDecodeError:
                        alt = raw.rstrip(b'\x00').decode('latin-1')
                    alt_printable = sum(1 for ch in alt if ch.isprintable() or ch in '\t\r\n')
                    if alt and alt_printable / len(alt) > printable / len(text):
                        text = alt
            return text

        def decode_value(raw, vtype):
            """Decode raw value bytes according to the Registry value type.

            Every returned object is JSON-serializable (binary payloads are
            returned as hex strings)."""
            if raw is None:
                return None
            if vtype in (1, 2, 6):          # REG_SZ / REG_EXPAND_SZ / REG_LINK
                return decode_string_value(raw)
            if vtype == 7:                  # REG_MULTI_SZ
                text = decode_string_value(raw, keep_internal_nuls=True)
                parts = text.split('\x00')
                while parts and parts[-1] == '':
                    parts.pop()
                return parts
            if vtype == 4:                  # REG_DWORD
                if len(raw) >= 4:
                    return struct.unpack_from('<I', raw, 0)[0]
                if raw:
                    return int.from_bytes(raw, 'little')
                return 0
            if vtype == 5:                  # REG_DWORD_BIG_ENDIAN
                if len(raw) >= 4:
                    return struct.unpack_from('>I', raw, 0)[0]
                if raw:
                    return int.from_bytes(raw, 'big')
                return 0
            if vtype == 11:                 # REG_QWORD
                if len(raw) >= 8:
                    return struct.unpack_from('<Q', raw, 0)[0]
                if raw:
                    return int.from_bytes(raw, 'little')
                return 0
            if vtype == 0:                  # REG_NONE
                return None if not raw else to_hex(raw)
            return to_hex(raw)               # REG_BINARY and every other type

        def filetime_to_iso(filetime):
            try:
                if not filetime:
                    return None
                dt = datetime(1601, 1, 1) + timedelta(microseconds=filetime / 10.0)
                if dt.year < 1601 or dt.year > 9999:
                    return None
                return dt.isoformat()
            except Exception:
                return None

        # ------------------------------------------------------------------
        # Walk the key path, starting at the hive's root key
        # ------------------------------------------------------------------
        root = parse_nk(root_cell_offset)
        if root is None:
            # tolerate hives whose header root offset is unusual: scan the first
            # few cells of the first hbin for a key node
            for candidate_offset in (0x20, 0x28, 0x30, 0x38, 0x40, 0x48, 0x50, 0x58, 0x60):
                root = parse_nk(candidate_offset)
                if root is not None:
                    log('root cell recovered at fallback offset %#x' % (candidate_offset,))
                    break
        if root is None:
            return make_error('could not locate the root key of hive %r '
                              '(root cell offset %#x)' % (hive_path, root_cell_offset))
        preferred_layout = root['layout']
        log('root key: %r (layout=%s, subkeys=%d, values=%d)'
            % (root['name'], root['layout'], root['nsub'], root['nval']))

        components = [c for c in key_path.replace('/', '\\').split('\\') if c]
        log('traversing key path components: %r' % (components,))

        current = root
        index = 0
        while index < len(components):
            component = components[index]
            child = None
            for subkey_offset in enumerate_subkey_offsets(current):
                child_nk = parse_nk(subkey_offset, preferred_layout)
                if child_nk is None:
                    continue
                if any(nm.casefold() == component.casefold() for nm in child_nk['names']):
                    child = child_nk
                    break
            if child is None and index == 0:
                # tolerate a leading component that names the hive/root itself
                # (e.g. a hive file name or an "HKEY_*" style prefix)
                basename = os.path.basename(hive_path)
                aliases = {basename.casefold(), os.path.splitext(basename)[0].casefold()}
                root_alias = root['name'].casefold() if root['name'] else ''
                if ((root_alias and component.casefold() == root_alias)
                        or component.casefold() in aliases
                        or component.casefold().startswith('hkey_')):
                    log('skipping leading root/hive-name component %r' % (component,))
                    index += 1
                    continue
            if child is None:
                return make_error('registry key not found in hive %r: component %d (%r) does not '
                                  'exist below key %r' % (hive_path, index + 1, component,
                                                          current['name'] or '<root>'))
            current = child
            index += 1
            log('descended into %r (layout=%s, subkeys=%d, values=%d)'
                % (current['name'], current['layout'], current['nsub'], current['nval']))

        # ------------------------------------------------------------------
        # Locate the requested value
        # ------------------------------------------------------------------
        value_entries = enumerate_value_offsets(current)
        log('key %r exposes %d value(s)' % (current['name'] or '<root>', len(value_entries)))
        target = None
        for entry in value_entries:
            vk = parse_vk(entry)
            if vk is None:
                continue
            if vk['name'].casefold() == value_name.casefold():
                target = vk
                break
        if target is None and value_name == '@':
            for entry in value_entries:
                vk = parse_vk(entry)
                if vk is not None and vk['name'] == '':
                    target = vk
                    break
        if target is None:
            available = []
            for entry in value_entries[:50]:
                vk = parse_vk(entry)
                if vk is not None:
                    available.append(vk['name'] if vk['name'] else '<default>')
            return make_error('registry value not found: %r under key %r in hive %r; '
                              'available values: %r' % (value_name, current['name'] or '<root>',
                                                        hive_path, available),
                              available_value_names=available)

        raw, data_len, inline = read_value_data(target)
        if raw is None:
            return make_error('unable to read the data of value %r under key %r in hive %r '
                              '(data size %d bytes is inconsistent with the hive)'
                              % (value_name, current['name'] or '<root>', hive_path, data_len))
        log('value %r: type=%d (%s), data size=%d, inline=%s'
            % (target['name'], target['type'],
               REG_TYPE_NAMES.get(target['type'], 'REG_UNKNOWN_%d' % target['type']),
               data_len, inline))
        decoded = decode_value(raw, target['type'])
        type_name = REG_TYPE_NAMES.get(target['type'], 'REG_UNKNOWN_%d' % target['type'])
        log('decoded value: %r (type %s)' % (decoded, type_name))

        result = {
            'found': True,
            'hive_path': hive_path,
            'key_path': key_path,
            'key_name': current['name'],
            'value_name': target['name'],
            'value': decoded,
            'type': target['type'],
            'type_name': type_name,
            'registry_type': type_name,
            'data_size': data_len,
            'data_hex': to_hex(raw),
            'last_write_timestamp': filetime_to_iso(current['timestamp']),
        }
    except Exception:
        pass  # Console diagnostics suppressed: evidence stays in the result contract.
        return make_error('unexpected error while reading registry value: %s'
                          % (traceback.format_exc().strip().splitlines()[-1],))
    log('final result: %s' % json.dumps(result))
    return {'result': result}