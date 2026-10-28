
def parse_e01_header(file_path: str) -> dict:
    """
    Parse structured acquisition metadata from an EnCase EWF/E01 forensic disk image header.

    Given the path to an E01 (Expert Witness Format) image file, the function reads the
    13-byte EWF file header (signature + version), walks the section chain to locate the
    'header' / 'header2' sections, inflates their (zlib compressed) payload, decodes it
    (UTF-16 for 'header2', 8-bit for classic 'header') and extracts the acquisition
    metadata blocks (main / srce / sub) into a structured dictionary.

    Args:
        file_path: Path to the E01 forensic disk image file.

    Returns:
        dict: ``{'result': {...}}``. On success the inner dict contains the parsed
        acquisition metadata (format, signature, sections, acquisition fields, device
        info, dates/timestamps, raw field blocks). If the input is not a valid EWF/E01
        image the inner dict is an explicit error structure:
        ``{'valid': False, 'error': <message>, 'error_type': <code>, ...}``.
    """
    import os
    import re
    import struct
    import sys
    import zlib
    from datetime import datetime, timezone

    def _log(message):
        try:
            pass  # Console diagnostics suppressed: evidence stays in the result contract.
        except Exception:
            pass

    def _fail(message, error_type='invalid_ewf_image', **extra):
        try:
            pass  # Console diagnostics suppressed: evidence stays in the result contract.
        except Exception:
            pass
        payload = {'valid': False, 'error': message, 'error_type': error_type}
        payload.update(extra)
        return {'result': payload}

    # ------------------------------------------------------------------ constants
    EWF_V1_SIGNATURE = b'EVF\x09\r\n\xff\x00'      # EWF/E01 (physical evidence, v1)
    EWF2_SIGNATURE = b'EVF2\r\n\x81\x00'           # EWF2/EX01
    LVF_V1_SIGNATURE = b'LVF\x09\r\n\xff\x00'      # EWF/L01 (logical evidence, v1)
    LVF2_SIGNATURE = b'LVF2\r\n\x81\x00'           # EWF2/L01
    FILE_HEADER_LEN = 13                           # EWF v1 file header size
    SECTION_HEADER_LEN = 76                        # EWF v1 section header size
    HEADER_SECTION_TYPES = ('header', 'header2', 'xheader', 'xheader2')
    MAX_SECTIONS = 64
    MAX_HEADER_DATA_BYTES = 4 * 1024 * 1024
    MAX_RAW_TEXT_CHARS = 4000

    # field-code -> descriptive name mapping used for the 'main' metadata block
    FIELD_CODE_MAP = {
        'a': 'description',
        'c': 'case_number',
        'n': 'notes',
        'e': 'evidence_number',
        'md': 'device_model',
        'sn': 'device_serial_number',
        'l': 'device_label',
        'an': 'acquisition_tool',
        'av': 'acquisition_tool_version',
        'ov': 'operating_system',
        'p': 'password',
        'pid': 'process_identifier',
        'dc': 'device_class',
        'ext': 'extensible',
    }
    # srce (source device) block field codes
    SRCE_CODE_MAP = {
        'mfr': 'device_manufacturer',
        'mo': 'device_model',
        'sn': 'device_serial_number',
        'se': 'device_serial_number',
        'loc': 'device_location',
        'id': 'device_identifier',
    }
    # classic (positional) EWF v1 header field order, used when a header block carries
    # values without an explicit field-name line
    CLASSIC_POSITIONAL_FIELDS = [
        'case_number', 'evidence_number', 'description', 'examiner_name', 'notes',
        'acquired_date', 'system_date', 'operating_system', 'hardware_identifier',
    ]

    # ------------------------------------------------------ 1. validate the path
    if not isinstance(file_path, str) or not file_path.strip():
        return _fail('A path to an E01 image file must be provided (got %r).' % (file_path,),
                     'invalid_argument')
    file_path = os.path.abspath(os.path.expanduser(file_path.strip()))
    _log('Parsing EWF/E01 acquisition header for: %s' % file_path)
    if not os.path.exists(file_path):
        return _fail('File not found: %s' % file_path, 'file_not_found', file_path=file_path)
    if os.path.isdir(file_path):
        return _fail('Path is a directory, not an EWF/E01 image file: %s' % file_path,
                     'is_directory', file_path=file_path)
    if not os.path.isfile(file_path):
        return _fail('Not a regular file: %s' % file_path, 'not_a_regular_file',
                     file_path=file_path)
    try:
        file_size = os.path.getsize(file_path)
    except OSError as exc:
        return _fail('Could not determine file size: %s' % exc, 'stat_error', file_path=file_path)
    _log('File size: %d bytes' % file_size)
    if file_size < FILE_HEADER_LEN + SECTION_HEADER_LEN:
        return _fail('File is only %d bytes long - too small to be a valid EWF/E01 image.'
                     % file_size, 'file_too_small', file_size_bytes=file_size)

    # ------------------------------------------- 2. read + validate EWF signature
    try:
        with open(file_path, 'rb') as fh:
            file_header = fh.read(FILE_HEADER_LEN)
    except OSError as exc:
        return _fail('Could not read the file header: %s' % exc, 'read_error',
                     file_path=file_path)
    signature = file_header[:8]
    _log('File header: %s (signature hex: %s)' % (file_header.hex(), signature.hex()))
    known_signatures = {
        EWF_V1_SIGNATURE: 'EWF-E01 (EnCase Expert Witness Format v1, physical evidence)',
        EWF2_SIGNATURE: 'EWF2-EX01 (EnCase Expert Witness Format 2)',
        LVF_V1_SIGNATURE: 'EWF-L01 (EnCase logical evidence file v1)',
        LVF2_SIGNATURE: 'EWF2-L01 (EnCase logical evidence file 2)',
    }
    ewf_kind = known_signatures.get(signature)
    if ewf_kind is None and signature[:4] in (b'EVF\x09', b'EVF2', b'LVF\x09', b'LVF2'):
        ewf_kind = 'EWF (Expert Witness Format) - variant signature'
    if ewf_kind is None:
        return _fail('File is not a valid EWF/E01 image: file signature %r does not match the '
                     'EnCase EWF signature %r.' % (signature, EWF_V1_SIGNATURE),
                     'invalid_signature', signature_hex=signature.hex(), file_path=file_path)
    version_major = file_header[8] if len(file_header) > 8 else 0
    version_minor = file_header[9] if len(file_header) > 9 else 0
    _log('Valid EWF signature detected (%s), version %d.%d' % (ewf_kind, version_major,
                                                                version_minor))

    # ------------------------------------------------- 3. walk the EWF sections
    sections_info = []
    header_sections = []
    volume_bytes = None
    offset = FILE_HEADER_LEN
    found_header = False

    try:
        with open(file_path, 'rb') as fh:
            for index in range(MAX_SECTIONS):
                fh.seek(offset)
                raw_section = fh.read(SECTION_HEADER_LEN)
                if len(raw_section) < SECTION_HEADER_LEN:
                    _log('End of file reached while reading a section header - stopping walk.')
                    break
                type_bytes = raw_section[:16].split(b'\x00', 1)[0]
                if not type_bytes or not all(32 <= b < 127 for b in type_bytes):
                    _log('Section type field is not printable ASCII - stopping section walk '
                         '(likely not a section header).')
                    break
                try:
                    section_type = type_bytes.decode('ascii')
                except UnicodeDecodeError:
                    break
                next_offset = struct.unpack('<Q', raw_section[16:24])[0]
                section_size = struct.unpack('<Q', raw_section[24:32])[0]
                header_crc = struct.unpack('<I', raw_section[72:76])[0]
                data_offset = offset + SECTION_HEADER_LEN
                data_end = None
                if data_offset < next_offset <= file_size:
                    data_end = next_offset
                elif section_size >= SECTION_HEADER_LEN and offset + section_size <= file_size:
                    data_end = offset + section_size
                    if not next_offset:
                        next_offset = data_end
                data_size = (data_end - data_offset) if data_end is not None else 0
                _log('Section #%d @ %d: type=%-10s data_offset=%d data_size=%d next=%d'
                     % (index, offset, section_type, data_offset, data_size, next_offset))
                info = {'index': index, 'type': section_type, 'offset': offset,
                        'data_offset': data_offset, 'data_size': data_size,
                        'section_size': section_size,
                        'next_section_offset': next_offset,
                        'header_crc32': '0x%08x' % header_crc}
                sections_info.append(info)
                if section_type in HEADER_SECTION_TYPES and data_size > 0:
                    fh.seek(data_offset)
                    header_sections.append({'info': info,
                                            'raw': fh.read(min(data_size,
                                                               MAX_HEADER_DATA_BYTES))})
                    _log('Captured %r section payload (%d bytes).'
                         % (section_type, min(data_size, MAX_HEADER_DATA_BYTES)))
                    found_header = True
                elif section_type in ('volume', 'disk') and data_size > 0:
                    fh.seek(data_offset)
                    volume_bytes = fh.read(min(data_size, 64))
                    _log('Captured %r section payload (%d bytes) for media information.'
                         % (section_type, len(volume_bytes)))
                if section_type in ('done', 'next'):
                    _log("Reached terminal %r section - stopping walk." % section_type)
                    break
                if found_header and section_type not in HEADER_SECTION_TYPES:
                    _log("Header section(s) parsed and non-header section %r reached - "
                         "stopping walk." % section_type)
                    break
                if not next_offset or next_offset <= offset or next_offset > file_size:
                    _log('Invalid next section offset %d - stopping walk.' % next_offset)
                    break
                offset = next_offset
    except Exception:
        import traceback
        pass  # Console diagnostics suppressed: evidence stays in the result contract.
        _log('Exception while walking sections - continuing with the data collected so far.')

    if not header_sections:
        return _fail('No header/header2 section could be parsed from %s - the file does not '
                     'appear to be a valid EWF/E01 image.' % file_path, 'no_header_section',
                     file_path=file_path, signature_hex=signature.hex(),
                     sections=sections_info)

    # ----------------------------------- 4. inflate + decode the header payloads
    parsed_header_sections = []

    def _inflate(raw):
        for compression, wbits in (('zlib', 15), ('raw-deflate', -15)):
            try:
                dec = zlib.decompressobj(wbits)
                out = dec.decompress(raw)
                if out:
                    return compression, out
            except Exception:
                continue
        return 'none', raw

    for entry in header_sections:
        try:
            compression, payload = _inflate(entry['raw'])
            if payload[:2] in (b'\xff\xfe', b'\xfe\xff'):
                encoding = 'utf-16'
                text = payload.decode('utf-16', 'replace')
            else:
                try:
                    text = payload.decode('utf-8')
                    encoding = 'utf-8'
                except UnicodeDecodeError:
                    text = payload.decode('latin-1')
                    encoding = 'latin-1'
            _log('Header section %r: compression=%s encoding=%s decoded_chars=%d'
                 % (entry['info']['type'], compression, encoding, len(text)))
            parsed_header_sections.append({'info': entry['info'],
                                           'compression': compression,
                                           'encoding': encoding,
                                           'text': text})
        except Exception:
            import traceback
            pass  # Console diagnostics suppressed: evidence stays in the result contract.
            _log('Failed to inflate/decode header section %r - skipping it.'
                 % entry['info']['type'])

    if not parsed_header_sections:
        return _fail('The header section(s) of %s could not be decompressed/decoded - the file '
                     'does not appear to be a valid EWF/E01 image.' % file_path,
                     'undecodable_header', file_path=file_path, sections=sections_info)

    # ------------------------------------------- 5. parse the metadata text blocks
    merged_blocks = {}
    main_date_values = {'m': [], 'u': []}

    for parsed in parsed_header_sections:
        normalized = parsed['text'].replace('\r\n', '\n').replace('\r', '\n')
        lines = [ln for ln in normalized.split('\n')]
        raw_blocks = []
        current = []
        for line in lines:
            if line.strip() == '':
                if current:
                    raw_blocks.append(current)
                    current = []
            else:
                current.append(line)
        if current:
            raw_blocks.append(current)
        parsed['blocks'] = {}
        parsed['record_count'] = None
        for block in raw_blocks:
            if (parsed['record_count'] is None and len(block) >= 2
                    and block[0].strip().isdigit() and not block[1].strip().isdigit()):
                parsed['record_count'] = int(block[0].strip())
                block = block[1:]
            if not block:
                continue
            block_name = block[0].strip()
            rest = block[1:]
            if not rest:
                continue
            values_line = rest[-1]
            middle = rest[:-1]
            names_line = None
            names_index = -1
            best = 0
            for position, candidate in enumerate(middle):
                tokens = len(candidate.split('\t'))
                if tokens > best:
                    best = tokens
                    names_index = position
                    names_line = candidate
            extra_lines = [c for position, c in enumerate(middle) if position != names_index]
            names = [n.strip() for n in names_line.split('\t')] if names_line else []
            values = [v.strip() for v in values_line.split('\t')]
            while len(values) < len(names):
                values.append('')
            if names:
                fields = dict(zip(names, values))
            else:
                fields = {}
                for pos, value in enumerate(values):
                    key = (CLASSIC_POSITIONAL_FIELDS[pos]
                           if pos < len(CLASSIC_POSITIONAL_FIELDS) else 'extra_value_%d' % pos)
                    fields[key] = value
            parsed['blocks'][block_name] = {'fields': fields, 'extra_lines': extra_lines,
                                            'field_names': names, 'values': values}
            if block_name == 'main':
                for code in ('m', 'u'):
                    value = fields.get(code, '')
                    if value:
                        main_date_values[code].append(value)
            merged = merged_blocks.setdefault(block_name, {})
            for key, value in fields.items():
                merged.setdefault(key, value)
            _log("Parsed block %r: %d fields, %d extra lines."
                 % (block_name, len(fields), len(extra_lines)))

    # ------------------------------------------------- 6. build acquisition fields
    main_fields = merged_blocks.get('main', {})
    srce_fields = merged_blocks.get('srce', {})

    def _classify_date(value):
        value = (value or '').strip()
        if not value:
            return None
        if value.isdigit() and 9 <= len(value) <= 11:
            try:
                stamp = int(value)
                return ('timestamp', stamp,
                        datetime.fromtimestamp(stamp, tz=timezone.utc)
                        .strftime('%Y-%m-%dT%H:%M:%SZ'))
            except (ValueError, OSError, OverflowError):
                return None
        match = re.match(r'^(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})$',
                         value)
        if match:
            year, month, day, hour, minute, second = (int(g) for g in match.groups())
            return ('date', value, '%04d-%02d-%02d %02d:%02d:%02d'
                    % (year, month, day, hour, minute, second))
        for fmt in ('%a %b %d %Y %H:%M:%S', '%a %b %d %H:%M:%S %Y', '%Y-%m-%d %H:%M:%S',
                    '%m/%d/%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S'):
            try:
                parsed = datetime.strptime(value, fmt)
                return ('date', value, parsed.strftime('%Y-%m-%d %H:%M:%S'))
            except ValueError:
                continue
        return None

    acquisition = {}
    for code, value in main_fields.items():
        if code in ('m', 'u'):
            continue
        name = FIELD_CODE_MAP.get(code, code)
        acquisition[name] = value
        if code == 'l' and value:
            acquisition['device_manufacturer'] = value
    for code, value in srce_fields.items():
        if not value:
            continue
        name = SRCE_CODE_MAP.get(code)
        if name and not acquisition.get(name):
            acquisition[name] = value

    for code, label in (('m', 'acquired'), ('u', 'system')):
        values = list(main_date_values.get(code, []))
        existing = acquisition.get('%s_date' % label)
        if existing and existing not in values:
            values.append(existing)
        raw_fallback = None
        for value in values:
            classified = _classify_date(value)
            if classified is None:
                raw_fallback = value
                continue
            kind, primary, secondary = classified
            if kind == 'timestamp':
                acquisition['%s_timestamp' % label] = primary
                acquisition['%s_timestamp_utc' % label] = secondary
            else:
                acquisition['%s_date' % label] = primary
                acquisition['%s_date_normalized' % label] = secondary
        if values and '%s_date' % label not in acquisition:
            acquisition['%s_date' % label] = raw_fallback if raw_fallback is not None else values[0]

    # ------------------------------------------------- 7. media info (volume sect.)
    media_info = None
    if volume_bytes and len(volume_bytes) >= 20:
        try:
            words = list(struct.unpack('<%dI' % (len(volume_bytes) // 4),
                                       volume_bytes[: (len(volume_bytes) // 4) * 4]))
            media_info = {'source_section': 'volume',
                          'volume_section_values': words}
            bytes_per_sector = words[3] if len(words) > 3 else 0
            total_sectors = words[4] if len(words) > 4 else 0
            if bytes_per_sector in (512, 1024, 2048, 4096) and 0 < total_sectors < (1 << 40):
                media_info['bytes_per_sector'] = bytes_per_sector
                media_info['total_sectors'] = total_sectors
                media_info['total_bytes'] = bytes_per_sector * total_sectors
                media_info['total_size_human'] = '%.2f GB' % (
                    (bytes_per_sector * total_sectors) / (1000.0 ** 3))
            _log('Media info parsed from the volume section: %s' % media_info)
        except Exception:
            import traceback
            pass  # Console diagnostics suppressed: evidence stays in the result contract.
            media_info = None

    # --------------------------------------------------------- 8. assemble result
    raw_text = parsed_header_sections[0]['text']
    result = {
        'valid': True,
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'file_size_bytes': file_size,
        'format': 'EWF/E01',
        'format_description': ewf_kind,
        'signature_hex': signature.hex(),
        'ewf_version': '%d.%d' % (version_major, version_minor),
        'ewf_version_major': version_major,
        'ewf_version_minor': version_minor,
        'sections': sections_info,
        'section_types': [s['type'] for s in sections_info],
        'header_section_types': [p['info']['type'] for p in parsed_header_sections],
        'acquisition_metadata': acquisition,
        'header_fields': merged_blocks,
        'parsed_header_sections': [
            {'type': p['info']['type'], 'offset': p['info']['offset'],
             'data_offset': p['info']['data_offset'], 'data_size': p['info']['data_size'],
             'compression': p['compression'], 'encoding': p['encoding'],
             'record_count': p['record_count'], 'blocks': p['blocks']}
            for p in parsed_header_sections
        ],
        'raw_header_text': raw_text[:MAX_RAW_TEXT_CHARS],
    }
    if media_info:
        result['media_info'] = media_info
    _log('Successfully parsed %d header section(s); %d acquisition field(s) extracted.'
         % (len(parsed_header_sections), len(acquisition)))
    for key in sorted(acquisition):
        _log('  %-28s = %r' % (key, acquisition[key]))
    return {'result': result}
