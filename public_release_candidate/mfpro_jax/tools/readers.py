"""Generic file reading, metadata, strings, and search primitives."""

from __future__ import annotations

import asyncio
import codecs
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import subprocess
from typing import Any

MAX_BOUNDED_READ_BYTES = 1024 * 1024
TEXT_ENCODING_SAMPLE_BYTES = 64 * 1024
SEARCH_FILE_MAX_BYTES = 256 * 1024 * 1024
SEARCH_FILE_CHUNK_BYTES = 1024 * 1024
SEARCH_FILE_MAX_OVERLAP_BYTES = 1024 * 1024
MAX_SEARCH_RESULTS = 1000
MAX_CONTEXT_CHARS = 4096
MAX_SEARCH_CONTEXT_VALUE_CHARS = 8192
MAX_STRING_MIN_LENGTH = 4096
MAX_STRING_VALUE_CHARS = 500


def _path(file_path: str) -> Path:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(file_path)
    return path


def _strict_int(value, field_name: str) -> tuple[int | None, str]:
    if isinstance(value, bool):
        return None, f"Error: {field_name} must be an integer."
    if isinstance(value, int):
        return value, ""
    return None, f"Error: {field_name} must be an integer."


def _bounded_range(offset, count, count_name: str) -> tuple[int | None, int | None, str]:
    parsed_offset, error = _strict_int(offset, "offset")
    if error:
        return None, None, error
    parsed_count, error = _strict_int(count, count_name)
    if error:
        return None, None, error
    if parsed_offset < 0:
        return None, None, "Error: offset must be >= 0."
    if parsed_count <= 0:
        return None, None, f"Error: {count_name} must be between 1 and {MAX_BOUNDED_READ_BYTES}."
    if parsed_count > MAX_BOUNDED_READ_BYTES:
        return None, None, f"Error: {count_name} must be between 1 and {MAX_BOUNDED_READ_BYTES}."
    return parsed_offset, parsed_count, ""


def _read_range(path: Path, offset: int, count: int) -> tuple[bytes, int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(count)
    return data, size


def _read_encoding_sample(path: Path) -> bytes:
    with path.open("rb") as stream:
        return stream.read(TEXT_ENCODING_SAMPLE_BYTES)


def _next_offset(offset: int, bytes_read: int, source_size: int | None = None, forced_truncated: bool = False) -> tuple[bool, int | None]:
    end = offset + bytes_read
    truncated = forced_truncated
    if source_size is not None:
        truncated = truncated or source_size > end
    return truncated, end if truncated else None


def _decode_text(data: bytes, encoding: str, detection_sample: bytes | None = None) -> tuple[str, str, bool, str]:
    if not isinstance(encoding, str):
        return "", "", False, "Error: encoding must be a string."
    encoding_name = encoding.strip()
    if encoding_name.lower() == "auto":
        if not data:
            return "", "unknown", False, ""
        sample = detection_sample if detection_sample is not None else data
        encoding_name = detect_encoding_bytes(sample or data)
    elif not encoding_name:
        return "", "", False, "Error: unknown encoding: "
    try:
        codec = codecs.lookup(encoding_name)
    except LookupError:
        return "", "", False, f"Error: unknown encoding: {encoding_name}"
    normalized = codec.name
    try:
        return data.decode(normalized, errors="strict"), normalized, False, ""
    except UnicodeDecodeError:
        return data.decode(normalized, errors="replace"), normalized, True, ""


async def read_text(file_path: str, offset: int = 0, max_bytes: int = 16000, encoding: str = "auto") -> str:
    parsed_offset, parsed_max, error = _bounded_range(offset, max_bytes, "max_bytes")
    if error:
        return error
    path = _path(file_path)
    if not isinstance(encoding, str):
        return "Error: encoding must be a string."
    detection_sample = _read_encoding_sample(path) if encoding.strip().lower() == "auto" else None
    data, source_size = _read_range(path, parsed_offset, parsed_max)
    text, enc, decode_lossy, error = _decode_text(data, encoding, detection_sample)
    if error:
        return error
    truncated, next_offset = _next_offset(parsed_offset, len(data), source_size)
    return json.dumps({
        "text": text,
        "encoding": enc,
        "offset": parsed_offset,
        "bytes_read": len(data),
        "decode_lossy": decode_lossy,
        "truncated": truncated,
        "next_offset": next_offset,
    }, ensure_ascii=False)


async def read_binary_range(file_path: str, offset: int = 0, length: int = 2048) -> str:
    parsed_offset, parsed_length, error = _bounded_range(offset, length, "length")
    if error:
        return error
    path = _path(file_path)
    data, source_size = _read_range(path, parsed_offset, parsed_length)
    lines = []
    for pos in range(0, len(data), 16):
        chunk = data[pos:pos + 16]
        lines.append(f"{parsed_offset + pos:08x}  {' '.join(f'{b:02x}' for b in chunk):<48}  {''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)}")
    truncated, next_offset = _next_offset(parsed_offset, len(data), source_size)
    return json.dumps({
        "hexdump": "\n".join(lines),
        "offset": parsed_offset,
        "bytes_read": len(data),
        "truncated": truncated,
        "next_offset": next_offset,
    }, ensure_ascii=False)


async def compute_hash(file_path: str, algorithms: list[str] | None = None) -> str:
    path = _path(file_path)
    names = algorithms or ["sha256"]
    hashes = {}
    for name in names:
        digest = hashlib.new(name)
        canonical = digest.name.lower()
        if canonical not in hashes:
            hashes[canonical] = digest
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            for digest in hashes.values():
                digest.update(chunk)
            await asyncio.sleep(0)
    return json.dumps({"digests": {name: digest.hexdigest() for name, digest in hashes.items()}}, indent=2)


async def identify_file_type(file_path: str) -> str:
    path = _path(file_path)
    mime, encoding = mimetypes.guess_type(path.name)
    proc = await asyncio.create_subprocess_exec("file", "--brief", "--mime-type", str(path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return "Error: file type dependency timed out."
    if proc.returncode:
        return f"Error: file type dependency failed: {stderr.decode(errors='replace').strip()}"
    detected = stdout.decode(errors="replace").strip() or None
    return json.dumps({
        "extension": path.suffix.lower(),
        "extension_mime": mime or None,
        "detected_mime": detected,
        "signature_mismatch": bool(mime and detected and mime != detected),
    }, indent=2)


def detect_encoding_bytes(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError as exc:
        if exc.end == len(data) and exc.reason == "unexpected end of data":
            return "utf-8"
        if data.count(b"\x00") > max(2, len(data) // 10):
            return "utf-16-le"
        return "latin-1"


async def detect_text_encoding(file_path: str, sample_bytes: int = 65536) -> str:
    path = _path(file_path)
    with path.open("rb") as stream:
        data = stream.read(max(0, min(sample_bytes, 1024 * 1024)))
    encoding = "unknown" if not data else detect_encoding_bytes(data)
    return json.dumps({"encoding": encoding, "sampled_bytes": len(data)})


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _bounded_int(value: Any, field_name: str, minimum: int, maximum: int) -> tuple[int | None, str]:
    parsed, error = _strict_int(value, field_name)
    if error:
        return None, error
    if parsed < minimum or parsed > maximum:
        return None, f"Error: {field_name} must be between {minimum} and {maximum}."
    return parsed, ""


def _require_bool(value: Any, field_name: str) -> tuple[bool | None, str]:
    if not isinstance(value, bool):
        return None, f"Error: {field_name} must be a boolean."
    return value, ""


def _compile_search_pattern(pattern: Any, regex: Any) -> tuple[re.Pattern[str] | None, str]:
    parsed_regex, error = _require_bool(regex, "regex")
    if error:
        return None, error
    if not isinstance(pattern, str):
        return None, "Error: pattern must be a string."
    if pattern == "":
        return None, "Error: pattern must be a non-empty string."
    try:
        return re.compile(pattern if parsed_regex else re.escape(pattern), re.IGNORECASE), ""
    except re.error as exc:
        return None, f"Error: invalid regular expression: {exc}"


def _read_search_prefix(path: Path, max_bytes: int = SEARCH_FILE_MAX_BYTES) -> tuple[bytes, int, bool]:
    file_size = path.stat().st_size
    scan_bytes = min(file_size, max_bytes)
    with path.open("rb") as stream:
        data = stream.read(scan_bytes)
    return data, file_size, file_size <= len(data)


def _decode_search_data(data: bytes) -> tuple[str, str, bool]:
    encoding = "utf-8" if not data else detect_encoding_bytes(data[:TEXT_ENCODING_SAMPLE_BYTES])
    try:
        return data.decode(encoding, errors="strict"), encoding, False
    except UnicodeDecodeError:
        return data.decode(encoding, errors="replace"), encoding, True


def _byte_offset_for_char(data: bytes, text: str, char_index: int, encoding: str, decode_lossy: bool) -> int | None:
    if decode_lossy:
        return None
    if encoding == "latin-1":
        return char_index
    if encoding == "utf-8":
        return len(text[:char_index].encode("utf-8"))
    if encoding == "utf-8-sig":
        return 3 + len(text[:char_index].encode("utf-8")) if data.startswith(b"\xef\xbb\xbf") else len(text[:char_index].encode("utf-8"))
    if encoding == "utf-16":
        if data.startswith(b"\xff\xfe"):
            return 2 + len(text[:char_index].encode("utf-16-le"))
        if data.startswith(b"\xfe\xff"):
            return 2 + len(text[:char_index].encode("utf-16-be"))
        return None
    if encoding == "utf-16-le":
        return len(text[:char_index].encode("utf-16-le"))
    return None


def _bounded_match_context(text: str, match_start: int, match_end: int, context_chars: int) -> tuple[str, bool]:
    ideal_start = max(0, match_start - context_chars)
    ideal_end = min(len(text), match_end + context_chars)
    if ideal_end - ideal_start <= MAX_SEARCH_CONTEXT_VALUE_CHARS:
        return text[ideal_start:ideal_end], False
    match_length = match_end - match_start
    if match_length >= MAX_SEARCH_CONTEXT_VALUE_CHARS:
        return text[match_start:match_start + MAX_SEARCH_CONTEXT_VALUE_CHARS], True
    budget = MAX_SEARCH_CONTEXT_VALUE_CHARS - match_length
    before = min(context_chars, budget // 2, match_start)
    after = min(context_chars, budget - before, len(text) - match_end)
    remaining = budget - before - after
    if remaining and before < min(context_chars, match_start):
        take = min(remaining, min(context_chars, match_start) - before)
        before += take
        remaining -= take
    if remaining and after < min(context_chars, len(text) - match_end):
        take = min(remaining, min(context_chars, len(text) - match_end) - after)
        after += take
    return text[match_start - before:match_end + after], True


def _search_file_core(path: Path, pattern: str, regex: bool, context_chars: int, max_results: int, max_bytes: int | None = None) -> tuple[dict | None, str]:
    compiled, error = _compile_search_pattern(pattern, regex)
    if error:
        return None, error
    data, file_size, full_prefix = _read_search_prefix(path, SEARCH_FILE_MAX_BYTES if max_bytes is None else max_bytes)
    text, encoding, decode_lossy = _decode_search_data(data)
    results = []
    extra_found = False
    for match in compiled.finditer(text):
        context, context_truncated = _bounded_match_context(text, match.start(), match.end(), context_chars)
        item = {
            "byte_offset": _byte_offset_for_char(data, text, match.start(), encoding, decode_lossy),
            "context": context,
            "context_truncated": context_truncated,
        }
        if len(results) < max_results:
            results.append(item)
        else:
            extra_found = True
            break
    return {
        "encoding": encoding,
        "decode_lossy": decode_lossy,
        "file_size": file_size,
        "scanned_bytes": len(data),
        "scan_complete": full_prefix and not extra_found,
        "results": results,
        "returned_matches": len(results),
        "truncated": extra_found,
    }, ""


def _scan_file_bytes(path: Path, max_bytes: int) -> tuple[bytes, int, bool]:
    file_size = path.stat().st_size
    scan_bytes = min(file_size, max_bytes)
    with path.open("rb") as stream:
        data = stream.read(scan_bytes)
    return data, file_size, file_size <= len(data)


def _iter_file_strings(data: bytes, min_length: int):
    ascii_re = re.compile(rb"[\x20-\x7e]{" + str(min_length).encode() + rb",}")
    utf16_re = re.compile(rb"(?:[\x20-\x7e]\x00){" + str(min_length).encode() + rb",}")
    items = []
    for match in ascii_re.finditer(data):
        items.append({
            "byte_offset": match.start(),
            "encoding": "ascii",
            "value": match.group().decode("ascii", errors="replace"),
        })
    for match in utf16_re.finditer(data):
        items.append({
            "byte_offset": match.start(),
            "encoding": "utf16le",
            "value": match.group().decode("utf-16-le", errors="replace"),
        })
    items.sort(key=lambda item: (item["byte_offset"], item["encoding"]))
    yield from items


def _string_result_value(value: str) -> tuple[str, bool]:
    if len(value) <= MAX_STRING_VALUE_CHARS:
        return value, False
    return value[:MAX_STRING_VALUE_CHARS], True


def _string_match_window(value: str, start: int, end: int) -> tuple[str, bool]:
    if len(value) <= MAX_STRING_VALUE_CHARS:
        return value, False
    match_len = max(1, end - start)
    if match_len >= MAX_STRING_VALUE_CHARS:
        return value[start:start + MAX_STRING_VALUE_CHARS], True
    before = max(0, (MAX_STRING_VALUE_CHARS - match_len) // 2)
    window_start = max(0, min(start - before, len(value) - MAX_STRING_VALUE_CHARS))
    return value[window_start:window_start + MAX_STRING_VALUE_CHARS], True


async def extract_file_strings(file_path: str, min_length: int = 4, max_results: int = 500, max_bytes: int = 268435456) -> str:
    parsed_min, error = _bounded_int(min_length, "min_length", 1, MAX_STRING_MIN_LENGTH)
    if error:
        return error
    parsed_max_results, error = _bounded_int(max_results, "max_results", 1, MAX_SEARCH_RESULTS)
    if error:
        return error
    parsed_max_bytes, error = _bounded_int(max_bytes, "max_bytes", 1, SEARCH_FILE_MAX_BYTES)
    if error:
        return error
    path = _path(file_path)
    data, file_size, full_scan = _scan_file_bytes(path, parsed_max_bytes)
    results = []
    extra_found = False
    for item in _iter_file_strings(data, parsed_min):
        if len(results) >= parsed_max_results:
            extra_found = True
            break
        value, value_truncated = _string_result_value(item["value"])
        results.append({
            "byte_offset": item["byte_offset"],
            "encoding": item["encoding"],
            "value": value,
            "value_truncated": value_truncated,
        })
    return _json_dumps({
        "file_size": file_size,
        "scanned_bytes": len(data),
        "scan_complete": full_scan and not extra_found,
        "results": results,
        "returned_strings": len(results),
        "truncated": extra_found,
    })


async def search_file_strings(
    file_path: str, pattern: str, regex: bool = False, min_length: int = 4,
    max_results: int = 200, max_bytes: int = 268435456,
) -> str:
    parsed_min, error = _bounded_int(min_length, "min_length", 1, MAX_STRING_MIN_LENGTH)
    if error:
        return error
    parsed_max_results, error = _bounded_int(max_results, "max_results", 1, MAX_SEARCH_RESULTS)
    if error:
        return error
    parsed_max_bytes, error = _bounded_int(max_bytes, "max_bytes", 1, SEARCH_FILE_MAX_BYTES)
    if error:
        return error
    compiled, error = _compile_search_pattern(pattern, regex)
    if error:
        return error
    path = _path(file_path)
    data, file_size, full_scan = _scan_file_bytes(path, parsed_max_bytes)
    results = []
    extra_found = False
    for item in _iter_file_strings(data, parsed_min):
        for match in compiled.finditer(item["value"]):
            byte_offset = item["byte_offset"] + (match.start() if item["encoding"] == "ascii" else match.start() * 2)
            if len(results) >= parsed_max_results:
                extra_found = True
                break
            value, value_truncated = _string_match_window(item["value"], match.start(), match.end())
            results.append({
                "byte_offset": byte_offset,
                "encoding": item["encoding"],
                "value": value,
                "value_truncated": value_truncated,
            })
        if extra_found:
            break
    return _json_dumps({
        "file_size": file_size,
        "scanned_bytes": len(data),
        "scan_complete": full_scan and not extra_found,
        "results": results,
        "returned_matches": len(results),
        "truncated": extra_found,
    })


async def search_file(file_path: str, pattern: str, regex: bool = False, context_chars: int = 80, max_results: int = 100) -> str:
    parsed_context, error = _bounded_int(context_chars, "context_chars", 0, MAX_CONTEXT_CHARS)
    if error:
        return error
    parsed_max_results, error = _bounded_int(max_results, "max_results", 1, MAX_SEARCH_RESULTS)
    if error:
        return error
    path = _path(file_path)
    result, error = _search_file_core(path, pattern, regex, parsed_context, parsed_max_results)
    if error:
        return error
    return _json_dumps(result)


async def search_files_content(root: str, pattern: str, regex: bool = False, extensions: list[str] | None = None, max_file_size: int = 10485760, max_results: int = 200) -> str:
    base = Path(root)
    if not base.is_dir():
        return f"Error: not a directory: {root}"
    parsed_max_file_size, error = _bounded_int(max_file_size, "max_file_size", 1, SEARCH_FILE_MAX_BYTES)
    if error:
        return error
    parsed_max_results, error = _bounded_int(max_results, "max_results", 1, MAX_SEARCH_RESULTS)
    if error:
        return error
    compiled, error = _compile_search_pattern(pattern, regex)
    if error:
        return error
    # The compiled object is only used for validation here; _search_file_core recompiles per call.
    del compiled
    if extensions is None:
        allowed = set()
    elif not isinstance(extensions, list):
        return "Error: extensions must be an array of strings."
    else:
        allowed = set()
        for item in extensions:
            if not isinstance(item, str):
                return "Error: extensions must be an array of strings."
            ext = item.strip()
            if not ext:
                return "Error: extensions must contain non-empty strings."
            if "/" in ext or "\\" in ext or "\x00" in ext:
                return "Error: extensions must not contain path separators."
            allowed.add(ext.lower() if ext.startswith(".") else f".{ext.lower()}")
    hits = []
    scanned_files = 0
    skipped_large_files = 0
    failed_files = 0
    scan_complete = True
    try:
        candidates = sorted(base.rglob("*"), key=lambda item: item.relative_to(base).as_posix())
    except OSError:
        return f"Error: unable to read directory: {root}"
    for path in candidates:
        try:
            if path.is_symlink() or not path.is_file() or (allowed and path.suffix.lower() not in allowed):
                continue
            if path.stat().st_size > parsed_max_file_size:
                skipped_large_files += 1
                scan_complete = False
                continue
            scanned_files += 1
            remaining = parsed_max_results + 1 - len(hits)
            result, error = _search_file_core(path, pattern, regex, 80, remaining, parsed_max_file_size)
            if error:
                failed_files += 1
                scan_complete = False
                continue
            if not result["scan_complete"]:
                scan_complete = False
            for hit in result["results"]:
                hits.append({"relative_path": path.relative_to(base).as_posix(), **hit})
            if len(hits) > parsed_max_results:
                return _json_dumps({
                    "results": hits[:parsed_max_results],
                    "scanned_files": scanned_files,
                    "skipped_large_files": skipped_large_files,
                    "failed_files": failed_files,
                    "scan_complete": False,
                    "returned_matches": parsed_max_results,
                    "truncated": True,
                })
        except (OSError, ValueError, UnicodeError):
            failed_files += 1
            scan_complete = False
            continue
        if scanned_files % 100 == 0:
            await asyncio.sleep(0)
    return _json_dumps({
        "results": hits,
        "scanned_files": scanned_files,
        "skipped_large_files": skipped_large_files,
        "failed_files": failed_files,
        "scan_complete": scan_complete,
        "returned_matches": len(hits),
        "truncated": False,
    })
