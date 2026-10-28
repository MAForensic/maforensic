"""Generic image metadata and timestamp normalization primitives."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import asyncio
import json
import math
from pathlib import Path
import re

MAX_EXIF_ITEMS = 64
MAX_EXIF_VALUE_CHARS = 512


async def read_image_metadata(file_path: str) -> str:
    path = Path(file_path)
    try:
        from PIL import Image, ExifTags
        with Image.open(path) as image:
            exif = image.getexif()
            exif_data, exif_truncated = _bounded_exif(exif, ExifTags.TAGS)
            result = {
                "format": image.format, "width": image.width,
                "height": image.height, "mode": image.mode,
                "exif": exif_data, "exif_truncated": exif_truncated,
            }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: image metadata read failed: {exc}"


def _bounded_exif(exif, tag_names: dict[int, str]) -> tuple[dict, bool]:
    items = []
    for key, value in exif.items():
        tag_name = str(tag_names.get(key, str(key)))
        items.append((tag_name, str(key), value))
    items.sort(key=lambda item: (item[0], item[1]))
    result = {}
    truncated = len(items) > MAX_EXIF_ITEMS
    for tag_name, tag_id, value in items[:MAX_EXIF_ITEMS]:
        output_key = tag_name if tag_name not in result else f"{tag_name}#{tag_id}"
        converted, value_truncated = _exif_value(value)
        result[output_key] = converted
        truncated = truncated or value_truncated
    return result, truncated


def _exif_value(value) -> tuple[object, bool]:
    if value is None or isinstance(value, (int, float, bool)):
        return value, False
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary {len(value)} bytes>", True
    if isinstance(value, str):
        return _limit_exif_text(value)
    text = str(value)
    limited, truncated = _limit_exif_text(text)
    return limited, True or truncated


def _limit_exif_text(value: str) -> tuple[str, bool]:
    if len(value) <= MAX_EXIF_VALUE_CHARS:
        return value, False
    return value[:MAX_EXIF_VALUE_CHARS], True


async def normalize_timestamp(value: float, format_hint: str = "auto", utc_offset_minutes: int = 0) -> str:
    if isinstance(value, bool):
        return "Error: timestamp value must be finite numeric."
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return "Error: timestamp value must be finite numeric."
    if not math.isfinite(raw):
        return "Error: timestamp value must be finite numeric."
    if not isinstance(format_hint, str):
        return "Error: format_hint must be a string."
    offset_minutes = _parse_integer_offset(utc_offset_minutes)
    if offset_minutes is None:
        return f"Error: invalid utc_offset_minutes: {utc_offset_minutes}"
    if not -1439 <= offset_minutes <= 1439:
        return f"Error: invalid utc_offset_minutes: {utc_offset_minutes}"
    hint = {
        "unix_seconds": "unix",
        "unix_milliseconds": "unix_ms",
        "unix_microseconds": "unix_us",
        "unix_nanoseconds": "unix_ns",
        "windows_filetime": "filetime",
    }.get(format_hint.lower(), format_hint.lower())
    known = {"auto", "unix", "unix_ms", "unix_us", "unix_ns", "filetime", "webkit", "apple"}
    if hint not in known:
        return f"Error: unknown format_hint: {format_hint}"
    candidates = []
    formats = [hint] if hint != "auto" else ["unix", "unix_ms", "unix_us", "unix_ns", "filetime", "webkit", "apple"]
    for fmt in formats:
        try:
            if fmt == "unix":
                dt = datetime.fromtimestamp(raw, timezone.utc)
            elif fmt == "unix_ms":
                dt = datetime.fromtimestamp(raw / 1e3, timezone.utc)
            elif fmt == "unix_us":
                dt = datetime.fromtimestamp(raw / 1e6, timezone.utc)
            elif fmt == "unix_ns":
                dt = datetime.fromtimestamp(raw / 1e9, timezone.utc)
            elif fmt == "filetime":
                dt = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=raw / 10)
            elif fmt == "webkit":
                dt = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=raw)
            elif fmt == "apple":
                dt = datetime(2001, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=raw)
            else:
                continue
            if 1970 <= dt.year <= 2200:
                local = dt.astimezone(timezone(timedelta(minutes=offset_minutes)))
                candidates.append({"format": fmt, "utc": _utc_iso(dt), "local": local.isoformat()})
        except (ValueError, OverflowError, OSError):
            continue
    if not candidates:
        return "Error: no valid timestamp interpretation."
    return json.dumps({"candidates": candidates, "ambiguous": len(candidates) > 1}, indent=2)


def _parse_integer_offset(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return None
    return None


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
