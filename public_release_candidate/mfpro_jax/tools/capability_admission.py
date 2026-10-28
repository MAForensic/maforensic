"""Thin host-path adapters; archived parser algorithms are installed locally."""
from __future__ import annotations

import json

from .read_windows_registry_value import read_windows_registry_value as _registry_value
from .parse_e01_header import parse_e01_header as _e01_header


def authorized_path(ctx, value: str) -> str:
    from ..tool_registry import _normalize_and_validate_paths
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty local evidence path is required")
    return _normalize_and_validate_paths({"file_path": value}, ctx)["file_path"]


def encode_result(raw: dict) -> str:
    result = raw.get("result", {})
    failed = bool(result.get("error")) or result.get("found") is False or result.get("valid") is False
    return ("Error: " if failed else "") + json.dumps(raw, ensure_ascii=False)


async def read_windows_registry_value(ctx, hive_path: str, key_path: str, value_name: str) -> str:
    try:
        path = authorized_path(ctx, hive_path)
    except (ValueError, OSError, RuntimeError):
        return encode_result({"result": {"found": False, "error": "Hive path is not accessible within case evidence roots", "error_type": "path_denied"}})
    return encode_result(_registry_value(path, key_path, value_name))


async def parse_e01_header(ctx, file_path: str) -> str:
    try:
        path = authorized_path(ctx, file_path)
    except (ValueError, OSError, RuntimeError):
        return encode_result({"result": {"valid": False, "error": "EWF path is not accessible within case evidence roots", "error_type": "path_denied"}})
    return encode_result(_e01_header(path))
