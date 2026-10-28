"""Generic structured-data parsers with no application semantics."""

from __future__ import annotations

import configparser
import csv
import io
import json
import math
from pathlib import Path
import plistlib
import re
import sqlite3
import time
from urllib.parse import quote
from xml.parsers import expat
import xml.etree.ElementTree as ET

MAX_STRUCTURED_INPUT_BYTES = 64 * 1024 * 1024
SQLITE_OPERATION_TIMEOUT_SEC = 55
MAX_STRUCTURED_ROWS = 10_000
MAX_STRUCTURED_RESULTS = 1_000
MAX_PREVIEW_DEPTH = 6
MAX_CONTAINER_ITEMS = 32
MAX_STRING_CHARS = 512
MAX_KEY_CHARS = 256
MAX_TOTAL_SCALAR_CHARS = 20_000
MAX_TOTAL_PREVIEW_NODES = 512
MAX_PLIST_DATA_BYTES = 256
MAX_SQLITE_OBJECTS = 128
MAX_SQL_CHARS = 4096
MAX_CSV_COLUMNS = 64
MAX_QUERY_ROWS = 1000
MAX_SEARCH_RESULTS = 1000
MAX_QUERY_TEXT_CHARS = 4096
MAX_SEARCH_VALUE_CHARS = 512
MAX_BINARY_PREVIEW_BYTES = 256
_OMITTED = object()
SQLITE_SCHEMA_DETAIL_TYPES = {"table", "view", "index", "trigger"}


def _bounded_path(file_path: str) -> Path:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(file_path)
    if path.stat().st_size > MAX_STRUCTURED_INPUT_BYTES:
        raise ValueError(f"structured input exceeds {MAX_STRUCTURED_INPUT_BYTES} byte limit")
    return path


def _readonly_connection(db_path: str) -> sqlite3.Connection:
    path = Path(db_path).resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(db_path)
    uri = path.as_uri()
    readonly_uri = f"{uri}{'&' if '?' in uri else '?'}mode=ro"
    conn = sqlite3.connect(readonly_uri, uri=True)
    deadline = time.monotonic() + SQLITE_OPERATION_TIMEOUT_SEC
    conn.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, 10000)
    conn.execute("PRAGMA query_only=ON")
    conn.set_authorizer(_sqlite_readonly_authorizer)
    return conn

def _render(value, limit: int = 32000) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=lambda obj: obj.isoformat() if hasattr(obj, "isoformat") else str(obj))
    return text[:limit] + (f"\n[truncated; total={len(text)} chars]" if len(text) > limit else "")


class _PreviewState:
    def __init__(self) -> None:
        self.truncated = False
        self.nodes = 0
        self.scalar_chars = 0

    def mark_truncated(self) -> None:
        self.truncated = True

    def take_node(self) -> bool:
        if self.nodes >= MAX_TOTAL_PREVIEW_NODES:
            self.mark_truncated()
            return False
        self.nodes += 1
        return True

    @property
    def remaining_chars(self) -> int:
        return max(0, MAX_TOTAL_SCALAR_CHARS - self.scalar_chars)

    def take_text(self, text: str, limit: int = MAX_STRING_CHARS):
        if text == "":
            return ""
        if self.remaining_chars <= 0:
            self.mark_truncated()
            return _OMITTED
        allowed = min(limit, max(0, MAX_TOTAL_SCALAR_CHARS - self.scalar_chars))
        if allowed <= 0:
            self.mark_truncated()
            return _OMITTED
        if len(text) > allowed:
            self.mark_truncated()
            result = text[:allowed]
        else:
            result = text
        self.scalar_chars += len(result)
        if len(text) > limit:
            self.mark_truncated()
        if self.scalar_chars >= MAX_TOTAL_SCALAR_CHARS and len(text) > len(result):
            self.mark_truncated()
        return result


def _json_dumps(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    except ValueError as exc:
        return f"Error: JSON serialization failed: {exc}"


def _compact_json_dumps(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except ValueError as exc:
        return f"Error: JSON serialization failed: {exc}"


def _root_type(value) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "data"
    if hasattr(value, "isoformat"):
        return "datetime"
    if _is_plist_uid(value):
        return "uid"
    return "unsupported"


def _is_plist_uid(value) -> bool:
    uid_type = getattr(plistlib, "UID", None)
    return uid_type is not None and isinstance(value, uid_type)


def _uid_value(value) -> int:
    return int(getattr(value, "data", value))


def _shorten_key(key: object, state: _PreviewState):
    text = str(key)
    return state.take_text(text, MAX_KEY_CHARS)


def _unique_key(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    index = 2
    while f"{base}#{index}" in used:
        index += 1
    value = f"{base}#{index}"
    used.add(value)
    return value


def _unique_budgeted_key(base: str, used: set[str], state: _PreviewState):
    value = _unique_key(base, used)
    if value != base:
        suffix = value[len(base):]
        if state.take_text(suffix, len(suffix)) is _OMITTED:
            used.discard(value)
            return _OMITTED
    return value


def _bounded_preview(value, state: _PreviewState | None = None, depth: int = 0):
    state = state or _PreviewState()
    if not state.take_node():
        return _OMITTED, state.truncated
    if depth > MAX_PREVIEW_DEPTH:
        state.mark_truncated()
        return _OMITTED, state.truncated
    if isinstance(value, bool) or value is None:
        return value, state.truncated
    if isinstance(value, (int, float)):
        return value, state.truncated
    if isinstance(value, str):
        return state.take_text(value), state.truncated
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        max_hex_chars = min(MAX_PLIST_DATA_BYTES * 2, state.remaining_chars)
        max_hex_chars -= max_hex_chars % 2
        if max_hex_chars <= 0 and data:
            state.mark_truncated()
            return _OMITTED, state.truncated
        preview = data[: max_hex_chars // 2]
        hex_preview = preview.hex()
        state.scalar_chars += len(hex_preview)
        truncated = len(data) > len(preview)
        if truncated:
            state.mark_truncated()
        return {
            "bytes_hex": hex_preview,
            "total_bytes": len(data),
            "truncated": truncated,
        }, state.truncated
    if _is_plist_uid(value):
        return {"uid": _uid_value(value)}, state.truncated
    if hasattr(value, "isoformat"):
        return state.take_text(value.isoformat()), state.truncated
    if isinstance(value, dict):
        if depth >= MAX_PREVIEW_DEPTH and value:
            state.mark_truncated()
            return _OMITTED, state.truncated
        result = {}
        used = set()
        added_items = 0
        for index, (key, child) in enumerate(value.items()):
            if index >= MAX_CONTAINER_ITEMS:
                state.mark_truncated()
                break
            short = _shorten_key(key, state)
            if short is _OMITTED:
                state.mark_truncated()
                if added_items == 0 and value:
                    return _OMITTED, state.truncated
                break
            short = _unique_budgeted_key(short, used, state)
            if short is _OMITTED:
                state.mark_truncated()
                if added_items == 0 and value:
                    return _OMITTED, state.truncated
                break
            child_preview, _ = _bounded_preview(child, state, depth + 1)
            if child_preview is _OMITTED:
                state.mark_truncated()
                if added_items == 0 and value:
                    return _OMITTED, state.truncated
                break
            result[short] = child_preview
            added_items += 1
        return result, state.truncated
    if isinstance(value, (list, tuple)):
        if depth >= MAX_PREVIEW_DEPTH and value:
            state.mark_truncated()
            return _OMITTED, state.truncated
        result = []
        added_items = 0
        for index, child in enumerate(value):
            if index >= MAX_CONTAINER_ITEMS:
                state.mark_truncated()
                break
            item, _ = _bounded_preview(child, state, depth + 1)
            if item is _OMITTED:
                state.mark_truncated()
                if added_items == 0 and value:
                    return _OMITTED, state.truncated
                break
            result.append(item)
            added_items += 1
        return result, state.truncated
    state.mark_truncated()
    return {"unsupported_type": type(value).__name__}, state.truncated


def _ensure_represented(preview, label: str):
    if preview is _OMITTED:
        raise ValueError(f"{label} cannot be represented within preview budget")
    return preview


def _parse_finite_json_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {raw}")
    return value


def _strict_int(value, name: str) -> tuple[int | None, str]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None, f"Error: {name} must be an integer."
    return value, ""


def _jsonable(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value[:128].hex(), "total_bytes": len(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sqlite_schema_columns(conn: sqlite3.Connection, object_name: str) -> list[list[object]]:
    columns = []
    try:
        rows = conn.execute(f"PRAGMA table_xinfo({_quote_identifier(object_name)})")
    except sqlite3.DatabaseError:
        return columns
    for row in rows:
        name = row[1]
        if name is None:
            continue
        column: list[object] = [str(name), str(row[2] or "")]
        pk = int(row[5] or 0)
        if pk:
            column.append(pk)
        columns.append(column)
    return columns


async def sqlite_schema(db_path: str) -> str:
    path = Path(db_path)
    try:
        conn = _readonly_connection(str(path))
        rows = conn.execute("SELECT type, name FROM sqlite_master WHERE type IN ('table','view') ORDER BY type, name").fetchall()
        objects = []
        truncated = False
        for type_, name in rows[:MAX_SQLITE_OBJECTS]:
            objects.append({
                "type": type_,
                "name": name,
                "columns": _sqlite_schema_columns(conn, str(name)),
            })
        if len(rows) > len(objects):
            truncated = True
        return _compact_json_dumps({"objects": objects, "total_objects": len(rows), "truncated": truncated})
    except sqlite3.DatabaseError as exc:
        return f"Error: SQLite open/schema failed: {exc}"
    finally:
        if "conn" in locals():
            conn.close()


async def sqlite_schema_detail(db_path: str, object_name: str, object_type: str = "") -> str:
    object_name, error = _nonempty_text(object_name, "object_name")
    if error:
        return error
    if not isinstance(object_type, str):
        return "Error: object_type must be a string."
    object_type = object_type.strip().lower()
    if object_type and object_type not in SQLITE_SCHEMA_DETAIL_TYPES:
        return "Error: object_type must be one of table, view, index, or trigger."
    try:
        conn = _readonly_connection(db_path)
        params: tuple[str, ...]
        if object_type:
            query = "SELECT type, name, sql FROM sqlite_master WHERE name=? AND type=?"
            params = (object_name, object_type)
        else:
            query = (
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name=? AND type IN ('table','view','index','trigger') ORDER BY type, name"
            )
            params = (object_name,)
        rows = conn.execute(query, params).fetchall()
        if not rows:
            return f"Error: SQLite schema object not found: {object_name}"
        if len(rows) > 1:
            types = ", ".join(str(row[0]) for row in rows)
            return f"Error: SQLite schema object name is ambiguous; provide object_type. matches={types}"
        type_, name, sql = rows[0]
        sql_truncated = isinstance(sql, str) and len(sql) > MAX_SQL_CHARS
        if sql_truncated:
            sql = sql[:MAX_SQL_CHARS]
        return _compact_json_dumps({
            "type": type_,
            "name": name,
            "sql": sql,
            "sql_truncated": sql_truncated,
        })
    except sqlite3.DatabaseError as exc:
        return f"Error: SQLite schema detail failed: {exc}"
    finally:
        if "conn" in locals():
            conn.close()


_WRITE_ACTIONS = {
    getattr(sqlite3, name) for name in (
        "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE",
        "SQLITE_CREATE_INDEX", "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE", "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_VIEW", "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE", "SQLITE_DROP_TEMP_INDEX", "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER", "SQLITE_DROP_TEMP_VIEW", "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW", "SQLITE_ALTER_TABLE", "SQLITE_REINDEX", "SQLITE_ANALYZE",
        "SQLITE_ATTACH", "SQLITE_DETACH", "SQLITE_TRANSACTION", "SQLITE_SAVEPOINT",
    ) if hasattr(sqlite3, name)
}


def _sqlite_readonly_authorizer(action, arg1, arg2, dbname, source):
    if action in _WRITE_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == getattr(sqlite3, "SQLITE_FUNCTION", -1):
        function_name = str(arg1 or arg2 or "").lower()
        if function_name == "load_extension":
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _strict_int_range(value, name: str, minimum: int, maximum: int) -> tuple[int | None, str]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None, f"Error: {name} must be an integer."
    if value < minimum or value > maximum:
        return None, f"Error: {name} must be between {minimum} and {maximum}."
    return value, ""


def _nonempty_text(value, name: str) -> tuple[str | None, str]:
    if not isinstance(value, str):
        return None, f"Error: {name} must be a string."
    if not value.strip():
        return None, f"Error: {name} must be a non-empty string."
    return value, ""


def _strip_sql_leading_comments(sql: str) -> str:
    index = 0
    length = len(sql)
    while index < length:
        while index < length and sql[index].isspace():
            index += 1
        if sql.startswith("--", index):
            newline = sql.find("\n", index + 2)
            index = length if newline == -1 else newline + 1
            continue
        if sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            if end == -1:
                return ""
            index = end + 2
            continue
        break
    return sql[index:]


def _validate_select_query(query: str) -> str:
    query, error = _nonempty_text(query, "query")
    if error:
        return error
    stripped = _strip_sql_leading_comments(query)
    first = re.match(r"([A-Za-z_]+)", stripped)
    token = first.group(1).upper() if first else ""
    if token not in {"SELECT", "WITH"}:
        return "Error: only one read-only SELECT or WITH statement is allowed."
    return ""


def _normalize_query_value(value):
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"number_text": "NaN", "non_finite": True}
        if math.isinf(value):
            return {"number_text": "Infinity" if value > 0 else "-Infinity", "non_finite": True}
        return value
    if isinstance(value, str):
        if len(value) <= MAX_QUERY_TEXT_CHARS:
            return value
        return {"text": value[:MAX_QUERY_TEXT_CHARS], "total_chars": len(value), "truncated": True}
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        preview = data[:MAX_BINARY_PREVIEW_BYTES]
        return {"bytes_hex": preview.hex(), "total_bytes": len(data), "truncated": len(data) > len(preview)}
    return str(value)


def _value_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "data"
    if _is_plist_uid(value) or (isinstance(value, (int, float)) and not isinstance(value, bool)):
        return "number"
    if isinstance(value, bool):
        return "boolean"
    return "string"


def _normalize_search_value(value):
    if _is_plist_uid(value):
        return _uid_value(value), "number", False
    if hasattr(value, "isoformat"):
        text = value.isoformat()
        if len(text) <= MAX_SEARCH_VALUE_CHARS:
            return text, "string", False
        return text[:MAX_SEARCH_VALUE_CHARS], "string", True
    value_type = _value_type(value)
    if isinstance(value, str):
        if len(value) <= MAX_SEARCH_VALUE_CHARS:
            return value, value_type, False
        return value[:MAX_SEARCH_VALUE_CHARS], value_type, True
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        preview = data[:MAX_BINARY_PREVIEW_BYTES]
        return {"bytes_hex": preview.hex(), "total_bytes": len(data), "truncated": len(data) > len(preview)}, value_type, len(data) > len(preview)
    normalized = _normalize_query_value(value)
    if isinstance(normalized, dict):
        return normalized, value_type, bool(normalized.get("truncated", False))
    return normalized, value_type, False


def _json_pointer_escape(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _json_pointer(parts: list[object]) -> str:
    return "" if not parts else "/" + "/".join(_json_pointer_escape(part) for part in parts)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _like_literal(keyword: str) -> str:
    return "%" + keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _execute_readonly_sql(db_path: str, query: str, max_rows: int, max_limit: int) -> tuple[dict | None, str]:
    error = _validate_select_query(query)
    if error:
        return None, error
    parsed_rows, error = _strict_int_range(max_rows, "max_rows", 1, max_limit)
    if error:
        return None, error
    conn = None
    try:
        conn = _readonly_connection(db_path)
        cur = conn.execute(query)
        columns = [item[0] for item in cur.description] if cur.description else []
        fetched = cur.fetchmany(parsed_rows + 1)
        truncated = len(fetched) > parsed_rows
        rows = [[_normalize_query_value(value) for value in row] for row in fetched[:parsed_rows]]
        return {
            "columns": columns,
            "rows": rows,
            "returned_rows": len(rows),
            "truncated": truncated,
            "max_rows": parsed_rows,
        }, ""
    except sqlite3.DatabaseError as exc:
        return None, f"Error: SQLite query failed: {exc}"
    finally:
        if conn is not None:
            conn.close()


async def sqlite_query(db_path: str, query: str, max_rows: int = 100) -> str:
    result, error = _execute_readonly_sql(db_path, query, max_rows, MAX_QUERY_ROWS)
    if error:
        return error
    return _json_dumps({
        "columns": result["columns"],
        "rows": result["rows"],
        "returned_rows": result["returned_rows"],
        "truncated": result["truncated"],
    })


def _table_is_without_rowid(conn: sqlite3.Connection, table: str) -> bool:
    try:
        cur = conn.execute("PRAGMA table_list")
    except sqlite3.DatabaseError:
        return False
    names = [item[0] for item in (cur.description or [])]
    try:
        schema_index = names.index("schema")
        name_index = names.index("name")
        wr_index = names.index("wr")
    except ValueError:
        return False
    for row in cur.fetchall():
        if row[schema_index] == "main" and row[name_index] == table:
            return bool(row[wr_index])
    return False


def _rowid_expr(conn: sqlite3.Connection, table: str) -> str | None:
    cols = {str(row[1]).casefold() for row in conn.execute(f"PRAGMA table_xinfo({_quote_identifier(table)})") if row[1] is not None}
    for candidate in ("rowid", "_rowid_", "oid"):
        if candidate.casefold() in cols:
            continue
        try:
            conn.execute(f"SELECT {_quote_identifier(candidate)} FROM {_quote_identifier(table)} LIMIT 0")
            return _quote_identifier(candidate)
        except sqlite3.DatabaseError:
            continue
    return None


async def sqlite_search(db_path: str, keyword: str, max_results: int = 100) -> str:
    keyword, error = _nonempty_text(keyword, "keyword")
    if error:
        return error
    parsed_limit, error = _strict_int_range(max_results, "max_results", 1, MAX_SEARCH_RESULTS)
    if error:
        return error
    conn = None
    try:
        conn = _readonly_connection(db_path)
        results = []
        truncated = False
        pattern = _like_literal(keyword)
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for table in tables:
            table_q = _quote_identifier(table)
            info = conn.execute(f"PRAGMA table_info({table_q})").fetchall()
            pk_cols = [(row[5], row[1]) for row in info if row[5]]
            pk_cols.sort()
            without_rowid = _table_is_without_rowid(conn, table)
            rowid_expr = None if without_rowid else _rowid_expr(conn, table)
            for col in info:
                col_name = col[1]
                select_parts = []
                if rowid_expr:
                    select_parts.append(f"{rowid_expr} AS __p0_rowid")
                for _, pk in pk_cols:
                    select_parts.append(f"{_quote_identifier(pk)} AS {_quote_identifier('__p0_pk_' + pk)}")
                select_parts.append(f"{_quote_identifier(col_name)} AS __p0_value")
                sql = (
                    f"SELECT {', '.join(select_parts)} FROM {table_q} "
                    f"WHERE CAST({_quote_identifier(col_name)} AS TEXT) LIKE ? ESCAPE '\\' "
                    f"ORDER BY {rowid_expr or ', '.join(_quote_identifier(pk) for _, pk in pk_cols) or _quote_identifier(col_name)} "
                    f"LIMIT ?"
                )
                for row in conn.execute(sql, (pattern, parsed_limit + 1 - len(results))):
                    values = list(row)
                    pos = 0
                    rowid = None
                    if rowid_expr:
                        rowid = values[pos]
                        if isinstance(rowid, bool) or not isinstance(rowid, int):
                            rowid = None
                        pos += 1
                    primary_key = None
                    if pk_cols:
                        primary_key = {}
                        for _, pk in pk_cols:
                            primary_key[pk] = _normalize_query_value(values[pos])
                            pos += 1
                    raw_value = values[pos]
                    norm_value, value_type, value_truncated = _normalize_search_value(raw_value)
                    if len(results) >= parsed_limit:
                        truncated = True
                        break
                    results.append({
                        "table": table,
                        "column": col_name,
                        "rowid": rowid,
                        "primary_key": primary_key,
                        "value": norm_value,
                        "value_type": value_type,
                        "value_truncated": value_truncated,
                    })
                if truncated:
                    break
            if truncated:
                break
        return _json_dumps({"results": results, "returned_matches": len(results), "truncated": truncated})
    except sqlite3.DatabaseError as exc:
        return f"Error: SQLite search failed: {exc}"
    finally:
        if conn is not None:
            conn.close()


async def sqlite_export_query(db_path: str, query: str, output_path: str, max_rows: int = 10000) -> str:
    result, error = _execute_readonly_sql(db_path, query, max_rows, MAX_STRUCTURED_ROWS)
    if error:
        return error
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps({
        "columns": result["columns"],
        "rows": result["rows"],
        "max_rows": result["max_rows"],
    }), encoding="utf-8")
    return _json_dumps({"output_path": str(path.resolve())})


async def parse_plist(file_path: str) -> str:
    try:
        with _bounded_path(file_path).open("rb") as stream:
            value = plistlib.load(stream)
        state = _PreviewState()
        preview, truncated = _bounded_preview(value, state)
        preview = _ensure_represented(preview, "plist preview")
        return _json_dumps({"root_type": _root_type(value), "preview": preview, "truncated": truncated})
    except Exception as exc:
        return f"Error: plist parse failed: {exc}"


def _scalar_search_text(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return str(value)
    if isinstance(value, str):
        return value
    if _is_plist_uid(value):
        return str(_uid_value(value))
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return None


def _hit(path: str, match_in: str, value):
    normalized, value_type, truncated = _normalize_search_value(value)
    return {
        "path": path,
        "match_in": match_in,
        "value": normalized,
        "value_type": value_type,
        "value_truncated": truncated,
    }


def _bounded_add(results: list[dict], hit: dict, max_results: int) -> bool:
    if len(results) >= max_results:
        return False
    results.append(hit)
    return True


def _search_json_like(value, keyword_fold: str, max_results: int, path_parts: list[object] | None = None) -> tuple[list[dict], bool]:
    path_parts = path_parts or []
    results: list[dict] = []
    truncated = False

    def add(hit: dict) -> bool:
        nonlocal truncated
        if len(results) >= max_results:
            truncated = True
            return False
        results.append(hit)
        return True

    def walk(current, parts: list[object]) -> bool:
        if isinstance(current, dict):
            for key, child in current.items():
                key_text = str(key)
                child_parts = [*parts, key_text]
                if keyword_fold in key_text.casefold():
                    if not add(_hit(_json_pointer(child_parts), "key", key_text)):
                        return False
                if not walk(child, child_parts):
                    return False
            return True
        if isinstance(current, (list, tuple)):
            for index, child in enumerate(current):
                if not walk(child, [*parts, index]):
                    return False
            return True
        text = _scalar_search_text(current)
        if text is not None and keyword_fold in text.casefold():
            return add(_hit(_json_pointer(parts), "value", current))
        return True

    walk(value, path_parts)
    return results, truncated


async def plist_search(file_path: str, keyword: str, max_results: int = 200) -> str:
    keyword, error = _nonempty_text(keyword, "keyword")
    if error:
        return error
    parsed_limit, error = _strict_int_range(max_results, "max_results", 1, MAX_SEARCH_RESULTS)
    if error:
        return error
    try:
        with _bounded_path(file_path).open("rb") as stream:
            value = plistlib.load(stream)
        results, truncated = _search_json_like(value, keyword.casefold(), parsed_limit)
        return _json_dumps({"results": results, "returned_matches": len(results), "truncated": truncated})
    except Exception as exc:
        return f"Error: plist search failed: {exc}"


async def inspect_json(file_path: str) -> str:
    try:
        def reject_constant(value: str):
            raise ValueError(f"non-standard JSON constant: {value}")

        def reject_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        value = json.loads(
            _bounded_path(file_path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
            parse_float=_parse_finite_json_float,
        )
        state = _PreviewState()
        preview, truncated = _bounded_preview(value, state)
        preview = _ensure_represented(preview, "JSON preview")
        return _json_dumps({"root_type": _root_type(value), "preview": preview, "truncated": truncated})
    except Exception as exc:
        return f"Error: JSON parse failed: {exc}"


async def inspect_xml(file_path: str) -> str:
    try:
        root = _load_xml_safe(_bounded_path(file_path))
        node_count = sum(1 for _ in root.iter())
        state = _PreviewState()

        def convert(node, depth: int = 0):
            if not state.take_node():
                return _OMITTED
            tag = state.take_text(str(node.tag), MAX_KEY_CHARS)
            if tag is _OMITTED:
                return _OMITTED
            attrs = {}
            for index, (key, value) in enumerate(node.attrib.items()):
                if index >= MAX_CONTAINER_ITEMS:
                    state.mark_truncated()
                    break
                short = _shorten_key(key, state)
                if short is _OMITTED:
                    state.mark_truncated()
                    break
                short = _unique_budgeted_key(short, set(attrs), state)
                if short is _OMITTED:
                    state.mark_truncated()
                    break
                attr_value = state.take_text(str(value))
                if attr_value is _OMITTED:
                    state.mark_truncated()
                    break
                attrs[short] = attr_value
            text = state.take_text((node.text or "").strip())
            if text is _OMITTED:
                state.mark_truncated()
                text = ""
            children = []
            child_nodes = list(node)
            if depth >= MAX_PREVIEW_DEPTH and child_nodes:
                state.mark_truncated()
            else:
                for index, child in enumerate(child_nodes):
                    if index >= MAX_CONTAINER_ITEMS:
                        state.mark_truncated()
                        break
                    child_preview = convert(child, depth + 1)
                    if child_preview is _OMITTED:
                        state.mark_truncated()
                        break
                    children.append(child_preview)
            return {
                "tag": tag,
                "attributes": attrs,
                "text": text,
                "children": children,
            }

        root_preview = _ensure_represented(convert(root), "XML root preview")
        return _json_dumps({"root": root_preview, "node_count": node_count, "truncated": state.truncated})
    except Exception as exc:
        return f"Error: XML parse failed: {exc}"


async def inspect_csv(file_path: str, max_rows: int = 200) -> str:
    try:
        parsed_rows, error = _strict_int(max_rows, "max_rows")
        if error:
            return error
        if parsed_rows < 1 or parsed_rows > MAX_STRUCTURED_RESULTS:
            return f"Error: max_rows must be between 1 and {MAX_STRUCTURED_RESULTS}."
        data = _bounded_path(file_path).read_bytes()
        decode_lossy = False
        try:
            text = data.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            text = data.decode("utf-8-sig", errors="replace")
            decode_lossy = True
        if text == "":
            return _json_dumps({"rows": [], "delimiter": None, "encoding": "utf-8-sig", "decode_lossy": False, "truncated": False})
        sample = text[:4096]
        delimiter = None
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(io.StringIO(text, newline=""), dialect)
        state = _PreviewState()
        rows = []
        truncated = False
        for index, row in enumerate(reader):
            if index >= parsed_rows:
                truncated = True
                break
            out_row = []
            stop_preview = False
            if len(row) > MAX_CSV_COLUMNS:
                truncated = True
            for cell in row[:MAX_CSV_COLUMNS]:
                cell_preview = state.take_text(cell, MAX_STRING_CHARS)
                if cell_preview is _OMITTED:
                    truncated = True
                    stop_preview = True
                    break
                out_row.append(cell_preview)
            if out_row or not stop_preview:
                rows.append(out_row)
            if stop_preview:
                break
        return _json_dumps({
            "rows": rows,
            "delimiter": delimiter,
            "encoding": "utf-8-sig",
            "decode_lossy": decode_lossy,
            "truncated": truncated or state.truncated,
        })
    except Exception as exc:
        return f"Error: CSV parse failed: {exc}"


async def inspect_ini(file_path: str) -> str:
    try:
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.optionxform = str
        parser.read(_bounded_path(file_path), encoding="utf-8")
        defaults = dict(parser.defaults())
        sections = {}
        for section in parser.sections():
            raw = dict(parser._sections.get(section, {}))
            raw.pop("__name__", None)
            sections[section] = raw
        state = _PreviewState()
        defaults_preview, _ = _bounded_preview(defaults, state)
        sections_preview, _ = _bounded_preview(sections, state)
        defaults_preview = _ensure_represented(defaults_preview, "INI defaults preview")
        sections_preview = _ensure_represented(sections_preview, "INI sections preview")
        return _json_dumps({"defaults": defaults_preview, "sections": sections_preview, "truncated": state.truncated})
    except Exception as exc:
        return f"Error: INI parse failed: {exc}"


def _load_json_strict(path: Path):
    def reject_constant(value: str):
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
        parse_float=_parse_finite_json_float,
    )


def _reject_xml_declarations_with_expat(data: bytes) -> None:
    parser = expat.ParserCreate()

    def reject(*_args):
        raise ValueError("XML DTD/entity declarations are not supported.")

    parser.StartDoctypeDeclHandler = reject
    parser.EntityDeclHandler = reject
    parser.ExternalEntityRefHandler = lambda *_args: (_ for _ in ()).throw(
        ValueError("XML DTD/entity declarations are not supported.")
    )
    parser.NotationDeclHandler = reject
    parser.Parse(data, True)


def _load_xml_safe(path: Path):
    data = path.read_bytes()
    _reject_xml_declarations_with_expat(data)
    try:
        from defusedxml import ElementTree as SafeET  # type: ignore
    except ImportError:
        return ET.fromstring(data)
    return SafeET.fromstring(data)


def _xml_path_segment(raw_tag: object) -> str:
    return quote(str(raw_tag), safe="")


def _load_ini_raw(path: Path):
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    parser.read(path, encoding="utf-8")
    defaults = dict(parser.defaults())
    sections = {}
    for section in parser.sections():
        raw = dict(parser._sections.get(section, {}))
        raw.pop("__name__", None)
        sections[section] = raw
    return defaults, sections


def _search_xml(root, keyword_fold: str, max_results: int) -> tuple[list[dict], bool]:
    results: list[dict] = []
    truncated = False

    def add(hit: dict) -> bool:
        nonlocal truncated
        if len(results) >= max_results:
            truncated = True
            return False
        results.append(hit)
        return True

    def walk(node, path: str) -> bool:
        tag_text = str(node.tag)
        if keyword_fold in tag_text.casefold() and not add(_hit(path, "tag", tag_text)):
            return False
        for key, value in node.attrib.items():
            if keyword_fold in str(key).casefold() and not add(_hit(path, "attribute_name", str(key))):
                return False
            if keyword_fold in str(value).casefold() and not add(_hit(path, "attribute_value", str(value))):
                return False
        text = (node.text or "").strip()
        if text and keyword_fold in text.casefold() and not add(_hit(path, "text", text)):
            return False
        counts: dict[str, int] = {}
        for child in list(node):
            child_tag = str(child.tag)
            counts[child_tag] = counts.get(child_tag, 0) + 1
            child_path = f"{path}/{_xml_path_segment(child_tag)}[{counts[child_tag]}]"
            if not walk(child, child_path):
                return False
        return True

    walk(root, f"/{_xml_path_segment(root.tag)}[1]")
    return results, truncated


def _search_csv_rows(path: Path, keyword_fold: str, max_results: int, delimiter: str | None) -> tuple[list[dict], bool]:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        text = data.decode("utf-8-sig", errors="replace")
    if text == "":
        return [], False
    if delimiter == "\t":
        dialect = csv.excel_tab
    else:
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample)
            if dialect.delimiter not in {",", ";", "\t", "|"}:
                dialect = csv.excel
        except csv.Error:
            dialect = csv.excel
    results: list[dict] = []
    truncated = False
    reader = csv.reader(io.StringIO(text, newline=""), dialect)
    for row_index, row in enumerate(reader):
        for col_index, cell in enumerate(row):
            if keyword_fold in cell.casefold():
                if len(results) >= max_results:
                    truncated = True
                    return results, truncated
                results.append(_hit(f"/rows/{row_index}/{col_index}", "value", cell))
    return results, truncated


def _search_ini(defaults: dict, sections: dict, keyword_fold: str, max_results: int) -> tuple[list[dict], bool]:
    results: list[dict] = []
    truncated = False

    def add(hit: dict) -> bool:
        nonlocal truncated
        if len(results) >= max_results:
            truncated = True
            return False
        results.append(hit)
        return True

    for key, value in defaults.items():
        path = _json_pointer(["defaults", key])
        if keyword_fold in str(key).casefold() and not add(_hit(path, "key", str(key))):
            return results, truncated
        if keyword_fold in str(value).casefold() and not add(_hit(path, "value", str(value))):
            return results, truncated
    for section, values in sections.items():
        section_path = _json_pointer(["sections", section])
        if keyword_fold in str(section).casefold() and not add(_hit(section_path, "section", str(section))):
            return results, truncated
        for key, value in values.items():
            path = _json_pointer(["sections", section, key])
            if keyword_fold in str(key).casefold() and not add(_hit(path, "key", str(key))):
                return results, truncated
            if keyword_fold in str(value).casefold() and not add(_hit(path, "value", str(value))):
                return results, truncated
    return results, truncated


async def search_structured_file(file_path: str, keyword: str, max_results: int = 200) -> str:
    keyword, error = _nonempty_text(keyword, "keyword")
    if error:
        return error
    parsed_limit, error = _strict_int_range(max_results, "max_results", 1, MAX_SEARCH_RESULTS)
    if error:
        return error
    path = _bounded_path(file_path)
    try:
        suffix = path.suffix.lower()
        if suffix == ".plist":
            with path.open("rb") as stream:
                value = plistlib.load(stream)
            fmt = "plist"
            results, truncated = _search_json_like(value, keyword.casefold(), parsed_limit)
        elif suffix == ".json":
            value = _load_json_strict(path)
            fmt = "json"
            results, truncated = _search_json_like(value, keyword.casefold(), parsed_limit)
        elif suffix in {".xml"}:
            root = _load_xml_safe(path)
            fmt = "xml"
            results, truncated = _search_xml(root, keyword.casefold(), parsed_limit)
        elif suffix in {".csv", ".tsv"}:
            fmt = "csv"
            results, truncated = _search_csv_rows(path, keyword.casefold(), parsed_limit, "\t" if suffix == ".tsv" else None)
        elif suffix in {".ini", ".cfg", ".conf"}:
            defaults, sections = _load_ini_raw(path)
            fmt = "ini"
            results, truncated = _search_ini(defaults, sections, keyword.casefold(), parsed_limit)
        else:
            return "Error: unsupported structured file type."
        return _json_dumps({"format": fmt, "results": results, "returned_matches": len(results), "truncated": truncated})
    except Exception as exc:
        return f"Error: structured search failed: {exc}"
