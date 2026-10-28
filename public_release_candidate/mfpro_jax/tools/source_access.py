"""Generic evidence-source probing and image access primitives."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from ..index import indexed_entry_kind
from ..runtime import RuntimeContext
from . import readers

INDEXED_BOUNDED_EXPORT_LIMIT = 1024**3


async def _run(args: list[str], timeout: float = 60, max_chars: int = 32000) -> str:
    proc = None
    try:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            proc = await asyncio.create_subprocess_exec(*args, stdout=stdout_file, stderr=stderr_file)
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(max_chars + 1)
            stderr = stderr_file.read(max_chars + 1)
    except FileNotFoundError:
        return f"Error: external dependency not found: {args[0]}"
    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return f"Error: command timed out after {timeout}s: {args[0]}"
    except asyncio.CancelledError:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise
    truncated = len(stdout) > max_chars
    text = stdout[:max_chars].decode("utf-8", errors="replace")
    if proc.returncode:
        return f"Error: command failed rc={proc.returncode}: {stderr.decode('utf-8', errors='replace')}\n{text}"[:max_chars]
    return text + ("\n[command output truncated]" if truncated else "")


def _tsk_partition_args(ctx: RuntimeContext) -> list[str]:
    state = ctx.access_state()
    return ["-b", str(state.get("sector_size", 512)), "-o", str(state.get("start_sector", 0))]


def _parse_mmls(text: str) -> list[dict]:
    sector_match = re.search(r"Units are in\s+(\d+)-byte sectors", text, re.IGNORECASE)
    sector_size = int(sector_match.group(1)) if sector_match else 512
    partitions = []
    for line in text.splitlines():
        match = re.match(
            r"^\s*(?P<slot>\d+):\s+(?P<meta>\S+)\s+(?P<start>\d+)\s+(?P<end>\d+)\s+(?P<length>\d+)\s*(?P<description>.*)$",
            line,
        )
        if not match:
            continue
        start = int(match.group("start"))
        partitions.append({
            "partition_id": f"part-{int(match.group('slot')):03d}",
            "slot": int(match.group("slot")),
            "metadata_slot": match.group("meta"),
            "start_sector": start,
            "end_sector": int(match.group("end")),
            "length_sectors": int(match.group("length")),
            "sector_size": sector_size,
            "byte_offset": start * sector_size,
            "description": match.group("description").strip(),
        })
    return partitions


async def probe_source(ctx: RuntimeContext) -> str:
    source = ctx.active_source
    if source is None:
        return "Error: no active source."
    path = Path(source.path)
    if not path.exists():
        status = "missing"
    elif not os.access(path, os.R_OK):
        status = "unreadable"
    else:
        status = "ready"
    result = {
        "access_mode": source.access_mode,
        "status": status,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


async def list_partitions(ctx: RuntimeContext) -> str:
    source = ctx.active_source
    if source is None:
        return "Error: no active source."
    if source.access_mode != "image":
        return "No partitions: active source is not an image."
    raw = await _run(["mmls", source.path], max_chars=256000)
    if raw.startswith("Error:"):
        return raw
    partitions = _parse_mmls(raw)
    source.meta["partitions"] = partitions
    return json.dumps({
        "partitions": [
            {
                "partition_id": item["partition_id"],
                "description": item.get("description", ""),
            }
            for item in partitions
        ],
    }, ensure_ascii=False, indent=2)


async def select_partition(ctx: RuntimeContext, partition_id: str) -> str:
    source = ctx.active_source
    if source is None:
        return "Error: no active source."
    partitions = source.meta.get("partitions")
    if not partitions:
        listed = await list_partitions(ctx)
        if listed.startswith("Error:"):
            return listed
        partitions = source.meta.get("partitions", [])
    selected = next((item for item in partitions if item["partition_id"] == partition_id), None)
    if selected is None:
        return f"Error: unknown partition_id: {partition_id}"
    ctx.source_access[source.id] = {
        "partition_id": selected["partition_id"],
        "sector_size": selected["sector_size"],
        "start_sector": selected["start_sector"],
        "byte_offset": selected["byte_offset"],
    }
    ctx.activate_cached_index(source)
    return json.dumps({"partition_id": selected["partition_id"]}, ensure_ascii=False, indent=2)


async def filesystem_info(ctx: RuntimeContext) -> str:
    source = ctx.active_source
    if source is None:
        return "Error: no active source."
    if source.access_mode != "image":
        return json.dumps({"filesystem_type": "host_tree"}, ensure_ascii=False, indent=2)
    raw = await _run(["fsstat", *_tsk_partition_args(ctx), source.path])
    if raw.startswith("Error:"):
        return raw
    result: dict[str, object] = {"filesystem_type": _parse_filesystem_type(raw)}
    geometry = _parse_filesystem_geometry(raw)
    if geometry:
        result["geometry"] = geometry
    return json.dumps(result, ensure_ascii=False, indent=2)


def _parse_filesystem_type(text: str) -> str:
    for line in text.splitlines():
        match = re.match(r"^\s*File\s+System\s+Type\s*:\s*(?P<value>.+?)\s*$", line, re.IGNORECASE)
        if match:
            value = match.group("value").strip()
            return value or "unknown"
    return "unknown"


def _parse_filesystem_geometry(text: str) -> dict[str, int]:
    bytes_per_sector = _parse_fsstat_positive_int(text, "Sector Size")
    bytes_per_cluster = _parse_fsstat_positive_int(text, "Cluster Size")
    geometry: dict[str, int] = {}
    if bytes_per_sector is not None:
        geometry["bytes_per_sector"] = bytes_per_sector
    if bytes_per_cluster is not None:
        geometry["bytes_per_cluster"] = bytes_per_cluster
    if (
        bytes_per_sector is not None
        and bytes_per_cluster is not None
        and bytes_per_cluster % bytes_per_sector == 0
    ):
        geometry["sectors_per_cluster"] = bytes_per_cluster // bytes_per_sector
    return geometry


def _parse_fsstat_positive_int(text: str, label: str) -> int | None:
    pattern = re.compile(rf"^\s*{re.escape(label)}\s*:\s*(?P<value>\d+)\s*$", re.IGNORECASE)
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        value = int(match.group("value"))
        if value > 0:
            return value
    return None


async def list_directory(
    ctx: RuntimeContext, path: str = ".", inode: str = "", recursive: bool = False,
) -> str:
    source = ctx.active_source
    if source is None:
        return "Error: no active source."
    if source.access_mode == "image":
        if path not in {"", ".", "/"} and not inode:
            return "Error: image directory listing requires an inode; use the file index to resolve a path."
        args = ["fls", *_tsk_partition_args(ctx)]
        if recursive:
            args.append("-r")
        args.append(source.path)
        if inode:
            args.append(inode)
        raw = await _run(args)
        if raw.startswith("Error:"):
            return raw
        entries = _parse_fls_directory(raw)
        truncated = "[command output truncated]" in raw
        return json.dumps({"entries": entries, "truncated": truncated}, ensure_ascii=False, indent=2)
    root = ctx.source_root.resolve()
    target = root if path in {"", ".", "/"} else (root / path).resolve()
    if target != root and root not in target.parents:
        return "Error: path escapes evidence root."
    if not target.is_dir():
        return f"Error: not a directory: {path}"
    entries = []
    truncated = False
    iterator = target.rglob("*") if recursive else target.iterdir()
    for item in iterator:
        try:
            entries.append({
                "name": item.name,
                "kind": _directory_entry_kind(item),
                "path": str(item.relative_to(root)),
            })
        except OSError:
            continue
        if len(entries) >= 1000:
            truncated = True
            break
        if len(entries) % 100 == 0:
            await asyncio.sleep(0)
    return json.dumps({"entries": entries, "truncated": truncated}, ensure_ascii=False, indent=2)


def _directory_entry_kind(path: Path) -> str:
    if path.is_symlink():
        return "other"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def _parse_fls_directory(text: str) -> list[dict]:
    entries: list[dict] = []
    pattern = re.compile(
        r"^\s*(?:\+\s*)*(?P<mode>[A-Za-z]/[A-Za-z])\s+"
        r"(?P<deleted>\*\s+)?(?P<inode>[0-9-]+)(?:\([^)]*\))?:\s*(?P<path>.+?)\s*$"
    )
    for line in text.splitlines():
        if not line.strip() or line.startswith("[command output truncated]"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        raw_path = match.group("path").strip()
        mode = match.group("mode").lower()
        kind = _fls_entry_kind(mode)
        entries.append({
            "name": Path(raw_path.rstrip("/")).name or raw_path,
            "kind": kind,
            "path": raw_path,
            "inode": match.group("inode"),
        })
    return entries


def _fls_entry_kind(mode: str) -> str:
    prefix = mode.split("/", 1)[0]
    if prefix == "d":
        return "directory"
    if prefix == "r":
        return "file"
    return "other"


async def extract_image_file(ctx: RuntimeContext, inode: str, output_name: str = "") -> str:
    source = ctx.active_source
    if source is None or source.access_mode != "image":
        return "Error: active source is not an image."
    output_dir = ctx.run_dir / "extracted" / source.id
    output_dir.mkdir(parents=True, exist_ok=True)
    name = output_name or inode.replace("-", "_")
    output = output_dir / Path(name).name
    result = await _export_image_inode(ctx, inode, output, int(ctx.config.get("io", {}).get("max_export_bytes", 1024**3)))
    if result.startswith("Error:"):
        return result
    data = json.loads(result)
    output_path = data.get("output_path")
    if not output_path:
        return "Error: image extraction did not produce an output path."
    return json.dumps({"output_path": str(Path(str(output_path)).resolve())}, ensure_ascii=False, indent=2)


async def _export_image_inode(
    ctx: RuntimeContext, inode: str, output: Path, max_bytes: int, allow_truncated: bool = False,
) -> str:
    source = ctx.active_source
    if source is None or source.access_mode != "image":
        return "Error: active source is not an image."
    proc = await asyncio.create_subprocess_exec(
        "icat", *_tsk_partition_args(ctx), source.path, inode,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    written = 0
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            while True:
                chunk = await proc.stdout.read(1024 * 1024)
                if not chunk:
                    break
                remaining = max_bytes - written
                if remaining <= 0:
                    proc.kill()
                    await proc.wait()
                    if not allow_truncated:
                        output.unlink(missing_ok=True)
                        return "Error: indexed member exceeds export limit."
                    return json.dumps({"output_path": str(output), "size": max_bytes, "truncated": True})
                if len(chunk) > remaining:
                    stream.write(chunk[:remaining])
                    written += remaining
                    proc.kill()
                    await proc.wait()
                    if not allow_truncated:
                        output.unlink(missing_ok=True)
                        return "Error: indexed member exceeds export limit."
                    return json.dumps({"output_path": str(output), "size": written, "truncated": True})
                stream.write(chunk)
                written += len(chunk)
                await asyncio.sleep(0)
        stderr = await proc.stderr.read(1024 * 1024)
        await proc.wait()
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        output.unlink(missing_ok=True)
        raise
    if proc.returncode:
        output.unlink(missing_ok=True)
        return f"Error: icat failed rc={proc.returncode}: {stderr.decode(errors='replace')}"
    return json.dumps({"output_path": str(output), "size": written, "truncated": False})


def _indexed_entry(ctx: RuntimeContext, source_id: str, file_id: str) -> tuple[object | None, dict | None]:
    source = ctx.case.get_source(source_id)
    entries = ctx.indexes.get(source_id, [])
    entry = next((item for item in entries if item.get("file_id") == file_id), None)
    if entry is None:
        cached = ctx.index_cache.get(source_id, {})
        entry = next((item for item in cached.get("entries", []) if item.get("file_id") == file_id), None)
    return source, entry


async def get_indexed_file_metadata(ctx: RuntimeContext, source_id: str, file_id: str) -> str:
    source, entry = _indexed_entry(ctx, source_id, file_id)
    if source is None:
        return f"Error: unknown source_id: {source_id}"
    if entry is None:
        return f"Error: unknown file_id for source: {file_id}"
    result = {
        "name": str(entry.get("name", "")),
        "evidence_path": str(entry.get("path", "")),
        "kind": indexed_entry_kind(entry),
        "size": entry.get("size") if entry.get("size") is not None else None,
        "inode": str(entry["inode"]) if entry.get("inode") is not None else None,
        "is_deleted": bool(entry.get("is_deleted", False)),
        "is_hidden": bool(entry.get("is_hidden", False)),
        "timestamps": {
            "created": entry.get("created_time") or None,
            "modified": entry.get("modified_time") or None,
            "accessed": entry.get("accessed_time") or None,
        },
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


async def export_indexed_file(
    ctx: RuntimeContext, source_id: str, file_id: str, output_name: str = "", max_bytes: int = 1073741824,
) -> str:
    max_bytes = min(int(ctx.config.get("io", {}).get("max_export_bytes", 1024**3)), max(0, max_bytes))
    source, entry = _indexed_entry(ctx, source_id, file_id)
    if source is None or entry is None:
        return "Error: unknown source_id or file_id."
    ctx.set_active_source(source)
    _apply_entry_partition(ctx, entry)
    output_dir = ctx.run_dir / "indexed_exports" / source.id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / Path(output_name or entry["name"] or file_id).name
    if source.access_mode == "image":
        result = await _export_image_inode(ctx, str(entry["inode"]), output, max_bytes)
    else:
        root = ctx.evidence_roots[source.id].resolve()
        source_path = (root / entry["path"]).resolve()
        if root not in source_path.parents or not source_path.is_file() or source_path.stat().st_size > max_bytes:
            return "Error: indexed member is unreadable or exceeds export limit."
        await _copy_file_bounded(source_path, output, max_bytes)
        result = json.dumps({"output_path": str(output), "size": output.stat().st_size})
    if result.startswith("Error:"):
        return result
    value = json.loads(result)
    output_path = value.get("output_path")
    if not output_path:
        return "Error: export did not produce an output path."
    return json.dumps({"output_path": str(Path(str(output_path)).resolve())}, ensure_ascii=False, indent=2)


def _apply_entry_partition(ctx: RuntimeContext, entry: dict) -> None:
    if not entry.get("partition_id"):
        return
    ctx.source_access[entry["source_evidence"]] = {
        "partition_id": entry["partition_id"],
        "sector_size": int(entry.get("sector_size") or 512),
        "start_sector": int(entry.get("start_sector") or 0),
        "byte_offset": int(entry.get("byte_offset") or 0),
    }


async def _with_indexed_export(ctx: RuntimeContext, source_id: str, file_id: str, limit: int, operation) -> str:
    source, entry = _indexed_entry(ctx, source_id, file_id)
    if source is None or entry is None:
        return "Error: unknown source_id or file_id."
    ctx.set_active_source(source)
    _apply_entry_partition(ctx, entry)
    temporary = source.access_mode == "image"
    if temporary:
        output_dir = ctx.run_dir / "indexed_temporary" / source.id
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{file_id}.tmp"
        result = await _export_image_inode(ctx, str(entry["inode"]), output, limit, allow_truncated=True)
        if result.startswith("Error:"):
            return result
        export_data = json.loads(result)
    else:
        root = ctx.evidence_roots[source.id].resolve()
        candidate = root / entry["path"]
        output = candidate.resolve()
        if candidate.is_symlink() or not output.is_file() or (output != root and root not in output.parents):
            return "Error: indexed member is outside the evidence root or unreadable."
        export_data = {"truncated": output.stat().st_size > limit if output.is_file() else False}
    try:
        result = await operation(str(output))
        try:
            value = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
        if isinstance(value, dict):
            value.update({
                "source_id": source_id, "partition_id": entry.get("partition_id", ""),
                "file_id": file_id, "access_mode": "indexed_member",
            })
            if export_data.get("truncated"):
                value["truncated"] = True
                value["indexed_member_size"] = entry.get("size")
            return json.dumps(value, ensure_ascii=False, indent=2)
        return result
    finally:
        if temporary:
            output.unlink(missing_ok=True)


async def search_indexed_file(
    ctx: RuntimeContext, source_id: str, file_id: str, pattern: str,
    regex: bool = False, context_chars: int = 80, max_results: int = 100,
) -> str:
    parsed_context, error = readers._bounded_int(context_chars, "context_chars", 0, readers.MAX_CONTEXT_CHARS)
    if error:
        return error
    parsed_max_results, error = readers._bounded_int(max_results, "max_results", 1, readers.MAX_SEARCH_RESULTS)
    if error:
        return error
    _, error = readers._compile_search_pattern(pattern, regex)
    if error:
        return error
    source, entry = _indexed_entry(ctx, source_id, file_id)
    if source is None or entry is None:
        return "Error: unknown source_id or file_id."
    ctx.set_active_source(source)
    _apply_entry_partition(ctx, entry)
    source_size = entry.get("size") if isinstance(entry.get("size"), int) else None
    temporary = source.access_mode == "image"
    export_truncated = False
    if temporary:
        output_dir = ctx.run_dir / "indexed_temporary" / source.id
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{file_id}.tmp"
    else:
        root = ctx.evidence_roots[source.id].resolve()
        candidate = root / entry["path"]
        output = candidate.resolve()
        if candidate.is_symlink() or not output.is_file() or (output != root and root not in output.parents):
            return "Error: indexed member is outside the evidence root or unreadable."
        source_size = output.stat().st_size
    try:
        if temporary:
            try:
                export_result = await _export_image_inode(
                    ctx, str(entry["inode"]), output, readers.SEARCH_FILE_MAX_BYTES, allow_truncated=True,
                )
                if export_result.startswith("Error:"):
                    return "Error: indexed member export failed."
                export_data = json.loads(export_result)
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
                return "Error: indexed member export failed."
            export_truncated = bool(export_data.get("truncated", False))
        try:
            result, error = readers._search_file_core(
                output, pattern, regex, parsed_context, parsed_max_results, readers.SEARCH_FILE_MAX_BYTES,
            )
        except (OSError, ValueError, RuntimeError):
            return "Error: indexed member search failed."
        if error:
            return error
        if source_size is not None:
            result["file_size"] = source_size
            if source_size > result["scanned_bytes"]:
                result["scan_complete"] = False
        if export_truncated:
            result["scan_complete"] = False
        return readers._json_dumps(result)
    finally:
        if temporary:
            output.unlink(missing_ok=True)


async def read_indexed_text(
    ctx: RuntimeContext, source_id: str, file_id: str,
    offset: int = 0, max_bytes: int = 16000, encoding: str = "auto",
) -> str:
    return await _with_indexed_bounded_read(
        ctx, source_id, file_id, offset, max_bytes, "max_bytes",
        lambda path: readers.read_text(path, offset, max_bytes, encoding),
    )


async def read_indexed_binary_range(
    ctx: RuntimeContext, source_id: str, file_id: str,
    offset: int = 0, length: int = 2048,
) -> str:
    return await _with_indexed_bounded_read(
        ctx, source_id, file_id, offset, length, "length",
        lambda path: readers.read_binary_range(path, offset, length),
    )


async def _with_indexed_bounded_read(
    ctx: RuntimeContext, source_id: str, file_id: str,
    offset: int, count: int, count_name: str, operation,
) -> str:
    parsed_offset, parsed_count, error = readers._bounded_range(offset, count, count_name)
    if error:
        return error
    source, entry = _indexed_entry(ctx, source_id, file_id)
    if source is None or entry is None:
        return "Error: unknown source_id or file_id."
    ctx.set_active_source(source)
    _apply_entry_partition(ctx, entry)
    source_size = entry.get("size") if isinstance(entry.get("size"), int) else None
    temporary = source.access_mode == "image"
    export_truncated = False
    if temporary:
        output_dir = ctx.run_dir / "indexed_temporary" / source.id
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{file_id}.tmp"
        limit = parsed_offset + parsed_count + 1
        if limit > INDEXED_BOUNDED_EXPORT_LIMIT:
            return "Error: requested indexed range exceeds export limit."
        result = await _export_image_inode(ctx, str(entry["inode"]), output, limit, allow_truncated=True)
        if result.startswith("Error:"):
            return result
        export_data = json.loads(result)
        export_truncated = bool(export_data.get("truncated", False))
    else:
        root = ctx.evidence_roots[source.id].resolve()
        candidate = root / entry["path"]
        output = candidate.resolve()
        if candidate.is_symlink() or not output.is_file() or (output != root and root not in output.parents):
            return "Error: indexed member is outside the evidence root or unreadable."
        source_size = output.stat().st_size
    try:
        result = await operation(str(output))
        if result.startswith("Error:"):
            return result
        try:
            value = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
        if isinstance(value, dict) and {"bytes_read", "offset", "truncated", "next_offset"} <= set(value):
            bytes_read = int(value.get("bytes_read") or 0)
            end = parsed_offset + bytes_read
            truncated = bool(export_truncated)
            if source_size is not None:
                truncated = source_size > end
            elif temporary and output.exists():
                truncated = truncated or output.stat().st_size > end
            value["offset"] = parsed_offset
            value["truncated"] = truncated
            value["next_offset"] = end if truncated else None
            return json.dumps(value, ensure_ascii=False, indent=2)
        return result
    finally:
        if temporary:
            output.unlink(missing_ok=True)


async def export_file(ctx: RuntimeContext, file_path: str, output_name: str = "", max_bytes: int = 1073741824) -> str:
    source = ctx.active_source
    if source is None or source.access_mode == "image":
        return "Error: active source is not tree evidence."
    max_bytes = min(int(ctx.config.get("io", {}).get("max_export_bytes", 1024**3)), max(0, max_bytes))
    path = Path(file_path).resolve()
    root = ctx.source_root.resolve()
    if not path.is_file() or path.is_symlink() or (path != root and root not in path.parents):
        return "Error: file is outside the active evidence root or unreadable."
    if path.stat().st_size > max_bytes:
        return "Error: file exceeds export limit."
    output_dir = ctx.run_dir / "exported" / source.id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / Path(output_name or path.name).name
    await _copy_file_bounded(path, output, max_bytes)
    shutil.copystat(path, output)
    return json.dumps({"output_path": str(output.resolve())}, ensure_ascii=False, indent=2)


async def _copy_file_bounded(path: Path, output: Path, max_bytes: int) -> None:
    written = 0
    with path.open("rb") as source_stream, output.open("wb") as output_stream:
        while chunk := source_stream.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                output.unlink(missing_ok=True)
                raise ValueError("file exceeds export limit")
            output_stream.write(chunk)
            await asyncio.sleep(0)
