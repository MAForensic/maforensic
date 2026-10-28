"""Unified structured file index for image and tree evidence."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .runtime import RuntimeContext

_FLS_RE = re.compile(r"^(?P<entry_type>\S+)\s+(?P<deleted>\*\s+)?(?P<inode>[0-9-]+)(?:\([^)]*\))?:\s*(?P<path>.*)$")
MAX_INDEX_ENTRIES = 1_000_000
MAX_INDEX_QUERY_RESULTS = 1_000
_MIME_TOP_LEVEL_TYPES = {
    "application", "audio", "font", "image", "message",
    "model", "multipart", "text", "video",
}


def indexed_entry_kind(entry: dict) -> str:
    file_type = str(entry.get("file_type", "") or "").strip().lower()
    if file_type == "directory" or file_type.startswith("d/"):
        return "directory"
    if file_type == "symlink" or file_type.startswith(("l/", "v/")):
        return "other"
    if file_type.startswith(("b/", "c/", "p/", "s/")):
        return "other"
    if file_type == "file" or file_type.startswith("r/"):
        return "file"
    if _is_explicit_mime_type(file_type):
        return "file"
    return "other"


def _is_explicit_mime_type(value: str) -> bool:
    top_level, separator, subtype = value.partition("/")
    return bool(separator and subtype and top_level in _MIME_TOP_LEVEL_TYPES)


def _iso(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts is not None else ""


async def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            await asyncio.sleep(0)
    return digest.hexdigest()


def _magic(path: Path) -> str:
    with path.open("rb") as stream:
        return stream.read(16).hex()


async def _tree_entry(root: Path, path: Path, source_id: str, compute_hash: bool) -> dict:
    stat = path.lstat()
    is_symlink = path.is_symlink()
    is_file = path.is_file() and not is_symlink
    mime, _ = mimetypes.guess_type(path.name)
    return {
        "file_id": hashlib.sha256(f"{source_id}:{path.relative_to(root)}".encode()).hexdigest()[:16],
        "source_evidence": source_id,
        "partition_id": "",
        "path": str(path.relative_to(root)),
        "name": path.name,
        "extension": path.suffix.lower(),
        "size": stat.st_size,
        "created_time": _iso(getattr(stat, "st_ctime", None)),
        "modified_time": _iso(stat.st_mtime),
        "accessed_time": _iso(stat.st_atime),
        "file_type": "symlink" if is_symlink else "directory" if path.is_dir() else (mime or "file"),
        "hash": await _sha256(path) if is_file and compute_hash else "",
        "magic_number": _magic(path) if is_file else "",
        "is_deleted": False,
        "is_hidden": path.name.startswith("."),
        "readable": not is_symlink and os.access(path, os.R_OK),
        "inode": str(stat.st_ino),
    }


async def _write_index(output: Path, entries: list[dict]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write("[\n")
            for index, entry in enumerate(entries):
                if index:
                    stream.write(",\n")
                json.dump(entry, stream, ensure_ascii=False)
                if index % 1000 == 0:
                    await asyncio.sleep(0)
            stream.write("\n]\n")
        temporary.replace(output)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


async def build_file_index(ctx: RuntimeContext, compute_hash: bool = False) -> str:
    source = ctx.active_source
    if source is None:
        return "Error: no active source."
    index_dir = ctx.run_dir / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    output = index_dir / f"{source.id}.json"
    if source.id in ctx.indexes and not compute_hash:
        metadata = ctx.index_metadata.get(source.id, {})
        return json.dumps({
            "entries": len(ctx.indexes[source.id]), "truncated": bool(metadata.get("truncated", False)),
        })
    entries: list[dict] = []
    if source.access_mode == "image":
        state = ctx.access_state(source)
        args = ["fls", "-r", "-m", "/", "-b", str(state["sector_size"]), "-o", str(state["start_sector"])]
        if compute_hash:
            args.insert(1, "-h")
        args.append(source.path)
        stderr_file = tempfile.TemporaryFile()
        proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=stderr_file)
        limit = min(MAX_INDEX_ENTRIES, max(1, int(ctx.config.get("index", {}).get("max_files", MAX_INDEX_ENTRIES))))
        truncated = False
        try:
            async with asyncio.timeout(300):
                async for raw_line in proc.stdout:
                    line = raw_line.decode(errors="replace").rstrip("\r\n")
                    fields = line.split("|")
                    if len(fields) != 11:
                        continue
                    digest, path, inode, mode, _uid, _gid, size, atime, mtime, ctime, crtime = fields
                    path = path.lstrip("/")
                    name = Path(path).name
                    entries.append({
                        "file_id": hashlib.sha256(f"{source.id}:{state['partition_id']}:{inode}:{path}".encode()).hexdigest()[:16],
                        "source_evidence": source.id,
                        "partition_id": state["partition_id"],
                        "sector_size": state["sector_size"],
                        "start_sector": state["start_sector"],
                        "byte_offset": state["byte_offset"],
                        "path": path,
                        "name": name,
                        "extension": Path(name).suffix.lower(),
                        "size": int(size) if size.isdigit() else None,
                        "created_time": _iso(float(crtime)) if crtime not in {"", "0"} else "",
                        "modified_time": _iso(float(mtime)) if mtime not in {"", "0"} else "",
                        "accessed_time": _iso(float(atime)) if atime not in {"", "0"} else "",
                        "file_type": mode,
                        "hash": digest if digest != "0" * 32 else "",
                        "magic_number": "",
                        "is_deleted": "(deleted)" in path.lower(),
                        "is_hidden": name.startswith("."), "readable": True,
                        "inode": inode,
                    })
                    if len(entries) >= limit:
                        truncated = True
                        proc.kill()
                        break
                    if len(entries) % 1000 == 0:
                        await asyncio.sleep(0)
                await proc.wait()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stderr_file.close()
            return "Error: fls index timed out."
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            stderr_file.close()
            raise
        if proc.returncode and not truncated:
            stderr_file.seek(0)
            stderr = stderr_file.read(32000)
            stderr_file.close()
            return f"Error: fls index failed: {stderr.decode(errors='replace')}"
        stderr_file.close()
    else:
        root = ctx.source_root
        if not root.is_dir():
            return f"Error: evidence root is not a directory: {root}"
        limit = min(MAX_INDEX_ENTRIES, max(1, int(ctx.config.get("index", {}).get("max_files", MAX_INDEX_ENTRIES))))
        for path in root.rglob("*"):
            try:
                entries.append(await _tree_entry(root, path, source.id, compute_hash))
            except OSError:
                continue
            if len(entries) >= limit:
                break
            if len(entries) % 1000 == 0:
                await asyncio.sleep(0)
    is_truncated = truncated if source.access_mode == "image" else len(entries) >= limit
    metadata = {
        "partition_id": ctx.access_state(source).get("partition_id", ""),
        "truncated": is_truncated,
        "compute_hash": compute_hash,
    }
    await _write_index(output, entries)
    ctx.cache_active_index(source, entries, metadata)
    ctx.indexed_sources.add(source.id)
    return json.dumps({
        "entries": len(entries), "truncated": is_truncated,
    })


async def query_file_index(
    ctx: RuntimeContext, path_pattern: str = "*", name_pattern: str = "*",
    extension: str = "", file_type: str = "", min_size: int = 0,
    max_size: int = -1, is_deleted: bool | None = None, time_field: str = "",
    time_after: str = "", time_before: str = "", max_results: int = 200,
) -> str:
    source = ctx.active_source
    if source is None:
        return "Error: no active source."
    if source.id not in ctx.indexes:
        result = await build_file_index(ctx)
        if result.startswith("Error:"):
            return result
    extension = extension.lower()
    max_results = min(MAX_INDEX_QUERY_RESULTS, max(0, max_results))
    hits = []
    total_matches = 0
    for item in ctx.indexes[source.id]:
        size = item.get("size")
        if not fnmatch.fnmatch(str(item.get("path", "")).lower(), path_pattern.lower()):
            continue
        if not fnmatch.fnmatch(str(item.get("name", "")).lower(), name_pattern.lower()):
            continue
        if extension and item.get("extension", "").lower() != extension:
            continue
        if file_type and file_type.lower() not in str(item.get("file_type", "")).lower():
            continue
        if size is not None and (size < min_size or (max_size >= 0 and size > max_size)):
            continue
        if is_deleted is not None and item.get("is_deleted") is not is_deleted:
            continue
        if time_field:
            field = {
                "created": "created_time", "modified": "modified_time", "accessed": "accessed_time"
            }.get(time_field, time_field)
            value = str(item.get(field, ""))
            if not value or (time_after and value < time_after) or (time_before and value > time_before):
                continue
        total_matches += 1
        if len(hits) < max_results:
            hits.append(_public_index_result(item))
    metadata = ctx.index_metadata.get(source.id, {})
    truncated = bool(metadata.get("truncated", False)) or total_matches > len(hits)
    return json.dumps({"matches": len(hits), "results": hits, "truncated": truncated}, ensure_ascii=False, indent=2)


def _public_index_result(item: dict) -> dict:
    return {
        "source_id": str(item.get("source_evidence", "")),
        "file_id": str(item.get("file_id", "")),
        "name": str(item.get("name", "")),
        "path": str(item.get("path", "")),
        "kind": indexed_entry_kind(item),
    }
