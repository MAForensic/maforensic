"""Generic archive listing and bounded extraction."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tarfile
import tempfile
import zipfile

_ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".7z", ".rar"}
_EXTERNAL_TIMEOUT_SEC = 300
MAX_ARCHIVE_CONTAINER_BYTES = 16 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
MAX_EXPANDED_BYTES = 10 * 1024 * 1024 * 1024


def _archive_path(archive_path: str) -> Path:
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(archive_path)
    if path.stat().st_size > MAX_ARCHIVE_CONTAINER_BYTES:
        raise ValueError("archive container exceeds configured limit")
    return path


async def _run_external(
    args: list[str], timeout: float = _EXTERNAL_TIMEOUT_SEC, max_output_bytes: int = 64 * 1024 * 1024,
) -> tuple[int, bytes, bytes, str]:
    stdout_file = tempfile.TemporaryFile()
    stderr_file = tempfile.TemporaryFile()
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(*args, stdout=stdout_file, stderr=stderr_file)
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(max_output_bytes + 1)
        stderr = stderr_file.read(1024 * 1024 + 1)
        if len(stdout) > max_output_bytes or len(stderr) > 1024 * 1024:
            return proc.returncode, stdout[:max_output_bytes], stderr[:1024 * 1024], "Error: archive dependency output exceeded limit."
        return proc.returncode, stdout, stderr, ""
    except FileNotFoundError:
        return -1, b"", b"", f"Error: archive dependency not found: {args[0]}"
    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return -1, b"", b"", f"Error: archive dependency timed out after {timeout}s."
    except asyncio.CancelledError:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise
    finally:
        stdout_file.close()
        stderr_file.close()


def _safe_target(root: Path, name: str) -> Path | None:
    if name.startswith(("/", "\\")) or ".." in Path(name).parts:
        return None
    target = (root / name).resolve()
    resolved = root.resolve()
    return target if target == resolved or resolved in target.parents else None


async def _copy_bounded(src, dst, limit: int) -> int:
    total = 0
    while True:
        chunk = src.read(min(1024 * 1024, limit + 1 - total))
        if not chunk:
            return total
        dst.write(chunk)
        total += len(chunk)
        if total > limit:
            return total
        await asyncio.sleep(0)


def _materialized_output(path: Path) -> str:
    return json.dumps({"output_path": str(path.resolve())}, ensure_ascii=False, indent=2)


async def list_archive(archive_path: str, max_members: int = 10000) -> str:
    try:
        max_members = min(MAX_ARCHIVE_MEMBERS, max(0, max_members))
        path = _archive_path(archive_path)
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                members = [_public_archive_entry(item.filename, item.file_size, item.is_dir()) for item in infos[:max_members]]
                truncated = len(infos) > max_members
        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                members = []
                truncated = False
                for item in archive:
                    if len(members) >= max_members:
                        truncated = True
                        break
                    members.append(_public_archive_entry(item.name, item.size if item.isfile() else None, item.isdir(), other=not (item.isfile() or item.isdir())))
                    await asyncio.sleep(0)
        elif path.suffix.lower() == ".gz":
            with path.open("rb") as stream:
                stream.seek(-4, 2)
                size = struct.unpack("<I", stream.read(4))[0]
            members = [_public_archive_entry(path.stem, size, False)]
            truncated = False
        elif path.suffix.lower() in {".7z", ".rar"} and shutil.which("7z"):
            entries, error, truncated = await _list_7z_entries(path, max_members)
            if error:
                return error
            members = [
                _public_archive_entry(
                    item["path"], item["size"], bool(item["is_dir"]), other=bool(item["is_link"])
                )
                for item in entries
            ]
        else:
            return "Error: unsupported or invalid archive."
        return json.dumps({"entries": members, "truncated": truncated}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: archive list failed: {exc}"


def _public_archive_entry(member: str, size: int | None, is_dir: bool, other: bool = False) -> dict:
    return {
        "member": member,
        "kind": "other" if other else "directory" if is_dir else "file",
        "size": None if is_dir else size,
    }


async def extract_archive_member(archive_path: str, member: str, output_dir: str, max_expanded_bytes: int = 1073741824) -> str:
    max_expanded_bytes = min(MAX_EXPANDED_BYTES, max(0, max_expanded_bytes))
    try:
        path = _archive_path(archive_path)
    except Exception as exc:
        return f"Error: archive member extraction failed: {exc}"
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = _safe_target(root, member)
    if target is None:
        return "Error: unsafe archive member path."
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo(member)
                if info.file_size > max_expanded_bytes:
                    return "Error: archive member exceeds expansion limit."
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, target.open("wb") as dst:
                    total = await _copy_bounded(src, dst, max_expanded_bytes)
                if total > max_expanded_bytes:
                    target.unlink()
                    return "Error: archive member exceeds expansion limit."
        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                info = archive.getmember(member)
                if not info.isfile() or info.size > max_expanded_bytes:
                    return "Error: archive member is not a bounded regular file."
                target.parent.mkdir(parents=True, exist_ok=True)
                src = archive.extractfile(info)
                if src is None:
                    return "Error: archive member is unreadable."
                with src, target.open("wb") as dst:
                    total = await _copy_bounded(src, dst, max_expanded_bytes)
                if total > max_expanded_bytes:
                    target.unlink()
                    return "Error: archive member exceeds expansion limit."
        elif path.suffix.lower() == ".gz" and member in {path.stem, path.name}:
            with gzip.open(path, "rb") as src, target.open("wb") as dst:
                total = await _copy_bounded(src, dst, max_expanded_bytes)
            if total > max_expanded_bytes:
                target.unlink()
                return "Error: archive member exceeds expansion limit."
        else:
            return "Error: member extraction supports ZIP and tar-family archives."
        return _materialized_output(target)
    except Exception as exc:
        return f"Error: archive member extraction failed: {exc}"


async def _expand_archive_detailed(
    archive_path: str, output_dir: str, max_members: int = 100000,
    max_expanded_bytes: int = 10737418240,
) -> dict | str:
    max_members = min(MAX_ARCHIVE_MEMBERS, max(0, max_members))
    max_expanded_bytes = min(MAX_EXPANDED_BYTES, max(0, max_expanded_bytes))
    try:
        path = _archive_path(archive_path)
    except Exception as exc:
        return f"Error: archive expansion failed: {exc}"
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    count = total = 0
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) > max_members or sum(item.file_size for item in infos) > max_expanded_bytes:
                    return "Error: archive exceeds member or expansion limit."
                for info in infos:
                    if info.is_dir():
                        continue
                    target = _safe_target(root, info.filename)
                    if target is None:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as src, target.open("wb") as dst:
                        written = await _copy_bounded(src, dst, max_expanded_bytes - total)
                    if written > max_expanded_bytes - total:
                        shutil.rmtree(root, ignore_errors=True)
                        return "Error: archive exceeds expansion limit."
                    count += 1
                    total += written
        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                for info in archive:
                    if not info.isfile():
                        continue
                    if count + 1 > max_members or total + info.size > max_expanded_bytes:
                        shutil.rmtree(root, ignore_errors=True)
                        return "Error: archive exceeds member or expansion limit."
                    target = _safe_target(root, info.name)
                    src = archive.extractfile(info)
                    if target is None or src is None:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with src, target.open("wb") as dst:
                        written = await _copy_bounded(src, dst, max_expanded_bytes - total)
                    if written > max_expanded_bytes - total:
                        shutil.rmtree(root, ignore_errors=True)
                        return "Error: archive exceeds expansion limit."
                    count += 1
                    total += written
                    await asyncio.sleep(0)
        elif path.suffix.lower() == ".gz":
            target = _safe_target(root, path.stem)
            if target is None:
                return "Error: unsafe archive member path."
            with gzip.open(path, "rb") as src, target.open("wb") as dst:
                total = await _copy_bounded(src, dst, max_expanded_bytes)
            if total > max_expanded_bytes:
                target.unlink()
                return "Error: archive exceeds expansion limit."
            count = 1
        elif path.suffix.lower() in {".7z", ".rar"} and shutil.which("7z"):
            entries, error, listing_truncated = await _list_7z_entries(path, max_members + 1)
            if error:
                return error
            if listing_truncated or len(entries) > max_members or sum(item["size"] for item in entries) > max_expanded_bytes:
                return "Error: archive exceeds member or expansion limit."
            if any(_safe_target(root, item["path"]) is None for item in entries):
                return "Error: unsafe archive member path."
            if any(item["is_link"] for item in entries):
                return "Error: archive contains links that are not extracted by Profile 0."
            temporary = Path(tempfile.mkdtemp(prefix="profile0-7z-", dir=root.parent))
            returncode, _, stderr, error = await _run_external(
                ["7z", "x", f"-o{temporary}", "-y", str(path)], max_output_bytes=1024 * 1024,
            )
            if error:
                shutil.rmtree(temporary, ignore_errors=True)
                return error
            if returncode:
                shutil.rmtree(temporary, ignore_errors=True)
                return f"Error: 7z extraction failed: {stderr.decode(errors='replace')}"
            if any(item.is_symlink() for item in temporary.rglob("*")):
                shutil.rmtree(temporary, ignore_errors=True)
                return "Error: extracted 7z contains symbolic links."
            files = [item for item in temporary.rglob("*") if item.is_file()]
            count, total = len(files), sum(item.stat().st_size for item in files)
            if count > max_members or total > max_expanded_bytes:
                shutil.rmtree(temporary, ignore_errors=True)
                return "Error: extracted 7z exceeds configured limits; output removed."
            shutil.rmtree(root, ignore_errors=True)
            temporary.replace(root)
        else:
            return "Error: unsupported or invalid archive."
        return {"output_path": str(root.resolve()), "files": count, "expanded_bytes": total}
    except Exception as exc:
        return f"Error: archive expansion failed: {exc}"


async def expand_archive(
    archive_path: str, output_dir: str, max_members: int = 100000,
    max_expanded_bytes: int = 10737418240,
) -> str:
    result = await _expand_archive_detailed(archive_path, output_dir, max_members, max_expanded_bytes)
    if isinstance(result, str):
        return result
    return _materialized_output(Path(result["output_path"]))


async def _list_7z_entries(path: Path, max_entries: int = MAX_ARCHIVE_MEMBERS) -> tuple[list[dict], str, bool]:
    returncode, stdout, stderr, error = await _run_external(["7z", "l", "-slt", str(path)])
    if error:
        return [], error, False
    if returncode:
        return [], f"Error: 7z list failed: {stderr.decode(errors='replace')}", False
    entries = []
    truncated = False
    current: dict[str, str] = {}

    def append_current() -> None:
        nonlocal truncated
        if "Path" not in current or "Folder" not in current or current["Path"] == str(path):
            return
        if len(entries) >= max_entries:
            truncated = True
            return
        entries.append({
            "path": current["Path"],
            "size": int(current.get("Size", "0") or 0),
            "is_dir": _truthy_7z_flag(current.get("Folder", "")) or str(current.get("Attributes", "")).startswith("D"),
            "is_link": bool(current.get("Symbolic Link") or current.get("Hard Link")),
        })

    for line in stdout.decode(errors="replace").splitlines():
        if not line.strip():
            append_current()
            current = {}
        elif " = " in line:
            key, value = line.split(" = ", 1)
            current[key] = value
    append_current()
    return entries, "", truncated


def _truthy_7z_flag(value: str) -> bool:
    return value.strip() in {"+", "1", "True", "true", "Yes", "yes"}


async def expand_nested_archives(
    root: str, output_root: str, max_depth: int = 2, max_members: int = 100000,
    max_expanded_bytes: int = 10737418240,
) -> str:
    max_members = min(MAX_ARCHIVE_MEMBERS, max(0, max_members))
    max_expanded_bytes = min(MAX_EXPANDED_BYTES, max(0, max_expanded_bytes))
    base = Path(root)
    if not base.is_dir():
        return f"Error: not a directory: {root}"
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    expanded = []
    total_files = total_bytes = 0
    frontier = [(path, 1) for path in base.rglob("*") if path.is_file() and path.suffix.lower() in _ARCHIVE_SUFFIXES]
    while frontier:
        path, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        archive_id = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        output = destination / f"{path.name}.{archive_id}.expanded"
        result = await _expand_archive_detailed(str(path), str(output), max_members - total_files, max_expanded_bytes - total_bytes)
        if isinstance(result, str):
            expanded.append({"archive_path": str(path), "depth": depth, "result": result})
            continue
        data = result
        total_files += int(data["files"])
        total_bytes += int(data["expanded_bytes"])
        expanded.append({"archive_path": str(path), "depth": depth, **data})
        if total_files >= max_members or total_bytes >= max_expanded_bytes:
            break
        frontier.extend((child, depth + 1) for child in output.rglob("*") if child.is_file() and child.suffix.lower() in _ARCHIVE_SUFFIXES)
    return _materialized_output(destination)
