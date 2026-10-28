"""Physical-minimum Profile 0 tool registry."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from .index import build_file_index, query_file_index
from .runtime import RuntimeContext
from .trace import utc_now
from .tools import archive, media_time, readers, source_access, structured, capability_admission, vision

ALLOWED_CATEGORIES = {"primitive", "generic_parser"}
HOST_PATH_KEYS = {"path", "file_path", "db_path", "archive_path", "root"}
HOST_PATH_NOTE = " Accepts only host/tree paths; for image-index members use indexed-member tools or export_indexed_file first."
INDEX_TOOL_TIMEOUT_SEC = 360


@dataclass
class ToolDefinition:
    name: str
    description: str
    category: str
    input_schema: dict
    executor: Callable[..., Any]
    dependencies: list[str] = field(default_factory=list)
    read_only: bool = True
    deterministic: bool = True
    timeout_sec: float = 60.0
    source_sha256: str = ""
    output_schema: dict | None = None

    def public_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def _schema(properties: dict | None = None, required: list[str] | None = None) -> dict:
    result = {"type": "object", "properties": properties or {}}
    if required:
        result["required"] = required
    return result


def _p(kind: str, description: str = "", **extra: Any) -> dict:
    return {"type": kind, "description": description, **extra}


def _resolve_registry_mode(ctx: RuntimeContext | None, mode: str | None) -> str:
    configured = getattr(ctx, "config", {}).get("processor_mode", "baseline") if ctx is not None else "baseline"
    if (mode is not None and mode != "baseline") or (mode is None and configured != "baseline"):
        raise ValueError("Only baseline output mode is distributed")
    return "baseline"


class ToolRegistry:
    _OUTPUT_PROFILE: str | None = None

    def __init__(self, ctx: RuntimeContext | None, *, mode: str | None = None) -> None:
        self._mode = _resolve_registry_mode(ctx, mode)
        self.ctx = ctx
        self.tools: dict[str, ToolDefinition] = {}
        self._register_all()
        self._output_contracts: dict[str, dict] = {}
        self._output_profile_error = ""
        self._load_output_profile()

    @property
    def mode(self) -> str:
        """Construction-time mode identity; does not select result processors."""
        return self._mode

    def _load_output_profile(self) -> None:
        """Attach descriptive metadata; unavailable/partial metadata never gates tools."""
        filename = self._OUTPUT_PROFILE or (
            "baseline_raw.v1.json"
        )
        representation = "raw" if self.mode == "baseline" else "active_first"
        lifecycle = "raw" if self.mode == "baseline" else "first_agent_visible"
        try:
            profile = json.loads((Path(__file__).parent / "tool_output_schemas" / filename).read_text(encoding="utf-8"))
            if not isinstance(profile, dict) or profile.get("mode") != self.mode:
                raise ValueError("Output profile mode mismatch")
            rows = profile["tools"]
            if not isinstance(rows, list):
                raise ValueError("Output profile tools must be a list")
            contracts = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Output contract must be an object")
                name = row["tool_id"]
                if not isinstance(name, str) or name in contracts:
                    raise ValueError("Invalid or duplicate output contract Tool ID")
                if (row.get("mode") != self.mode
                        or row.get("representation") != representation
                        or row.get("lifecycle") != lifecycle
                        or not isinstance(row.get("output_schema"), dict)
                        or not isinstance(row.get("model_description"), str)
                        or not row["model_description"].strip()):
                    raise ValueError("Invalid output contract mode/lifecycle/content")
                contracts[name] = deepcopy(row)
            missing = sorted(set(self.tools) - set(contracts))
            extra = sorted(set(contracts) - set(self.tools))
            if missing or extra:
                self._output_profile_error = f"Output metadata coverage: missing={missing}, extra={extra}"
            self._output_contracts = {name: row for name, row in contracts.items() if name in self.tools}
            for name, row in self._output_contracts.items():
                self.tools[name].output_schema = deepcopy(row["output_schema"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            # Metadata failure changes only the advisory description/status.
            # Registration and executor availability remain usable.
            self._output_contracts = {}
            self._output_profile_error = f"Output metadata unavailable: {exc}"

    @property
    def output_profile_error(self) -> str:
        """Empty for complete metadata; advisory failure never rejects execution."""
        return self._output_profile_error

    def output_contracts(self) -> dict[str, dict]:
        """Independent mode-specific copies keyed by Tool ID, including evidence/status."""
        return deepcopy(self._output_contracts)

    def output_schema_for(self, tool_name: str) -> dict | None:
        """Copy the wire schema, or None if descriptive metadata is unavailable."""
        contract = self._output_contracts.get(tool_name)
        return deepcopy(contract["output_schema"]) if contract is not None else None

    def model_visible_definitions(self) -> list[dict]:
        """Provider-compatible derived descriptions; original definitions stay intact.

        Agent retrieval filters this view before the existing provider call.
        Never put output metadata in input_schema or provider-specific fields.
        """
        definitions = []
        for name, tool in self.tools.items():
            item = deepcopy(tool.public_dict())
            contract = self._output_contracts.get(name)
            if contract is None:
                label = "baseline RAW"
                detail = f"Output ({label}): field contract unavailable; no output fields are guaranteed by metadata."
            else:
                detail = contract["model_description"]
            item["description"] = tool.description + "\n\n" + detail
            definitions.append(item)
        return definitions

    def add(self, definition: ToolDefinition) -> None:
        if definition.category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Profile 0 rejects tool category: {definition.category}")
        if definition.name in self.tools:
            raise ValueError(f"Duplicate tool: {definition.name}")
        properties = set(definition.input_schema.get("properties", {}))
        if properties & HOST_PATH_KEYS and "file_id" not in properties:
            definition.description += HOST_PATH_NOTE
        try:
            definition.source_sha256 = hashlib.sha256(inspect.getsource(definition.executor).encode()).hexdigest()
        except (OSError, TypeError):
            definition.source_sha256 = ""
        definition.executor = self._instrument(definition)
        self.tools[definition.name] = definition

    def _instrument(self, definition: ToolDefinition) -> Callable[..., Any]:
        original = definition.executor

        async def wrapped(**kwargs: Any) -> str:
            started = utc_now()
            t0 = time.monotonic()
            input_paths = _resolve_input_paths(kwargs, self.ctx)
            admitted_path = {"read_windows_registry_value": "hive_path",
                             "parse_e01_header": "file_path", "analyze_image": "image_path"}.get(definition.name)
            if admitted_path is not None:
                # These adapters authorize before reading. Do not fingerprint an
                # untrusted path before that boundary, even for trace metadata.
                try:
                    input_paths = [capability_admission.authorized_path(self.ctx, kwargs.get(admitted_path))]
                except (ValueError, OSError, RuntimeError):
                    input_paths = []
            evidence_paths = [path for path in input_paths if Path(path).exists()]
            input_hashes = _input_fingerprints(evidence_paths, self.ctx)
            provenance = _access_provenance(kwargs, self.ctx)
            target_source_id = str(kwargs.get("source_id") or (self.ctx.active_source.id if self.ctx.active_source else ""))
            self.ctx.trace.emit(
                "tool_call_started", tool_name=definition.name,
                capability_category=definition.category, args=kwargs,
                status="started", input_paths=input_paths, started_at=started,
                evidence_paths=evidence_paths, source_id=target_source_id,
                details={"input_fingerprints": input_hashes, **provenance},
            )
            try:
                normalized = _normalize_and_validate_paths(kwargs, self.ctx)
                from .integration.retry_boundary import retry_boundary_guard
                boundary = retry_boundary_guard.get()
                if boundary is None:
                    result = await asyncio.wait_for(original(**normalized), timeout=definition.timeout_sec)
                else:
                    async def guarded_retry():
                        boundary(definition, normalized)
                        return await original(**normalized)
                    result = await asyncio.wait_for(guarded_retry(), timeout=definition.timeout_sec)
            except asyncio.CancelledError as exc:
                self.ctx.trace.emit(
                    "tool_call_failed", tool_name=definition.name,
                    capability_category=definition.category, args=kwargs,
                    source_id=target_source_id, status="cancelled",
                    reason_code="tool_call_cancelled", input_paths=input_paths,
                    started_at=started, finished_at=utc_now(),
                    duration_ms=(time.monotonic() - t0) * 1000, error=str(exc),
                    details=provenance,
                )
                raise
            except asyncio.TimeoutError:
                self.ctx.trace.emit(
                    "tool_call_timed_out", tool_name=definition.name,
                    capability_category=definition.category, args=kwargs,
                    source_id=target_source_id,
                    status="timed_out", input_paths=input_paths, started_at=started,
                    finished_at=utc_now(), duration_ms=(time.monotonic() - t0) * 1000,
                    error=f"Tool timed out after {definition.timeout_sec}s",
                    reason_code="tool_timed_out", details=provenance,
                )
                return f"Error: tool timed out after {definition.timeout_sec}s"
            except Exception as exc:
                self.ctx.trace.emit(
                    "tool_call_failed", tool_name=definition.name,
                    capability_category=definition.category, args=kwargs,
                    source_id=target_source_id,
                    status="failed", input_paths=input_paths, started_at=started,
                    finished_at=utc_now(), duration_ms=(time.monotonic() - t0) * 1000,
                    error=str(exc),
                    reason_code=(
                        "unreadable_file" if isinstance(exc, (FileNotFoundError, PermissionError))
                        else _failure_reason(str(exc))
                    ), details=provenance,
                )
                if admitted_path is not None:
                    return capability_admission.encode_result({"result": {
                        "valid": False, "error": f"{definition.name} failed: {exc}",
                        "error_type": "path_denied" if isinstance(exc, ValueError) else "execution_error"}})
                return f"Error: {definition.name} failed: {exc}"
            text = str(result)
            terminal_provenance = _access_provenance(kwargs, self.ctx)
            event_type = "tool_call_failed" if text.startswith("Error:") else "tool_call_empty" if _is_empty(text) else "tool_call_succeeded"
            status = event_type.removeprefix("tool_call_")
            output_path = _extract_output_path(text)
            self.ctx.trace.emit(
                event_type, tool_name=definition.name,
                capability_category=definition.category, args=kwargs,
                source_id=target_source_id,
                status=status, input_paths=input_paths, output_preview=text[:4000],
                output_path=output_path, evidence_paths=evidence_paths,
                output_sha256=hashlib.sha256(text.encode(errors="replace")).hexdigest(),
                started_at=started, finished_at=utc_now(),
                duration_ms=(time.monotonic() - t0) * 1000,
                error=text[:1000] if event_type == "tool_call_failed" else "",
                reason_code=_result_reason(text, event_type, definition.name),
                details={"input_fingerprints": input_hashes, **terminal_provenance, **_structured_result_details(text)},
            )
            return text
        setattr(wrapped, "_profile0_context", self.ctx)
        return wrapped

    def definitions(self) -> list[dict]:
        return [tool.public_dict() for tool in self.tools.values()]

    def executors(self) -> dict[str, Callable[..., Any]]:
        return {name: tool.executor for name, tool in self.tools.items()}

    def write_manifest(self, system_prompt: str, config: dict) -> Path:
        items = []
        for tool in self.tools.values():
            items.append({
                "name": tool.name, "category": tool.category,
                "description": tool.description, "read_only": tool.read_only,
                "deterministic": tool.deterministic,
                "dependencies": {dep: _dependency_available(dep) for dep in tool.dependencies},
                "source_sha256": tool.source_sha256,
                "description_sha256": hashlib.sha256(tool.description.encode()).hexdigest(),
            })
        manifest = {
            "profile_id": "profile0",
            "profile_version": "qualified-profile0-v1",
            "tools": items,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "tool_descriptions_sha256": hashlib.sha256(json.dumps([item["description"] for item in items], ensure_ascii=False).encode()).hexdigest(),
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "runtime_source_sha256": {
                str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.rglob("*.py"))
            },
        }
        path = self.ctx.run_dir / "capability_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _register_all(self) -> None:
        ctx = self.ctx
        add = self.add
        add(ToolDefinition("probe_source", "Report whether the active evidence source is ready, missing, or unreadable for the current access mode.", "primitive", _schema(), lambda: source_access.probe_source(ctx)))
        add(ToolDefinition("list_partitions", "List selectable partition identifiers and descriptions for the active disk image.", "primitive", _schema(), lambda: source_access.list_partitions(ctx), ["mmls"]))
        add(ToolDefinition("select_partition", "Select a structured partition_id returned by list_partitions; sector conversion is handled internally.", "primitive", _schema({"partition_id": _p("string", "Partition identifier returned by list_partitions.")}, ["partition_id"]), lambda partition_id: source_access.select_partition(ctx, partition_id), read_only=False))
        add(ToolDefinition("filesystem_info", "Report structured filesystem metadata for the active evidence source, including filesystem type and available sector/cluster geometry.", "primitive", _schema(), lambda: source_access.filesystem_info(ctx), ["fsstat"]))
        add(ToolDefinition("list_directory", "List candidate files and directories from the active tree or image evidence source.", "primitive", _schema({"path": _p("string", default="."), "inode": _p("string", default=""), "recursive": _p("boolean", default=False)}), lambda path=".", inode="", recursive=False: source_access.list_directory(ctx, path, inode, recursive), ["fls"]))
        add(ToolDefinition("extract_image_file", "Export one file from an image by inode into the run directory. On success, returns only {\"output_path\": \"...\"} for the materialized host file.", "primitive", _schema({"inode": _p("string"), "output_name": _p("string", default="")}, ["inode"]), lambda inode, output_name="": source_access.extract_image_file(ctx, inode, output_name), ["icat"], read_only=False))
        add(ToolDefinition("export_file", "Copy one readable tree-evidence file into the run output directory within a byte limit. On success, returns only {\"output_path\": \"...\"} for the materialized host file.", "primitive", _schema({"file_path": _p("string"), "output_name": _p("string", default=""), "max_bytes": _p("integer", default=1073741824)}, ["file_path"]), lambda file_path, output_name="", max_bytes=1073741824: source_access.export_file(ctx, file_path, output_name, max_bytes), read_only=False))
        add(ToolDefinition("build_file_index", "Build or reuse a structured file index for the active image or tree evidence source.", "primitive", _schema(), lambda: build_file_index(ctx, False), ["fls"], read_only=False, timeout_sec=INDEX_TOOL_TIMEOUT_SEC))
        add(ToolDefinition("query_file_index", "Query the active source file index and return stable source_id/file_id references for matching candidates.", "primitive", _schema({
            "path_pattern": _p("string", default="*"), "name_pattern": _p("string", default="*"), "extension": _p("string", default=""),
            "file_type": _p("string", default=""), "min_size": _p("integer", default=0), "max_size": _p("integer", default=-1),
            "is_deleted": _p("boolean"), "time_field": _p("string", default=""),
            "time_after": _p("string", default=""), "time_before": _p("string", default=""), "max_results": _p("integer", default=200),
        }), lambda **kwargs: query_file_index(ctx, **kwargs), timeout_sec=INDEX_TOOL_TIMEOUT_SEC))
        indexed_ids = {"source_id": _p("string"), "file_id": _p("string")}
        add(ToolDefinition("get_indexed_file_metadata", "Return bounded metadata for one indexed evidence member; evidence_path is an evidence/internal path, not a host path.", "primitive", _schema(indexed_ids, ["source_id", "file_id"]), lambda source_id, file_id: source_access.get_indexed_file_metadata(ctx, source_id, file_id)))
        add(ToolDefinition("export_indexed_file", "Export one indexed evidence member within a bounded byte limit. On success, returns only {\"output_path\": \"...\"} for the materialized host file.", "primitive", _schema({**indexed_ids, "output_name": _p("string", default=""), "max_bytes": _p("integer", default=1073741824)}, ["source_id", "file_id"]), lambda source_id, file_id, output_name="", max_bytes=1073741824: source_access.export_indexed_file(ctx, source_id, file_id, output_name, max_bytes), read_only=False, timeout_sec=300))
        add(ToolDefinition("read_indexed_text", "Read bounded text from one indexed evidence member using byte offset/max_bytes; returns text, actual encoding, bytes_read, decode_lossy, truncated, and next_offset without exposing temporary paths.", "primitive", _schema({**indexed_ids, "offset": _p("integer", "Byte offset from the start of the indexed member.", default=0, minimum=0), "max_bytes": _p("integer", "Maximum bytes to read before decoding.", default=16000, minimum=1, maximum=1048576), "encoding": _p("string", "Python codec name or auto.", default="auto")}, ["source_id", "file_id"]), lambda source_id, file_id, offset=0, max_bytes=16000, encoding="auto": source_access.read_indexed_text(ctx, source_id, file_id, offset, max_bytes, encoding), timeout_sec=300))
        add(ToolDefinition("read_indexed_binary_range", "Read a bounded binary range from one indexed evidence member using byte offset/length; returns JSON hexdump, bytes_read, truncated, and next_offset without exposing temporary paths.", "primitive", _schema({**indexed_ids, "offset": _p("integer", "Byte offset from the start of the indexed member.", default=0, minimum=0), "length": _p("integer", "Number of bytes to read.", default=2048, minimum=1, maximum=1048576)}, ["source_id", "file_id"]), lambda source_id, file_id, offset=0, length=2048: source_access.read_indexed_binary_range(ctx, source_id, file_id, offset, length), timeout_sec=300))
        add(ToolDefinition("search_indexed_file", "Search one indexed evidence member for a case-insensitive literal or regex pattern; tree/image outputs match search_file, offsets are member-relative, and no temporary path is returned.", "primitive", _schema({**indexed_ids, "pattern": _p("string"), "regex": _p("boolean"), "context_chars": _p("integer", minimum=0, maximum=4096, default=80), "max_results": _p("integer", minimum=1, maximum=1000, default=100)}, ["source_id", "file_id", "pattern"]), lambda source_id, file_id, pattern, regex=False, context_chars=80, max_results=100: source_access.search_indexed_file(ctx, source_id, file_id, pattern, regex, context_chars, max_results), timeout_sec=300))
        add(ToolDefinition("read_text", "Read a bounded text range from a file using byte offset/max_bytes; returns text, actual encoding, bytes_read, decode_lossy, truncated, and next_offset.", "primitive", _schema({"file_path": _p("string"), "offset": _p("integer", "Byte offset from the start of the file.", default=0, minimum=0), "max_bytes": _p("integer", "Maximum bytes to read before decoding.", default=16000, minimum=1, maximum=1048576), "encoding": _p("string", "Python codec name or auto.", default="auto")}, ["file_path"]), readers.read_text))
        add(ToolDefinition("read_binary_range", "Read a bounded binary range using byte offset/length and return JSON hexdump plus continuation metadata.", "primitive", _schema({"file_path": _p("string"), "offset": _p("integer", "Byte offset from the start of the file.", default=0, minimum=0), "length": _p("integer", "Number of bytes to read.", default=2048, minimum=1, maximum=1048576)}, ["file_path"]), readers.read_binary_range))
        add(ToolDefinition("compute_hash", "Compute one or more standard cryptographic hashes for a file and return them under digests.", "primitive", _schema({"file_path": _p("string"), "algorithms": {"type": "array", "items": {"type": "string"}, "default": ["sha256"]}}, ["file_path"]), readers.compute_hash))
        add(ToolDefinition("identify_file_type", "Compare extension-derived MIME information with generic file-signature MIME detection.", "primitive", _schema({"file_path": _p("string")}, ["file_path"]), readers.identify_file_type, ["file"]))
        add(ToolDefinition("detect_text_encoding", "Detect a basic text encoding from a bounded file sample and report sampled_bytes.", "primitive", _schema({"file_path": _p("string"), "sample_bytes": _p("integer", minimum=1, maximum=1048576, default=65536)}, ["file_path"]), readers.detect_text_encoding))
        add(ToolDefinition("extract_file_strings", "Extract ASCII printable and ASCII-range UTF-16LE strings with byte offsets, value truncation flags, scan_complete, and result-count truncated.", "primitive", _schema({"file_path": _p("string"), "min_length": _p("integer", minimum=1, maximum=4096, default=4), "max_results": _p("integer", minimum=1, maximum=1000, default=500), "max_bytes": _p("integer", minimum=1, maximum=268435456, default=268435456)}, ["file_path"]), readers.extract_file_strings))
        add(ToolDefinition("search_file_strings", "Search the complete extracted ASCII/UTF-16LE string stream and return match-start byte offsets with value_truncated; does not prelimit to max_results*20 strings.", "primitive", _schema({"file_path": _p("string"), "pattern": _p("string"), "regex": _p("boolean"), "min_length": _p("integer", minimum=1, maximum=4096, default=4), "max_results": _p("integer", minimum=1, maximum=1000, default=200), "max_bytes": _p("integer", minimum=1, maximum=268435456, default=268435456)}, ["file_path", "pattern"]), readers.search_file_strings))
        add(ToolDefinition("search_file", "Search one readable file for a case-insensitive literal or regex pattern; returns decode_lossy, exact byte_offset or null, scan_complete, and result-count truncated.", "primitive", _schema({"file_path": _p("string"), "pattern": _p("string"), "regex": _p("boolean"), "context_chars": _p("integer", minimum=0, maximum=4096, default=80), "max_results": _p("integer", minimum=1, maximum=1000, default=100)}, ["file_path", "pattern"]), readers.search_file))
        add(ToolDefinition("search_files_content", "Recursively search readable files below a directory; hidden files are included, symlinks skipped, results use relative_path, and skipped/failed files affect scan_complete.", "primitive", _schema({"root": _p("string"), "pattern": _p("string"), "regex": _p("boolean"), "extensions": {"type": "array", "items": {"type": "string"}}, "max_file_size": _p("integer", minimum=1, maximum=268435456, default=10485760), "max_results": _p("integer", minimum=1, maximum=1000, default=200)}, ["root", "pattern"]), readers.search_files_content, timeout_sec=120))
        add(ToolDefinition("sqlite_schema", "Return a bounded SQLite table/view structure overview with object type, name, and compact columns [name,type,pk?] to guide follow-up queries; does not return table content or CREATE SQL.", "generic_parser", _schema({"db_path": _p("string")}, ["db_path"]), structured.sqlite_schema))
        add(ToolDefinition("sqlite_schema_detail", "Return exact or bounded CREATE SQL for one selected SQLite table, view, index, or trigger by object name; use object_type to resolve ambiguous names.", "generic_parser", _schema({"db_path": _p("string"), "object_name": _p("string"), "object_type": _p("string", "Optional object type: table, view, index, or trigger.", default="", enum=["", "table", "view", "index", "trigger"])}, ["db_path", "object_name"]), structured.sqlite_schema_detail))
        add(ToolDefinition("sqlite_query", "Run one single read-only SQLite SELECT/WITH query and return bounded rows with returned_rows and truncated; PRAGMA, DDL, DML, and multi-statement SQL are rejected.", "generic_parser", _schema({"db_path": _p("string"), "query": _p("string"), "max_rows": _p("integer", "Maximum rows to return.", default=100, minimum=1, maximum=1000)}, ["db_path", "query"]), structured.sqlite_query))
        add(ToolDefinition("sqlite_search", "Search user SQLite table columns for a literal substring; percent and underscore are escaped, rowid/primary_key identify hits, and returned_matches/truncated describe result bounds.", "generic_parser", _schema({"db_path": _p("string"), "keyword": _p("string"), "max_results": _p("integer", "Maximum matches to return.", default=100, minimum=1, maximum=1000)}, ["db_path", "keyword"]), structured.sqlite_search))
        add(ToolDefinition("sqlite_export_query", "Export the result of one read-only SQLite query. On success, returns only {\"output_path\": \"...\"} for the materialized host file.", "generic_parser", _schema({"db_path": _p("string"), "query": _p("string"), "output_path": _p("string"), "max_rows": _p("integer", default=10000, minimum=1, maximum=10000)}, ["db_path", "query", "output_path"]), structured.sqlite_export_query, read_only=False))
        add(ToolDefinition("parse_plist", "Parse XML or binary plist and return a bounded root preview with root_type and truncated for depth, item, string, or Data limits.", "generic_parser", _schema({"file_path": _p("string")}, ["file_path"]), structured.parse_plist))
        add(ToolDefinition("plist_search", "Search plist keys and scalar values using JSON Pointer paths; containers are not string-searched and returned_matches/truncated describe result bounds.", "generic_parser", _schema({"file_path": _p("string"), "keyword": _p("string"), "max_results": _p("integer", "Maximum matches to return.", default=200, minimum=1, maximum=1000)}, ["file_path", "keyword"]), structured.plist_search))
        add(ToolDefinition("inspect_json", "Parse standard JSON and return a bounded root preview; rejects duplicate keys and non-standard NaN/Infinity constants.", "generic_parser", _schema({"file_path": _p("string")}, ["file_path"]), structured.inspect_json))
        add(ToolDefinition("inspect_xml", "Parse XML and return a bounded element-tree preview plus complete node_count; rejects DTD/entity declarations.", "generic_parser", _schema({"file_path": _p("string")}, ["file_path"]), structured.inspect_xml))
        add(ToolDefinition("inspect_csv", "Parse UTF-8 delimited text and return bounded row preview with delimiter, encoding, decode_lossy, and truncated; header is not inferred.", "generic_parser", _schema({"file_path": _p("string"), "max_rows": _p("integer", "Rows to preview.", default=200, minimum=1, maximum=1000)}, ["file_path"]), structured.inspect_csv))
        add(ToolDefinition("inspect_ini", "Parse INI and return raw defaults and sections without interpolation while preserving key case.", "generic_parser", _schema({"file_path": _p("string")}, ["file_path"]), structured.inspect_ini))
        add(ToolDefinition("search_structured_file", "Search real structure positions in JSON, XML, CSV/TSV, INI, or Plist files using bounded JSON Pointer or XML/CSV paths and returned_matches/truncated.", "generic_parser", _schema({"file_path": _p("string"), "keyword": _p("string"), "max_results": _p("integer", "Maximum matches to return.", default=200, minimum=1, maximum=1000)}, ["file_path", "keyword"]), structured.search_structured_file))
        add(ToolDefinition("list_archive", "List extractable member references, kinds, and sizes in a supported generic archive.", "primitive", _schema({"archive_path": _p("string"), "max_members": _p("integer", default=10000)}, ["archive_path"]), archive.list_archive, ["7z"]))
        add(ToolDefinition("extract_archive_member", "Safely extract one bounded regular file from an archive. On success, returns only {\"output_path\": \"...\"} for the materialized host file.", "primitive", _schema({"archive_path": _p("string"), "member": _p("string"), "output_dir": _p("string"), "max_expanded_bytes": _p("integer", default=1073741824)}, ["archive_path", "member", "output_dir"]), archive.extract_archive_member, read_only=False))
        add(ToolDefinition("expand_archive", "Safely expand a ZIP, tar-family, or 7z archive within configured limits. On success, returns only {\"output_path\": \"...\"} for the materialized output directory.", "primitive", _schema({"archive_path": _p("string"), "output_dir": _p("string"), "max_members": _p("integer", default=100000), "max_expanded_bytes": _p("integer", default=10737418240)}, ["archive_path", "output_dir"]), archive.expand_archive, ["7z"], read_only=False, timeout_sec=300))
        add(ToolDefinition("expand_nested_archives", "Recursively expand generic archives below a directory into the run directory within depth, member, and byte limits. On success, returns only {\"output_path\": \"...\"} for the materialized output directory.", "primitive", _schema({"root": _p("string"), "max_depth": _p("integer", default=2), "max_members": _p("integer", default=100000), "max_expanded_bytes": _p("integer", default=10737418240)}, ["root"]), lambda root, max_depth=2, max_members=100000, max_expanded_bytes=10737418240: archive.expand_nested_archives(root, str(ctx.run_dir / "nested_archive_expansions"), max_depth, max_members, max_expanded_bytes), ["7z"], read_only=False, timeout_sec=300))
        add(ToolDefinition("read_image_metadata", "Read generic image dimensions, format, mode, and bounded EXIF fields.", "generic_parser", _schema({"file_path": _p("string")}, ["file_path"]), media_time.read_image_metadata, ["PIL"]))
        timestamp_hints = [
            "auto", "unix", "unix_seconds", "unix_ms", "unix_milliseconds",
            "unix_us", "unix_microseconds", "unix_ns", "unix_nanoseconds",
            "filetime", "windows_filetime", "webkit", "apple",
        ]
        add(ToolDefinition("normalize_timestamp", "Convert a numeric timestamp into possible UTC/local candidate interpretations; candidates are not confidence-ranked.", "primitive", _schema({"value": _p("number"), "format_hint": _p("string", enum=timestamp_hints, default="auto"), "utc_offset_minutes": _p("integer", minimum=-1439, maximum=1439, default=0)}, ["value"]), media_time.normalize_timestamp))
        add(ToolDefinition("read_windows_registry_value", "Read a caller-specified registry key path and registry value from an offline Windows Registry hive; returns a decoded typed registry value. hive_path must be an authorized host/tree hive path; export image members first.", "generic_parser", _schema({"hive_path": _p("string", "Accessible local Windows Registry hive path."), "key_path": _p("string", "Hive-relative registry key path."), "value_name": _p("string", "Registry value name; empty string denotes the default value.")}, ["hive_path", "key_path", "value_name"]), lambda hive_path, key_path, value_name: capability_admission.read_windows_registry_value(ctx, hive_path, key_path, value_name)))
        add(ToolDefinition("parse_e01_header", "Parse an E01/EWF forensic image header (header/header2) into acquisition metadata, device information and acquisition timestamps; requires an accessible authorized EWF image host path.", "generic_parser", _schema({"file_path": _p("string", "Authorized local E01/EWF forensic image path.")}, ["file_path"]), lambda file_path: capability_admission.parse_e01_header(ctx, file_path)))
        add(ToolDefinition("analyze_image", "Analyze pixels and clearly visible content of one local image artifact: visible text, objects and interface elements. This is visual observation, distinct from read_image_metadata EXIF/format attributes; unsupported hidden facts must not be inferred. image_path must be an authorized host/tree image path; export image members first.", "generic_parser", _schema({"image_path": _p("string", "Authorized local PNG/JPEG/WEBP image path; maximum 10 MiB and 20 million pixels."), "query": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Optional local visible-content instruction; null or omitted uses a general low-inference description."}}, ["image_path"]), lambda image_path, query=None: vision.analyze_image(ctx, image_path, query), ["PIL"], deterministic=False, timeout_sec=300))


def create_tool_registry(ctx: RuntimeContext | None, *, mode: str | None = None) -> ToolRegistry:
    _resolve_registry_mode(ctx, mode)
    from .tool_registry_baseline import BaselineToolRegistry
    return BaselineToolRegistry(ctx)


def _is_empty(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return True
    return any(marker in stripped for marker in ('"matches": 0', '"rows": []', '"results": []', '"entries": []', "no matches", "no partitions"))


def _dependency_available(name: str) -> bool:
    return bool(shutil.which(name) or importlib.util.find_spec(name))


def _resolve_input_paths(args: dict[str, Any], ctx: RuntimeContext) -> list[str]:
    input_keys = {"path", "file_path", "db_path", "archive_path", "root"}
    result = []
    for key, value in args.items():
        if key not in input_keys or not isinstance(value, str):
            continue
        path = Path(value)
        if ctx.active_source and ctx.source_root.is_dir():
            if key == "path" and value in {"", ".", "/"}:
                path = ctx.source_root
            elif not path.is_absolute():
                path = ctx.source_root / path
        result.append(str(path))
    return result


def _normalize_and_validate_paths(args: dict[str, Any], ctx: RuntimeContext) -> dict[str, Any]:
    normalized = dict(args)
    input_keys = {"file_path", "db_path", "archive_path", "root"}
    output_keys = {"output_path", "output_dir"}
    allowed_inputs = [Path(source.path).resolve() for source in ctx.case.sources]
    allowed_inputs.extend(path.resolve() for path in ctx.evidence_roots.values())
    allowed_inputs.append(ctx.run_dir.resolve())
    for key in input_keys:
        value = normalized.get(key)
        if not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_absolute() and ctx.active_source and ctx.source_root.is_dir():
            path = ctx.source_root / path
        resolved = path.resolve()
        if not any(_within(resolved, allowed) for allowed in allowed_inputs):
            hint = _indexed_path_hint(ctx, value)
            if hint:
                raise ValueError(
                    f"image/internal path is unreadable as a host path: {value}; {hint}"
                )
            raise ValueError(f"input path is outside case evidence roots: {value}")
        normalized[key] = str(resolved)
    for key in output_keys:
        value = normalized.get(key)
        if not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = ctx.run_dir / path
        resolved = path.resolve()
        if not _within(resolved, ctx.run_dir.resolve()):
            raise ValueError(f"output path is outside run directory: {value}")
        normalized[key] = str(resolved)
    return normalized


def _indexed_path_hint(ctx: RuntimeContext, value: str) -> str:
    source = ctx.active_source
    if source is None or source.access_mode != "image":
        return ""
    requested = value.replace("\\", "/").lstrip("/").casefold()
    active_entries = ctx.indexes.get(source.id, [])
    cached = ctx.index_cache.get(source.id, {}).get("entries", [])
    entry_groups = (active_entries,) if cached is active_entries else (active_entries, cached)
    matches = []
    seen = set()
    for entries in entry_groups:
        for entry in entries:
            if str(entry.get("path", "")).replace("\\", "/").lstrip("/").casefold() != requested:
                continue
            file_id = str(entry.get("file_id", ""))
            if not file_id or file_id in seen:
                continue
            seen.add(file_id)
            matches.append(f"source_id={source.id}, file_id={file_id}")
            if len(matches) >= 3:
                return "use indexed-member tools with " + " or ".join(matches)
    if matches:
        return "use indexed-member tools with " + " or ".join(matches)
    return (
        f"build/query the file index for source_id={source.id} and use its file_id, "
        "or use extract_image_file with an observed inode"
    )


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _input_fingerprints(paths: list[str], ctx: RuntimeContext, full_hash_limit: int = 1024 * 1024) -> dict[str, dict]:
    result = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            continue
        size = path.stat().st_size
        if raw in ctx.file_hashes:
            result[raw] = {"sha256": ctx.file_hashes[raw], "size": size, "complete": True}
            continue
        if size > full_hash_limit:
            result[raw] = {"sha256": "", "size": size, "complete": False, "reason": "full_hash_deferred_large_input"}
            continue
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            value = digest.hexdigest()
            result[raw] = {"sha256": value, "size": size, "complete": True}
            ctx.file_hashes[raw] = value
        except OSError:
            continue
    return result


def _access_provenance(args: dict[str, Any], ctx: RuntimeContext) -> dict:
    state = ctx.access_state()
    indexed = bool(args.get("file_id"))
    if indexed:
        _, entry = source_access._indexed_entry(ctx, str(args.get("source_id", "")), str(args.get("file_id", "")))
        entry = entry or {}
        state = {
            "partition_id": entry.get("partition_id", ""),
            "sector_size": entry.get("sector_size"),
            "start_sector": entry.get("start_sector"),
            "byte_offset": entry.get("byte_offset"),
        }
    source_mode = ctx.active_source.access_mode if ctx.active_source else ""
    access_mode = "indexed_member" if indexed else f"{source_mode}_source" if source_mode in {"image", "archive"} else "host_path"
    return {
        "access_mode": access_mode,
        "partition_id": state.get("partition_id", ""),
        "sector_size": state.get("sector_size"),
        "start_sector": state.get("start_sector"),
        "byte_offset": state.get("byte_offset"),
        "file_id": args.get("file_id", ""),
        "indexed_source_id": args.get("source_id", "") if indexed else "",
    }


def _structured_result_details(text: str) -> dict:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        key: value[key] for key in (
            "file_size", "scanned_bytes", "truncated", "scan_complete",
            "source_id", "partition_id", "file_id",
        ) if key in value
    }


def _failure_reason(text: str) -> str:
    lowered = text.casefold()
    if "dependency not found" in lowered:
        return "dependency_missing"
    if "outside case evidence roots" in lowered or "escapes evidence root" in lowered or "unreadable" in lowered or "no such file" in lowered:
        return "unreadable_file"
    if "archive member exceeds" in lowered or "archive exceeds" in lowered:
        return "archive_member_too_large"
    if "encrypted" in lowered:
        return "encrypted_database"
    if "unsupported" in lowered or "invalid archive" in lowered:
        return "unsupported_format"
    return "tool_call_failed"


def _result_reason(text: str, event_type: str, tool_name: str) -> str:
    if event_type == "tool_call_failed":
        return _failure_reason(text)
    details = _structured_result_details(text)
    if details.get("truncated") or details.get("scan_complete") is False or "[command output truncated]" in text:
        return "search_truncated" if "search" in tool_name else "tool_output_truncated"
    return ""


def _extract_output_path(text: str) -> str:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(value, dict):
        return ""
    for key in ("output_path", "index_path", "evidence_root"):
        if value.get(key):
            return str(value[key])
    return ""
