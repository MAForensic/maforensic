"""Baseline-only runtime identity; no processor discovery or plugin loader."""
from pathlib import Path
CENTRAL_TOOL_IDS = frozenset(['analyze_image', 'build_file_index', 'compute_hash', 'detect_text_encoding', 'expand_archive', 'expand_nested_archives', 'export_file', 'export_indexed_file', 'extract_archive_member', 'extract_file_strings', 'extract_image_file', 'filesystem_info', 'get_indexed_file_metadata', 'identify_file_type', 'inspect_csv', 'inspect_ini', 'inspect_json', 'inspect_xml', 'list_archive', 'list_directory', 'list_partitions', 'normalize_timestamp', 'parse_e01_header', 'parse_plist', 'plist_search', 'probe_source', 'query_file_index', 'read_binary_range', 'read_image_metadata', 'read_indexed_binary_range', 'read_indexed_text', 'read_text', 'read_windows_registry_value', 'search_file', 'search_file_strings', 'search_files_content', 'search_indexed_file', 'search_structured_file', 'select_partition', 'sqlite_export_query', 'sqlite_query', 'sqlite_schema', 'sqlite_schema_detail', 'sqlite_search'])
class RunProcessorRouter:
    def __init__(self, processor_mode="baseline"):
        if processor_mode != "baseline":
            raise ValueError("This source candidate supports baseline output only")
    @property
    def processor_mode(self): return "baseline"
    @property
    def identity(self): return {"processor_mode":"baseline","output_policy":"raw-baseline-v1"}
    def provider(self, envelope, mode): return None
    def preflight_run(self, run_dir):
        return {"identity":self.identity,"processor_mode":"baseline"}
    def bind_run(self, run_dir):
        return self.preflight_run(run_dir)
