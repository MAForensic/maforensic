"""Fixed-baseline registry entry point using the shared tool construction source."""

from __future__ import annotations

from .runtime import RuntimeContext
from .tool_registry import ToolRegistry


class BaselineToolRegistry(ToolRegistry):
    """Fresh shared tool definitions bound once, with fixed baseline identity."""

    _OUTPUT_PROFILE = "baseline_raw.v1.json"

    def __init__(self, ctx: RuntimeContext | None) -> None:
        super().__init__(ctx, mode="baseline")


def build_registry(ctx: RuntimeContext | None) -> BaselineToolRegistry:
    """Build a new baseline instance without mutating the context configuration."""
    return BaselineToolRegistry(ctx)
