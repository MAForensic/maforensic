"""Retry-only synchronous check inside Registry's scheduled executor task."""
from contextvars import ContextVar
retry_boundary_guard = ContextVar('profile0_retry_boundary_guard', default=None)
