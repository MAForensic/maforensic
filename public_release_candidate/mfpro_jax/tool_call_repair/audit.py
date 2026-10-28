"""Independent redacted events/metrics. No A/B model usage mutation or raw data."""
from dataclasses import dataclass, field
from collections import Counter
from .contracts import freeze, plain


@dataclass(frozen=True)
class RepairEvent:
    event_id: str
    kind: str
    original_attempt: str
    reason: str
    details: object = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'details', freeze(self.details))


class MemoryAudit:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


class RepairMetrics:
    def __init__(self):
        self.counts = Counter()
        self.usage = []
        self.context_chars = []
        self.wall_seconds = 0.0
        self.cpu_seconds = 0.0

    def snapshot(self):
        return {'role': 'repair', 'counts': dict(self.counts), 'usage': plain(self.usage),
                'context_chars': list(self.context_chars), 'wall_seconds': self.wall_seconds,
                'cpu_seconds': self.cpu_seconds}
