"""In-memory rate limits for the public demo, so strangers cannot exhaust the free LLM quota.

Single-process only (the Space runs one worker); a multi-replica deployment would need Redis.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        per_client_per_minute: int,
        per_day: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.per_client_per_minute = per_client_per_minute
        self.per_day = per_day
        self.clock = clock
        self._client_hits: dict[str, deque[float]] = defaultdict(deque)
        self._day_hits: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self, client: str) -> str | None:
        """Record a request. Returns None if allowed, or a reason string if it must be rejected."""
        now = self.clock()
        with self._lock:
            hits = self._client_hits[client]
            while hits and now - hits[0] >= 60:
                hits.popleft()
            while self._day_hits and now - self._day_hits[0] >= 86_400:
                self._day_hits.popleft()
            if len(self._day_hits) >= self.per_day:
                return "The demo's daily question budget is used up; please try again tomorrow."
            if len(hits) >= self.per_client_per_minute:
                return "Too many questions in a minute; please wait a moment."
            hits.append(now)
            self._day_hits.append(now)
            return None
