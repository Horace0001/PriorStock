"""Simple rate-limiter used by the text-generation client."""

from __future__ import annotations

import threading
import time


class FixedIntervalRateLimiter:
    """Guarantee a minimum wall-clock interval between outgoing API requests."""

    def __init__(self, requests_per_second: int) -> None:
        """Initialize the limiter from one explicit request budget."""

        self._minimum_interval_seconds = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed_timestamp = 0.0

    def acquire(self) -> None:
        """Block the current thread until one request slot is available."""

        with self._lock:
            current_timestamp = time.perf_counter()
            wait_seconds = self._next_allowed_timestamp - current_timestamp
            if wait_seconds > 0.0:
                time.sleep(wait_seconds)
                current_timestamp = time.perf_counter()
            self._next_allowed_timestamp = current_timestamp + self._minimum_interval_seconds
