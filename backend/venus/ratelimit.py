"""A per-client token bucket for the screening endpoint.

In process and in memory on purpose: this serves one district laptop, so a
Redis dependency would buy nothing and cost a moving part at the demo. One
bucket per client address, refilling at `per_minute` tokens a minute up to
`burst`; a request with no token left is refused with 429 and a Retry-After the
caller can obey.

What this protects against is a stuck retry loop or an open browser tab eating
the single CPU that the operator at the camera is waiting on. It is not a
security control — there is no authentication to attach an identity to (see
docs/PRIVACY.md) and a client can change address.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, per_minute: int, burst: int) -> None:
        self.per_minute = per_minute
        self.burst = max(burst, 1)
        self.rate = per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.per_minute > 0

    def check(self, client: str, now: float | None = None) -> float:
        """0.0 if the request may proceed, else seconds until it may."""
        if not self.enabled:
            return 0.0
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(client, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self._buckets[client] = (tokens, now)
                return max((1.0 - tokens) / self.rate, 0.001)
            self._buckets[client] = (tokens - 1.0, now)
            if len(self._buckets) > 1024:          # a laptop never has this many clients
                self._prune(now)
            return 0.0

    def _prune(self, now: float) -> None:
        """Drop buckets that have refilled completely; they carry no state."""
        full = self.burst / self.rate
        for key in [k for k, (_, last) in self._buckets.items() if now - last > full]:
            self._buckets.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
