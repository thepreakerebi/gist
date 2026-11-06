"""An in-process sliding-window rate limiter.

The API runs as a single uvicorn process, so counts held in memory are exact.
They reset when the process restarts, which is acceptable for limits that exist
to stop bursts rather than to bill anyone. The daily question cap is *not* kept
here for that reason: it lives in the database (see ``store.charge_question``).

Running several workers would give each its own counts, multiplying every limit
by the worker count; at that point the counters belong in Postgres or Redis.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class Limit:
    """At most ``count`` events per ``seconds``."""

    count: int
    seconds: float


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._lock = Lock()

    def hit(self, key: str, *limits: Limit) -> float | None:
        """Record an event for ``key`` if every limit allows it.

        Returns None when the event is allowed, otherwise the number of seconds
        until it would be. A refused event is not recorded, so a client that
        keeps retrying is not locked out for longer than the window.
        """

        if not limits:
            return None
        now = self._clock()
        longest = max(limit.seconds for limit in limits)
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= now - longest:
                events.popleft()

            wait = 0.0
            for limit in limits:
                recent = [stamp for stamp in events if stamp > now - limit.seconds]
                if len(recent) >= limit.count:
                    # The window frees up when the oldest event in it expires.
                    oldest = recent[len(recent) - limit.count]
                    wait = max(wait, oldest + limit.seconds - now)
            if wait > 0:
                return max(1.0, math.ceil(wait))

            events.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


limiter = RateLimiter()

# Guessing codes: per client address.
LOGIN_LIMITS = (Limit(5, 60), Limit(20, 3600))
# Adding a video starts minutes of CPU-bound transcription: per code.
INGEST_LIMITS = (Limit(3, 3600),)
# Asking questions, on top of the daily cap: per code.
QUERY_LIMITS = (Limit(10, 60),)
# Every other authenticated request, including the media byte ranges a video
# player makes while scrubbing: per code, and generous.
GENERAL_LIMITS = (Limit(600, 60),)
