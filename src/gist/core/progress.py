import sys
import time
from collections.abc import Callable

ProgressCallback = Callable[[str], None]


class StepLogger:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._start_time = time.monotonic()

    def __call__(self, message: str) -> None:
        if not self.enabled:
            return
        elapsed = time.monotonic() - self._start_time
        print(f"[gist +{elapsed:6.1f}s] {message}", file=sys.stderr, flush=True)
