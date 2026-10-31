"""Per-process UTC-day guard for paid hosted inference calls in a public Space."""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone


class ModelCallBudgetExceeded(RuntimeError):
    pass


class ModelCallBudget:
    def __init__(self, daily_limit: int):
        if daily_limit < 0:
            raise ValueError("daily_limit must be non-negative")
        self.daily_limit = daily_limit
        self._day = None
        self._used = 0
        self._lock = threading.Lock()

    def reserve(self) -> None:
        today = datetime.now(timezone.utc).date()
        with self._lock:
            if today != self._day:
                self._day, self._used = today, 0
            if self._used >= self.daily_limit:
                raise ModelCallBudgetExceeded(
                    f"This Space has reached its daily Beam-call limit ({self.daily_limit} calls). Try again after the UTC day resets."
                )
            self._used += 1


def configured_daily_limit() -> int:
    raw = os.getenv("PRAMAAN_DAILY_MODEL_CALL_CAP", "40")
    try:
        value = int(raw)
    except ValueError:
        value = 40
    return max(0, min(value, 1000))


MODEL_CALL_BUDGET = ModelCallBudget(configured_daily_limit())
