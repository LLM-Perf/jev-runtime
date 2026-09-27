"""Bounded per-request runtime observations, independent of a metrics backend."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Literal

Stage = Literal["pin", "compile", "journal", "queue", "execute", "finalize", "release", "total"]
STAGES = ("pin", "compile", "journal", "queue", "execute", "finalize", "release", "total")


class DecisionTrace:
    """Serial wall-clock phases partition runtime time, including failure cleanup.

    Work samples can overlap across branches and are deliberately separate from
    that partition. No prompts, IDs or user-provided label values are retained.
    """

    def __init__(self):
        self.seconds: dict[str, float] = defaultdict(float)
        self.work: dict[str, list[float]] = defaultdict(list)
        self._started = self._previous = None
        self._stage = None

    def switch(self, stage: Stage) -> None:
        now = time.perf_counter()
        if self._started is None:
            self._started = now
        if self._stage is not None:
            self.seconds[self._stage] += now - self._previous
        self._previous, self._stage = now, stage

    def finish(self) -> None:
        if self._stage is None:
            return
        now = time.perf_counter()
        self.seconds[self._stage] += now - self._previous
        self.seconds["total"] = now - self._started
        self._stage = None

    @contextmanager
    def measure_work(self, kind: Literal["branch_queue", "engine_call", "assembly", "abort"]):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.work[kind].append(time.perf_counter() - start)

    def header(self) -> str:
        return ", ".join(
            f"jev_{stage};dur={self.seconds[stage] * 1000:.6f}"
            for stage in STAGES
            if stage in self.seconds
        )


def parse_timing_header(value: str | None) -> dict[Stage, float]:
    """Parse our opt-in header strictly; an omitted phase remains unobserved."""
    import math

    if not value or len(value) > 2048:
        raise ValueError("Runtime timing header missing or oversized")
    result = {}
    for item in value.split(","):
        name, separator, duration = item.strip().partition(";dur=")
        stage = name.removeprefix("jev_")
        if not name.startswith("jev_") or stage not in STAGES or not separator or stage in result:
            raise ValueError("Invalid or duplicate runtime timing stage")
        duration_ms = float(duration)
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError("Invalid runtime timing duration")
        result[stage] = duration_ms
    if (
        "total" not in result
        or abs(result["total"] - sum(v for k, v in result.items() if k != "total")) > 0.00001
    ):
        raise ValueError("Runtime phases do not partition total time")
    return result
