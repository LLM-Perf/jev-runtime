from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager

from jev_runtime.errors import JevError


class Admission:
    """Cancellation-safe FIFO admission with request and expanded-token budgets."""

    def __init__(self, max_requests: int = 64, max_tokens: int = 1_048_576, max_queue: int = 256):
        if min(max_requests, max_tokens, max_queue) <= 0:
            raise ValueError("Admission limits must be positive")
        self.max_requests, self.max_tokens, self.max_queue = max_requests, max_tokens, max_queue
        self.requests = 0
        self.tokens = 0
        self._condition = asyncio.Condition()
        self._queue: deque[object] = deque()

    @asynccontextmanager
    async def acquire(self, tokens: int):
        if tokens > self.max_tokens:
            raise JevError("engine_token_budget", "Request exceeds engine admission budget", 413)
        ticket = object()
        acquired = False
        async with self._condition:
            if len(self._queue) >= self.max_queue:
                raise JevError("queue_full", "Engine admission queue is full", 429)
            self._queue.append(ticket)
            try:
                await self._condition.wait_for(
                    lambda: (
                        self._queue[0] is ticket
                        and self.requests < self.max_requests
                        and self.tokens + tokens <= self.max_tokens
                    )
                )
                self._queue.popleft()
                self.requests += 1
                self.tokens += tokens
                acquired = True
                self._condition.notify_all()
            except BaseException:
                self._queue.remove(ticket)
                self._condition.notify_all()
                raise
        try:
            yield
        finally:
            if acquired:
                async with self._condition:
                    self.requests -= 1
                    self.tokens -= tokens
                    self._condition.notify_all()
