from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager

from jev_runtime.errors import JevError


class Admission:
    """Round-robin between authenticated tenants; FIFO within each tenant.

    Fairness is by admitted request, not by GPU time. Expanded-token limits bound
    each tenant's resource occupancy. An ineligible tenant does not block others.
    """

    def __init__(
        self,
        max_requests: int = 64,
        max_tokens: int = 1_048_576,
        max_queue: int = 256,
        max_tenant_requests: int | None = None,
        max_tenant_tokens: int | None = None,
        max_tenant_queue: int | None = None,
    ):
        self.max_requests, self.max_tokens, self.max_queue = max_requests, max_tokens, max_queue
        self.max_tenant_requests = (
            max_tenant_requests if max_tenant_requests is not None else max_requests
        )
        self.max_tenant_tokens = max_tenant_tokens if max_tenant_tokens is not None else max_tokens
        self.max_tenant_queue = max_tenant_queue if max_tenant_queue is not None else max_queue
        if (
            min(
                max_requests,
                max_tokens,
                max_queue,
                self.max_tenant_requests,
                self.max_tenant_tokens,
                self.max_tenant_queue,
            )
            <= 0
        ):
            raise ValueError("Admission limits must be positive")
        self.requests = self.tokens = self.queued = 0
        self._condition = asyncio.Condition()
        self._queues: dict[str, deque[tuple[object, int]]] = {}
        self._round_robin: deque[str] = deque()
        self._tenant_usage: dict[str, list[int]] = {}

    def _next_ticket(self):
        if self.requests >= self.max_requests:
            return None
        for tenant in self._round_robin:
            ticket, tokens = self._queues[tenant][0]
            requests_used, tokens_used = self._tenant_usage.get(tenant, [0, 0])
            if (
                requests_used < self.max_tenant_requests
                and self.tokens + tokens <= self.max_tokens
                and tokens_used + tokens <= self.max_tenant_tokens
            ):
                return ticket
        return None

    def _remove(self, tenant: str, item: tuple[object, int], rotate: bool):
        queue = self._queues[tenant]
        queue.remove(item)
        self.queued -= 1
        if not queue:
            self._queues.pop(tenant)
            self._round_robin.remove(tenant)
        elif rotate:
            self._round_robin.remove(tenant)
            self._round_robin.append(tenant)

    @asynccontextmanager
    async def acquire(self, tokens: int, tenant: str = "default"):
        if tokens <= 0:
            raise JevError("invalid_token_budget", "Token demand must be positive", 400)
        if tokens > min(self.max_tokens, self.max_tenant_tokens):
            raise JevError("engine_token_budget", "Request exceeds admission token budget", 413)
        ticket = object()
        item = (ticket, tokens)
        async with self._condition:
            if (
                self.queued >= self.max_queue
                or len(self._queues.get(tenant, ())) >= self.max_tenant_queue
            ):
                raise JevError("queue_full", "Admission queue is full", 429)
            if tenant not in self._queues:
                self._queues[tenant] = deque()
                self._round_robin.append(tenant)
            self._queues[tenant].append(item)
            self.queued += 1
            self._condition.notify_all()
            try:
                await self._condition.wait_for(lambda: self._next_ticket() is ticket)
                self._remove(tenant, item, rotate=True)
                self.requests += 1
                self.tokens += tokens
                usage = self._tenant_usage.setdefault(tenant, [0, 0])
                usage[0] += 1
                usage[1] += tokens
                self._condition.notify_all()
            except BaseException:
                self._remove(tenant, item, rotate=False)
                self._condition.notify_all()
                raise
        try:
            yield
        finally:
            async with self._condition:
                self.requests -= 1
                self.tokens -= tokens
                usage = self._tenant_usage[tenant]
                usage[0] -= 1
                usage[1] -= tokens
                if usage == [0, 0]:
                    self._tenant_usage.pop(tenant)
                self._condition.notify_all()
