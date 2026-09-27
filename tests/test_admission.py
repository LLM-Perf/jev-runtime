import asyncio

import pytest

from jev_runtime.admission import Admission
from jev_runtime.errors import JevError


async def wait_queued(pool, count):
    async with asyncio.timeout(1):
        async with pool._condition:
            await pool._condition.wait_for(lambda: pool.queued == count)


async def test_tenant_round_robin_prevents_fifo_flood_starvation():
    pool = Admission(max_requests=1)
    order = []

    async def run(tenant, label):
        async with pool.acquire(1, tenant):
            order.append(label)

    async with pool.acquire(1, "initial"):
        a1 = asyncio.create_task(run("a", "a1"))
        a2 = asyncio.create_task(run("a", "a2"))
        await wait_queued(pool, 2)
        b1 = asyncio.create_task(run("b", "b1"))
        await wait_queued(pool, 3)
    await asyncio.gather(a1, a2, b1)
    assert order == ["a1", "b1", "a2"]
    assert pool.requests == pool.tokens == pool.queued == 0


async def test_tenant_limit_does_not_block_eligible_other_tenant():
    pool = Admission(max_requests=2, max_tenant_requests=1, max_tenant_tokens=10)
    b_started = asyncio.Event()

    async def run(tenant):
        async with pool.acquire(5, tenant):
            if tenant == "b":
                b_started.set()

    async with pool.acquire(8, "a"):
        a = asyncio.create_task(run("a"))
        await wait_queued(pool, 1)
        b = asyncio.create_task(run("b"))
        await asyncio.wait_for(b_started.wait(), 1)
        assert not a.done()
    await asyncio.gather(a, b)
    assert pool.requests == pool.tokens == pool.queued == 0


async def test_cancelled_waiter_releases_queue_and_tenant_state():
    pool = Admission(max_requests=1, max_tenant_queue=1)

    async def wait():
        async with pool.acquire(1, "queued"):
            pytest.fail("Cancelled request should never be admitted")

    async with pool.acquire(1):
        queued = asyncio.create_task(wait())
        await wait_queued(pool, 1)
        with pytest.raises(JevError) as error:
            async with pool.acquire(1, "queued"):
                pass
        assert error.value.code == "queue_full"
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert pool.queued == 0
    assert pool._tenant_usage == {}
