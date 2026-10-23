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


@pytest.mark.parametrize("binding", ["requests", "tokens", "branches"])
async def test_full_queue_does_not_block_an_eligible_tenant(binding):
    limits = dict(max_requests=3, max_tokens=10, max_branches=6, max_queue=1)
    limits["max_tenant_" + binding] = {"requests": 1, "tokens": 5, "branches": 2}[binding]
    pool = Admission(**limits)

    async def blocked():
        async with pool.acquire(2, "a", branches=1):
            pytest.fail("Tenant a must remain blocked")

    async with pool.acquire(4, "a", branches=2):
        waiter = asyncio.create_task(blocked())
        try:
            await wait_queued(pool, 1)
            # The waiting limit still applies to work that cannot run immediately.
            with pytest.raises(JevError) as exc:
                async with pool.acquire(1, "a"):
                    pytest.fail("Same-tenant FIFO must not be bypassed")
            assert exc.value.code == "queue_full"
            assert pool.queued == 1
            async with pool.acquire(2, "b"):
                assert (pool.requests, pool.tokens, pool.branches, pool.queued) == (2, 6, 3, 1)
            assert (pool.requests, pool.tokens, pool.branches, pool.queued) == (1, 4, 2, 1)
        finally:
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert pool.queued == 0
    assert pool.requests == pool.tokens == pool.branches == pool.queued == 0
    assert not pool._tenant_usage and not pool._queues and not pool._round_robin


@pytest.mark.parametrize("new_tenant", ["waiting", "newcomer"])
async def test_full_queue_newcomer_cannot_pass_an_eligible_waiter(new_tenant):
    pool = Admission(max_requests=1, max_queue=1)
    admitted = []

    async def waiting():
        async with pool.acquire(1, "waiting"):
            admitted.append("waiting")

    async with pool.acquire(1, "held"):
        waiter = asyncio.create_task(waiting())
        await wait_queued(pool, 1)
    try:
        # Capacity has just become available, but the notified waiter has not run yet.
        with pytest.raises(JevError) as exc:
            async with pool.acquire(1, new_tenant):
                admitted.append("newcomer")
        assert exc.value.code == "queue_full"
        assert pool.queued == 1
        await asyncio.wait_for(waiter, 1)
        assert admitted == ["waiting"]
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
    assert pool.requests == pool.tokens == pool.branches == pool.queued == 0
