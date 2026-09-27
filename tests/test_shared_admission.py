import asyncio
import multiprocessing
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import pytest

from jev_runtime.api import create_app
from jev_runtime.config import Settings, build_runtime
from jev_runtime.errors import JevError
from jev_runtime.registry import Registry
from jev_runtime.runtime import Runtime
from jev_runtime.schema import DecisionRequest, ExecutionOptions, TextInput
from jev_runtime.shared_admission import SharedAdmission
from tests.conftest import ControlledEngine


def serving(registry, limits):
    pool = SharedAdmission(registry, "fixture:0", **limits)
    pool.start(registry, "fixture:0")
    if registry.list()["routes"]:
        registry.record_worker_prepared("test@1")
    assert registry.serve_worker("fixture:0")
    return pool


def setup_pair(tmp_path, bundle, **limits):
    a = Registry(tmp_path / "shared.db")
    first = serving(a, limits)
    a.upload(bundle)
    _, lease = a.begin_prepare(bundle.reference, "fixture:0", "prepare")
    a.finish_prepare(bundle.reference, lease)
    a.record_worker_prepared(bundle.reference)
    a.activate("model", bundle.reference, 0)
    b = Registry(a.path)
    second = serving(b, limits)
    return a, b, first, second


@asynccontextmanager
async def request(registry, pool, tenant="default", tokens=1, branches=1):
    lease = registry.acquire("model", uuid.uuid4().hex, None, "fixture:0", tenant).lease_id
    try:
        async with pool.acquire(tokens, tenant, branches=branches, lease_id=lease):
            yield lease
    finally:
        registry.release(lease)


async def queued(pool, count):
    async with asyncio.timeout(2):
        while pool.snapshot()["queued_requests"] != count:  # noqa: ASYNC110 - durable cross-worker state
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("binding", ["requests", "tokens", "branches"])
async def test_shared_capacity_cannot_multiply_with_workers(tmp_path, bundle, binding):
    limits = dict(max_requests=10, max_tokens=100, max_branches=100)
    limits["max_" + binding] = 1
    a, b, first, second = setup_pair(tmp_path, bundle, **limits)
    entered = asyncio.Event()

    async def peer():
        async with request(b, second):
            entered.set()

    async with request(a, first):
        work = asyncio.create_task(peer())
        await queued(first, 1)
        assert not entered.is_set()
        assert first.snapshot() == second.snapshot()
        assert first.snapshot()["requests"] == 1
    await asyncio.wait_for(work, 2)
    assert entered.is_set() and first.snapshot()["requests"] == 0
    a.close()
    b.close()


async def test_shared_tenant_fairness_bypasses_ineligible_tenant(tmp_path, bundle):
    a, b, first, second = setup_pair(tmp_path, bundle, max_requests=2, max_tenant_requests=1)
    order = []

    async def peer(tenant):
        async with request(b, second, tenant):
            order.append(tenant)

    async with request(a, first, "a"):
        blocked = asyncio.create_task(peer("a"))
        await queued(first, 1)
        await asyncio.wait_for(peer("b"), 2)
        assert order == ["b"] and not blocked.done()
    await asyncio.wait_for(blocked, 2)
    assert order == ["b", "a"]
    a.close()
    b.close()


async def test_shared_queue_limit_and_cancel_are_global(tmp_path, bundle):
    a, b, first, second = setup_pair(tmp_path, bundle, max_requests=1, max_queue=1)

    async def peer():
        async with request(b, second):
            pytest.fail("never admitted")

    async with request(a, first):
        work = asyncio.create_task(peer())
        await queued(first, 1)
        with pytest.raises(JevError, match="queue") as exc:
            async with request(a, first, "other"):
                pass
        assert exc.value.code == "queue_full"
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
        assert first.snapshot()["queued_requests"] == 0
        assert first.snapshot()["requests"] == 1
    a.close()
    b.close()


async def test_ticket_keeps_capacity_until_explicit_abort_recovery(tmp_path, bundle):
    a, b, first, second = setup_pair(tmp_path, bundle, max_requests=1)
    snap = a.acquire("model", "uncertain", None, "fixture:0")
    a.record_branches(snap.lease_id, ["branch"])
    async with first.acquire(9, branches=3, lease_id=snap.lease_id):
        pass
    # Merely leaving an execution context cannot prove GPU drain.
    a.mark_abort_pending(snap.lease_id, {"branch"})
    assert second.snapshot()["requests"] == 1
    assert second.snapshot()["expanded_tokens"] == 9
    a.stop_worker()
    a.close()
    reopened = Registry(b.path)
    assert (
        SharedAdmission(reopened, "fixture:0", max_requests=1).snapshot()["expanded_branches"] == 3
    )
    entered = asyncio.Event()

    async def peer():
        async with request(b, second):
            entered.set()

    work = asyncio.create_task(peer())
    await queued(second, 1)
    assert not entered.is_set()
    recovery = b.recovery_snapshot("uncertain", "fixture:0")
    b.release_recovered(recovery)
    await asyncio.wait_for(work, 2)
    assert second.snapshot()["requests"] == 0
    reopened.close()
    b.close()


def test_limits_cannot_diverge_or_change_with_retained_work(tmp_path, bundle):
    a, b, first, second = setup_pair(tmp_path, bundle, max_requests=1)
    c = Registry(a.path)
    with pytest.raises(JevError) as exc:
        serving(c, dict(max_requests=2))
    assert exc.value.code == "admission_policy_conflict"
    # A stopped worker with a retained lease still prevents policy migration.
    snap = a.acquire("model", "retained", None, "fixture:0")
    a.stop_worker()
    b.stop_worker()
    with pytest.raises(JevError):
        serving(c, dict(max_requests=2))
    a.release(snap.lease_id)
    assert serving(c, dict(max_requests=2)).snapshot()["limits"]["max_requests"] == 2
    a.close()
    b.close()
    c.close()


async def test_shared_admission_enforces_tenant_owner_and_oversized_branch_budget(tmp_path, bundle):
    a, b, first, second = setup_pair(tmp_path, bundle, max_branches=2)
    snap = a.acquire("model", "owned", None, "fixture:0", "a")
    for pool, tenant, branches, expected in [
        (first, "b", 1, "admission_lease_mismatch"),
        (second, "a", 1, "admission_lease_mismatch"),
        (first, "a", 3, "engine_branch_budget"),
    ]:
        with pytest.raises(JevError) as exc:
            async with pool.acquire(1, tenant, branches=branches, lease_id=snap.lease_id):
                pass
        assert exc.value.code == expected
    assert first.snapshot()["requests"] == first.snapshot()["queued_requests"] == 0
    a.release(snap.lease_id)
    a.close()
    b.close()


async def test_runtime_uncertain_abort_retains_shared_budget_and_recovery_unblocks(
    tmp_path, compiler, bundle, question
):
    registry = Registry(tmp_path / "runtime.db")
    engine = ControlledEngine()
    pool = SharedAdmission(registry, "fixture:0", max_requests=1)
    runtime = Runtime(engine, compiler, registry, "fixture:0", "fixture", admission=pool)
    await runtime.start()
    registry.upload(bundle)
    await runtime.prepare(bundle.reference)
    runtime.activate("model", bundle.reference, 0)
    engine.gate.clear()
    engine.started.clear()
    engine.fail_cancel = True
    body = DecisionRequest(model="model", input=TextInput(text="refund"), questions=(question,))
    work = asyncio.create_task(runtime.decide(body, "uncertain"))
    await engine.started.wait()
    work.cancel()
    with pytest.raises(JevError):
        await work
    assert pool.snapshot()["requests"] == 1 and not runtime._active
    second = asyncio.create_task(runtime.decide(body, "waiting"))
    await queued(pool, 1)
    assert not second.done()
    engine.fail_cancel = False
    assert await runtime.recover_cancelled("uncertain")
    engine.gate.set()
    assert (await asyncio.wait_for(second, 2)).status == "completed"
    assert pool.snapshot()["requests"] == 0
    await runtime.close()


def process_contender(path, start, release, messages):
    async def run():
        registry = Registry(path)
        pool = serving(registry, dict(max_requests=2, max_branches=3))
        messages.put(("ready", registry.owner))
        await asyncio.to_thread(start.wait)
        async with request(registry, pool, branches=2):
            messages.put(("admitted", pool.snapshot()))
            await asyncio.to_thread(release.acquire)
        registry.stop_worker()
        registry.close()
        messages.put(("done", None))

    asyncio.run(run())


def test_real_processes_atomically_reserve_expanded_branch_budget(tmp_path, bundle):
    a, b, first, second = setup_pair(tmp_path, bundle, max_requests=2, max_branches=3)
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    release = ctx.Semaphore(0)
    messages = ctx.Queue()
    processes = [
        ctx.Process(target=process_contender, args=(str(a.path), start, release, messages))
        for _ in range(3)
    ]
    try:
        for p in processes:
            p.start()
        assert [messages.get(timeout=15)[0] for _ in processes] == ["ready"] * 3
        start.set()
        done = 0
        admitted = 0
        while done < 3:
            kind, value = messages.get(timeout=15)
            if kind == "admitted":
                admitted += 1
                assert value["requests"] == 1 and value["expanded_branches"] == 2
                release.release()
            else:
                assert kind == "done"
                done += 1
        assert admitted == 3 and first.snapshot()["requests"] == 0
        for p in processes:
            p.join(5)
            assert p.exitcode == 0
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(5)
        a.close()
        b.close()


async def test_configured_runtime_uses_shared_admission_and_global_metrics(
    tmp_path, compiler, bundle
):
    settings = Settings(
        backend="vllm",
        model_id="fixture",
        model_revision="a" * 40,
        registry_path=str(tmp_path / "config.db"),
    )
    runtime = await build_runtime(settings, native_backend=ControlledEngine(), compiler=compiler)
    try:
        assert isinstance(runtime.admission, SharedAdmission)
        await runtime.start()
        bundle = bundle.model_copy(update={"model": runtime.expected_model})
        runtime.registry.upload(bundle)
        await runtime.prepare(bundle.reference)
        runtime.activate("model", bundle.reference, 0)
        app = create_app(instance=runtime)
        app.state.jev_runtime = runtime
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            snap = runtime.registry.acquire("model", "held", None, runtime.backend_identity)
            async with runtime.admission.acquire(7, branches=2, lease_id=snap.lease_id):
                text = (await c.get("/metrics")).text
                assert 'jev_shared_admission{quantity="expanded_tokens"} 7.0' in text
                assert 'jev_shared_admission{quantity="expanded_branches"} 2.0' in text
                assert "jev_admission{quantity=" not in text
            runtime.registry.release(snap.lease_id)
    finally:
        await runtime.close()


async def test_sync_compile_cannot_dispatch_after_deadline(runtime, question, monkeypatch):
    original = runtime._compile_request

    def slow_compile(*args):
        time.sleep(0.02)
        return original(*args)

    monkeypatch.setattr(runtime, "_compile_request", slow_compile)
    body = DecisionRequest(
        model="model",
        input=TextInput(text="refund"),
        questions=(question,),
        execution=ExecutionOptions(timeout_ms=1),
    )
    with pytest.raises(JevError) as exc:
        await runtime.decide(body)
    assert exc.value.code == "deadline_exceeded"
    assert not runtime.backend.calls and not runtime.registry.list()["leases"]


async def test_round_robin_is_shared_across_workers(tmp_path, bundle):
    a, b, first, second = setup_pair(tmp_path, bundle, max_requests=1)
    order = []

    async def run(registry, pool, tenant, label):
        async with request(registry, pool, tenant):
            order.append(label)

    async with request(a, first, "initial"):
        a1 = asyncio.create_task(run(a, first, "a", "a1"))
        await queued(first, 1)
        a2 = asyncio.create_task(run(b, second, "a", "a2"))
        await queued(first, 2)
        b1 = asyncio.create_task(run(b, second, "b", "b1"))
        await queued(first, 3)
    await asyncio.wait_for(asyncio.gather(a1, a2, b1), 2)
    assert order == ["a1", "b1", "a2"]
    a.close()
    b.close()
