import asyncio

import httpx
import pytest
from conftest import ControlledEngine

from jev_runtime.api import create_app
from jev_runtime.backends.base import ScoreInput
from jev_runtime.errors import JevError
from jev_runtime.registry import Registry
from jev_runtime.runtime import Runtime
from jev_runtime.schema import DecisionRequest, TextInput


def decision(question, rid="request"):
    return DecisionRequest(
        model="model", request_id=rid, input=TextInput(text="ready"), questions=(question,)
    )


def sibling(runtime, *, recovery_only=False):
    return Runtime(
        ControlledEngine(),
        runtime.compiler,
        Registry(runtime.registry.path),
        runtime.backend_identity,
        runtime.model_id,
        recovery_only=recovery_only,
    )


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():  # noqa: ASYNC110 - observes persisted admission/control state
            await asyncio.sleep(0.005)


async def test_cross_worker_drain_preserves_routes_and_requires_offline_resume(runtime, question):
    other = sibling(runtime)
    await other.start()
    routes = runtime.registry.list()["routes"]
    runtime.backend.gate.clear()
    other.backend.gate.clear()
    other.backend.started.clear()
    typed = asyncio.create_task(runtime.decide(decision(question)))
    raw = asyncio.create_task(other.score_raw(ScoreInput("raw", "q", (1, 2), (3, 4))))
    try:
        await runtime.backend.started.wait()
        await other.backend.started.wait()
        state = await runtime.quiesce(0, 0)
        assert not state["drained"]
        assert state["outstanding"] == {
            "leases": 1,
            "raw_work": 1,
            "adapter_operations": 0,
            "recovery_operations": 0,
        }
        assert state["generation"] == 1 and state["state"] == "QUIESCING"
        assert runtime.registry.list()["routes"] == routes
        newcomer = sibling(runtime)
        try:
            with pytest.raises(JevError, match="draining"):
                await newcomer.start()
            assert not newcomer.backend.calls
        finally:
            await newcomer.close()
        with pytest.raises(JevError, match="draining"):
            await other.decide(decision(question, "late"))
        with pytest.raises(JevError, match="draining"):
            await other.score_raw(ScoreInput("late-raw", "q", (1,), (2,)))
        runtime.backend.gate.set()
        other.backend.gate.set()
        assert (await typed).status == "completed"
        assert (await raw).request_id == "raw"
        final = await other.quiesce(1, 1)
        assert final["drained"] and not final["waiting_workers"]
        assert all(row["state"] in {"QUIESCED", "STOPPED"} for row in final["workers"])
        assert runtime._health_task.done() and other._health_task.done()
        assert runtime.registry.list()["routes"] == routes
        with pytest.raises(JevError, match="Stop all previous"):
            runtime.registry.resume_backend(runtime.backend_identity, 1)
    finally:
        runtime.backend.gate.set()
        other.backend.gate.set()
        await asyncio.gather(typed, raw, return_exceptions=True)
        await other.close()
        await runtime.close()
    registry = Registry(runtime.registry.path)
    try:
        with pytest.raises(JevError, match="generation"):
            registry.resume_backend(runtime.backend_identity, 0)
        registry.resume_backend(runtime.backend_identity, 1)
        assert registry.backend_control(runtime.backend_identity)["generation"] == 2
        restarted = sibling(runtime)
        try:
            await restarted.start()
            assert (await restarted.decide(decision(question))).status == "completed"
            assert restarted.registry.list()["routes"] == routes
        finally:
            await restarted.close()
    finally:
        registry.close()


async def test_gate_is_atomic_for_prepare_publication_and_acquire(runtime, bundle):
    reg, backend = runtime.registry, runtime.backend_identity
    fresh = bundle.model_copy(update={"version": 2})
    reg.upload(fresh)
    resolved = reg.resolve("model", None, backend)
    reg.begin_quiesce(backend, 0)
    # Simulate a worker which has not yet observed its control poll.
    assert runtime._quiescing is False
    operations = (
        lambda: reg.acquire("model", "late", None, backend),
        lambda: reg.begin_prepare(fresh.reference, backend, "new-prepare"),
        lambda: reg.pin_revalidation(bundle.reference, "new-canary", backend),
        lambda: reg.activate("new-alias", bundle.reference, 0),
        lambda: reg.serve_worker(backend),
        lambda: reg.begin_raw_work(backend, "raw-late"),
        lambda: runtime.admission.reserve(resolved, "model", "late", None, "default", 2, ["b"]),
    )
    for operation in operations:
        with pytest.raises(JevError) as error:
            operation()
        assert error.value.code == "backend_quiescing"
    assert reg.inspect(fresh.reference)["state"] == "VALIDATED"
    assert not reg.list()["leases"]
    assert (await runtime.quiesce(1, 1))["drained"]


async def test_waiting_prepare_cannot_outlive_a_successful_drain(runtime, bundle):
    await runtime._management_lock.acquire()
    task = asyncio.create_task(runtime.prepare(bundle.reference))
    try:
        await until(lambda: bool(runtime._prepare_tasks))
        state = await runtime.quiesce(0, 0.05)
        assert not state["drained"] and state["waiting_workers"]
        calls = len(runtime.backend.calls)
        runtime._management_lock.release()
        with pytest.raises(JevError, match="draining"):
            await task
        assert (await runtime.quiesce(1, 1))["drained"]
        assert len(runtime.backend.calls) == calls
    finally:
        if runtime._management_lock.locked():
            runtime._management_lock.release()
        await asyncio.gather(task, return_exceptions=True)


async def test_health_probe_already_running_finishes_before_ack(runtime):
    # Wake the real monitor quickly; do not manually substitute a health task.
    runtime.health.settings = runtime.health.settings.model_copy(update={"interval_seconds": 0.01})
    runtime._health_task.cancel()
    await asyncio.gather(runtime._health_task, return_exceptions=True)
    runtime._health_task = asyncio.create_task(runtime._watch_health())
    runtime.backend.gate.clear()
    try:
        await runtime.backend.started.wait()
        assert not (await runtime.quiesce(0, 0.05))["drained"]
        calls = len(runtime.backend.calls)
        runtime.backend.gate.set()
        assert (await runtime.quiesce(1, 1))["drained"]
        await asyncio.sleep(0.03)
        assert len(runtime.backend.calls) == calls and not runtime.backend.cancelled
        assert runtime._health_task.done()
    finally:
        runtime.backend.gate.set()


@pytest.mark.parametrize("runtime", [{"max_requests": 1}], indirect=True)
async def test_previously_queued_requests_drain_normally(runtime, question):
    runtime.backend.gate.clear()
    first = asyncio.create_task(runtime.decide(decision(question, "one")))
    await runtime.backend.started.wait()
    second = asyncio.create_task(runtime.decide(decision(question, "two")))
    try:
        await until(lambda: runtime.admission.snapshot()["queued_requests"] == 1)
        assert not (await runtime.quiesce(0, 0.05))["drained"]
        runtime.backend.gate.set()
        assert all(result.status == "completed" for result in await asyncio.gather(first, second))
        assert (await runtime.quiesce(1, 1))["drained"]
        assert runtime.admission.snapshot()["requests"] == 0
    finally:
        runtime.backend.gate.set()
        await asyncio.gather(first, second, return_exceptions=True)


async def test_uncertain_raw_abort_survives_and_requires_quiesced_explicit_recovery(runtime):
    runtime.backend.gate.clear()
    runtime.backend.fail_cancel = True
    task = asyncio.create_task(runtime.score_raw(ScoreInput("raw", "q", (1,), (2, 3))))
    await runtime.backend.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = runtime.registry.raw_recovery_candidates(runtime.backend_identity)[0]
    assert row["phase"] == "abort_pending" and row["recoverable"]
    with pytest.raises(JevError, match="Quiesce"):
        await runtime.recover_raw(row["id"])
    state = await runtime.quiesce(0, 0.05)
    assert not state["drained"] and state["outstanding"]["raw_work"] == 1
    with pytest.raises(RuntimeError, match="abort failure"):
        await runtime.recover_raw(row["id"])
    recovered = sibling(runtime, recovery_only=True)
    try:
        await recovered.start()
        assert recovered.registry.raw_recovery_candidates(runtime.backend_identity) == [row]
        assert await recovered.recover_raw(row["id"])
        assert not await recovered.recover_raw(row["id"])
        assert (await runtime.quiesce(1, 1))["drained"]
    finally:
        runtime.backend.gate.set()
        await recovered.close()


async def test_live_raw_work_cannot_be_recovered_and_duplicate_id_is_rejected(runtime):
    runtime.backend.gate.clear()
    task = asyncio.create_task(runtime.score_raw(ScoreInput("raw", "q", (1,), (2,))))
    try:
        await runtime.backend.started.wait()
        with pytest.raises(JevError, match="already in flight"):
            await runtime.score_raw(ScoreInput("raw", "q", (1,), (2,)))
        await runtime.quiesce(0, 0)
        row = runtime.registry.raw_recovery_candidates(runtime.backend_identity)[0]
        with pytest.raises(JevError, match="live or unverifiable"):
            await runtime.recover_raw(row["id"])
    finally:
        runtime.backend.gate.set()
        await task


async def test_http_auth_readiness_and_cancellation_remain_correct_during_quiesce(
    runtime, question
):
    app = create_app(instance=runtime, api_key="data", admin_key="admin")
    app.state.jev_runtime = runtime
    runtime.backend.gate.clear()
    task = asyncio.create_task(runtime.decide(decision(question)))
    try:
        await runtime.backend.started.wait()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            body = {"expected_generation": 0, "timeout_seconds": 0}
            assert (await client.post("/admin/quiescence", json=body)).status_code == 401
            response = await client.post(
                "/admin/quiescence", json=body, headers={"Authorization": "Bearer admin"}
            )
            assert response.status_code == 200 and not response.json()["drained"]
            assert (
                await client.get("/ready", headers={"Authorization": "Bearer data"})
            ).status_code == 503
            response = await client.post(
                "/v1/requests/request/cancel", headers={"Authorization": "Bearer data"}
            )
            assert response.status_code == 200 and response.json()["cancelled"]
            assert (await runtime.quiesce(1, 1))["drained"]
    finally:
        runtime.backend.gate.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_legacy_workers_prevent_protocol_acceptance(runtime):
    with runtime.registry._transaction() as db:
        db.execute("DELETE FROM worker_quiescence WHERE owner=?", (runtime.registry.owner,))
    with pytest.raises(JevError, match="legacy"):
        await runtime.quiesce(0, 0)
    assert runtime.registry.backend_control(runtime.backend_identity)["state"] == "OPEN"


async def test_dead_raw_owner_blocks_drain_until_recovered(runtime, monkeypatch):
    row_id = runtime.registry.begin_raw_work(runtime.backend_identity, "orphan")
    # Actual Linux death identity is covered separately; force only this owner for this unit case.
    monkeypatch.setattr(Registry, "_owner_status", staticmethod(lambda identity: "dead"))
    assert not (await runtime.quiesce(0, 0))["drained"]
    assert await runtime.recover_raw(row_id)
    assert (await runtime.quiesce(1, 1))["drained"]


@pytest.mark.parametrize("kind", ["raw", "lease"])
async def test_recovery_claim_prevents_duplicate_abort_and_premature_drain(runtime, kind):
    if kind == "raw":
        identifier = runtime.registry.begin_raw_work(runtime.backend_identity, "orphan")
        runtime.registry.finish_raw_work(identifier, unconfirmed=True)
        recover = runtime.recover_raw
    else:
        snap = runtime.registry.acquire("model", "orphan", None, runtime.backend_identity)
        runtime.registry.record_branches(snap.lease_id, ["orphan.branch"])
        runtime.registry.mark_abort_pending(snap.lease_id, {"orphan.branch"})
        identifier, recover = "orphan", runtime.recover_cancelled
    await runtime.quiesce(0, 0)
    entered, release = asyncio.Event(), asyncio.Event()

    async def abort(request_id):
        entered.set()
        await release.wait()

    runtime.backend.cancel = abort
    task = asyncio.create_task(recover(identifier))
    try:
        await entered.wait()
        with pytest.raises(JevError, match="already owns"):
            await recover(identifier)
        state = runtime.registry.quiescence_status(runtime.backend_identity)
        assert not state["drained"] and state["outstanding"]["recovery_operations"] == 1
        release.set()
        assert await task
        assert (await runtime.quiesce(1, 1))["drained"]
    finally:
        release.set()
        await task


async def test_startup_cannot_rejoin_after_quiesce_during_prewarm(runtime):
    newcomer = sibling(runtime)
    newcomer.backend.gate.clear()
    startup = asyncio.create_task(newcomer.start())
    try:
        await newcomer.backend.started.wait()
        assert not (await runtime.quiesce(0, 0))["drained"]
        newcomer.backend.gate.set()
        with pytest.raises(JevError, match="draining"):
            await startup
    finally:
        newcomer.backend.gate.set()
        await asyncio.gather(startup, return_exceptions=True)
        await newcomer.close()
    assert (await runtime.quiesce(1, 1))["drained"]


async def test_unknown_adapter_blocks_until_explicit_reconciliation(runtime):
    from test_adapter_lifecycle import artifact, enable_runtime

    await enable_runtime(runtime)
    ref = runtime.registry.register_adapter(artifact(), runtime.backend_identity)["reference"]
    operation, _ = runtime.registry.begin_adapter_operation(ref, runtime.backend_identity, "load")
    runtime.registry.finish_adapter_operation(ref, operation, "injected uncertainty")
    state = await runtime.quiesce(0, 0.05)
    assert not state["drained"] and state["outstanding"]["adapter_operations"] == 1
    await runtime.change_adapter(ref, "unload", recover=True)
    assert (await runtime.quiesce(1, 1))["drained"]
    with pytest.raises(JevError, match="draining"):
        await runtime.change_adapter(ref, "unload", recover=True)
