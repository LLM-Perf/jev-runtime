import asyncio

import pytest

from jev_runtime.errors import JevError
from jev_runtime.registry import Registry
from jev_runtime.runtime import Runtime
from jev_runtime.schema import DecisionRequest, ExecutionOptions, TextInput


async def test_failed_abort_journal_survives_new_runtime_instance(runtime, question):
    runtime.backend.gate.clear()
    runtime.backend.fail_cancel = True
    body = DecisionRequest(
        model="model",
        request_id="durable-abort",
        input=TextInput(text="refund"),
        questions=(question,),
        execution=ExecutionOptions(timeout_ms=500),
    )
    task = asyncio.create_task(runtime.decide(body))
    await asyncio.wait_for(runtime.backend.started.wait(), 1)
    with pytest.raises(JevError):
        await task
    recovered = Runtime(
        runtime.backend,
        runtime.compiler,
        Registry(runtime.registry.path),
        runtime.backend_identity,
        runtime.model_id,
    )
    await recovered.start()
    pending = recovered.pending_cancellations()
    assert len(pending) == 1 and pending[0]["phase"] == "abort_pending"
    assert pending[0]["engine_request_ids"]
    runtime.backend.fail_cancel = False
    runtime.backend.gate.set()
    assert await recovered.recover_cancelled("durable-abort")
    assert not runtime.registry.list()["leases"]
    await recovered.close()


async def test_crash_recovery_requires_dead_owner_matching_backend_and_abort(runtime, monkeypatch):
    snapshot = runtime.registry.acquire("model", "orphan", None, runtime.backend_identity)
    runtime.registry.record_branches(snapshot.lease_id, ["jev-orphan.q.0", "jev-orphan.q.1"])
    with pytest.raises(JevError) as error:
        await runtime.recover_cancelled("orphan")
    assert error.value.code == "recovery_not_confirmed"
    assert not runtime.backend.cancelled
    monkeypatch.setattr(Registry, "_owner_status", staticmethod(lambda identity: "dead"))
    with pytest.raises(JevError) as error:
        runtime.registry.recovery_snapshot("orphan", "different-engine")
    assert error.value.code == "recovery_backend_mismatch"
    runtime.backend.fail_cancel = True
    with pytest.raises(RuntimeError):
        await runtime.recover_cancelled("orphan")
    assert len(runtime.registry.list()["leases"]) == 1
    runtime.backend.fail_cancel = False
    assert await runtime.recover_cancelled("orphan")
    assert runtime.backend.cancelled == ["jev-orphan.q.0", "jev-orphan.q.1"]
    assert not runtime.registry.list()["leases"]


async def test_prepare_cannot_reuse_failed_bundle_with_unconfirmed_work(runtime, bundle):
    next_bundle = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(next_bundle)
    runtime.registry.begin_prepare(next_bundle.reference, runtime.backend_identity)
    lease = runtime.registry.pin_preparation(
        next_bundle.reference, "canary-unknown", runtime.backend_identity
    )
    runtime.registry.record_branches(lease, ["canary-unknown.q.0"])
    runtime.registry.mark_abort_pending(lease, {"canary-unknown.q.0"})
    runtime.registry.finish_prepare(next_bundle.reference, "injected failure")
    with pytest.raises(JevError) as error:
        await runtime.prepare(next_bundle.reference)
    assert error.value.code == "bundle_in_use"
    assert await runtime.recover_cancelled("canary-unknown")
    assert (await runtime.prepare(next_bundle.reference))["state"] == "READY"
