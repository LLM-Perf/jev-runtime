import asyncio

import pytest

from jev_runtime.errors import JevError
from jev_runtime.registry import Registry
from jev_runtime.runtime import Runtime
from jev_runtime.schema import DecisionRequest, TextInput


@pytest.mark.parametrize("abort_fails", [False, True])
async def test_peer_worker_cancel_uses_owner_and_retains_failed_abort(
    runtime, question, abort_fails
):
    peer = Runtime(
        runtime.backend,
        runtime.compiler,
        Registry(runtime.registry.path),
        runtime.backend_identity,
        runtime.model_id,
    )
    await peer.start()
    runtime.backend.started.clear()
    runtime.backend.gate.clear()
    runtime.backend.fail_cancel = abort_fails
    body = DecisionRequest(model="model", input=TextInput(text="refund"), questions=(question,))
    work = asyncio.create_task(runtime.decide(body, "cross-worker", tenant="a"))
    try:
        await asyncio.wait_for(runtime.backend.started.wait(), 1)
        assert not await peer.cancel("cross-worker", tenant="b")
        assert not work.done()
        if abort_fails:
            with pytest.raises(JevError) as exc:
                await peer.cancel("cross-worker", tenant="a")
            assert exc.value.code == "cancellation_unconfirmed"
            assert runtime.registry.recovery_candidates()[0]["phase"] == "abort_pending"
        else:
            assert await peer.cancel("cross-worker", tenant="a")
            assert not runtime.registry.list()["leases"]
            assert runtime.backend.cancelled
        with pytest.raises(JevError) as exc:
            await work
        assert exc.value.code == "request_cancelled"
    finally:
        runtime.backend.gate.set()
        await peer.close()


def test_completed_lease_cancellation_cannot_affect_reused_public_id(runtime):
    registry = runtime.registry
    old = registry.acquire("model", "reusable", None, runtime.backend_identity, "a")
    command = registry.request_cancel("reusable", "a", runtime.backend_identity)
    registry.release(old.lease_id)
    new = registry.acquire("model", "reusable", None, runtime.backend_identity, "a")
    assert registry.cancel_status(command) == "COMPLETED"
    assert not registry.claim_cancel(command)
    assert registry.list()["leases"][0]["id"] == new.lease_id
    new_command = registry.request_cancel("reusable", "a", runtime.backend_identity)
    assert new_command != command
    registry.release(new.lease_id)


def test_queued_cancellation_can_only_be_claimed_by_lease_owner(runtime):
    owner = runtime.registry
    peer = Registry(owner.path)
    try:
        snapshot = owner.acquire("model", "owned", None, runtime.backend_identity, "a")
        assert peer.request_cancel("owned", "b", runtime.backend_identity) is None
        assert peer.request_cancel("owned", "a", "another-engine") is None
        command = peer.request_cancel("owned", "a", runtime.backend_identity)
        assert not peer.pending_cancel_commands()
        assert not peer.claim_cancel(command)
        assert owner.claim_cancel(command)
        assert not owner.claim_cancel(command)
        owner.release(snapshot.lease_id)
        assert owner.cancel_status(command) == "RUNNING"
        owner.finish_cancel(command, "CONFIRMED")
        assert peer.cancel_status(command) == "CONFIRMED"
    finally:
        peer.close()
