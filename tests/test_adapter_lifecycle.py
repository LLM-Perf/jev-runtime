import asyncio
from types import SimpleNamespace

import pytest

from jev_runtime.adapters import AdapterArtifact
from jev_runtime.errors import JevError
from jev_runtime.registry import Registry
from jev_runtime.schema import DecisionRequest, TextInput


def artifact():
    # Registry state fixture, not an executable or GPU-certified adapter.
    return AdapterArtifact(
        id="task",
        revision="sha256:" + "c" * 64,
        base_model_id="fixture",
        base_model_revision="a" * 40,
        path="/fixture",
        rank=2,
        targets=("q_proj",),
        files={},
        bytes=16,
    )


def adapted(bundle, version=2):
    return bundle.model_copy(
        update={
            "version": version,
            "model": bundle.model.model_copy(
                update={"adapter_id": "task", "adapter_revision": artifact().revision}
            ),
        }
    )


def ready_adapter(registry, backend):
    row = registry.register_adapter(artifact(), backend)
    operation, binding = registry.begin_adapter_operation(row["reference"], backend, "load")
    registry.finish_adapter_operation(row["reference"], operation)
    return binding


async def enable_runtime(runtime):
    runtime.capabilities = runtime.capabilities.model_copy(update={"lora": True})
    runtime.adapter_store = SimpleNamespace(verify=lambda value: None)

    async def confirmed(binding):
        pass

    runtime.backend.load_adapter = runtime.backend.unload_adapter = confirmed


async def test_all_bundle_routes_and_leases_block_unload(runtime, bundle, question):
    await enable_runtime(runtime)
    binding = ready_adapter(runtime.registry, runtime.backend_identity)
    first, second = adapted(bundle), adapted(bundle, 3)
    for item in (first, second):
        runtime.registry.upload(item)
        await runtime.prepare(item.reference)
    runtime.activate("model", first.reference, 1)
    runtime.activate("another", second.reference, 0)
    runtime.backend.started.clear()
    runtime.backend.gate.clear()
    work = asyncio.create_task(
        runtime.decide(
            DecisionRequest(model="model", input=TextInput(text="x"), questions=(question,))
        )
    )
    await runtime.backend.started.wait()
    runtime.registry.disable("model", 2)
    with pytest.raises(JevError) as error:
        await runtime.change_adapter(binding.artifact.reference, "unload")
    assert error.value.code == "adapter_in_use"  # another alias still receives traffic
    runtime.registry.disable("another", 1)
    with pytest.raises(JevError) as error:
        await runtime.change_adapter(binding.artifact.reference, "unload")
    assert error.value.code == "adapter_in_use"  # detached old request still owns its weights
    runtime.backend.gate.set()
    assert (await work).bundle == first.reference
    assert runtime.backend.calls[-1].adapter_id == binding.artifact.reference
    assert (await runtime.change_adapter(binding.artifact.reference, "unload"))[
        "state"
    ] == "UNLOADED"
    assert all(
        runtime.registry.inspect(item.reference)["state"] == "VALIDATED" for item in (first, second)
    )
    assert not runtime.is_prepared(first.reference)
    with pytest.raises(JevError):
        await runtime.prepare(first.reference)


async def test_unload_transaction_excludes_new_prepare_and_activation(runtime, bundle):
    await enable_runtime(runtime)
    binding = ready_adapter(runtime.registry, runtime.backend_identity)
    item = adapted(bundle)
    runtime.registry.upload(item)
    await runtime.prepare(item.reference)
    operation, _ = runtime.registry.begin_adapter_operation(
        binding.artifact.reference, runtime.backend_identity, "unload"
    )
    peer = Registry(runtime.registry.path)
    try:
        with pytest.raises(JevError) as error:
            peer.activate("other", item.reference, 0)
        assert error.value.code == "adapter_not_ready"
        with pytest.raises(JevError):
            peer.pin_revalidation(item.reference, "race", runtime.backend_identity)
        with pytest.raises(JevError):
            peer.finish_adapter_operation(binding.artifact.reference, operation)
    finally:
        peer.close()
    runtime.registry.finish_adapter_operation(binding.artifact.reference, operation)


async def test_cancelled_load_is_quarantined_until_explicit_unload(runtime):
    await enable_runtime(runtime)
    ref = runtime.registry.register_adapter(artifact(), runtime.backend_identity)["reference"]
    started = asyncio.Event()

    async def stalled(binding):
        started.set()
        await asyncio.Event().wait()

    runtime.backend.load_adapter = stalled
    work = asyncio.create_task(runtime.change_adapter(ref, "load"))
    await started.wait()
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    assert runtime.registry.inspect_adapter(ref)["state"] == "UNKNOWN"
    with pytest.raises(JevError) as error:
        await runtime.change_adapter(ref, "load")
    assert error.value.code == "adapter_state"
    assert (await runtime.change_adapter(ref, "unload"))["state"] == "UNLOADED"
    await enable_runtime(runtime)
    assert (await runtime.change_adapter(ref, "load"))["state"] == "READY"


async def test_restart_pauses_routes_preserves_leases_and_requires_canary(runtime, bundle):
    await enable_runtime(runtime)
    binding = ready_adapter(runtime.registry, runtime.backend_identity)
    item = adapted(bundle)
    runtime.registry.upload(item)
    await runtime.prepare(item.reference)
    runtime.activate("adapter", item.reference, 0)
    snapshot = runtime.registry.acquire("adapter", "unfinished", None, runtime.backend_identity)
    peer = Registry(runtime.registry.path)
    peer.start_worker(runtime.backend_identity)
    try:
        with pytest.raises(JevError) as error:
            peer.start_adapter_session(runtime.backend_identity)
        assert error.value.code == "adapter_coordinator_active"
        runtime.registry.stop_worker()
        assert peer.start_adapter_session(runtime.backend_identity) == [binding.artifact.reference]
        routes = {row["alias"]: row for row in peer.list()["routes"]}
        assert routes["adapter"]["ref"] is None and routes["adapter"]["generation"] == 2
        assert routes["model"]["ref"] == bundle.reference
        assert peer.has_lease(snapshot.lease_id)
        with pytest.raises(JevError) as error:
            peer.begin_adapter_operation(
                binding.artifact.reference, runtime.backend_identity, "unload"
            )
        assert error.value.code == "adapter_in_use"
        runtime.registry.release(snapshot.lease_id)
        operation, _ = peer.begin_adapter_operation(
            binding.artifact.reference, runtime.backend_identity, "unload"
        )
        peer.finish_adapter_operation(binding.artifact.reference, operation)
    finally:
        peer.stop_worker()
        peer.close()


def test_adapter_base_binding_and_immutable_registration(tmp_path, bundle):
    registry = Registry(tmp_path / "registry.db")
    try:
        item = adapted(bundle)
        with pytest.raises(JevError) as error:
            registry.upload(item)
        assert error.value.code == "adapter_not_found"
        original = registry.register_adapter(artifact(), "engine")
        assert registry.register_adapter(artifact(), "engine") == original
        with pytest.raises(JevError) as error:
            registry.register_adapter(artifact().model_copy(update={"rank": 4}), "engine")
        assert error.value.code == "immutable_adapter"
        with pytest.raises(JevError) as error:
            registry.upload(
                item.model_copy(
                    update={"model": item.model.model_copy(update={"revision": "b" * 40})}
                )
            )
        assert error.value.code == "adapter_base_mismatch"
    finally:
        registry.close()


def test_adapter_ids_fit_int32_and_collisions_never_alias_weights(tmp_path, monkeypatch):
    from jev_runtime import registry as module

    registry = Registry(tmp_path / "registry.db")
    try:
        first = registry.register_adapter(artifact(), "engine")
        identifier = first["binding"]["engine_id"]
        assert 1 <= identifier <= 2**31 - 1
        monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(int=identifier))
        with pytest.raises(JevError) as error:
            registry.register_adapter(artifact().model_copy(update={"id": "other"}), "engine")
        assert error.value.code == "adapter_id_exhausted"
        assert len(registry.list_adapters()) == 1
    finally:
        registry.close()
