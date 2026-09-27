import asyncio

import pytest

from jev_runtime.config import Settings, bootstrap
from jev_runtime.errors import JevError
from jev_runtime.registry import Registry
from jev_runtime.runtime import Runtime
from jev_runtime.schema import DecisionRequest, TextInput


def replica(runtime):
    return Runtime(
        type(runtime.backend)(),
        runtime.compiler,
        Registry(runtime.registry.path),
        runtime.backend_identity,
        runtime.model_id,
    )


async def test_new_worker_revalidates_persisted_active_version(runtime, bundle):
    worker = replica(runtime)
    await worker.start()
    assert worker.is_prepared(bundle.reference)
    assert worker.backend.calls and worker.backend.calls[0].question_id == "canary"
    assert not runtime.registry.list()["leases"]
    await worker.close()


async def test_failed_new_worker_canary_preserves_existing_route(runtime, bundle):
    worker = replica(runtime)
    worker.backend.fail_question = "canary"
    with pytest.raises(RuntimeError):
        await worker.start()
    assert not worker.is_prepared(bundle.reference)
    assert runtime.registry.list()["routes"][0]["ref"] == bundle.reference
    assert runtime.registry.inspect(bundle.reference)["state"] == "ACTIVE"
    assert runtime.is_prepared(bundle.reference)
    await worker.close()


async def test_activation_waits_for_all_serving_workers(runtime, bundle, question):
    worker = replica(runtime)
    await worker.start()
    second = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(second)
    await runtime.prepare(second.reference)
    body = DecisionRequest(model="model", input=TextInput(text="refund"), questions=(question,))
    with pytest.raises(JevError) as error:
        runtime.activate("model", second.reference, 1)
    assert error.value.code == "replicas_not_ready"
    assert (await worker.decide(body)).bundle == bundle.reference
    assert (await runtime.decide(body)).bundle == bundle.reference
    await worker.prepare(second.reference)
    runtime.activate("model", second.reference, 1)
    assert (await runtime.decide(body)).bundle == second.reference
    assert (await worker.decide(body)).bundle == second.reference
    assert runtime.registry.list()["routes"][0]["generation"] == 2
    await worker.close()


async def test_stopped_worker_does_not_block_publication(runtime, bundle):
    worker = replica(runtime)
    await worker.start()
    second = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(second)
    await runtime.prepare(second.reference)
    await worker.close()
    assert runtime.activate("model", second.reference, 1)["generation"] == 2


async def test_verified_dead_worker_does_not_block_publication(runtime, bundle, monkeypatch):
    worker = replica(runtime)
    await worker.start()
    second = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(second)
    await runtime.prepare(second.reference)
    monkeypatch.setattr(Registry, "_owner_status", staticmethod(lambda identity: "dead"))
    assert runtime.activate("model", second.reference, 1)["generation"] == 2
    await worker.close()


async def test_concurrent_worker_bootstrap_prepares_each_worker(runtime, tmp_path):
    path = tmp_path / "shared-boot.db"
    first = Runtime(
        type(runtime.backend)(), runtime.compiler, Registry(path), "fixture:boot", "fixture"
    )
    second = Runtime(
        type(runtime.backend)(), runtime.compiler, Registry(path), "fixture:boot", "fixture"
    )
    await first.start()
    await second.start()
    first.backend.gate.clear()
    settings = Settings(
        backend="sglang",
        model_id="fixture",
        model_revision="a" * 40,
        bootstrap_alias="model",
        bootstrap_bundle_id="boot",
    )
    a = asyncio.create_task(bootstrap(first, settings))
    await asyncio.wait_for(first.backend.started.wait(), 1)
    b = asyncio.create_task(bootstrap(second, settings))
    first.backend.gate.set()
    await asyncio.wait_for(asyncio.gather(a, b), 2)
    assert first.is_prepared("boot@1") and second.is_prepared("boot@1")
    assert first.backend.calls and second.backend.calls
    assert first.registry.list()["routes"] == [{"alias": "model", "ref": "boot@1", "generation": 1}]
    assert not first.registry.list()["leases"]
    await first.close()
    await second.close()
