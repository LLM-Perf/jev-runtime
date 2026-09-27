import asyncio

import pytest

from jev_runtime.errors import JevError
from jev_runtime.registry import Registry
from jev_runtime.schema import Bundle, DecisionRequest, ExecutionOptions, Policy, TextInput


def request(question, **kwargs):
    return DecisionRequest(
        model="model", input=TextInput(text="refund"), questions=(question,), **kwargs
    )


async def test_switch_pins_old_request_until_drain(runtime, bundle, question):
    engine = runtime.backend
    engine.gate.clear()
    work = asyncio.create_task(runtime.decide(request(question), "old"))
    await engine.started.wait()
    second = bundle.model_copy(update={"version": 2, "policy": Policy(min_probability=0.99)})
    runtime.registry.upload(second)
    runtime.registry.begin_prepare(second.reference, runtime.backend_identity)
    runtime.registry.finish_prepare(second.reference)
    runtime.registry.activate("model", second.reference, 1)
    with pytest.raises(JevError, match="in-flight"):
        runtime.registry.retire(bundle.reference)
    engine.gate.set()
    old = await work
    new = await runtime.decide(request(question))
    assert old.bundle == "test@1" and not old.answers[question.id].abstained
    assert new.bundle == "test@2" and new.answers[question.id].abstained
    assert runtime.registry.retire(bundle.reference)["state"] == "RETIRED"


async def test_timeout_cancels_branches_and_releases_lease(runtime, question):
    runtime.backend.gate.clear()
    work = asyncio.create_task(
        runtime.decide(request(question, execution=ExecutionOptions(timeout_ms=500)))
    )
    await asyncio.wait_for(runtime.backend.started.wait(), timeout=1)
    with pytest.raises(JevError) as error:
        await work
    assert error.value.code == "deadline_exceeded"
    assert len(runtime.backend.cancelled) == 1
    assert runtime.registry.list()["leases"] == []
    assert runtime.admission.tokens == 0


async def test_failed_abort_keeps_lease_until_confirmed(runtime, question):
    runtime.backend.gate.clear()
    runtime.backend.fail_cancel = True
    work = asyncio.create_task(
        runtime.decide(request(question, execution=ExecutionOptions(timeout_ms=500)), "cancel-me")
    )
    await asyncio.wait_for(runtime.backend.started.wait(), timeout=1)
    with pytest.raises(JevError):
        await work
    assert len(runtime.registry.list()["leases"]) == 1
    runtime.backend.fail_cancel = False
    assert await runtime.recover_cancelled("cancel-me")
    assert not runtime.registry.list()["leases"]


async def test_compare_and_swap_across_registry_instances(runtime, bundle):
    another = Registry(runtime.registry.path)
    with pytest.raises(JevError) as error:
        another.activate("model", bundle.reference, 0)
    assert error.value.code == "generation_conflict"


async def test_bundle_version_is_immutable(runtime, bundle):
    changed = bundle.model_copy(update={"policy": Policy(min_probability=0.9)})
    with pytest.raises(JevError) as error:
        runtime.registry.upload(changed)
    assert error.value.code == "immutable_version"


async def test_failed_preparation_preserves_old_route(runtime, bundle):
    invalid = Bundle.model_validate(
        {
            **bundle.model_dump(),
            "version": 2,
            "model": {**bundle.model.model_dump(), "tokenizer_digest": "different"},
        }
    )
    runtime.registry.upload(invalid)
    with pytest.raises(JevError):
        await runtime.prepare(invalid.reference)
    assert runtime.registry.inspect(invalid.reference)["state"] == "FAILED"
    assert runtime.registry.list()["routes"][0]["ref"] == bundle.reference


async def test_disable_stops_new_requests(runtime, question):
    runtime.registry.disable("model", 1)
    with pytest.raises(JevError) as error:
        await runtime.decide(request(question))
    assert error.value.code == "route_unavailable"


async def test_partial_failure_is_not_reported_success(runtime, question):
    other = question.model_copy(update={"id": "broken"})
    runtime.backend.fail_question = "broken"
    response = await runtime.decide(
        DecisionRequest(
            model="model",
            input=TextInput(text="x"),
            questions=(question, other),
            execution=ExecutionOptions(allow_partial=True),
        )
    )
    assert response.status == "partial"
    assert response.answers["broken"].status == "failed"
    assert response.usage.successful_questions == 1
    assert response.usage.engine_prompt_tokens is None


async def test_1000_atomic_config_switches(runtime, bundle):
    second = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(second)
    await runtime.prepare(second.reference)
    for generation in range(1, 1001):
        target = second if generation % 2 else bundle
        result = runtime.registry.activate("model", target.reference, generation)
        assert result["generation"] == generation + 1
    assert runtime.registry.list()["leases"] == []


@pytest.mark.parametrize("abort_fails", [False, True])
async def test_prepare_cancellation_pins_canary_until_abort_confirmed(runtime, bundle, abort_fails):
    next_bundle = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(next_bundle)
    runtime.backend.gate.clear()
    runtime.backend.fail_cancel = abort_fails
    task = asyncio.create_task(runtime.prepare(next_bundle.reference))
    await asyncio.wait_for(runtime.backend.started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.registry.inspect(next_bundle.reference)["state"] == "FAILED"
    leases = runtime.registry.list()["leases"]
    if abort_fails:
        assert len(leases) == 1
        with pytest.raises(JevError, match="in-flight"):
            runtime.registry.retire(next_bundle.reference)
        runtime.backend.fail_cancel = False
        assert await runtime.recover_cancelled(leases[0]["request_id"])
    else:
        assert not leases and runtime.backend.cancelled
    assert runtime.registry.retire(next_bundle.reference)["state"] == "RETIRED"


async def test_caller_id_cancellation_confirms_abort_and_prevents_cross_worker_duplicates(
    runtime, question
):
    runtime.backend.gate.clear()
    body = request(question, request_id="client-selected-id")
    task = asyncio.create_task(runtime.decide(body))
    await asyncio.wait_for(runtime.backend.started.wait(), 1)
    another = Registry(runtime.registry.path)
    with pytest.raises(JevError) as error:
        another.acquire("model", "client-selected-id", None, runtime.backend_identity)
    assert error.value.code == "duplicate_request"
    assert await runtime.cancel("client-selected-id")
    with pytest.raises(JevError) as error:
        await task
    assert error.value.code == "request_cancelled"
    assert not runtime.registry.list()["leases"]
    assert runtime.backend.cancelled


async def test_caller_cancellation_does_not_claim_confirmed_on_failed_abort(runtime, question):
    runtime.backend.gate.clear()
    runtime.backend.fail_cancel = True
    task = asyncio.create_task(runtime.decide(request(question, request_id="uncertain-abort")))
    await asyncio.wait_for(runtime.backend.started.wait(), 1)
    with pytest.raises(JevError) as error:
        await runtime.cancel("uncertain-abort")
    assert error.value.code == "cancellation_unconfirmed"
    with pytest.raises(JevError):
        await task
    assert len(runtime.registry.list()["leases"]) == 1
    runtime.backend.fail_cancel = False
    assert await runtime.recover_cancelled("uncertain-abort")
