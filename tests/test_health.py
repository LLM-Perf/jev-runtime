import asyncio

import httpx
import pytest
from pydantic import ValidationError

from jev_runtime.api import create_app
from jev_runtime.errors import JevError
from jev_runtime.health import HealthSettings
from jev_runtime.schema import DecisionRequest, TextInput


def body(question):
    return DecisionRequest(model="model", input=TextInput(text="refund"), questions=(question,))


async def test_periodic_canary_failure_opens_circuit_then_recovers(runtime, question, bundle):
    before = runtime.registry.list()["routes"]
    runtime.backend.fail_question = "canary"
    await runtime.check_health()
    assert not runtime.health.status(bundle.reference)["ready"]
    calls = len(runtime.backend.calls)
    with pytest.raises(JevError) as error:
        await runtime.decide(body(question))
    assert error.value.code == "engine_unavailable"
    assert len(runtime.backend.calls) == calls
    assert not runtime.registry.list()["leases"]
    runtime.backend.fail_question = None
    await runtime.check_health()
    assert runtime.registry.list()["routes"] == before
    assert (await runtime.decide(body(question))).status == "completed"


async def test_api_requires_every_active_bundle_but_other_bundle_keeps_serving(
    runtime, question, bundle
):
    second = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(second)
    await runtime.prepare(second.reference)
    runtime.activate("second", second.reference, 0)
    runtime.health.failed(second.reference, "engine_score_failed")
    app = create_app(instance=runtime, api_key="data", admin_key="admin")
    app.state.jev_runtime = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ready", headers={"Authorization": "Bearer data"})
        assert response.status_code == 503
        profile = await client.get("/admin/profile", headers={"Authorization": "Bearer admin"})
        assert profile.json()["health"]["bundles"][bundle.reference]["ready"]
        assert not profile.json()["health"]["bundles"][second.reference]["ready"]
        assert (await runtime.decide(body(question))).status == "completed"
        await runtime.check_health()
        assert (
            await client.get("/ready", headers={"Authorization": "Bearer data"})
        ).status_code == 200


async def test_stale_canary_rejects_traffic_and_publication(runtime, question, bundle, monkeypatch):
    successful_at = runtime.health.observations[bundle.reference].last_success
    monkeypatch.setattr("jev_runtime.health.monotonic", lambda: successful_at + 91)
    with pytest.raises(JevError) as error:
        await runtime.decide(body(question))
    assert error.value.code == "engine_unavailable"
    with pytest.raises(JevError) as error:
        runtime.activate("new", bundle.reference, 0)
    assert error.value.code == "engine_unavailable"
    assert not runtime.backend.calls
    assert not runtime.registry.list()["leases"]


async def test_unconfirmed_probe_is_not_retried_until_automatic_recovery(runtime, question, bundle):
    runtime.health.settings = HealthSettings(interval_seconds=1, timeout_seconds=0.1)
    runtime.backend.gate.clear()
    runtime.backend.fail_cancel = True
    await runtime.check_health()
    first = runtime.registry.list()["leases"]
    assert len(first) == 1
    calls = len(runtime.backend.calls)
    for _ in range(3):
        await runtime.check_health()
    assert len(runtime.backend.calls) == calls
    assert runtime.registry.list()["leases"] == first
    assert runtime.pending_cancellations()[0]["phase"] == "abort_pending"
    assert runtime.health.status(bundle.reference)["error"] == "health_cleanup_pending"
    runtime.backend.fail_cancel = False
    runtime.backend.gate.set()
    await runtime.check_health()
    assert runtime.health.status(bundle.reference)["ready"]
    assert (await runtime.decide(body(question))).status == "completed"
    assert not runtime.registry.list()["leases"]


async def test_scoring_error_opens_circuit_and_old_probe_cannot_close_it(runtime, bundle, question):
    started, release = asyncio.Event(), asyncio.Event()
    original = runtime.backend.score

    async def controlled(sequence):
        if sequence.question_id == "canary":
            started.set()
            await release.wait()
        elif sequence.question_id == question.id:
            raise RuntimeError("injected transport error")
        return await original(sequence)

    runtime.backend.score = controlled
    probe = asyncio.create_task(runtime.check_health())
    await asyncio.wait_for(started.wait(), 1)
    with pytest.raises(RuntimeError):
        await runtime.decide(body(question))
    release.set()
    await probe
    assert not runtime.health.status(bundle.reference)["ready"]
    assert bundle.reference not in runtime.registry.worker_status()[0]["prepared"]
    runtime.backend.score = original
    await runtime.check_health()
    assert (await runtime.decide(body(question))).status == "completed"


async def test_admission_queue_checks_circuit_again_before_dispatch(runtime, question, bundle):
    entered, release = asyncio.Event(), asyncio.Event()
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def queued(*args, **kwargs):
        entered.set()
        await release.wait()
        yield

    runtime.admission.wait_reserved = queued
    task = asyncio.create_task(runtime.decide(body(question)))
    await asyncio.wait_for(entered.wait(), 1)
    runtime.health.failed(bundle.reference, "engine_score_failed")
    release.set()
    with pytest.raises(JevError) as error:
        await task
    assert error.value.code == "engine_unavailable"
    assert not runtime.backend.calls and not runtime.registry.list()["leases"]


async def test_minimal_monitor_preserves_full_initial_task_validation(runtime, bundle, question):
    second = bundle.model_copy(
        update={"version": 2, "questions": (question, question.model_copy(update={"id": "q2"}))}
    )
    runtime.registry.upload(second)
    await runtime.prepare(second.reference)
    assert [call.question_id for call in runtime.backend.calls] == [question.id, "q2"]
    runtime.activate("model", second.reference, 1)
    runtime.backend.calls.clear()
    await runtime.check_health()
    assert [call.question_id for call in runtime.backend.calls] == [question.id, "canary"]


async def test_monitor_shutdown_waits_for_probe_abort(runtime, bundle):
    runtime.backend.gate.clear()
    runtime._health_task.cancel()
    await asyncio.gather(runtime._health_task, return_exceptions=True)
    runtime._health_task = asyncio.create_task(runtime.check_health())
    await asyncio.wait_for(runtime.backend.started.wait(), 1)
    await runtime.close()
    assert runtime._health_task.done()
    assert runtime.backend.cancelled
    assert not runtime._active


async def test_live_monitor_refreshes_after_an_engine_error(runtime, bundle, question):
    runtime._health_task.cancel()
    await asyncio.gather(runtime._health_task, return_exceptions=True)
    runtime.health.settings = HealthSettings(interval_seconds=1)
    runtime.backend.fail_question = question.id
    with pytest.raises(RuntimeError):
        await runtime.decide(body(question))
    runtime.backend.fail_question = None
    refreshed = asyncio.Event()
    original = runtime.check_health

    async def observed():
        await original()
        refreshed.set()

    runtime.check_health = observed
    runtime._health_task = asyncio.create_task(runtime._watch_health())
    await asyncio.wait_for(refreshed.wait(), 2)
    assert (await runtime.decide(body(question))).status == "completed"


async def test_changed_engine_profile_stays_closed_without_scoring(runtime, bundle):
    original = runtime.backend.probe

    async def changed():
        return (await original()).model_copy(update={"version": "changed"})

    runtime.backend.probe = changed
    await runtime.check_health()
    assert runtime.health.status(bundle.reference)["error"] == "engine_profile_changed"
    assert not runtime.backend.calls
    assert not runtime.registry.list()["leases"]


async def test_invalid_engine_scores_trip_circuit(runtime, bundle, question):
    from jev_runtime.backends.base import ScoreResult

    async def invalid(sequence):
        return ScoreResult(sequence.request_id, (float("nan"), float("nan")))

    runtime.backend.score = invalid
    with pytest.raises(JevError, match="non-finite"):
        await runtime.decide(body(question))
    assert runtime.health.status(bundle.reference)["error"] == "engine_score_contract"


def test_health_interval_timeout_and_expiry_are_consistent():
    with pytest.raises(ValidationError, match="max age"):
        HealthSettings(interval_seconds=30, timeout_seconds=10, max_age_seconds=40)
    with pytest.raises(ValidationError, match="max age"):
        HealthSettings(startup_grace_seconds=60, timeout_seconds=10, max_age_seconds=70)


async def test_monitor_starts_explicitly_and_honors_startup_grace(tmp_path, compiler, monkeypatch):
    from conftest import ControlledEngine

    from jev_runtime.registry import Registry
    from jev_runtime.runtime import Runtime

    instance = Runtime(
        ControlledEngine(),
        compiler,
        Registry(tmp_path / "monitor.db"),
        "fixture:monitor",
        "fixture",
        health_settings=HealthSettings(startup_grace_seconds=2),
    )
    await instance.start()
    assert instance._health_task is None

    delays = []

    async def wait_for(awaitable, wait_seconds):
        awaitable.close()
        delays.append(wait_seconds)
        instance._health_stop.set()

    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    instance.start_health_monitor()
    first = instance._health_task
    instance.start_health_monitor()
    assert instance._health_task is first
    await first
    assert delays == [2]
    await instance.close()


async def test_inactive_prepared_versions_stay_fresh_across_expiry(runtime, bundle, monkeypatch):
    clock = [runtime.health.observations[bundle.reference].last_success]
    monkeypatch.setattr("jev_runtime.health.monotonic", lambda: clock[0])
    second = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(second)
    await runtime.prepare(second.reference)
    # Polls always observe v1. v2 must still be probed despite never being active.
    for _ in range(4):
        clock[0] += 30
        await runtime.check_health()
        assert runtime.health.status(second.reference)["ready"]
    assert runtime.activate("model", second.reference, 1)["generation"] == 2
    assert bundle.reference in runtime.health_references()  # DRAINING is eligible for rollback.


async def test_monitor_skips_unprepared_and_retired_versions(runtime, bundle):
    second = bundle.model_copy(update={"version": 2})
    third = bundle.model_copy(update={"version": 3})
    runtime.registry.upload(second)
    runtime.registry.upload(third)
    await runtime.prepare(second.reference)
    runtime.registry.retire(second.reference)
    assert runtime.health_references() == [bundle.reference]
    runtime.backend.calls.clear()
    await runtime.check_health()
    assert len(runtime.backend.calls) == 1
    assert runtime.registry.inspect(second.reference)["state"] == "RETIRED"
    assert runtime.registry.inspect(third.reference)["state"] == "VALIDATED"


async def test_monitor_snapshot_cannot_resurrect_concurrently_retired_standby(runtime, bundle):
    second = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(second)
    await runtime.prepare(second.reference)
    entered, release = asyncio.Event(), asyncio.Event()
    original = runtime.backend.probe

    async def blocked():
        entered.set()
        await release.wait()
        return await original()

    runtime.backend.probe = blocked
    runtime.backend.calls.clear()
    task = asyncio.create_task(runtime.check_health())
    await asyncio.wait_for(entered.wait(), 1)
    runtime.registry.retire(second.reference)
    release.set()
    await task
    assert runtime.registry.inspect(second.reference)["state"] == "RETIRED"
    assert len(runtime.backend.calls) == 1
    assert not runtime.registry.list()["leases"]
