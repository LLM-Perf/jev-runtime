import asyncio

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from jev_runtime.api import create_app
from jev_runtime.errors import JevError
from jev_runtime.performance import Measurement, summarize_cohort
from jev_runtime.schema import DecisionRequest
from jev_runtime.telemetry import STAGES, DecisionTrace, parse_timing_header


def body(question, **values):
    return {
        "model": "model",
        "input": {"text": "private example"},
        "questions": [question.model_dump()],
        **values,
    }


def metric(text, name, labels):
    return next(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name and sample.labels == labels
    )


def test_trace_partitions_phases_and_separates_overlapping_work(monkeypatch):
    ticks = iter([10, 11, 12, 13, 15, 16, 20])
    monkeypatch.setattr("jev_runtime.telemetry.time.perf_counter", lambda: next(ticks))
    trace = DecisionTrace()
    trace.switch("pin")
    trace.switch("execute")
    with trace.measure_work("engine_call"):
        with trace.measure_work("engine_call"):
            pass
    trace.finish()
    assert trace.seconds == {"pin": 1, "execute": 9, "total": 10}
    assert trace.work["engine_call"] == [2, 4]
    assert parse_timing_header(trace.header()) == {"pin": 1000, "execute": 9000, "total": 10000}
    trace.finish()  # Recording cleanup twice cannot create duplicate observations.
    assert trace.seconds["total"] == 10


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "jev_total;dur=nan",
        "jev_total;dur=-1",
        "jev_pin;dur=1, jev_total;dur=2",
        "jev_pin;dur=1, jev_pin;dur=1, jev_total;dur=2",
        "jev_unknown;dur=1, jev_total;dur=1",
        "other;dur=1",
        "jev_pin;dur=1",
    ],
)
def test_bad_timing_headers_are_not_silently_accepted(value):
    with pytest.raises(ValueError):
        parse_timing_header(value)


async def test_success_and_pre_dispatch_error_have_bounded_observations(runtime, question):
    app = create_app(instance=runtime)
    app.state.jev_runtime = runtime
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as c:
        normal = await c.post("/v1/decisions", json=body(question))
        assert normal.status_code == 200 and "Server-Timing" not in normal.headers
        success = await c.post("/v1/decisions", json=body(question), headers={"X-Jev-Timing": "1"})
        timing = parse_timing_header(success.headers["Server-Timing"])
        assert set(timing) == set(STAGES)
        assert sum(v for k, v in timing.items() if k != "total") == pytest.approx(timing["total"])
        failed = await c.post(
            "/v1/decisions", json=body(question, bundle="test@99"), headers={"X-Jev-Timing": "1"}
        )
        assert failed.status_code == 409
        assert set(parse_timing_header(failed.headers["Server-Timing"])) == {
            "pin",
            "release",
            "total",
        }
        text = (await c.get("/metrics")).text
        assert "private example" not in text and question.id not in text
        assert (
            metric(
                text, "jev_runtime_stage_seconds_count", {"stage": "total", "outcome": "completed"}
            )
            == 2
        )
        assert (
            metric(text, "jev_runtime_stage_seconds_count", {"stage": "total", "outcome": "error"})
            == 1
        )
        assert metric(text, "jev_request_errors_total", {"category": "validation"}) == 1
        assert (
            metric(text, "jev_observed_tokens_total", {"quantity": "engine_completion_tokens"}) == 0
        )
        assert (
            metric(text, "jev_token_observations_total", {"quantity": "engine_completion_tokens"})
            == 2
        )
        assert metric(text, "jev_shared_admission", {"quantity": "requests"}) == 0


async def test_partial_metrics_preserve_unknown_usage(runtime, question):
    runtime.backend.fail_question = "bad"
    app = create_app(instance=runtime)
    app.state.jev_runtime = runtime
    payload = body(question, execution={"allow_partial": True})
    payload["questions"].append({**question.model_dump(), "id": "bad"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as c:
        response = await c.post("/v1/decisions", json=payload, headers={"X-Jev-Timing": "1"})
        assert response.json()["status"] == "partial"
        parse_timing_header(response.headers["Server-Timing"])
        text = (await c.get("/metrics")).text
        assert metric(text, "jev_questions_total", {"outcome": "failed"}) == 1
        assert metric(text, "jev_questions_total", {"outcome": "answered"}) == 1
        assert 'quantity="engine_prompt_tokens"' not in text
        assert metric(text, "jev_branch_work_seconds_count", {"kind": "engine_call"}) == 2
        assert metric(text, "jev_branch_work_seconds_count", {"kind": "abort"}) == 1


@pytest.mark.parametrize("runtime", [{"max_requests": 1}], indirect=True)
async def test_cancellation_while_queued_has_no_engine_time(runtime, question):
    runtime.backend.gate.clear()
    request = DecisionRequest.model_validate(body(question))
    first = asyncio.create_task(runtime.decide(request, "first"))
    await runtime.backend.started.wait()
    trace = DecisionTrace()
    second = asyncio.create_task(runtime.decide(request, "second", trace=trace))
    async with asyncio.timeout(2):
        while runtime.admission.snapshot()["queued_requests"] != 1:  # noqa: ASYNC110 - shared DB
            await asyncio.sleep(0.001)
        assert await runtime.cancel("second")
    with pytest.raises(JevError, match="cancelled"):
        await second
    assert set(trace.seconds) == {"pin", "compile", "journal", "queue", "release", "total"}
    assert trace.work == {}
    assert runtime.admission.snapshot()["queued_requests"] == 0
    assert len(runtime.registry.list()["leases"]) == 1
    runtime.backend.gate.set()
    await first
    assert not runtime.registry.list()["leases"]


async def test_release_failure_retains_journal_but_finishes_trace_and_local_ownership(
    runtime, question, monkeypatch
):
    trace = DecisionTrace()
    original = runtime.registry.release

    def fail(_):
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(runtime.registry, "release", fail)
    with pytest.raises(RuntimeError, match="disk unavailable"):
        await runtime.decide(
            DecisionRequest.model_validate(body(question)), "failed-release", trace=trace
        )
    assert "total" in trace.seconds and "release" in trace.seconds
    assert (
        "failed-release" not in runtime._active and "failed-release" not in runtime._active_leases
    )
    assert runtime.request_progress("failed-release")["request"] is None
    leases = runtime.registry.list()["leases"]
    assert len(leases) == 1
    monkeypatch.setattr(runtime.registry, "release", original)
    original(leases[0]["id"])


def test_cohort_timing_denominators_include_errors_and_missing_stages():
    base = dict(
        started_ms=0, finished_ms=10, questions=1, scoring_sequences=1, logical_prompt_tokens=10
    )
    rows = [
        Measurement(
            request_id="ok",
            **base,
            outcome="completed",
            successful_questions=1,
            runtime_timings_ms={"pin": 1, "execute": 3, "total": 4},
        ),
        Measurement(
            request_id="error",
            **base,
            outcome="failed",
            successful_questions=0,
            runtime_timings_ms={"pin": 2, "total": 2},
        ),
        Measurement(request_id="old-server", **base, outcome="completed", successful_questions=1),
    ]
    all_rows = summarize_cohort(rows, 0.01, 0.01)["runtime_timings"]["all_attempts"]
    assert all_rows["request_denominator"] == 3
    assert all_rows["stages"]["total"]["observed_requests"] == 2
    assert all_rows["stages"]["total"]["mean_ms"] == 3
    assert all_rows["stages"]["execute"]["observed_requests"] == 1
    assert all_rows["stages"]["compile"]["mean_ms"] is None


def test_live_work_counters_distinguish_completion_failure_and_overlap():
    trace = DecisionTrace()
    trace.switch("execute")
    with trace.measure_work("engine_call"):
        with trace.measure_work("engine_call"):
            current = trace.progress()["work"]["engine_call"]
            assert current == {"started": 2, "active": 2, "succeeded": 0, "failed_or_cancelled": 0}
        try:
            with trace.measure_work("engine_call"):
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            pass
        current = trace.progress()["work"]["engine_call"]
        assert current == {"started": 3, "active": 1, "succeeded": 1, "failed_or_cancelled": 1}
    trace.finish()
    snapshot = trace.progress()
    assert snapshot["stage"] is None and snapshot["elapsed_seconds"] == trace.seconds["total"]
    assert snapshot["work"]["engine_call"] == {
        "started": 3,
        "active": 0,
        "succeeded": 2,
        "failed_or_cancelled": 1,
    }


async def test_admin_progress_pins_bundle_and_counts_real_calls_without_input(
    runtime, bundle, question
):
    entered, release = asyncio.Event(), asyncio.Event()
    original = runtime.backend.score

    async def gated(request):
        if request.question_id == "blocked":
            entered.set()
            await release.wait()
        return await original(request)

    runtime.backend.score = gated
    app = create_app(instance=runtime, api_key="api-key", admin_key="admin-key")
    app.state.jev_runtime = runtime
    trace = DecisionTrace()
    request = DecisionRequest.model_validate(
        {
            **body(question, request_id="live-progress"),
            "questions": [question.model_dump(), {**question.model_dump(), "id": "blocked"}],
        }
    )
    work = asyncio.create_task(runtime.decide(request, trace=trace))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            path = "/admin/requests/live-progress/progress"
            assert (await c.get(path)).status_code == 401
            assert (
                await c.get(path, headers={"Authorization": "Bearer api-key"})
            ).status_code == 401
            current = (await c.get(path, headers={"Authorization": "Bearer admin-key"})).json()
            assert (
                current["scope"] == "local_worker"
                and current["worker_id"] == runtime.registry.owner
            )
            observed = current["request"]
            assert (
                observed["bundle"] == bundle.reference
                and observed["bundle_digest"] == bundle.digest
            )
            assert observed["scoring_sequences"] == 2 and observed["stage"] == "execute"
            assert observed["work"]["engine_call"] == {
                "started": 2,
                "active": 1,
                "succeeded": 1,
                "failed_or_cancelled": 0,
            }
            assert "private example" not in str(current) and question.instruction not in str(
                current
            )
            next_bundle = bundle.model_copy(update={"version": 2})
            runtime.registry.upload(next_bundle)
            await runtime.prepare(next_bundle.reference)
            runtime.activate("model", next_bundle.reference, 1)
            pinned = runtime.request_progress("live-progress")["request"]
            assert pinned["bundle"] == bundle.reference and pinned["generation"] == 1
            assert pinned["lease_id"] == observed["lease_id"]
            result = await c.post(
                "/v1/requests/live-progress/cancel", headers={"Authorization": "Bearer api-key"}
            )
            assert result.json() == {"cancelled": True}
            assert result.headers["X-Jev-Worker"] == runtime.registry.owner
            with pytest.raises(JevError, match="cancelled"):
                await work
            assert runtime.request_progress("live-progress")["request"] is None
            assert trace.progress()["work"]["engine_call"]["failed_or_cancelled"] == 1
            assert trace.progress()["work"]["abort"]["succeeded"] == 1
            assert not runtime.registry.list()["leases"]
    finally:
        release.set()
        if not work.done():
            await runtime.cancel("live-progress")
        await asyncio.gather(work, return_exceptions=True)
