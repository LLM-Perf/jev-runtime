import httpx

from jev_runtime.api import create_app


async def test_liveness_does_not_score_or_restart_on_unready_gpu(runtime):
    app = create_app(instance=runtime, api_key="data")
    app.state.jev_runtime = runtime
    runtime.health.failed("test@1", "engine_score_failed")
    before = len(runtime.backend.calls)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/live")).status_code == 401
        headers = {"Authorization": "Bearer data"}
        assert (await client.get("/live", headers=headers)).json()["live"]
        assert (await client.get("/ready", headers=headers)).status_code == 503
    assert len(runtime.backend.calls) == before


async def test_metrics_expose_retained_work_and_worker_identity(runtime):
    lease = runtime.registry.acquire("model", "retained", None, runtime.backend_identity)
    app = create_app(instance=runtime, api_key="data")
    app.state.jev_runtime = runtime
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/metrics", headers={"Authorization": "Bearer data"})
        assert response.status_code == 200
        assert response.headers["X-Jev-Worker"] == runtime.registry.owner
        assert 'jev_registry_state{quantity="leases"} 1.0' in response.text
        assert 'jev_registry_state{quantity="database_bytes"}' in response.text
        assert 'jev_registry_state{quantity="oldest_lease_seconds"}' in response.text
        assert 'jev_worker_health{quantity="active_bundles"} 1.0' in response.text
        assert "retained" not in response.text
    finally:
        runtime.registry.release(lease.lease_id)
