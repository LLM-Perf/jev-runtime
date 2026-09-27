from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

from fastapi import FastAPI
from jev_vllm.endpoint import JevEndpointPlugin


async def test_vllm_plugin_closes_runtime_and_preserves_host_lifespan(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    events = []

    @asynccontextmanager
    async def host_lifespan(app):
        events.append("host-start")
        yield
        events.append("host-stop")

    app = FastAPI(lifespan=host_lifespan)
    plugin = JevEndpointPlugin()
    plugin.attach_router(app)
    context = app.router.lifespan_context
    plugin.attach_router(app)
    assert context is app.router.lifespan_context
    runtime = AsyncMock()
    app.state.jev_runtime = runtime
    async with app.router.lifespan_context(app):
        assert events == ["host-start"]
        runtime.close.assert_not_awaited()
    assert events == ["host-start", "host-stop"]
    runtime.close.assert_awaited_once()
    assert app.state.jev_runtime is None
