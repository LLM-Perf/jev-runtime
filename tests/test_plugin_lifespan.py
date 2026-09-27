from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from jev_sglang.plugin import _configure_uvicorn_workers
from jev_vllm.endpoint import JevEndpointPlugin


async def test_vllm_plugin_closes_runtime_and_preserves_host_lifespan(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    events = []

    @asynccontextmanager
    async def host_lifespan(app):
        events.append("host-start")
        yield
        assert app.state.jev_runtime is None
        events.append("host-stop")

    app = FastAPI(lifespan=host_lifespan)
    plugin = JevEndpointPlugin()
    plugin.attach_router(app)
    context = app.router.lifespan_context
    plugin.attach_router(app)
    assert context is app.router.lifespan_context
    runtime = AsyncMock()
    runtime.close.side_effect = lambda: events.append("runtime-close")
    app.state.jev_runtime = runtime
    async with app.router.lifespan_context(app):
        assert events == ["host-start"]
        runtime.close.assert_not_awaited()
    assert events == ["host-start", "runtime-close", "host-stop"]
    runtime.close.assert_awaited_once()
    assert app.state.jev_runtime is None


async def test_vllm_host_may_delete_state_after_plugin_drains(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    runtime = AsyncMock()

    @asynccontextmanager
    async def host_lifespan(app):
        try:
            yield
        finally:
            runtime.close.assert_awaited_once()
            del app.state

    app = FastAPI(lifespan=host_lifespan)
    app.state.jev_runtime = runtime
    JevEndpointPlugin().attach_router(app)
    async with app.router.lifespan_context(app):
        pass


async def test_vllm_plugin_uses_actual_host_tokenizer(monkeypatch, compiler, tmp_path):
    from jev_vllm import endpoint

    runtime = AsyncMock()

    async def build(settings, native_backend, compiler):
        assert compiler.tokenizer is engine.renderer.tokenizer
        assert compiler._cache_tokens == 0
        return runtime

    config = tmp_path / "config.json"
    config.write_text(
        '{"backend":"vllm","model_id":"fixture","model_revision":"'
        + "a" * 40
        + '","compiler_cache_tokens":0}'
    )
    engine = SimpleNamespace(
        model_config=SimpleNamespace(model="fixture", max_model_len=2048),
        renderer=SimpleNamespace(tokenizer=compiler.tokenizer),
    )
    monkeypatch.setenv("JEV_CONFIG", str(config))
    monkeypatch.setattr(endpoint, "version", lambda name: "fixture")
    monkeypatch.setattr(endpoint, "build_runtime", build)
    monkeypatch.setattr(endpoint, "bootstrap", AsyncMock())
    state = SimpleNamespace()
    await JevEndpointPlugin().init_state(engine, state, SimpleNamespace(api_server_count=1))
    assert state.jev_runtime is runtime
    runtime.start.assert_awaited_once()


def test_sglang_spawn_target_preserves_other_uvicorn_apps():
    assert _configure_uvicorn_workers("other.application:app", workers=2) is None
    assert _configure_uvicorn_workers("sglang.srt.entrypoints.http_server:app", workers=1) is None
    args, kwargs = _configure_uvicorn_workers(
        "sglang.srt.entrypoints.http_server:app", workers=2, host="127.0.0.1", port=1234
    )
    assert args == ("jev_sglang.worker:app",)
    assert kwargs == {"workers": 2, "host": "127.0.0.1", "port": 1234}


async def test_sglang_worker_wraps_lifespan_before_manager_exists(monkeypatch, compiler, tmp_path):
    import sys

    from jev_sglang import plugin

    events = []
    global_state = None

    @asynccontextmanager
    async def host_lifespan(app):
        nonlocal global_state
        global_state = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(tokenizer=compiler.tokenizer)
        )
        events.append("host-start")
        yield
        events.append("host-stop")

    host = SimpleNamespace(
        app=FastAPI(lifespan=host_lifespan), get_global_state=lambda: global_state
    )
    for name in ("sglang", "sglang.srt", "sglang.srt.entrypoints"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["sglang.srt.entrypoints"].http_server = host
    runtime = AsyncMock()
    runtime.start.side_effect = lambda: events.append("runtime-start")
    runtime.close.side_effect = lambda: events.append("runtime-close")

    async def build(*args, **kwargs):
        assert global_state is not None
        assert kwargs["compiler"].tokenizer is global_state.tokenizer_manager.tokenizer
        return runtime

    async def bootstrap(*args):
        events.append("bootstrap")

    monkeypatch.setattr(plugin, "build_runtime", build)
    monkeypatch.setattr(plugin, "bootstrap", bootstrap)
    monkeypatch.setattr(plugin, "version", lambda name: "fixture")
    config = tmp_path / "config.json"
    config.write_text(
        '{"backend":"sglang","model_id":"fixture","model_revision":"' + "a" * 40 + '"}'
    )
    monkeypatch.setenv("JEV_CONFIG", str(config))
    monkeypatch.setenv("JEV_API_KEY", "fixture-key")
    plugin.configure_http_app()
    wrapper = host.app.router.lifespan_context
    plugin.configure_http_app()
    assert host.app.router.lifespan_context is wrapper
    assert global_state is None
    async with wrapper(host.app):
        assert host.app.state.jev_runtime is runtime
        assert events == ["host-start", "runtime-start", "bootstrap"]
    assert events == ["host-start", "runtime-start", "bootstrap", "runtime-close", "host-stop"]
