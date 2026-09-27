from __future__ import annotations

import os
from contextlib import asynccontextmanager
from importlib.metadata import version

from jev_runtime.backends.sglang import SGLangNative
from jev_runtime.compiler import Compiler
from jev_runtime.config import bootstrap, build_runtime, load_settings
from jev_runtime.plugin_api import install_plugin_routes

_registered = False


def configure_http_app():
    from sglang.srt.entrypoints import http_server

    app = http_server.app
    if getattr(app.state, "jev_lifespan_installed", False):
        return
    app.state.jev_lifespan_installed = True
    install_plugin_routes(app)
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(fastapi_app):
        async with original(fastapi_app):
            manager = http_server.get_global_state().tokenizer_manager
            backend = SGLangNative(manager, version("sglang"))
            fastapi_app.state.jev_backend = backend
            runtime = None
            try:
                path = os.environ.get("JEV_CONFIG")
                if path:
                    settings = load_settings(path)
                    if settings.backend != "sglang":
                        raise ValueError("JEV_CONFIG backend must match the host engine")
                    runtime = await build_runtime(
                        settings,
                        native_backend=backend,
                        compiler=Compiler(
                            manager.tokenizer,
                            settings.compiler_cache_tokens,
                            settings.compiler_cache_entries,
                        ),
                    )
                    await runtime.start()
                    await bootstrap(runtime, settings)
                    fastapi_app.state.jev_runtime = runtime
                yield
            finally:
                if runtime is not None:
                    await runtime.close()

    app.router.lifespan_context = lifespan


def _after_global_state(result, *args, **kwargs):
    configure_http_app()


def _configure_uvicorn_workers(*args, **kwargs):
    """SGLang 0.5.19 imports the bare ASGI app in spawned tokenizer workers."""
    target = args[0] if args else kwargs.get("app")
    if target != "sglang.srt.entrypoints.http_server:app" or kwargs.get("workers", 1) <= 1:
        return None
    target = "jev_sglang.worker:app"
    if args:
        return (target, *args[1:]), kwargs
    return args, {**kwargs, "app": target}


def _check_granian_workers(*args, **kwargs):
    if kwargs.get("tokenizer_worker_num", 1) > 1:
        raise ValueError("Jev multi-tokenizer integration currently requires the Uvicorn HTTP path")


def register():
    global _registered
    if _registered:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.entrypoints.http_server.set_global_state", _after_global_state, HookType.AFTER
    )
    HookRegistry.register("uvicorn.run", _configure_uvicorn_workers, HookType.BEFORE)
    HookRegistry.register(
        "sglang.srt.entrypoints.http_server._run_granian_server",
        _check_granian_workers,
        HookType.BEFORE,
    )
    _registered = True
