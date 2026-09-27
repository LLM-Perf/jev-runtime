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
                        settings, native_backend=backend, compiler=Compiler(manager.tokenizer)
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


def register():
    global _registered
    if _registered:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.entrypoints.http_server.set_global_state", _after_global_state, HookType.AFTER
    )
    _registered = True
