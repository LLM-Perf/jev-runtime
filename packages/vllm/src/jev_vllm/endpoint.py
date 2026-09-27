from __future__ import annotations

import os
from contextlib import asynccontextmanager
from importlib.metadata import version

from jev_runtime.backends.vllm import VLLMNative
from jev_runtime.config import bootstrap, build_runtime, compiler_for_tokenizer, load_settings
from jev_runtime.plugin_api import install_plugin_routes
from jev_vllm.tokenizer import compiler_tokenizer


class JevEndpointPlugin:
    name = "jev_runtime_api"
    required_tasks = ("generate",)

    def attach_router(self, app):
        if getattr(app.state, "jev_lifespan_installed", False):
            return
        install_plugin_routes(app)
        original = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(fastapi_app):
            async with original(fastapi_app):
                try:
                    yield
                finally:
                    # Drain while the host engine and app.state still exist.
                    # vLLM's outer lifespan deletes state during its teardown.
                    runtime = getattr(fastapi_app.state, "jev_runtime", None)
                    if runtime is not None:
                        await runtime.close()
                        fastapi_app.state.jev_runtime = None

        app.router.lifespan_context = lifespan
        app.state.jev_lifespan_installed = True

    async def init_state(self, engine_client, state, args):
        if engine_client is None:
            return
        config = engine_client.model_config
        backend = VLLMNative(
            engine_client,
            config.model,
            config.max_model_len,
            version("vllm"),
            api_workers=getattr(args, "api_server_count", None) or 1,
        )
        state.jev_backend = backend
        path = os.environ.get("JEV_CONFIG")
        if path:
            settings = load_settings(path)
            if settings.backend != "vllm":
                raise ValueError("JEV_CONFIG backend must match the host engine")
            # vLLM's renderer owns the tokenizer actually used for text input.
            # A separately loaded AutoTokenizer can miss engine adjustments.
            tokenizer = engine_client.renderer.tokenizer
            if tokenizer is None:
                raise ValueError("Jev typed compilation requires the host tokenizer")
            tokenizer, copied = compiler_tokenizer(tokenizer)
            compiler = compiler_for_tokenizer(settings, tokenizer)
            compiler.copied_from_host_pool = copied
            runtime = await build_runtime(settings, native_backend=backend, compiler=compiler)
            try:
                await runtime.start()
                await bootstrap(runtime, settings)
            except BaseException:
                await runtime.close()
                raise
            state.jev_runtime = runtime
