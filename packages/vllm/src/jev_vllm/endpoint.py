from __future__ import annotations

import os
from importlib.metadata import version

from jev_runtime.backends.vllm import VLLMNative
from jev_runtime.config import bootstrap, build_runtime, load_settings
from jev_runtime.plugin_api import install_plugin_routes


class JevEndpointPlugin:
    name = "jev_runtime_api"
    required_tasks = ("generate",)

    def attach_router(self, app):
        install_plugin_routes(app)

    async def init_state(self, engine_client, state, args):
        if engine_client is None:
            return
        config = engine_client.model_config
        backend = VLLMNative(engine_client, config.model, config.max_model_len, version("vllm"))
        state.jev_backend = backend
        path = os.environ.get("JEV_CONFIG")
        if path:
            settings = load_settings(path)
            if settings.backend != "vllm":
                raise ValueError("JEV_CONFIG backend must match the host engine")
            runtime = await build_runtime(settings, native_backend=backend)
            await runtime.start()
            await bootstrap(runtime, settings)
            state.jev_runtime = runtime
