from __future__ import annotations

import os
from contextlib import asynccontextmanager
from importlib.metadata import version

from jev_runtime.backends.sglang import SGLangNative
from jev_runtime.config import bootstrap, build_runtime, compiler_for_tokenizer, load_settings
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
                        compiler=compiler_for_tokenizer(settings, manager.tokenizer),
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


def _before_lora_unload(*args, **kwargs):
    import torch

    torch.cuda.synchronize()


def _after_lora_update(result, *args, **kwargs):
    if result.success:
        import torch

        torch.cuda.synchronize()
        # The native coordinator verifies this marker in the scheduler reply;
        # an uninstalled/failed hook cannot advertise completed GPU cleanup.
        result.error_message = "jev_gpu_barrier_v1"
    return result


class _HostLogprobRow(list):
    """Preserve SGLang's documented host-list rows at its tensor-only boundary."""

    def tolist(self):
        return list(self)


def _before_logprob_rows(*args, **kwargs):
    # SGLang 0.5.19 get_token_ids_logprobs_raw returns [] for a request
    # without selected IDs. Mixed decode/prefill normalization still calls
    # .tolist() on every row. Keep tensor rows and their device copies intact;
    # host rows need no tensor allocation, precision change or fabricated score.
    if not kwargs["batch"].return_logprob:
        return
    output = kwargs["logits_output"]
    rows = output.next_token_token_ids_logprobs_val
    if rows:
        for index, row in enumerate(rows):
            if type(row) is list:
                rows[index] = _HostLogprobRow(row)


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
    if version("sglang").split("+")[0] == "0.5.19":
        target = (
            "sglang.srt.managers.scheduler_components.batch_result_processor."
            "SchedulerBatchResultProcessor."
        )
        for method in ("move_logprobs_to_cpu", "_normalize_decode_outputs"):
            HookRegistry.register(target + method, _before_logprob_rows, HookType.BEFORE)
    path = os.environ.get("JEV_CONFIG")
    if path and load_settings(path).adapters.enabled:
        target = "sglang.srt.managers.scheduler.Scheduler."
        HookRegistry.register(target + "unload_lora_adapter", _before_lora_unload, HookType.BEFORE)
        for method in ("load_lora_adapter", "unload_lora_adapter"):
            HookRegistry.register(target + method, _after_lora_update, HookType.AFTER)
    _registered = True
