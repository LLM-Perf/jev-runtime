"""Native Engine lifespan integration without modifying TokenSpeed source files."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from jev_runtime.config import bootstrap, build_runtime, compiler_for_tokenizer, model_identity
from jev_runtime.errors import JevError
from jev_runtime.plugin_api import install_plugin_routes
from jev_runtime.schema import content_digest
from jev_tokenspeed.backend import TokenSpeedNative, validate_profile
from jev_tokenspeed.receipts import CompletionReceipts


def verify_source(root: Path) -> str:
    package = Path(__file__).parent
    profiles = [
        package / "source-profile.json",
        *sorted((package / "source-profiles").glob("*.json")),
    ]
    mismatches = {}
    for source in profiles:
        profile = json.loads(source.read_text())
        for relative, digest in profile["files"].items():
            path = root / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                mismatches[profile["revision"]] = relative
                break
        else:
            return profile["revision"]
    raise JevError(
        "tokenspeed_source_mismatch",
        f"TokenSpeed differs from the supported readout profiles: {mismatches}",
        409,
    )


async def submit_to_engine(loop, coroutine):
    """Use the pinned Engine's owner loop without a blocking thread per request."""
    try:
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    except BaseException:
        coroutine.close()
        raise
    return await asyncio.wrap_future(future)


def create_app(settings, engine_options: dict) -> FastAPI:
    if settings.backend != "tokenspeed" or settings.workers != 1 or settings.adapters.enabled:
        raise ValueError("TokenSpeed native plugin requires backend=tokenspeed, workers=1, no LoRA")

    @asynccontextmanager
    async def lifespan(app):
        import tokenspeed

        revision = verify_source(Path(tokenspeed.__file__).parent)
        from jev_tokenspeed.preflight import check_hardware

        check_hardware(engine_options.get("base_gpu_id", 0))
        from tokenspeed.runtime.entrypoints.engine import Engine
        from tokenspeed.runtime.utils.server_args import ServerArgs
        from tokenspeed.version import __version__

        args = ServerArgs(**engine_options)
        validate_profile(args)
        validate_binding(settings, args)
        # Engine owns its subprocesses. Only this launcher-created instance is
        # shut down here; embedding callers may use TokenSpeedNative separately.
        engine = Engine(server_args=args)
        runtime = None
        try:

            async def submit(coro):
                return await submit_to_engine(engine.llm._loop, coro)

            manager = engine.tokenizer_manager
            compiler = compiler_for_tokenizer(settings, manager.tokenizer)
            receipt_path = Path(settings.registry_path).with_suffix(".tokenspeed-receipts.db")
            namespace = content_digest(
                {
                    "engine_url": settings.engine_url,
                    "model": model_identity(settings, compiler).model_dump(mode="json"),
                    "upstream": revision,
                    "engine_options": engine_options,
                }
            )
            receipts = await asyncio.to_thread(CompletionReceipts, receipt_path, namespace)
            backend = TokenSpeedNative(
                manager, f"{__version__}+{revision[:12]}", submit, receipts=receipts
            )
            app.state.jev_backend = backend
            runtime = await build_runtime(
                settings,
                native_backend=backend,
                compiler=compiler,
            )
            await runtime.start()
            await bootstrap(runtime, settings)
            runtime.start_health_monitor()
            app.state.jev_runtime = runtime
            yield
        finally:
            # A failed drain is deliberately not swallowed to advertise a
            # successful unload. The durable journals remain for diagnosis.
            try:
                if runtime is not None:
                    await runtime.close()
            finally:
                # Closing an owned engine must not be skipped after a failed
                # drain. Unconfirmed journals/receipts remain unresolved.
                engine.shutdown()

    app = FastAPI(title="Jev TokenSpeed native plugin", lifespan=lifespan)
    install_plugin_routes(app)
    return app


def validate_binding(settings, args) -> None:
    if str(args.model) != settings.model_id:
        raise ValueError(
            "Engine model must match model_id; use its absolute path for local weights"
        )
    if args.revision != settings.model_revision:
        raise ValueError("Engine revision must match the immutable configured model revision")
    if str(args.tokenizer) != (settings.tokenizer or settings.model_id):
        raise ValueError("Engine tokenizer must match the configured tokenizer profile")
