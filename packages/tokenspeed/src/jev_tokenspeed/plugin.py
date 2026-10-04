"""Native Engine lifespan integration without modifying TokenSpeed source files."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from jev_runtime.config import bootstrap, build_runtime, compiler_for_tokenizer
from jev_runtime.errors import JevError
from jev_runtime.plugin_api import install_plugin_routes
from jev_tokenspeed.backend import TokenSpeedNative, validate_profile


def verify_source(root: Path) -> str:
    profile = json.loads(Path(__file__).with_name("source-profile.json").read_text())
    for relative, digest in profile["files"].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise JevError(
                "tokenspeed_source_mismatch",
                f"TokenSpeed source differs from the inspected readout contract: {relative}",
                409,
            )
    return profile["revision"]


def create_app(settings, engine_options: dict) -> FastAPI:
    if settings.backend != "tokenspeed" or settings.workers != 1 or settings.adapters.enabled:
        raise ValueError("TokenSpeed native plugin requires backend=tokenspeed, workers=1, no LoRA")

    @asynccontextmanager
    async def lifespan(app):
        import tokenspeed

        revision = verify_source(Path(tokenspeed.__file__).parent)
        from tokenspeed.runtime.entrypoints.engine import Engine
        from tokenspeed.runtime.utils.server_args import ServerArgs
        from tokenspeed.version import __version__

        args = ServerArgs(**engine_options)
        validate_profile(args)
        if str(args.model) not in {settings.model_id, settings.tokenizer}:
            raise ValueError("Engine model must match the configured model or local tokenizer path")
        if args.revision != settings.model_revision:
            raise ValueError("Engine revision must match the immutable configured model revision")
        # Engine owns its subprocesses. Only this launcher-created instance is
        # shut down here; embedding callers may use TokenSpeedNative separately.
        engine = Engine(server_args=args)
        runtime = None
        try:

            async def submit(coro):
                return await asyncio.to_thread(engine.llm.run, coro)

            manager = engine.tokenizer_manager
            backend = TokenSpeedNative(manager, f"{__version__}+{revision[:12]}", submit)
            app.state.jev_backend = backend
            runtime = await build_runtime(
                settings,
                native_backend=backend,
                compiler=compiler_for_tokenizer(settings, manager.tokenizer),
            )
            await runtime.start()
            await bootstrap(runtime, settings)
            runtime.start_health_monitor()
            app.state.jev_runtime = runtime
            yield
        finally:
            # A failed drain is deliberately not swallowed to advertise a
            # successful unload. The durable journals remain for diagnosis.
            if runtime is not None:
                await runtime.close()
            engine.shutdown()

    app = FastAPI(title="Jev TokenSpeed native plugin", lifespan=lifespan)
    install_plugin_routes(app)
    return app
