from __future__ import annotations

import asyncio
import hmac
import os
from dataclasses import asdict

from fastapi import APIRouter, Depends, FastAPI, Header, Request
from pydantic import Field, model_validator

from jev_runtime.api import disconnect_guard, install_routes
from jev_runtime.backends.base import ScoreInput
from jev_runtime.config import load_settings, tenant_keys
from jev_runtime.errors import JevError
from jev_runtime.schema import Contract


class ScoreWire(Contract):
    request_id: str = Field(min_length=1, max_length=512)
    question_id: str = Field(min_length=1, max_length=128)
    input_ids: tuple[int, ...] = Field(min_length=1, max_length=131072)
    label_ids: tuple[int, ...] = Field(min_length=1, max_length=128)
    candidate_id: str | None = None
    adapter_id: str | None = None

    @model_validator(mode="after")
    def valid_tokens(self):
        if any(token < 0 for token in (*self.input_ids, *self.label_ids)):
            raise ValueError("Token IDs must be nonnegative")
        if len(set(self.label_ids)) != len(self.label_ids):
            raise ValueError("Label token IDs must be distinct")
        return self


class CancelWire(Contract):
    request_id: str = Field(min_length=1, max_length=512)


def install_plugin_routes(app: FastAPI) -> None:
    if getattr(app.state, "jev_routes_installed", False):
        return
    prefix = "/plugins/jev-runtime"
    api_key = os.environ.get("JEV_API_KEY")
    admin_key = os.environ.get("JEV_ADMIN_KEY")
    if not api_key:
        raise ValueError("Set JEV_API_KEY before enabling the Jev engine plugin")
    app.state.jev_routes_installed = True
    config_path = os.environ.get("JEV_CONFIG")
    tenants = tenant_keys(load_settings(config_path)) if config_path else None
    install_routes(app, prefix=prefix, api_key=api_key, admin_key=admin_key, tenants=tenants)

    async def authorize(authorization: str | None = Header(default=None)):
        if not api_key or not hmac.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise JevError("unauthorized", "Plugin scoring requires JEV_API_KEY", 401)

    router = APIRouter(prefix=prefix + "/v1", dependencies=[Depends(authorize)])

    def backend(request: Request):
        adapter = getattr(request.app.state, "jev_backend", None)
        if adapter is None:
            raise JevError("not_ready", "Plugin backend is not initialized", 503)
        return adapter

    @router.get("/scoring-capabilities")
    async def capabilities(request: Request):
        return await backend(request).probe()

    @router.post("/scores")
    async def score(body: ScoreWire, request: Request):
        runtime = getattr(request.app.state, "jev_runtime", None)
        if runtime is not None:
            runtime.require_serving_mode()
        if body.adapter_id:
            raise JevError(
                "lora_unsupported", "Managed LoRA requires the leased typed decision endpoint", 409
            )
        adapter = backend(request)
        if runtime is not None:

            async def managed_execute():
                return asdict(await runtime.score_raw(ScoreInput(**body.model_dump())))

            return await disconnect_guard(request, managed_execute())
        capabilities = await adapter.probe()
        if len(body.input_ids) + 1 > capabilities.max_context_tokens:
            raise JevError("context_budget", "Prompt exceeds the native engine context limit", 413)
        if len(body.label_ids) > capabilities.max_label_tokens:
            raise JevError("label_budget", "Too many labels for this native engine", 413)

        async def execute():
            try:
                async with asyncio.timeout(300):
                    return asdict(await adapter.score(ScoreInput(**body.model_dump())))
            except BaseException:
                await adapter.cancel(body.request_id)
                raise

        return await disconnect_guard(request, execute())

    @router.post("/scores/cancel")
    async def cancel(body: CancelWire, request: Request):
        await backend(request).cancel(body.request_id)
        return {"cancelled": True}

    app.include_router(router)
