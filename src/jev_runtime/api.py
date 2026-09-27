from __future__ import annotations

import asyncio
import hmac
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import Field

from jev_runtime.config import Settings, bootstrap, build_runtime, load_settings, tenant_keys
from jev_runtime.errors import JevError
from jev_runtime.lifecycle import cancel_and_drain
from jev_runtime.runtime import Runtime
from jev_runtime.schema import Bundle, Contract, DecisionRequest
from jev_runtime.systemone import SystemOneRequest, from_decision


class Activation(Contract):
    alias: str = Field(min_length=1, max_length=256)
    reference: str
    expected_generation: int = Field(ge=0)


class Disable(Contract):
    alias: str
    expected_generation: int = Field(ge=0)


class Reference(Contract):
    reference: str


async def disconnect_guard(request: Request, coroutine):
    work = asyncio.create_task(coroutine)

    async def watch():
        while True:
            event = await request.receive()
            if event["type"] == "http.disconnect":
                return

    disconnected = asyncio.create_task(watch())
    try:
        done, _ = await asyncio.wait((work, disconnected), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return await work
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        raise JevError("client_disconnected", "Client disconnected", 499)
    finally:
        await cancel_and_drain((work, disconnected))


def install_routes(
    app: FastAPI,
    *,
    prefix: str = "",
    api_key: str | None = None,
    admin_key: str | None = None,
    tenants: dict[str, str] | None = None,
) -> None:
    identities = dict(tenants or {})
    if api_key is not None:
        if api_key in identities.values():
            raise ValueError("Default and tenant API keys must be distinct")
        identities["default"] = api_key

    async def authorize(request: Request, authorization: str | None = Header(default=None)):
        if not identities:
            request.state.jev_tenant = "default"
            return
        for tenant, key in identities.items():
            if hmac.compare_digest(authorization or "", f"Bearer {key}"):
                request.state.jev_tenant = tenant
                return
        raise JevError("unauthorized", "A valid API key is required", 401)

    async def admin(authorization: str | None = Header(default=None)):
        if not admin_key or not hmac.compare_digest(authorization or "", f"Bearer {admin_key}"):
            raise JevError("admin_unauthorized", "A separate admin key is required", 401)

    def runtime(request: Request) -> Runtime:
        instance = getattr(request.app.state, "jev_runtime", None)
        if instance is None or instance.capabilities is None:
            raise JevError("not_ready", "Decision runtime is not ready", 503)
        return instance

    @app.exception_handler(JevError)
    async def jev_error_handler(request: Request, exc: JevError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.as_dict()})

    metrics_registry = CollectorRegistry()
    count = Counter(
        "jev_decisions_total", "Decision HTTP outcomes", ["outcome"], registry=metrics_registry
    )
    latency = Histogram(
        "jev_request_seconds",
        "Decision endpoint latency",
        registry=metrics_registry,
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
    )
    router = APIRouter(prefix=prefix, dependencies=[Depends(authorize)])
    management = APIRouter(prefix=prefix + "/admin", dependencies=[Depends(admin)])

    @router.post("/v1/decisions")
    async def decisions(body: DecisionRequest, request: Request, http_response: Response):
        http_response.headers["X-Jev-Worker"] = runtime(request).registry.owner
        with latency.time():
            try:
                response = await disconnect_guard(
                    request, runtime(request).decide(body, tenant=request.state.jev_tenant)
                )
                count.labels(response.status).inc()
                return response
            except BaseException:
                count.labels("error").inc()
                raise

    @router.post("/v1/systemone")
    async def systemone(body: SystemOneRequest, request: Request):
        return from_decision(
            await disconnect_guard(
                request,
                runtime(request).decide(body.to_decision(), tenant=request.state.jev_tenant),
            )
        )

    @router.get("/v1/capabilities")
    async def capabilities(request: Request):
        return runtime(request).capabilities

    @router.get("/v1/models")
    async def models(request: Request):
        return {
            "object": "list",
            "data": [
                {"id": route["alias"], "object": "model"}
                for route in runtime(request).registry.list()["routes"]
                if route["ref"] and runtime(request).is_prepared(route["ref"])
            ],
        }

    @router.post("/v1/requests/{request_id}/cancel")
    async def cancel(request_id: str, request: Request):
        return {"cancelled": await runtime(request).cancel(request_id, request.state.jev_tenant)}

    @router.get("/ready")
    async def ready(request: Request):
        instance = runtime(request)
        if not instance.control_healthy:
            raise JevError("control_unavailable", "Worker cancellation control is unhealthy", 503)
        routes = instance.registry.list()["routes"]
        prepared = [
            route["ref"] for route in routes if route["ref"] and instance.is_prepared(route["ref"])
        ]
        if not prepared:
            raise JevError("no_active_bundle", "No decision bundle is active", 503)
        return {
            "ready": True,
            "engine": instance.capabilities.engine,
            "prepared_bundles": sorted(set(prepared)),
            "worker_id": instance.registry.owner,
        }

    @router.get("/metrics")
    async def metrics():
        return Response(generate_latest(metrics_registry), media_type="text/plain; version=0.0.4")

    @management.get("/bundles")
    async def list_bundles(request: Request):
        return runtime(request).registry.list()

    @management.get("/workers")
    async def workers(request: Request):
        return {"workers": runtime(request).registry.worker_status()}

    @management.get("/profile")
    async def profile(request: Request):
        instance = runtime(request)
        return {
            "worker_id": instance.registry.owner,
            "model": instance.expected_model,
            "tokenizer_implementation_verified": (
                instance.compiler.tokenizer_implementation_digest is not None
            ),
            "engine_identity_verified": instance.capabilities.verified,
            "capabilities": instance.capabilities,
        }

    @management.post("/compile")
    async def compile_preview(body: DecisionRequest, request: Request):
        return runtime(request).compile_preview(body)

    @management.post("/bundles")
    async def upload(body: Bundle, request: Request):
        return runtime(request).registry.upload(body)

    @management.post("/bundles/prepare")
    async def prepare(body: Reference, request: Request):
        return await runtime(request).prepare(body.reference)

    @management.post("/bundles/activate")
    async def activate(body: Activation, request: Request):
        return runtime(request).activate(body.alias, body.reference, body.expected_generation)

    @management.post("/bundles/disable")
    async def disable(body: Disable, request: Request):
        return runtime(request).registry.disable(body.alias, body.expected_generation)

    @management.post("/bundles/retire")
    async def retire(body: Reference, request: Request):
        return runtime(request).registry.retire(body.reference)

    @management.get("/requests/recovery")
    async def recovery_requests(request: Request):
        return {"requests": runtime(request).pending_cancellations()}

    @management.post("/requests/{request_id}/recover")
    async def recover_request(request_id: str, request: Request):
        return {"recovered": await runtime(request).recover_cancelled(request_id)}

    app.include_router(router)
    app.include_router(management)


def create_app(
    settings: Settings | None = None,
    instance: Runtime | None = None,
    api_key: str | None = None,
    admin_key: str | None = None,
) -> FastAPI:
    if settings:
        api_key = api_key if api_key is not None else os.environ.get(settings.api_key_env)
        admin_key = admin_key if admin_key is not None else os.environ.get(settings.admin_key_env)
        if settings.host not in {"127.0.0.1", "localhost", "::1"} and not (
            api_key or settings.tenant_key_envs
        ):
            raise ValueError("A public bind requires an API key")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        current = instance or await build_runtime(settings)
        app.state.jev_runtime = current
        try:
            await current.start()
            if settings:
                await bootstrap(current, settings)
            yield
        finally:
            await current.close()

    app = FastAPI(title="Jev Runtime", version="0.1.0a1", lifespan=lifespan)
    app.state.instance_id = uuid.uuid4().hex
    install_routes(
        app,
        api_key=api_key,
        admin_key=admin_key,
        tenants=tenant_keys(settings) if settings else None,
    )
    return app


def create_app_from_env() -> FastAPI:
    """Uvicorn factory: create an independent runtime after each worker is spawned."""
    path = os.environ.get("JEV_CONFIG")
    if not path:
        raise ValueError("JEV_CONFIG is required for a multi-worker gateway")
    return create_app(load_settings(path))
