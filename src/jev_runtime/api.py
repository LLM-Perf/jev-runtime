from __future__ import annotations

import asyncio
import hmac
import json
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import Field

from jev_runtime.config import Settings, bootstrap, build_runtime, load_settings, tenant_keys
from jev_runtime.errors import JevError
from jev_runtime.lifecycle import cancel_and_drain
from jev_runtime.runtime import Runtime
from jev_runtime.schema import Bundle, Contract, DecisionRequest, DecisionResponse, Identifier
from jev_runtime.systemone import SystemOneRequest, from_decision
from jev_runtime.telemetry import DecisionTrace


class Activation(Contract):
    alias: str = Field(min_length=1, max_length=256)
    reference: str
    expected_generation: int = Field(ge=0)


class Disable(Contract):
    alias: str
    expected_generation: int = Field(ge=0)


class Quiesce(Contract):
    expected_generation: int = Field(ge=0)
    timeout_seconds: float = Field(default=30, ge=0, le=300)


class Reference(Contract):
    reference: str


class AdapterRegistration(Contract):
    id: Identifier
    source: str = Field(min_length=1, max_length=4096)


class AdapterRemoval(Reference):
    recover: bool = False


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
    auth_policy = (
        hmac.new(
            admin_key.encode(), json.dumps(identities, sort_keys=True).encode(), "sha256"
        ).hexdigest()
        if admin_key
        else None
    )

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

    async def recovery_write_guard(request: Request):
        if (
            runtime(request).recovery_only
            and request.method not in {"GET", "HEAD", "OPTIONS"}
            and request.scope["route"].path
            not in {
                prefix + "/admin/requests/{request_id}/recover",
                prefix + "/admin/raw-requests/{work_id}/recover",
                prefix + "/admin/quiescence",
            }
        ):
            runtime(request).require_serving_mode()

    @app.exception_handler(JevError)
    async def jev_error_handler(request: Request, exc: JevError):
        trace = getattr(request.state, "jev_trace", None)
        headers = {}
        if trace and request.headers.get("X-Jev-Timing") == "1" and trace.seconds:
            headers["Server-Timing"] = trace.header()
        return JSONResponse(
            status_code=exc.status_code, content={"error": exc.as_dict()}, headers=headers
        )

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
    router = APIRouter(
        prefix=prefix, dependencies=[Depends(authorize), Depends(recovery_write_guard)]
    )
    management = APIRouter(
        prefix=prefix + "/admin", dependencies=[Depends(admin), Depends(recovery_write_guard)]
    )

    stages = Histogram(
        "jev_runtime_stage_seconds",
        "Serial runtime phases; total includes durable lease cleanup, excludes HTTP serialization",
        ["stage", "outcome"],
        registry=metrics_registry,
        buckets=(0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.1, 0.5, 1, 5, 30, 120),
    )
    work = Histogram(
        "jev_branch_work_seconds",
        "Individual branch operations; concurrent samples overlap and are not wall-time phases",
        ["kind"],
        registry=metrics_registry,
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.025, 0.1, 0.5, 1, 5, 30, 120),
    )
    errors = Counter(
        "jev_request_errors_total",
        "Runtime failures by bounded category",
        ["category"],
        registry=metrics_registry,
    )
    questions = Counter(
        "jev_questions_total", "Returned question outcomes", ["outcome"], registry=metrics_registry
    )
    tokens = Counter(
        "jev_observed_tokens_total",
        "Known usage values only; no estimates for missing values",
        ["quantity"],
        registry=metrics_registry,
    )
    token_observations = Counter(
        "jev_token_observations_total",
        "Responses with a known usage value",
        ["quantity"],
        registry=metrics_registry,
    )
    admission = Gauge(
        "jev_admission",
        "Current API-process admission occupancy",
        ["quantity"],
        registry=metrics_registry,
    )
    shared_admission = Gauge(
        "jev_shared_admission",
        "Registry-wide engine admission; do not sum across workers",
        ["quantity"],
        registry=metrics_registry,
    )

    def observe(trace, outcome, response):
        count.labels(outcome).inc()
        for stage, seconds in trace.seconds.items():
            stages.labels(stage, outcome).observe(seconds)
        for kind, samples in trace.work.items():
            for seconds in samples:
                work.labels(kind).observe(seconds)
        if response is not None:
            for answer in response.answers.values():
                questions.labels(answer.status).inc()
            for quantity in (
                "logical_prompt_tokens",
                "engine_prompt_tokens",
                "engine_completion_tokens",
                "cached_prompt_tokens",
            ):
                value = getattr(response.usage, quantity)
                if value is not None:
                    tokens.labels(quantity).inc(value)
                    token_observations.labels(quantity).inc()

    async def execute(body, request, http_response):
        instance = runtime(request)
        http_response.headers["X-Jev-Worker"] = instance.registry.owner
        trace = DecisionTrace()
        request.state.jev_trace = trace
        response, outcome = None, "error"
        with latency.time():
            try:
                response = await disconnect_guard(
                    request, instance.decide(body, tenant=request.state.jev_tenant, trace=trace)
                )
                outcome = response.status
                if request.headers.get("X-Jev-Timing") == "1":
                    http_response.headers["Server-Timing"] = trace.header()
                return response
            except BaseException as exc:
                status = exc.status_code if isinstance(exc, JevError) else 500
                category = (
                    "cancelled"
                    if status == 499
                    else "deadline"
                    if status == 504
                    else "budget"
                    if status in {413, 429}
                    else "validation"
                    if 400 <= status < 500
                    else "engine_or_internal"
                )
                errors.labels(category).inc()
                raise
            finally:
                observe(trace, outcome, response)

    @router.post("/v1/decisions")
    async def decisions(
        body: DecisionRequest, request: Request, http_response: Response
    ) -> DecisionResponse:
        return await execute(body, request, http_response)

    @router.post("/v1/systemone")
    async def systemone(body: SystemOneRequest, request: Request, http_response: Response):
        return from_decision(await execute(body.to_decision(), request, http_response))

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
    async def cancel(request_id: str, request: Request, http_response: Response):
        instance = runtime(request)
        http_response.headers["X-Jev-Worker"] = instance.registry.owner
        return {"cancelled": await instance.cancel(request_id, request.state.jev_tenant)}

    @router.get("/ready")
    async def ready(request: Request):
        instance = runtime(request)
        instance.require_serving_mode()
        instance.registry.require_backend_open(instance.backend_identity)
        if not instance.control_healthy:
            raise JevError("control_unavailable", "Worker cancellation control is unhealthy", 503)
        profile = instance.health_profile()
        if not profile["bundles"]:
            raise JevError("no_active_bundle", "No decision bundle is active", 503)
        if not profile["monitor_running"] or any(
            not bundle["ready"] or not bundle["prepared"] for bundle in profile["bundles"].values()
        ):
            raise JevError("engine_unavailable", "Active bundles lack current engine canaries", 503)
        capabilities = instance.capabilities
        assert capabilities is not None  # runtime(request) above guarantees a probed engine
        return {
            "ready": True,
            "engine": capabilities.engine,
            "prepared_bundles": sorted(profile["bundles"]),
            "worker_id": instance.registry.owner,
        }

    @router.get("/metrics")
    async def metrics(request: Request):
        current = runtime(request).admission.snapshot()
        metric = shared_admission if current["scope"] == "shared_registry_engine" else admission
        for quantity in ("requests", "expanded_tokens", "expanded_branches", "queued_requests"):
            metric.labels(quantity).set(current[quantity])
        return Response(generate_latest(metrics_registry), media_type="text/plain; version=0.0.4")

    @management.get("/bundles")
    async def list_bundles(request: Request):
        return runtime(request).registry.list()

    @management.get("/adapters")
    async def adapters(request: Request):
        return runtime(request).registry.list_adapters()

    @management.post("/adapters/register")
    async def register_adapter(body: AdapterRegistration, request: Request):
        return await runtime(request).register_adapter(body.id, body.source)

    @management.post("/adapters/load")
    async def load_adapter(body: Reference, request: Request):
        return await runtime(request).change_adapter(body.reference, "load")

    @management.post("/adapters/unload")
    async def unload_adapter(body: AdapterRemoval, request: Request):
        return await runtime(request).change_adapter(body.reference, "unload", body.recover)

    @management.get("/workers")
    async def workers(request: Request):
        return {"workers": runtime(request).registry.worker_status()}

    @management.get("/quiescence")
    async def quiescence_status(request: Request):
        instance = runtime(request)
        return instance.registry.quiescence_status(instance.backend_identity)

    @management.post("/quiescence")
    async def quiesce(body: Quiesce, request: Request):
        return await runtime(request).quiesce(body.expected_generation, body.timeout_seconds)

    @management.get("/raw-requests/recovery")
    async def raw_recovery(request: Request):
        instance = runtime(request)
        return {"requests": instance.registry.raw_recovery_candidates(instance.backend_identity)}

    @management.get("/recovery-operations")
    async def recovery_operations(request: Request):
        instance = runtime(request)
        return {"operations": instance.registry.pending_recoveries(instance.backend_identity)}

    @management.post("/raw-requests/{work_id}/recover")
    async def recover_raw(work_id: str, request: Request):
        return {"recovered": await runtime(request).recover_raw(work_id)}

    @management.get("/profile")
    async def profile(request: Request):
        instance = runtime(request)
        capabilities = instance.capabilities
        assert capabilities is not None  # runtime(request) above guarantees a probed engine
        return {
            "worker_id": instance.registry.owner,
            "model": instance.expected_model,
            "recovery_only": instance.recovery_only,
            "deployment": instance.registry.deployment_profile(),
            "auth_policy": auth_policy,
            "control_healthy": instance.control_healthy,
            "tokenizer_implementation_verified": (
                instance.compiler.tokenizer_implementation_digest is not None
            ),
            "engine_identity_verified": capabilities.verified,
            "capabilities": capabilities,
            "compiler": instance.compiler.profile(),
            "admission": instance.admission.snapshot(),
            "health": instance.health_profile(),
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

    @management.get("/requests/{request_id}/progress")
    async def request_progress(request_id: str, request: Request):
        return runtime(request).request_progress(request_id)

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
        if instance is not None:
            current = instance
        else:
            # create_app() without an instance requires settings to build one.
            assert settings is not None
            current = await build_runtime(settings)
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
