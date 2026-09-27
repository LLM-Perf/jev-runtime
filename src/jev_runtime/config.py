from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from jev_runtime.adapters import AdapterStore, validate_lora_base
from jev_runtime.backends.sglang import SGLangHTTP
from jev_runtime.backends.vllm import VLLMHTTP
from jev_runtime.compiler import Compiler
from jev_runtime.errors import JevError
from jev_runtime.registry import Registry
from jev_runtime.runtime import Runtime
from jev_runtime.schema import Bundle, Contract, ModelIdentity
from jev_runtime.shared_admission import SharedAdmission


class AdmissionSettings(Contract):
    max_requests: int = Field(default=64, gt=0)
    max_tokens: int = Field(default=1_048_576, gt=0)
    max_queue: int = Field(default=256, gt=0)
    max_tenant_requests: int = Field(default=16, gt=0)
    max_tenant_tokens: int = Field(default=262_144, gt=0)
    max_tenant_queue: int = Field(default=64, gt=0)
    max_branches: int = Field(default=1024, gt=0)
    max_tenant_branches: int = Field(default=256, gt=0)


class AdapterSettings(Contract):
    enabled: bool = False
    store_path: str = ".jev/adapters"
    allowed_roots: tuple[str, ...] = ()
    max_bytes: int = Field(default=268435456, ge=1, le=2147483648)
    operation_timeout_seconds: int = Field(default=120, ge=1, le=300)


class Settings(Contract):
    backend: Literal["sglang", "vllm"]
    engine_url: str = "http://127.0.0.1:30000"
    model_id: str
    model_revision: str = Field(pattern=r"^(?:[a-fA-F0-9]{40,64}|local-sha256:[a-fA-F0-9]{64})$")
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "bfloat16"
    quantization: str | None = None
    registry_path: str = ".jev/registry.db"
    host: str = "127.0.0.1"
    port: int = Field(default=8795, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=128)
    compiler_cache_tokens: int = Field(default=262144, ge=0)
    compiler_cache_entries: int = Field(default=256, ge=0)
    api_key_env: str = "JEV_API_KEY"
    admin_key_env: str = "JEV_ADMIN_KEY"
    engine_key_env: str = "JEV_ENGINE_API_KEY"
    bootstrap_alias: str | None = None
    bootstrap_bundle_id: str = "default"
    tenant_key_envs: dict[str, str] = Field(default_factory=dict)
    admission: AdmissionSettings = Field(default_factory=AdmissionSettings)
    adapters: AdapterSettings = Field(default_factory=AdapterSettings)


def tenant_keys(settings: Settings) -> dict[str, str]:
    keys = {}
    for tenant, env_name in settings.tenant_key_envs.items():
        key = os.environ.get(env_name)
        if not tenant or tenant == "default" or not key:
            raise ValueError("Tenant names must be unique, non-default and have a configured key")
        keys[tenant] = key
    if len(set(keys.values())) != len(keys):
        raise ValueError("Tenant API keys must be distinct")
    return keys


def load_settings(path: str | Path) -> Settings:
    with Path(path).open() as file:
        return Settings.model_validate(yaml.safe_load(file))


def load_compiler(settings: Settings) -> Compiler:
    from transformers import AutoTokenizer

    kwargs = {"trust_remote_code": False}
    revision = settings.tokenizer_revision or settings.model_revision
    if not Path(settings.tokenizer or settings.model_id).is_dir():
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer or settings.model_id, **kwargs)
    return Compiler(tokenizer, settings.compiler_cache_tokens, settings.compiler_cache_entries)


def model_identity(settings: Settings, compiler: Compiler) -> ModelIdentity:
    return ModelIdentity(
        id=settings.model_id,
        revision=settings.model_revision,
        tokenizer_digest=compiler.tokenizer_digest,
        tokenizer_implementation_digest=compiler.tokenizer_implementation_digest,
        template_digest=compiler.template_digest,
        dtype=settings.dtype,
        quantization=settings.quantization,
    )


async def build_runtime(
    settings: Settings, native_backend=None, compiler: Compiler | None = None
) -> Runtime:
    compiler = compiler or await asyncio.to_thread(load_compiler, settings)
    if native_backend is not None:
        backend = native_backend
    elif settings.backend == "sglang":
        backend = SGLangHTTP(
            settings.engine_url, settings.model_id, os.environ.get(settings.engine_key_env)
        )
    else:
        backend = VLLMHTTP(settings.engine_url, os.environ.get(settings.engine_key_env))
    identity = (
        f"{settings.backend}:{settings.engine_url}:{settings.model_id}:{settings.model_revision}"
    )
    store = None
    if settings.adapters.enabled:
        if native_backend is None or settings.workers != 1:
            raise JevError(
                "adapter_profile", "Managed LoRA requires a native single-worker plugin", 409
            )
        validate_lora_base(settings.model_id, settings.model_revision)
        store = AdapterStore(
            settings.adapters.store_path,
            settings.adapters.allowed_roots,
            settings.adapters.max_bytes,
        )
        backend.managed_lora = True
    registry = Registry(settings.registry_path)
    return Runtime(
        backend,
        compiler,
        registry,
        identity,
        settings.model_id,
        admission=SharedAdmission(registry, identity, **settings.admission.model_dump()),
        expected_model=model_identity(settings, compiler),
        adapter_store=store,
        adapter_timeout=settings.adapters.operation_timeout_seconds,
    )


async def bootstrap(runtime: Runtime, settings: Settings) -> None:
    if settings.bootstrap_alias is None:
        return
    bundle = Bundle(
        id=settings.bootstrap_bundle_id, version=1, model=model_identity(settings, runtime.compiler)
    )
    runtime.registry.upload(bundle)
    async with asyncio.timeout(180):
        while not runtime.is_prepared(bundle.reference):
            state = runtime.registry.inspect(bundle.reference)["state"]
            if state == "PREPARING":
                # Another local API worker owns the initial preparation. Once
                # published, this worker still runs its own engine canary.
                await asyncio.sleep(0.05)
                continue
            try:
                await runtime.prepare(bundle.reference)
            except JevError as exc:
                if exc.code not in {"invalid_state", "not_ready"}:
                    raise
                if runtime.registry.inspect(bundle.reference)["state"] not in {
                    "PREPARING",
                    "READY",
                    "ACTIVE",
                    "DRAINING",
                }:
                    raise
                await asyncio.sleep(0.05)
    async with asyncio.timeout(180):
        while True:
            routes = {route["alias"]: route for route in runtime.registry.list()["routes"]}
            if settings.bootstrap_alias in routes:
                return
            try:
                runtime.activate(settings.bootstrap_alias, bundle.reference, 0)
                return
            except JevError as exc:
                if exc.code == "replicas_not_ready":
                    await asyncio.sleep(0.05)
                    continue
                if exc.code != "generation_conflict":
                    raise
                current = {r["alias"]: r for r in runtime.registry.list()["routes"]}
                if current.get(settings.bootstrap_alias, {}).get("ref") != bundle.reference:
                    raise
                return
