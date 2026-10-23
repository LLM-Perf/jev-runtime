import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from jev_runtime import config
from jev_runtime.adapters import LORA_BASE_PROFILES
from jev_runtime.backends.vllm import VLLMHTTP
from jev_runtime.errors import JevError


@pytest.fixture
def settings(tmp_path):
    return config.Settings(
        backend="vllm",
        model_id="fixture",
        model_revision="a" * 40,
        registry_path=str(tmp_path / "registry.db"),
    )


@pytest.fixture
async def resources(monkeypatch):
    """Retain failed constructors' real resources so GC cannot hide a leak."""
    original_registry = config.Registry
    registries, connections, backends = [], [], []

    def registry(*args, **kwargs):
        result = original_registry(*args, **kwargs)
        registries.append(result)
        with result._connection() as connection:
            connections.append(connection)
        return result

    def backend(settings, api_key):
        result = VLLMHTTP(settings.engine_url, api_key)
        backends.append(result)
        return result

    registry_factory = Mock(side_effect=registry)
    backend_factory = Mock(side_effect=backend)
    monkeypatch.setattr(config, "Registry", registry_factory)
    monkeypatch.setattr(config, "create_backend", backend_factory)
    yield SimpleNamespace(
        registries=registries,
        connections=connections,
        backends=backends,
        registry_factory=registry_factory,
        backend_factory=backend_factory,
        native=SimpleNamespace(close=AsyncMock(), managed_lora=False),
    )
    for result in backends:
        await result.client.aclose()
    for result in registries:
        result.close()


def assert_connections_closed(resources):
    for connection in resources.connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")


async def test_deployment_pair_is_checked_before_loading_or_allocating(
    settings, resources, monkeypatch
):
    settings = settings.model_copy(update={"deployment_id": "blue"})
    loader = Mock(side_effect=AssertionError("must not load the tokenizer"))
    monkeypatch.setattr(config, "load_compiler", loader)

    with pytest.raises(ValueError, match="requires an immutable release_id"):
        await config.build_runtime(settings)

    loader.assert_not_called()
    resources.backend_factory.assert_not_called()
    resources.registry_factory.assert_not_called()
    assert not await asyncio.to_thread(Path(settings.registry_path).exists)


@pytest.mark.parametrize("native,workers", [(False, 1), (True, 2)])
async def test_adapter_topology_is_checked_before_loading_or_allocating(
    settings, resources, monkeypatch, native, workers
):
    settings = settings.model_copy(
        update={"workers": workers, "adapters": config.AdapterSettings(enabled=True)}
    )
    loader = Mock(side_effect=AssertionError("must not load the tokenizer"))
    monkeypatch.setattr(config, "load_compiler", loader)

    with pytest.raises(JevError, match="native single-worker") as error:
        await config.build_runtime(settings, native_backend=resources.native if native else None)

    assert error.value.code == "adapter_profile"
    loader.assert_not_called()
    resources.backend_factory.assert_not_called()
    resources.registry_factory.assert_not_called()
    resources.native.close.assert_not_awaited()
    assert resources.native.managed_lora is False


async def test_unsupported_adapter_base_is_checked_before_loading(settings, resources, monkeypatch):
    settings = settings.model_copy(update={"adapters": config.AdapterSettings(enabled=True)})
    loader = Mock(side_effect=AssertionError("must not load the tokenizer"))
    monkeypatch.setattr(config, "load_compiler", loader)

    with pytest.raises(JevError, match="lifecycle-tested checkpoint"):
        await config.build_runtime(settings, native_backend=resources.native)

    loader.assert_not_called()
    resources.registry_factory.assert_not_called()
    resources.native.close.assert_not_awaited()
    assert resources.native.managed_lora is False


async def test_model_identity_is_validated_before_resource_creation(settings, resources, compiler):
    settings = settings.model_copy(update={"model_id": ""})

    with pytest.raises(ValidationError):
        await config.build_runtime(settings, compiler=compiler)

    resources.backend_factory.assert_not_called()
    resources.registry_factory.assert_not_called()
    assert not await asyncio.to_thread(Path(settings.registry_path).exists)


async def test_unpublished_registry_migration_closes_created_transport(
    settings, resources, compiler
):
    Path(settings.registry_path).with_name("migration-intent.json").write_text(
        '{"database_name":"registry.db","operation_id":"incomplete"}'
    )

    with pytest.raises(ValueError, match="migration is incomplete"):
        await config.build_runtime(settings, compiler=compiler)

    assert len(resources.backends) == 1
    assert resources.backends[0].client.is_closed
    assert not resources.connections


async def test_registry_cleanup_error_does_not_skip_transport_cleanup(
    settings, resources, compiler, monkeypatch
):
    from jev_runtime.registry import Registry

    construction_error = RuntimeError("construction failed")
    cleanup_error = RuntimeError("registry cleanup failed")
    original_close = Registry.close

    def close(registry):
        original_close(registry)
        raise cleanup_error

    with monkeypatch.context() as patch:
        patch.setattr(config, "Runtime", Mock(side_effect=construction_error))
        patch.setattr(Registry, "close", close)
        with pytest.raises(RuntimeError, match="registry cleanup failed") as error:
            await config.build_runtime(settings, compiler=compiler)

    assert error.value is cleanup_error
    assert error.value.__context__ is construction_error
    assert_connections_closed(resources)
    assert resources.backends[0].client.is_closed


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("stage", ["registry", "deployment", "admission", "runtime", "cancelled"])
async def test_failed_construction_closes_only_owned_resources(
    settings, resources, compiler, monkeypatch, native, stage
):
    settings = settings.model_copy(update={"deployment_id": "blue", "release_id": "release-1"})
    if native:
        model, revision = next(iter(LORA_BASE_PROFILES))
        settings = settings.model_copy(
            update={
                "model_id": model,
                "model_revision": revision,
                "adapters": config.AdapterSettings(enabled=True),
            }
        )
    failure = asyncio.CancelledError() if stage == "cancelled" else RuntimeError(stage)
    if stage == "registry":
        resources.registry_factory.side_effect = failure
    elif stage == "deployment":
        from jev_runtime.registry import Registry

        monkeypatch.setattr(Registry, "tag_deployment", Mock(side_effect=failure))
    else:
        target = "SharedAdmission" if stage == "admission" else "Runtime"
        monkeypatch.setattr(config, target, Mock(side_effect=failure))

    with pytest.raises(type(failure)) as error:
        await config.build_runtime(
            settings, native_backend=resources.native if native else None, compiler=compiler
        )

    assert error.value is failure
    if stage != "registry":
        assert len(resources.connections) == 1
    assert_connections_closed(resources)
    if native:
        resources.backend_factory.assert_not_called()
        resources.native.close.assert_not_awaited()
        assert resources.native.managed_lora is False
    else:
        assert len(resources.backends) == 1
        assert resources.backends[0].client.is_closed


@pytest.mark.parametrize("native", [False, True])
async def test_success_transfers_resource_ownership_to_runtime(
    settings, resources, compiler, native
):
    settings = settings.model_copy(update={"deployment_id": "blue", "release_id": "release-1"})
    if native:
        model, revision = next(iter(LORA_BASE_PROFILES))
        settings = settings.model_copy(
            update={
                "model_id": model,
                "model_revision": revision,
                "adapters": config.AdapterSettings(enabled=True),
            }
        )
    runtime = await config.build_runtime(
        settings, native_backend=resources.native if native else None, compiler=compiler
    )
    try:
        assert runtime.registry.deployment_profile()["release"] == "release-1"
        assert resources.connections[0].execute("SELECT 1").fetchone()[0] == 1
        assert runtime.expected_model == config.model_identity(settings, compiler)
        if native:
            assert runtime.backend is resources.native
            resources.native.close.assert_not_awaited()
            assert resources.native.managed_lora is True
            assert runtime.adapter_store is not None
        else:
            assert not resources.backends[0].client.is_closed
    finally:
        await runtime.close()

    assert_connections_closed(resources)
    if native:
        resources.native.close.assert_awaited_once()
    else:
        assert resources.backends[0].client.is_closed
