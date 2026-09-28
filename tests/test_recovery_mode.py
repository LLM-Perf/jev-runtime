import httpx
import pytest
from conftest import ControlledEngine
from fastapi import FastAPI

from jev_runtime.api import create_app
from jev_runtime.backends.base import Capabilities
from jev_runtime.config import Settings, bootstrap
from jev_runtime.errors import JevError
from jev_runtime.plugin_api import install_plugin_routes
from jev_runtime.registry import Registry
from jev_runtime.runtime import Runtime
from jev_runtime.schema import DecisionRequest, TextInput


def recovery_instance(runtime):
    return Runtime(
        ControlledEngine(),
        runtime.compiler,
        Registry(runtime.registry.path),
        runtime.backend_identity,
        runtime.model_id,
        recovery_only=True,
    )


async def test_recovery_mode_preserves_active_and_preparing_state_until_confirmed_abort(
    runtime, bundle, question, monkeypatch
):
    pending = bundle.model_copy(update={"version": 2})
    runtime.registry.upload(pending)
    _, lease = runtime.registry.begin_prepare(pending.reference, runtime.backend_identity, "boot")
    branches = ["boot.branch.0", "boot.branch.1"]
    runtime.registry.record_branches(lease, branches)
    before = runtime.registry.list()
    instance = recovery_instance(runtime)
    try:
        await instance.start()
        await bootstrap(
            instance,
            Settings(
                backend="sglang",
                model_id="fixture",
                model_revision="a" * 40,
                bootstrap_alias="should-not-publish",
                bootstrap_bundle_id="should-not-upload",
            ),
        )
        assert not instance.backend.calls
        assert instance.registry.list() == before
        assert not instance._prepared and not instance.control_healthy
        assert instance._control_task is None and instance._health_task is None
        assert instance.registry.owner not in {
            row["worker_id"] for row in instance.registry.worker_status()
        }
        with pytest.raises(RuntimeError, match="already started"):
            await instance.start()
        with pytest.raises(JevError, match="Owner is alive"):
            await instance.recover_cancelled("boot")
        assert not instance.backend.cancelled
        monkeypatch.setattr(Registry, "_owner_status", staticmethod(lambda identity: "dead"))
        instance.backend.fail_cancel = True
        with pytest.raises(RuntimeError, match="abort failure"):
            await instance.recover_cancelled("boot")
        assert instance.registry.list() == before
        instance.backend.fail_cancel = False
        assert await instance.recover_cancelled("boot")
        assert instance.backend.cancelled == branches
        assert not instance.registry.list()["leases"]
        assert instance.registry.inspect(pending.reference)["state"] == "FAILED"
        assert instance.registry.list()["routes"] == before["routes"]
        assert not await instance.recover_cancelled("boot")
        body = DecisionRequest(model="model", input=TextInput(text="ready"), questions=(question,))
        for operation in (
            lambda: instance.activate("model", bundle.reference, 1),
            lambda: instance.compile_preview(body),
        ):
            with pytest.raises(JevError) as exc:
                operation()
            assert exc.value.code == "recovery_only"
        for operation in (
            lambda: instance.decide(body),
            lambda: instance.prepare(bundle.reference),
            lambda: instance.register_adapter("adapter", "/unused"),
            lambda: instance.change_adapter("adapter@1", "unload", True),
        ):
            with pytest.raises(JevError) as exc:
                await operation()
            assert exc.value.code == "recovery_only"
        assert not instance.backend.calls and not instance.registry.list()["leases"]
    finally:
        await instance.close()


@pytest.mark.parametrize("plugin", [False, True])
async def test_recovery_http_blocks_writes_and_readiness_but_allows_admin_recovery(
    runtime, bundle, question, monkeypatch, plugin
):
    instance = recovery_instance(runtime)
    await instance.start()
    if plugin:
        monkeypatch.setenv("JEV_API_KEY", "data")
        monkeypatch.setenv("JEV_ADMIN_KEY", "admin")
        monkeypatch.delenv("JEV_CONFIG", raising=False)
        app = FastAPI()
        app.state.jev_backend = instance.backend
        install_plugin_routes(app)
        prefix = "/plugins/jev-runtime"
    else:
        app = create_app(instance=instance, api_key="data", admin_key="admin")
        prefix = ""
    app.state.jev_runtime = instance
    before = runtime.registry.list()
    body = {"model": "model", "input": {"text": "ready"}, "questions": [question.model_dump()]}
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test" + prefix
        ) as client:
            profile = await client.get("/admin/profile", headers={"Authorization": "Bearer admin"})
            assert profile.status_code == 200 and profile.json()["recovery_only"] is True
            assert profile.json()["control_healthy"] is False
            response = await client.get("/ready", headers={"Authorization": "Bearer data"})
            assert (
                response.status_code == 503 and response.json()["error"]["code"] == "recovery_only"
            )
            writes = [
                ("/v1/decisions", body, "data"),
                ("/v1/systemone", {"model": "model", "state": "ready", "questions": {}}, "data"),
                ("/v1/requests/unknown/cancel", {}, "data"),
                ("/admin/bundles", bundle.model_dump(), "admin"),
                ("/admin/bundles/prepare", {"reference": bundle.reference}, "admin"),
                (
                    "/admin/bundles/activate",
                    {"alias": "model", "reference": bundle.reference, "expected_generation": 1},
                    "admin",
                ),
                ("/admin/bundles/disable", {"alias": "model", "expected_generation": 1}, "admin"),
                ("/admin/bundles/retire", {"reference": bundle.reference}, "admin"),
                ("/admin/compile", body, "admin"),
                ("/admin/adapters/register", {"id": "adapter", "source": "/unused"}, "admin"),
                ("/admin/adapters/load", {"reference": "adapter@1"}, "admin"),
                ("/admin/adapters/unload", {"reference": "adapter@1", "recover": True}, "admin"),
            ]
            if plugin:
                writes.append(
                    (
                        "/v1/scores",
                        {
                            "request_id": "raw",
                            "question_id": "q",
                            "input_ids": [1],
                            "label_ids": [2],
                        },
                        "data",
                    )
                )
            for path, payload, key in writes:
                denied = await client.post(path, json=payload)
                assert denied.status_code == 401, path
                response = await client.post(
                    path, json=payload, headers={"Authorization": "Bearer " + key}
                )
                assert response.status_code == 503, (path, response.text)
                assert response.json()["error"]["code"] == "recovery_only"
            assert instance.registry.list() == before and not instance.backend.calls
            for key, status in [("data", 401), ("admin", 200)]:
                response = await client.post(
                    "/admin/requests/missing/recover", headers={"Authorization": "Bearer " + key}
                )
                assert response.status_code == status
            assert response.json() == {"recovered": False}
            assert not instance.backend.cancelled
    finally:
        await instance.close()


@pytest.mark.parametrize("mismatch", ["precision", "mode", "cancellation"])
async def test_recovery_start_keeps_engine_contract_guards(runtime, bundle, mismatch):
    instance = recovery_instance(runtime)
    instance.expected_model = bundle.model
    values = {
        "engine": "sglang",
        "version": "0.5.19",
        "model_id": "fixture",
        "model_dtype": "bfloat16",
        "readout_dtype": "bfloat16",
        "batch_invariant": False,
    }
    values.update(
        {"model_dtype": "float32"}
        if mismatch == "precision"
        else {"batch_invariant": True}
        if mismatch == "mode"
        else {"cancellation": False}
    )

    async def probe():
        return Capabilities(**values)

    instance.backend.probe = probe
    try:
        with pytest.raises(JevError) as error:
            await instance.start()
        assert (
            error.value.code
            == {
                "precision": "engine_precision_mismatch",
                "mode": "engine_execution_mismatch",
                "cancellation": "unsupported_engine",
            }[mismatch]
        )
        assert not instance.backend.calls
        assert instance.registry.owner not in {
            row["worker_id"] for row in instance.registry.worker_status()
        }
    finally:
        await instance.close()
