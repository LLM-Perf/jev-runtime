import httpx
import pytest
from fastapi import FastAPI

from jev_runtime.api import create_app, install_routes
from jev_runtime.plugin_api import install_plugin_routes


@pytest.fixture
def protected_app(request, runtime, monkeypatch):
    surface = request.param
    if surface == "tenant":
        app = FastAPI()
        install_routes(app, tenants={"first": "tenant-key", "second": "other-key"})
        path, key, error = "/ready", "tenant-key", "unauthorized"
    elif surface == "plugin":
        monkeypatch.setenv("JEV_API_KEY", "plugin-key")
        monkeypatch.setenv("JEV_ADMIN_KEY", "admin-key")
        monkeypatch.delenv("JEV_CONFIG", raising=False)
        app = FastAPI()
        install_plugin_routes(app)
        app.state.jev_backend = runtime.backend
        path, key, error = (
            "/plugins/jev-runtime/v1/scoring-capabilities",
            "plugin-key",
            "unauthorized",
        )
    else:
        app = create_app(instance=runtime, api_key="data-key", admin_key="admin-key")
        if surface == "admin":
            path, key, error = "/admin/bundles", "admin-key", "admin_unauthorized"
        else:
            path, key, error = "/ready", "data-key", "unauthorized"
    app.state.jev_runtime = runtime
    return app, path, key, error


@pytest.mark.parametrize("protected_app", ["default", "tenant", "admin", "plugin"], indirect=True)
@pytest.mark.parametrize("authorization", [b"Bearer \xff", b"Bearer \xc3\xa9", b"\xff"])
async def test_non_ascii_authorization_is_unauthorized(protected_app, authorization, runtime):
    app, path, _, code = protected_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.get(path, headers=[(b"authorization", authorization)])
    assert response.status_code == 401
    assert response.json()["error"]["code"] == code
    assert not runtime.backend.calls


@pytest.mark.parametrize("protected_app", ["plugin"], indirect=True)
@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("scores", {"request_id": "req", "question_id": "q", "input_ids": [1], "label_ids": [2]}),
        ("scores/cancel", {"request_id": "req"}),
    ],
)
async def test_non_ascii_plugin_authorization_cannot_dispatch_or_cancel(
    protected_app, route, body, runtime
):
    app, _, _, _ = protected_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/plugins/jev-runtime/v1/{route}",
            json=body,
            headers=[(b"authorization", b"Bearer \xff")],
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert not runtime.backend.calls
    assert not runtime.backend.cancelled


@pytest.mark.parametrize("protected_app", ["default", "tenant", "admin", "plugin"], indirect=True)
async def test_ascii_authorization_still_requires_an_exact_match(protected_app):
    app, path, key, code = protected_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for value in (b"", b"Basic " + key.encode(), b"Bearer " + key.encode() + b"-wrong"):
            denied = await client.get(path, headers=[(b"authorization", value)])
            assert denied.status_code == 401
            assert denied.json()["error"]["code"] == code
        missing = await client.get(path)
        assert missing.status_code == 401
        allowed = await client.get(path, headers=[(b"authorization", f"Bearer {key}".encode())])
        assert allowed.status_code == 200


@pytest.mark.parametrize("surface", ["default", "tenant", "admin", "plugin"])
async def test_non_ascii_configured_key_never_authenticates(surface, runtime, monkeypatch):
    key = "caf\u00e9"
    if surface == "tenant":
        app = FastAPI()
        install_routes(app, tenants={"first": key})
        path, code = "/ready", "unauthorized"
    elif surface == "plugin":
        monkeypatch.setenv("JEV_API_KEY", key)
        monkeypatch.setenv("JEV_ADMIN_KEY", "admin-key")
        monkeypatch.delenv("JEV_CONFIG", raising=False)
        app = FastAPI()
        install_plugin_routes(app)
        app.state.jev_backend = runtime.backend
        path, code = "/plugins/jev-runtime/v1/scoring-capabilities", "unauthorized"
    else:
        app = create_app(instance=runtime, api_key=key, admin_key=key)
        path = "/admin/bundles" if surface == "admin" else "/ready"
        code = "admin_unauthorized" if surface == "admin" else "unauthorized"
    app.state.jev_runtime = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        for value in (b"Bearer cafe", b"Bearer caf\xe9", b"Bearer caf\xc3\xa9"):
            response = await client.get(path, headers=[(b"authorization", value)])
            assert response.status_code == 401
            assert response.json()["error"]["code"] == code


async def test_non_ascii_tenant_key_does_not_block_valid_tenants(runtime):
    app = FastAPI()
    app.state.jev_runtime = runtime
    install_routes(app, tenants={"unsupported": "caf\u00e9", "valid": "tenant-key"})

    @app.middleware("http")
    async def record_tenant(request, call_next):
        response = await call_next(request)
        response.headers["X-Test-Tenant"] = getattr(request.state, "jev_tenant", "none")
        return response

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.get("/ready", headers=[(b"authorization", b"Bearer tenant-key")])
    assert response.status_code == 200
    assert response.headers["X-Test-Tenant"] == "valid"
