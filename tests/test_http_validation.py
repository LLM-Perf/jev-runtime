import json
import math
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from jev_runtime.api import create_app
from jev_runtime.plugin_api import install_plugin_routes
from jev_runtime.schema import DecisionRequest, TextInput


def decision_json(value="0.5"):
    return (
        '{"model":"model","input":{"text":"refund"},"questions":'
        '[{"id":"q","type":"score","instruction":"Rate","options":'
        '[{"id":"a","description":"A","value":' + value + "},"
        '{"id":"b","description":"B","value":1}]}]}'
    )


@pytest.fixture
def validation_app(request, runtime, monkeypatch):
    if request.param == "plugin":
        monkeypatch.setenv("JEV_API_KEY", "data")
        monkeypatch.setenv("JEV_ADMIN_KEY", "admin")
        monkeypatch.delenv("JEV_CONFIG", raising=False)
        app = FastAPI()
        install_plugin_routes(app)
        app.state.jev_backend = runtime.backend
        prefix = "/plugins/jev-runtime"
    else:
        app = create_app(instance=runtime, api_key="data", admin_key="admin")
        prefix = ""
    app.state.jev_runtime = runtime
    return app, prefix


@pytest.mark.parametrize("validation_app", ["gateway", "plugin"], indirect=True)
@pytest.mark.parametrize("token, rendered", [("1e999", "inf"), ("-1e999", "-inf"), ("NaN", "nan")])
@pytest.mark.parametrize("path", ["/v1/decisions", "/admin/compile", "/admin/quiescence"])
async def test_nonfinite_validation_errors_are_json_client_errors(
    validation_app, runtime, monkeypatch, token, rendered, path
):
    app, prefix = validation_app
    decide = AsyncMock(wraps=runtime.decide)
    quiesce = AsyncMock(wraps=runtime.quiesce)
    monkeypatch.setattr(runtime, "decide", decide)
    monkeypatch.setattr(runtime, "quiesce", quiesce)
    body = (
        '{"expected_generation":0,"timeout_seconds":' + token + "}"
        if path.endswith("quiescence")
        else decision_json(token)
    )
    key = "admin" if path.startswith("/admin") else "data"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post(
            prefix + path,
            content=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert len(detail) == 1
    assert detail[0]["type"] == "finite_number"
    assert detail[0]["input"] == rendered
    assert detail[0]["loc"] == (
        ["body", "timeout_seconds"]
        if path.endswith("quiescence")
        else ["body", "questions", 0, "options", 0, "value"]
    )
    decide.assert_not_called()
    quiesce.assert_not_called()
    assert not runtime.backend.calls
    assert not runtime.registry.list()["leases"]


@pytest.mark.parametrize("validation_app", ["plugin"], indirect=True)
@pytest.mark.parametrize("token, rendered", [("1e999", "inf"), ("-1e999", "-inf"), ("NaN", "nan")])
async def test_nonfinite_raw_token_ids_do_not_reach_engine(
    validation_app, runtime, monkeypatch, token, rendered
):
    app, prefix = validation_app
    score_raw = AsyncMock(wraps=runtime.score_raw)
    monkeypatch.setattr(runtime, "score_raw", score_raw)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post(
            prefix + "/v1/scores",
            content='{"request_id":"r","question_id":"q","input_ids":['
            + token
            + '],"label_ids":[2]}',
            headers={"Authorization": "Bearer data", "Content-Type": "application/json"},
        )
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["body", "input_ids", 0]
    assert response.json()["detail"][0]["input"] == rendered
    score_raw.assert_not_called()
    assert not runtime.backend.calls
    assert not runtime.registry.list()["leases"]


@pytest.mark.parametrize("validation_app", ["gateway", "plugin"], indirect=True)
async def test_nonfinite_input_nested_in_extra_field_is_serializable(validation_app, runtime):
    app, prefix = validation_app
    body = decision_json()[:-1] + ',"unexpected":{"values":[1e999,0.5,-1e999,NaN]}}'
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post(
            prefix + "/v1/decisions",
            content=body,
            headers={"Authorization": "Bearer data", "Content-Type": "application/json"},
        )
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == [
        {
            "type": "extra_forbidden",
            "loc": ["body", "unexpected"],
            "msg": "Extra inputs are not permitted",
            "input": {"values": ["inf", 0.5, "-inf", "nan"]},
        }
    ]
    assert not runtime.backend.calls
    assert not runtime.registry.list()["leases"]


@pytest.mark.parametrize("validation_app", ["gateway", "plugin"], indirect=True)
@pytest.mark.parametrize(
    "body",
    [
        "{}",
        '{"model":7.5,"input":{"text":"refund"}}',
        '{"model":',
        '{"model":"model","input":{"text":"refund"},"questions":'
        '[{"id":"q","type":"choice","instruction":"Choose"}]}',
    ],
)
async def test_regular_validation_keeps_fastapi_response(validation_app, body):
    app, prefix = validation_app
    reference = FastAPI()

    @reference.post("/reference")
    async def reference_route(body: DecisionRequest):
        raise AssertionError("Invalid requests must not run the endpoint")

    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=reference), base_url="http://test"
        ) as reference_client,
    ):
        expected = await reference_client.post(
            "/reference", content=body, headers={"Content-Type": "application/json"}
        )
        actual = await client.post(
            prefix + "/v1/decisions",
            content=body,
            headers={"Authorization": "Bearer data", "Content-Type": "application/json"},
        )
    assert actual.status_code == expected.status_code == 422
    assert actual.json() == expected.json()


@pytest.mark.parametrize("register_first", [False, True])
async def test_host_validation_handler_is_preserved(runtime, monkeypatch, register_first):
    monkeypatch.setenv("JEV_API_KEY", "data")
    monkeypatch.setenv("JEV_ADMIN_KEY", "admin")
    monkeypatch.delenv("JEV_CONFIG", raising=False)
    app = FastAPI()
    calls = []

    async def host_handler(request, exc):
        calls.append(request.url.path)
        return JSONResponse({"host_validation": True}, status_code=400)

    if register_first:
        app.add_exception_handler(RequestValidationError, host_handler)
    install_plugin_routes(app)
    if not register_first:
        app.add_exception_handler(RequestValidationError, host_handler)
    app.state.jev_runtime = runtime

    @app.post("/plugins/jev-runtime/native")
    async def native_route(body: TextInput):
        raise AssertionError("Invalid requests must not run the endpoint")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        native = await client.post(
            "/plugins/jev-runtime/native",
            content='{"text":1e999}',
            headers={"Content-Type": "application/json"},
        )
        ordinary = await client.post(
            "/plugins/jev-runtime/v1/decisions",
            json={},
            headers={"Authorization": "Bearer data"},
        )
        overflow = await client.post(
            "/plugins/jev-runtime/v1/decisions",
            content=decision_json("1e999"),
            headers={"Authorization": "Bearer data", "Content-Type": "application/json"},
        )
    assert native.status_code == ordinary.status_code == 400
    assert native.json() == ordinary.json() == {"host_validation": True}
    assert overflow.status_code == 422
    assert calls == ["/plugins/jev-runtime/native", "/plugins/jev-runtime/v1/decisions"]
    assert app.exception_handlers[RequestValidationError] is host_handler


@pytest.mark.parametrize("response_error", [False, True])
async def test_internal_validation_errors_remain_server_errors(
    runtime, monkeypatch, response_error
):
    if response_error:
        error = ResponseValidationError(
            [{"type": "finite_number", "loc": ("response", "value"), "input": math.inf}]
        )
    else:
        with pytest.raises(ValidationError) as validation:
            TextInput(text="")
        error = validation.value
    monkeypatch.setattr(runtime, "decide", AsyncMock(side_effect=error))
    app = create_app(instance=runtime)
    app.state.jev_runtime = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/decisions", content=decision_json(), headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 500


@pytest.mark.parametrize("validation_app", ["gateway", "plugin"], indirect=True)
async def test_finite_score_values_still_execute(validation_app, runtime):
    app, prefix = validation_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            prefix + "/v1/decisions",
            json=json.loads(decision_json()),
            headers={"Authorization": "Bearer data"},
        )
    assert response.status_code == 200, response.text
    assert isinstance(response.json()["answers"]["q"]["value"], float)
    assert runtime.backend.calls
    assert not runtime.registry.list()["leases"]
