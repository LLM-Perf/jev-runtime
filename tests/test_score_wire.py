from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from jev_runtime.plugin_api import ScoreWire, install_plugin_routes


def score_body(**overrides):
    return {
        "request_id": "raw-request",
        "question_id": "question",
        "input_ids": [0, 42, 42],
        "label_ids": [0, 2],
        **overrides,
    }


COERCED_TOKENS = [
    ("input_ids", True),
    ("input_ids", 1.0),
    ("input_ids", "2"),
    ("label_ids", False),
    ("label_ids", 1.0),
    ("label_ids", "2"),
]


@pytest.mark.parametrize("field, token", COERCED_TOKENS)
def test_score_wire_rejects_coerced_token_ids(field, token):
    with pytest.raises(ValidationError) as exc:
        ScoreWire(**score_body(**{field: [token]}))
    assert ((field, 0), "int_type") in [
        (error["loc"], error["type"]) for error in exc.value.errors()
    ]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"input_ids": [-1]}, "Token IDs must be nonnegative"),
        ({"label_ids": [-1]}, "Token IDs must be nonnegative"),
        ({"label_ids": [2, 2]}, "Label token IDs must be distinct"),
        ({"input_ids": []}, "at least 1 item"),
        ({"label_ids": []}, "at least 1 item"),
    ],
)
def test_score_wire_retains_token_constraints(overrides, message):
    with pytest.raises(ValidationError, match=message):
        ScoreWire(**score_body(**overrides))


def plugin_app(monkeypatch, runtime, managed):
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.delenv("JEV_CONFIG", raising=False)
    app = FastAPI()
    app.state.jev_backend = runtime.backend
    if managed:
        app.state.jev_runtime = runtime
    install_plugin_routes(app)
    return app


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("field, token", COERCED_TOKENS)
async def test_raw_http_rejects_coercion_before_execution(
    monkeypatch, runtime, managed, field, token
):
    app = plugin_app(monkeypatch, runtime, managed)
    score_raw = AsyncMock(wraps=runtime.score_raw)
    begin_raw_work = Mock(wraps=runtime.registry.begin_raw_work)
    monkeypatch.setattr(runtime, "score_raw", score_raw)
    monkeypatch.setattr(runtime.registry, "begin_raw_work", begin_raw_work)
    before = runtime.registry.list()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/plugins/jev-runtime/v1/scores",
            json=score_body(**{field: [token]}),
            headers={"Authorization": "Bearer test-key"},
        )
    assert response.status_code == 422
    assert (["body", field, 0], "int_type") in [
        (error["loc"], error["type"]) for error in response.json()["detail"]
    ]
    score_raw.assert_not_awaited()
    begin_raw_work.assert_not_called()
    assert not runtime.backend.calls
    assert runtime.registry.list() == before


@pytest.mark.parametrize("managed", [False, True])
async def test_raw_http_preserves_integer_token_ids(monkeypatch, runtime, managed):
    app = plugin_app(monkeypatch, runtime, managed)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/plugins/jev-runtime/v1/scores",
            json=score_body(),
            headers={"Authorization": "Bearer test-key"},
        )
    assert response.status_code == 200
    assert response.json()["request_id"] == "raw-request"
    assert len(runtime.backend.calls) == 1
    actual = runtime.backend.calls[0]
    assert actual.input_ids == (0, 42, 42)
    assert actual.label_ids == (0, 2)
    assert all(type(token) is int for token in (*actual.input_ids, *actual.label_ids))
