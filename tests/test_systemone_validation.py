from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from jev_runtime.api import create_app
from jev_runtime.errors import JevError
from jev_runtime.schema import TextInput
from jev_runtime.systemone import SystemOneRequest


def request_body():
    return {
        "model": "model",
        "state": "refund",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Classify",
                "criteria": {"a": "Refund", "b": "Other"},
            }
        },
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        pytest.param(("questions", "q", "criteria"), {"a": "Only"}, id="one-choice-option"),
        pytest.param(("questions", "q", "criteria"), {}, id="no-choice-options"),
        pytest.param(
            ("questions", "q", "criteria"), {"bad id": "Refund", "b": "Other"}, id="option-id"
        ),
        pytest.param(
            ("questions", "q", "criteria"), {"a": "", "b": "Other"}, id="empty-description"
        ),
        pytest.param(("questions", "q", "instructions"), "x" * 32769, id="long-instruction"),
        pytest.param(
            ("questions",),
            {"bad id": {"type": "noul", "instructions": "Check"}},
            id="question-id",
        ),
        pytest.param(
            ("questions", "q"),
            {"type": "rank", "instructions": "Rank", "criteria": {"a": "Only"}},
            id="one-rank-option",
        ),
        pytest.param(
            ("questions", "q"),
            {"type": "score", "instructions": "Score", "levels": ["Only"]},
            id="one-score-level",
        ),
        pytest.param(
            ("questions", "q"),
            {
                "type": "score",
                "instructions": "Score",
                "levels": ["Low", "High"],
                "values": ["not a number", 1],
            },
            id="non-numeric-score",
        ),
        pytest.param(
            ("questions", "q"),
            {
                "type": "score",
                "instructions": "Score",
                "levels": ["Low", "High"],
                "values": ["NaN", 1],
            },
            id="non-finite-score",
        ),
        pytest.param(
            ("questions", "q"),
            {
                "type": "score",
                "instructions": "Score",
                "levels": ["Low", "High"],
                "values": [None, 1],
            },
            id="partial-score-values",
        ),
        pytest.param(("state",), "", id="empty-state"),
        pytest.param(("state",), "x" * 1000001, id="long-state"),
        pytest.param(("model",), "", id="empty-model"),
        pytest.param(("model",), "x" * 257, id="long-model"),
        pytest.param(("bundle",), "", id="empty-bundle"),
    ],
)
async def test_conversion_errors_are_client_errors_before_execution(
    runtime, monkeypatch, path, value
):
    body = request_body()
    target = body
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    decide = AsyncMock(wraps=runtime.decide)
    monkeypatch.setattr(runtime, "decide", decide)
    app = create_app(instance=runtime)
    app.state.jev_runtime = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post("/v1/systemone", json=body)
    assert response.status_code == 422, response.text
    assert response.json() == {
        "error": {
            "code": "invalid_request",
            "message": "System One request does not satisfy the decision schema",
        }
    }
    decide.assert_not_called()
    assert not runtime.backend.calls
    assert not runtime.registry.list()["leases"]


@pytest.mark.parametrize(
    ("spec", "code"),
    [
        ({"type": "noul", "instructions": "Check", "extra": True}, "unsupported_field"),
        ({"type": "noul", "instructions": " "}, "invalid_question"),
        ({"type": "noul", "instructions": "Check", "levels": ["No"]}, "unsupported_field"),
        ({"type": "choice", "instructions": "Choose", "criteria": []}, "invalid_criteria"),
        ({"type": "score", "instructions": "Score", "levels": {}}, "invalid_levels"),
        (
            {"type": "score", "instructions": "Score", "levels": ["Low", "High"], "values": []},
            "invalid_values",
        ),
        ({"type": "other", "instructions": "Check"}, "unsupported_type"),
    ],
)
def test_existing_conversion_errors_keep_their_contract(spec, code):
    body = request_body()
    body["questions"] = {"q": spec}
    with pytest.raises(JevError) as error:
        SystemOneRequest.model_validate(body).to_decision()
    assert error.value.code == code
    assert error.value.status_code == 400


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "choice", "criteria": {"a": "Refund", "b": "Other"}},
        {"type": "rank", "criteria": {"a": "Refund", "b": "Other"}},
        {"type": "noul"},
        {"type": "boolean"},
        {"type": "score", "levels": ["Low", "High"]},
        {"type": "score", "levels": ["Low", "High"], "values": [0, 1]},
    ],
)
async def test_valid_conversions_still_execute(runtime, spec):
    body = request_body()
    body["questions"] = {"q": {**spec, "instructions": "Classify"}}
    app = create_app(instance=runtime)
    app.state.jev_runtime = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/systemone", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["compatibility"] == "jev-runtime-systemone-v1"
    answer = response.json()["answers"]["q"]
    assert answer["status"] == "answered"
    assert answer["type"] == ("noul" if spec["type"] == "boolean" else spec["type"])
    if spec["type"] == "choice":
        assert answer["choice"] == "a"
    elif spec["type"] in {"noul", "boolean"}:
        assert answer["p_true"] == answer["probabilities"]["true"]
    elif spec["type"] == "score":
        assert answer["score" if "values" in spec else "level"] == answer["value"]
    else:
        assert answer["value"] == ["a", "b"]
    assert runtime.backend.calls
    assert not runtime.registry.list()["leases"]


async def test_runtime_validation_errors_remain_server_errors(runtime, monkeypatch):
    with pytest.raises(ValidationError) as validation:
        TextInput(text="")
    monkeypatch.setattr(runtime, "decide", AsyncMock(side_effect=validation.value))
    app = create_app(instance=runtime)
    app.state.jev_runtime = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post("/v1/systemone", json=request_body())
    assert response.status_code == 500
