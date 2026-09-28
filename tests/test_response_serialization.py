"""HTTP response schema/serialization across the shared SDK contract fixtures."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.encoders import jsonable_encoder

from jev_runtime.api import create_app
from jev_runtime.schema import DecisionResponse
from jev_runtime.telemetry import parse_timing_header

CASES = json.loads((Path(__file__).parent / "fixtures/decision-responses.json").read_text())[
    "valid"
]


@pytest.mark.parametrize("name", CASES)
async def test_declared_response_preserves_wire_contract(runtime, question, monkeypatch, name):
    payload = {**CASES[name], "request_id": '响应-"\\\n🙂'}
    result = DecisionResponse.model_validate(payload)

    async def decide(*args, trace, **kwargs):
        trace.switch("pin")
        trace.finish()
        return result

    monkeypatch.setattr(runtime, "decide", decide)
    app = create_app(instance=runtime, api_key="data", admin_key="admin")
    app.state.jev_runtime = runtime
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/decisions",
            headers={"Authorization": "Bearer data", "X-Jev-Timing": "1"},
            json={
                "model": "model",
                "input": {"text": "退款"},
                "questions": [question.model_dump()],
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.headers["X-Jev-Worker"] == runtime.registry.owner
        parse_timing_header(response.headers["Server-Timing"])
        assert response.json() == jsonable_encoder(result)
        assert DecisionResponse.model_validate_json(response.content) == result
        schema = (await client.get("/openapi.json")).json()
        assert schema["paths"]["/v1/decisions"]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"] == {"$ref": "#/components/schemas/DecisionResponse"}
        assert set(schema["components"]["schemas"]["DecisionResponse"]["properties"]) == set(
            payload
        )
