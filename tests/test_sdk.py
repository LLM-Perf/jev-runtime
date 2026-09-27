import copy
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from jev_runtime.errors import JevError
from jev_runtime.schema import DecisionResponse
from jev_runtime.sdk import AsyncJevClient, JevClient, _decode

CASES = json.loads((Path(__file__).parent / "fixtures/decision-responses.json").read_text())


@pytest.mark.parametrize("name", CASES["valid"])
def test_valid_response_semantics(name):
    payload = CASES["valid"][name]
    assert _decode(httpx.Response(200, json=payload)).model_dump(mode="json") == payload


@pytest.mark.parametrize("case", CASES["invalid"], ids=lambda case: case["name"])
def test_corrupt_response_is_not_a_success(case):
    payload = copy.deepcopy(CASES["valid"][case["template"]])
    for change in case["changes"]:
        target = payload
        for part in change["path"][:-1]:
            target = target[part]
        target[change["path"][-1]] = change["value"]
    with pytest.raises(ValidationError):
        DecisionResponse.model_validate(payload)
    with pytest.raises(JevError) as exc:
        _decode(httpx.Response(200, json=payload))
    assert (exc.value.status_code, exc.value.code) == (502, "invalid_response")


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_public_clients_preserve_prefix_and_validate_cancellation(asynchronous):
    calls = []
    replies = iter(
        [
            httpx.Response(200, json=CASES["valid"]["boolean"]),
            httpx.Response(200, json={"cancelled": False}),
            httpx.Response(200, json={"cancelled": True}),
            httpx.Response(200, json={"cancelled": "false"}),
            httpx.Response(200, content=b"not json"),
            httpx.Response(
                409, json={"error": {"code": "generation_conflict", "message": "Changed"}}
            ),
            httpx.Response(503, content=b"gateway unavailable"),
            httpx.Response(502, json={"error": []}),
        ]
    )

    def handle(request):
        calls.append(request)
        return next(replies)

    base = "http://localhost/plugins/jev-runtime"
    client = AsyncJevClient(base) if asynchronous else JevClient(base)
    if asynchronous:
        await client.close()
        client.http = httpx.AsyncClient(base_url=base, transport=httpx.MockTransport(handle))
    else:
        client.close()
        client.http = httpx.Client(base_url=base, transport=httpx.MockTransport(handle))

    async def call(method, argument):
        value = getattr(client, method)(argument)
        return await value if asynchronous else value

    try:
        result = await call("decide", {"model": "fixture", "input": {"text": "sample"}})
        assert result.answers["q"].value is True
        assert not await call("cancel", "owned:request")
        assert await call("cancel", "owned:request")
        for status, code in [
            (502, "invalid_response"),
            (502, "invalid_response"),
            (409, "generation_conflict"),
            (503, "http_error"),
            (502, "http_error"),
        ]:
            with pytest.raises(JevError) as exc:
                await call("cancel", "owned:request")
            assert (exc.value.status_code, exc.value.code) == (status, code)
        assert len(calls) == 8  # Mutations are not retried automatically.
        assert calls[0].url.path == "/plugins/jev-runtime/v1/decisions"
        assert all(c.url.path.endswith("/v1/requests/owned:request/cancel") for c in calls[1:])
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()
