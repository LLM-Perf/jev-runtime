import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jev_runtime.api import install_routes
from jev_runtime.errors import JevError
from jev_runtime.schema import DecisionRequest
from jev_runtime.sdk import AsyncJevClient, JevClient


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("prefix", ["", "/plugins/jev-runtime"], ids=["root", "plugin"])
@pytest.mark.parametrize(
    ("request_id", "segment"),
    [
        (".", b"%2E"),
        ("..", b"%2E%2E"),
        ("...", b"..."),
        ("owned:request.v1", b"owned%3Arequest.v1"),
        ("请求", b"%E8%AF%B7%E6%B1%82"),
    ],
)
async def test_cancel_preserves_identifier_through_httpx_and_asgi(
    asynchronous, prefix, request_id, segment
):
    # All these identifiers are accepted for submitted decisions.
    DecisionRequest(request_id=request_id, model="model", input={"text": "sample"})
    app = FastAPI()
    instance = SimpleNamespace(
        capabilities=object(),
        recovery_only=False,
        registry=SimpleNamespace(owner="fixture"),
        cancel=AsyncMock(return_value=True),
    )
    app.state.jev_runtime = instance
    install_routes(app, prefix=prefix, api_key="data-key")
    raw_paths = []

    async def capture(scope, receive, send):
        if scope["type"] == "http":
            raw_paths.append(scope["raw_path"])
        await app(scope, receive, send)

    base = "http://test" + prefix
    headers = {"Authorization": "Bearer data-key"}
    if asynchronous:
        async with AsyncJevClient(base) as client:
            await client.close()
            client.http = httpx.AsyncClient(
                base_url=base, transport=httpx.ASGITransport(app=capture), headers=headers
            )
            assert await client.cancel(request_id)
    else:
        with JevClient(base) as client:
            client.close()
            client.http = TestClient(capture, base_url=base, headers=headers)
            assert client.cancel(request_id)

    instance.cancel.assert_awaited_once_with(request_id, "default")
    assert raw_paths == [prefix.encode() + b"/v1/requests/" + segment + b"/cancel"]


@pytest.mark.parametrize("prefix", ["", "/plugins/jev-runtime"], ids=["root", "plugin"])
@pytest.mark.parametrize("request_id", [".", ".."])
async def test_dot_identifier_cancels_the_submitted_decision(runtime, question, prefix, request_id):
    app = FastAPI()
    app.state.jev_runtime = runtime
    install_routes(app, prefix=prefix, api_key="data-key")
    runtime.backend.gate.clear()
    base = "http://test" + prefix
    body = {
        "request_id": request_id,
        "model": "model",
        "input": {"text": "refund"},
        "questions": [question.model_dump(mode="json")],
    }
    async with AsyncJevClient(base) as client:
        await client.close()
        client.http = httpx.AsyncClient(
            base_url=base,
            transport=httpx.ASGITransport(app=app),
            headers={"Authorization": "Bearer data-key"},
        )
        work = asyncio.create_task(client.decide(body))
        try:
            await asyncio.wait_for(runtime.backend.started.wait(), 1)
            assert await client.cancel(request_id)
            with pytest.raises(JevError) as exc:
                await asyncio.wait_for(work, 1)
            assert (exc.value.status_code, exc.value.code) == (499, "request_cancelled")
            assert runtime.backend.cancelled
            assert not runtime.registry.list()["leases"]
            assert not await client.cancel(request_id)
        finally:
            if not work.done():
                work.cancel()
            await asyncio.gather(work, return_exceptions=True)
