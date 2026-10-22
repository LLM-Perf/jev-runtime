import asyncio
import json

import httpx
import pytest

from jev_runtime.backends.remote import ScoringHTTP
from jev_runtime.errors import JevError


def make_backend(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fixture")
    return ScoringHTTP("http://fixture", "fixture", client=client)


def check_unconfirmed(error, cause=None):
    assert error.code == "cancellation_unconfirmed"
    assert error.message == "Engine did not confirm drain"
    assert error.status_code == 503
    if cause is None:
        assert error.__cause__ is None
    else:
        assert isinstance(error.__cause__, cause)


async def test_cancel_accepts_exact_confirmation_envelope():
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"cancelled": True})

    backend = make_backend(handler)
    assert await backend.cancel("r-1") is None
    (request,) = calls
    assert request.method == "POST"
    assert request.url.path == "/plugins/jev-runtime/v1/scores/cancel"
    assert json.loads(request.content) == {"request_id": "r-1"}
    await backend.close()


@pytest.mark.parametrize(
    "value",
    [False, 0, 1, 1.0, -1, "true", "1", None],
    ids=["false", "zero", "int-one", "float-one", "negative", "string-true", "string-one", "null"],
)
async def test_cancel_rejects_non_bool_cancelled_values(value):
    async def handler(request):
        return httpx.Response(200, json={"cancelled": value})

    backend = make_backend(handler)
    with pytest.raises(JevError) as error:
        await backend.cancel("r")
    check_unconfirmed(error.value)
    await backend.close()


@pytest.mark.parametrize(
    "envelope",
    [
        {},
        {"cancelled": True, "extra": 1},
        {"cancelled": True, "request_id": "r"},
        {"cancel": True},
    ],
    ids=["empty", "extra-field", "echoed-request-id", "wrong-key"],
)
async def test_cancel_rejects_malformed_envelopes(envelope):
    async def handler(request):
        return httpx.Response(200, json=envelope)

    backend = make_backend(handler)
    with pytest.raises(JevError) as error:
        await backend.cancel("r")
    check_unconfirmed(error.value)
    await backend.close()


@pytest.mark.parametrize(
    "body",
    [b"[true]", b'"cancelled"', b"1", b"true", b"null"],
    ids=["list", "string", "number", "bool", "null"],
)
async def test_cancel_rejects_wrong_root_type(body):
    async def handler(request):
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    backend = make_backend(handler)
    with pytest.raises(JevError) as error:
        await backend.cancel("r")
    check_unconfirmed(error.value)
    await backend.close()


async def test_cancel_rejects_malformed_json():
    async def handler(request):
        return httpx.Response(
            200, content=b'{"cancelled": tru', headers={"content-type": "application/json"}
        )

    backend = make_backend(handler)
    with pytest.raises(JevError) as error:
        await backend.cancel("r")
    check_unconfirmed(error.value, ValueError)
    await backend.close()


@pytest.mark.parametrize("status", [400, 409, 500])
async def test_cancel_rejects_unsuccessful_http_even_with_valid_envelope(status):
    async def handler(request):
        return httpx.Response(status, json={"cancelled": True})

    backend = make_backend(handler)
    with pytest.raises(JevError) as error:
        await backend.cancel("r")
    check_unconfirmed(error.value, httpx.HTTPStatusError)
    await backend.close()


async def test_cancel_rejects_transport_failure():
    async def handler(request):
        raise httpx.ConnectError("unreachable", request=request)

    backend = make_backend(handler)
    with pytest.raises(JevError) as error:
        await backend.cancel("r")
    check_unconfirmed(error.value, httpx.ConnectError)
    await backend.close()


async def test_cancel_propagates_parent_task_cancellation_while_transport_blocks():
    received = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        received.set()
        await release.wait()
        return httpx.Response(200, json={"cancelled": True})

    backend = make_backend(handler)
    parent = asyncio.create_task(backend.cancel("r"))
    await received.wait()
    parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent
    assert parent.cancelled()
    await backend.close()
