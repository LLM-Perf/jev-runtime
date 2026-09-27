from __future__ import annotations

import httpx

from jev_runtime.errors import JevError
from jev_runtime.schema import DecisionRequest, DecisionResponse


def _payload(response: httpx.Response):
    if response.is_error:
        try:
            body = response.json()
            error = body.get("error", {}) if isinstance(body, dict) else {}
        except ValueError:
            error = {}
        if not isinstance(error, dict):
            error = {}
        raise JevError(
            error["code"] if isinstance(error.get("code"), str) and error["code"] else "http_error",
            error["message"]
            if isinstance(error.get("message"), str) and error["message"]
            else "Decision service returned an error",
            response.status_code,
        )
    try:
        return response.json()
    except ValueError as exc:
        raise JevError("invalid_response", "Decision service response is not JSON", 502) from exc


def _decode(response: httpx.Response) -> DecisionResponse:
    try:
        return DecisionResponse.model_validate(_payload(response))
    except ValueError as exc:
        raise JevError(
            "invalid_response", "Decision service violated its typed response contract", 502
        ) from exc


def _cancelled(response: httpx.Response) -> bool:
    value = _payload(response)
    if not isinstance(value, dict) or type(value.get("cancelled")) is not bool:
        raise JevError("invalid_response", "Cancellation response is not boolean", 502)
    return value["cancelled"]


class JevClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def decide(self, request: DecisionRequest | dict) -> DecisionResponse:
        request = DecisionRequest.model_validate(request)
        return _decode(
            self.http.post(
                "/v1/decisions",
                json=request.model_dump(mode="json"),
                timeout=request.execution.timeout_ms / 1000 + 10,
            )
        )

    def close(self):
        self.http.close()

    def cancel(self, request_id: str) -> bool:
        from urllib.parse import quote

        response = self.http.post(f"/v1/requests/{quote(request_id, safe='')}/cancel", timeout=15)
        return _cancelled(response)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncJevClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    async def decide(self, request: DecisionRequest | dict) -> DecisionResponse:
        request = DecisionRequest.model_validate(request)
        return _decode(
            await self.http.post(
                "/v1/decisions",
                json=request.model_dump(mode="json"),
                timeout=request.execution.timeout_ms / 1000 + 10,
            )
        )

    async def close(self):
        await self.http.aclose()

    async def cancel(self, request_id: str) -> bool:
        from urllib.parse import quote

        response = await self.http.post(
            f"/v1/requests/{quote(request_id, safe='')}/cancel", timeout=15
        )
        return _cancelled(response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
