from __future__ import annotations

import httpx

from jev_runtime.errors import JevError
from jev_runtime.schema import DecisionRequest, DecisionResponse


def _decode(response: httpx.Response) -> DecisionResponse:
    if response.is_error:
        try:
            error = response.json().get("error", {})
        except ValueError:
            error = {}
        raise JevError(
            error.get("code", "http_error"),
            error.get("message", "Decision service returned an error"),
            response.status_code,
        )
    return DecisionResponse.model_validate(response.json())


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

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
