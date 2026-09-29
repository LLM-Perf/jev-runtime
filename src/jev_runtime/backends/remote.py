"""Transport for engines explicitly exporting the Jev raw scoring contract."""

from dataclasses import asdict
from typing import Annotated

import httpx
from pydantic import Field

from jev_runtime.backends.base import Capabilities, ScoreInput, ScoreResult
from jev_runtime.errors import JevError
from jev_runtime.schema import Contract


class ScoreResponse(Contract):
    request_id: str
    logprobs: tuple[Annotated[float, Field(allow_inf_nan=False)], ...] = Field(
        min_length=1, max_length=128
    )
    prompt_tokens: int | None = Field(default=None, ge=0, strict=True)
    completion_tokens: int | None = Field(default=None, ge=0, strict=True)
    cached_tokens: int | None = Field(default=None, ge=0, strict=True)
    raw_logprobs: bool = Field(strict=True)


class ScoringHTTP:
    prefix = "/plugins/jev-runtime/v1"

    def __init__(self, base_url, engine, api_key=None, client=None):
        self.engine = engine
        self.client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(300, connect=10),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

    async def probe(self) -> Capabilities:
        try:
            response = await self.client.get(self.prefix + "/scoring-capabilities")
            response.raise_for_status()
            capabilities = Capabilities.model_validate(response.json())
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise JevError(
                "engine_probe_failed", "Engine requires a Jev scoring plugin", 503
            ) from exc
        if capabilities.engine != self.engine:
            raise JevError("engine_mismatch", "Scoring plugin reports another engine", 409)
        if capabilities.api_workers != 1 or not capabilities.cancellation:
            raise JevError(
                "cancellation_routing_unsupported", "Bridge requires one cancellation owner", 503
            )
        return capabilities.model_copy(update={"lora": False})

    async def score(self, request: ScoreInput) -> ScoreResult:
        if request.adapter_id:
            raise JevError("lora_unsupported", "Bridge does not manage LoRA", 409)
        try:
            response = await self.client.post(self.prefix + "/scores", json=asdict(request))
            response.raise_for_status()
            data = ScoreResponse.model_validate(response.json())
            if (
                data.request_id != request.request_id
                or len(data.logprobs) != len(request.label_ids)
                or not data.raw_logprobs
                or any(value > 1e-5 for value in data.logprobs)
                or (
                    data.cached_tokens is not None
                    and data.prompt_tokens is not None
                    and data.cached_tokens > data.prompt_tokens
                )
            ):
                raise ValueError("Scoring response does not match request")
            return ScoreResult(**data.model_dump())
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise JevError("score_contract", "Invalid engine scoring response", 502) from exc

    async def cancel(self, request_id: str) -> None:
        response = await self.client.post(
            self.prefix + "/scores/cancel", json={"request_id": request_id}
        )
        response.raise_for_status()
        if response.json() != {"cancelled": True}:
            raise JevError("cancellation_unconfirmed", "Engine did not confirm drain", 503)

    async def close(self) -> None:
        await self.client.aclose()
