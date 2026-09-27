from __future__ import annotations

from typing import Any

import httpx

from jev_runtime.backends.base import Capabilities, ScoreInput, ScoreResult
from jev_runtime.errors import JevError


class VLLMNative:
    def __init__(
        self,
        engine_client: Any,
        model_id: str,
        max_context: int,
        version: str,
        api_workers: int = 1,
    ):
        self.engine_client = engine_client
        self.model_id, self.max_context, self.version = model_id, max_context, version
        self.api_workers = api_workers

    async def probe(self) -> Capabilities:
        model = self.engine_client.model_config
        configured_limit = getattr(model, "max_logprobs", 0)
        cache = getattr(getattr(self.engine_client, "vllm_config", None), "cache_config", None)
        return Capabilities(
            engine="vllm",
            version=self.version,
            model_id=self.model_id,
            max_context_tokens=self.max_context,
            max_label_tokens=128 if configured_limit < 0 else min(128, configured_limit),
            raw_logprobs=getattr(model, "logprobs_mode", None) == "raw_logprobs",
            prefix_cache=getattr(cache, "enable_prefix_caching", None),
            api_workers=self.api_workers,
        )

    async def score(self, request: ScoreInput) -> ScoreResult:
        from vllm.inputs import tokens_input
        from vllm.sampling_params import SamplingParams

        if request.adapter_id:
            raise JevError("lora_unsupported", "This adapter has no certified LoRA resolver", 409)
        params = SamplingParams(
            max_tokens=1,
            logprobs=len(request.label_ids),
            logprob_token_ids=list(request.label_ids),
            n=1,
        )
        generator = self.engine_client.generate(
            tokens_input(list(request.input_ids)), params, request.request_id
        )
        result = None
        try:
            async for output in generator:
                result = output
        finally:
            await generator.aclose()
        if result is None or not result.finished or len(result.outputs) != 1:
            raise JevError(
                "score_contract", "vLLM scoring did not return one completed sequence", 502
            )
        output = result.outputs[0]
        if output.finish_reason in {"error", "abort"} or len(output.token_ids) != 1:
            raise JevError(
                "score_contract", "vLLM scoring did not complete its one-token readout", 502
            )
        if not output.logprobs or len(output.logprobs) != 1:
            raise JevError("score_position", "Expected one vLLM answer-position distribution", 502)
        try:
            scores = tuple(float(output.logprobs[0][token].logprob) for token in request.label_ids)
        except (KeyError, TypeError, ValueError) as exc:
            raise JevError(
                "missing_labels", "vLLM response omitted required token scores", 502
            ) from exc
        return ScoreResult(
            request.request_id,
            scores,
            len(result.prompt_token_ids),
            len(output.token_ids),
            getattr(result, "num_cached_tokens", None),
        )

    async def cancel(self, request_id: str) -> None:
        await self.engine_client.abort(request_id)

    async def close(self) -> None:
        pass


class VLLMHTTP:
    """Attach to the explicit scoring contract exported by the vLLM plugin."""

    prefix = "/plugins/jev-runtime/v1"

    def __init__(
        self, base_url: str, api_key: str | None = None, client: httpx.AsyncClient | None = None
    ):
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
            if capabilities.engine == "vllm" and capabilities.api_workers != 1:
                # AsyncLLM.abort resolves external IDs in this API worker's
                # OutputProcessor. Sending cancellation to another frontend
                # can acknowledge an empty abort list. Do not treat that as
                # confirmed cleanup for a gateway lease.
                raise JevError(
                    "cancellation_routing_unsupported",
                    "The vLLM HTTP bridge requires one verified engine API worker; "
                    "use native typed endpoints for multiple engine API workers",
                    503,
                )
            return capabilities
        except (httpx.HTTPError, ValueError) as exc:
            raise JevError(
                "engine_probe_failed", "vLLM requires the Jev scoring endpoint plugin", 503
            ) from exc

    async def score(self, request: ScoreInput) -> ScoreResult:
        try:
            response = await self.client.post(
                self.prefix + "/scores",
                json={
                    "request_id": request.request_id,
                    "question_id": request.question_id,
                    "input_ids": request.input_ids,
                    "label_ids": request.label_ids,
                    "candidate_id": request.candidate_id,
                    "adapter_id": request.adapter_id,
                },
            )
            response.raise_for_status()
            data = response.json()
            data["logprobs"] = tuple(data["logprobs"])
            return ScoreResult(**data)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise JevError(
                "engine_request_failed", "vLLM plugin scoring request failed", 502
            ) from exc

    async def cancel(self, request_id: str) -> None:
        response = await self.client.post(
            self.prefix + "/scores/cancel", json={"request_id": request_id}
        )
        response.raise_for_status()

    async def close(self) -> None:
        await self.client.aclose()
