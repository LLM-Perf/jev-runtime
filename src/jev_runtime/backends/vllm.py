from __future__ import annotations

from typing import Any

import httpx

from jev_runtime.adapters import AdapterBinding
from jev_runtime.backends.base import Capabilities, ScoreInput, ScoreResult, reported_dtype
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
        self.managed_lora = False
        self._adapter_bindings: dict[str, AdapterBinding] = {}

    def _lora_profile(self) -> bool:
        config = getattr(self.engine_client, "vllm_config", None)
        parallel = getattr(config, "parallel_config", None)
        model = self.engine_client.model_config
        architectures = getattr(getattr(model, "hf_config", None), "architectures", ()) or ()
        return bool(
            self.managed_lora
            and self.version.split("+")[0] == "0.30.0"
            and self.api_workers == 1
            and getattr(config, "lora_config", None) is not None
            and getattr(parallel, "tensor_parallel_size", None) == 1
            and getattr(parallel, "pipeline_parallel_size", None) == 1
            and getattr(parallel, "data_parallel_size", None) == 1
            and getattr(parallel, "worker_extension_cls", None)
            == "jev_vllm.worker.LoRAWorkerExtension"
            and str(getattr(model, "dtype", None)) in {"torch.bfloat16", "bfloat16"}
            and getattr(model, "quantization", None) is None
            and reported_dtype(getattr(model, "head_dtype", model.dtype)) == "bfloat16"
            and set(architectures) <= {"LlamaForCausalLM"}
            and architectures
        )

    @staticmethod
    def _lora_request(binding: AdapterBinding):
        from vllm.lora.request import LoRARequest

        return LoRARequest(
            lora_name=binding.engine_name,
            lora_int_id=binding.engine_id,
            lora_path=binding.artifact.path,
            load_inplace=False,
        )

    async def _lora_barrier(self) -> None:
        results = await self.engine_client.collective_rpc("jev_lora_barrier", timeout=30)
        if not results or not all(value is True for value in results):
            raise JevError("adapter_barrier", "GPU completion barrier was not confirmed", 503)

    async def load_adapter(self, binding: AdapterBinding) -> None:
        if not self._lora_profile():
            raise JevError("adapter_profile", "Unsupported vLLM managed LoRA profile", 409)
        if binding.engine_id in await self.engine_client.list_loras():
            raise JevError(
                "adapter_collision",
                "Engine adapter ID is already resident; reconcile before loading",
                409,
            )
        if not await self.engine_client.add_lora(self._lora_request(binding)):
            raise JevError("adapter_load", "vLLM did not confirm LoRA loading", 503)
        if not await self.engine_client.pin_lora(binding.engine_id):
            raise JevError("adapter_pin", "vLLM did not pin the managed adapter", 503)
        await self._lora_barrier()
        self._adapter_bindings[binding.artifact.reference] = binding

    async def unload_adapter(self, binding: AdapterBinding) -> None:
        if not self._lora_profile():
            raise JevError("adapter_profile", "Unsupported vLLM managed LoRA profile", 409)
        await self._lora_barrier()
        if binding.engine_id in await self.engine_client.list_loras():
            if not await self.engine_client.remove_lora(binding.engine_id):
                raise JevError("adapter_unload", "vLLM did not confirm LoRA removal", 503)
        await self._lora_barrier()
        if binding.engine_id in await self.engine_client.list_loras():
            raise JevError("adapter_unload", "Adapter remains resident", 503)
        self._adapter_bindings.pop(binding.artifact.reference, None)

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
            model_dtype=reported_dtype(getattr(model, "dtype", None)),
            readout_dtype=(
                reported_dtype(getattr(model, "head_dtype", getattr(model, "dtype", None)))
                if getattr(model, "quantization", None) is None
                else None
            ),
            lora=self._lora_profile(),
        )

    async def score(self, request: ScoreInput) -> ScoreResult:
        from vllm.inputs import tokens_input
        from vllm.sampling_params import SamplingParams

        extra = {}
        if request.adapter_id:
            binding = self._adapter_bindings.get(request.adapter_id)
            if binding is None or not self._lora_profile():
                raise JevError(
                    "adapter_not_ready", "Adapter has not been loaded by this worker", 503
                )
            extra["lora_request"] = self._lora_request(binding)
        params = SamplingParams(
            max_tokens=1,
            logprobs=len(request.label_ids),
            logprob_token_ids=list(request.label_ids),
            n=1,
        )
        generator = self.engine_client.generate(
            tokens_input(list(request.input_ids)), params, request.request_id, **extra
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
            return capabilities.model_copy(update={"lora": False})
        except (httpx.HTTPError, ValueError) as exc:
            raise JevError(
                "engine_probe_failed", "vLLM requires the Jev scoring endpoint plugin", 503
            ) from exc

    async def score(self, request: ScoreInput) -> ScoreResult:
        if request.adapter_id:
            raise JevError(
                "lora_unsupported", "Managed LoRA requires the native typed endpoint", 409
            )
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
