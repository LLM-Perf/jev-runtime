from __future__ import annotations

import asyncio
import inspect
from typing import Any

import httpx

from jev_runtime.adapters import AdapterBinding
from jev_runtime.backends.base import Capabilities, ScoreInput, ScoreResult, reported_dtype
from jev_runtime.errors import JevError


def parse_sglang(result: dict, request: ScoreInput) -> ScoreResult:
    try:
        meta = result["meta_info"]
        finish = meta.get("finish_reason") or {}
        if isinstance(finish, dict) and finish.get("type") in {"abort", "error"}:
            raise JevError("engine_aborted", "SGLang did not complete scoring", 502)
        if meta["completion_tokens"] != 0:
            raise JevError("score_position", "Expected SGLang zero-output scoring", 502)
        positions = meta["output_token_ids_logprobs"]
        if len(positions) != 1:
            raise JevError(
                "score_position", "Expected exactly one answer-position distribution", 502
            )
        entries = positions[0]
        token_map = {int(entry[1]): float(entry[0]) for entry in entries}
        if len(token_map) != len(entries):
            raise JevError("score_contract", "Duplicate token scores in engine output", 502)
        scores = tuple(token_map[token] for token in request.label_ids)
        return ScoreResult(
            request.request_id, scores, meta.get("prompt_tokens"), 0, meta.get("cached_tokens")
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise JevError(
            "score_contract", "SGLang response is missing required label scores", 502
        ) from exc


def generate_payload(request: ScoreInput) -> dict:
    payload = {
        "rid": request.request_id,
        "input_ids": list(request.input_ids),
        "token_ids_logprob": list(request.label_ids),
        "return_logprob": True,
        "logprob_start_len": -1,
        "stream": False,
        "sampling_params": {"max_new_tokens": 0, "temperature": 1.0},
    }
    if request.adapter_id:
        payload["lora_path"] = request.adapter_id
    return payload


class SGLangHTTP:
    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.model_id = model_id
        self.client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(300, connect=10),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

    async def probe(self) -> Capabilities:
        try:
            response = await self.client.get("/get_model_info")
            response.raise_for_status()
            info = response.json()
            server = await self.client.get("/get_server_info")
            server.raise_for_status()
            config = server.json()
            return Capabilities(
                engine="sglang",
                version=str(config.get("version", "unknown")),
                model_id=str(info.get("model_path", self.model_id)),
                max_context_tokens=int(
                    info.get("context_length") or config.get("context_length") or 32768
                ),
                max_label_tokens=128,
                lora=False,
                prefix_cache=not bool(config.get("disable_radix_cache", False)),
                batch_invariant=config.get("enable_deterministic_inference"),
                verified=False,
                model_dtype=reported_dtype(config.get("dtype")),
                readout_dtype=(
                    "float32"
                    if config.get("enable_fp32_lm_head")
                    else reported_dtype(config.get("dtype"))
                )
                if config.get("quantization") is None
                else None,
            )
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            raise JevError(
                "engine_probe_failed", "Cannot inspect SGLang engine capabilities", 503
            ) from exc

    async def score(self, request: ScoreInput) -> ScoreResult:
        if request.adapter_id:
            raise JevError(
                "lora_unsupported", "Managed LoRA requires the native typed endpoint", 409
            )
        try:
            response = await self.client.post("/generate", json=generate_payload(request))
            response.raise_for_status()
            return parse_sglang(response.json(), request)
        except (httpx.HTTPError, ValueError) as exc:
            raise JevError("engine_request_failed", "SGLang scoring request failed", 502) from exc

    async def cancel(self, request_id: str) -> None:
        response = await self.client.post(
            "/abort_request", json={"rid": request_id, "abort_all": False}
        )
        response.raise_for_status()

    async def close(self) -> None:
        await self.client.aclose()


class SGLangNative:
    def __init__(self, manager: Any, version: str):
        self.manager, self.version = manager, version
        self.managed_lora = False
        self._adapter_bindings: dict[str, AdapterBinding] = {}
        self._adapter_attempts: set[str] = set()
        self._scoring: dict[str, asyncio.Task] = {}

    def _score_done(self, request_id: str, task: asyncio.Task) -> None:
        if self._scoring.get(request_id) is task:
            self._scoring.pop(request_id)
        if not task.cancelled():
            task.exception()

    def _lora_profile(self) -> bool:
        args = self.manager.server_args
        model = self.manager.model_config
        architectures = getattr(getattr(model, "hf_config", None), "architectures", ()) or ()
        return bool(
            self.managed_lora
            and not getattr(args, "enable_deterministic_inference", False)
            and self.version.split("+")[0] == "0.5.19"
            and getattr(args, "enable_lora", False)
            and getattr(args, "tokenizer_worker_num", None) == 1
            and getattr(args, "tp_size", None) == 1
            and getattr(args, "dp_size", None) == 1
            and getattr(args, "pp_size", None) == 1
            and str(getattr(model, "dtype", None)) in {"torch.bfloat16", "bfloat16"}
            and getattr(args, "quantization", None) is None
            and getattr(model, "quantization", None) is None
            and not getattr(args, "enable_fp32_lm_head", False)
            and set(architectures) <= {"LlamaForCausalLM"}
            and architectures
        )

    def _check_adapter_capacity(self) -> None:
        limit = getattr(self.manager.server_args, "max_loaded_loras", None)
        if limit is not None and self.manager.lora_registry.num_registered_loras >= limit:
            raise JevError(
                "adapter_capacity", "Unload an adapter before exceeding engine capacity", 409
            )
        pinned = sum(
            bool(item.pinned) for item in self.manager.lora_registry.get_all_adapters().values()
        )
        if pinned >= self.manager.server_args.max_loras_per_batch - 1:
            raise JevError(
                "adapter_capacity",
                "SGLang reserves one LoRA pool slot for base/unpinned requests; "
                "increase max_loras_per_batch",
                409,
            )

    async def load_adapter(self, binding: AdapterBinding) -> None:
        from sglang.srt.managers.io_struct import LoadLoRAAdapterReqInput

        if not self._lora_profile():
            raise JevError("adapter_profile", "Unsupported SGLang managed LoRA profile", 409)
        if binding.engine_name in self.manager.lora_registry.get_all_adapters():
            raise JevError("adapter_collision", "Engine adapter name is already resident", 409)
        self._check_adapter_capacity()
        self._adapter_attempts.add(binding.engine_name)
        result = await self.manager.load_lora_adapter(
            LoadLoRAAdapterReqInput(
                lora_name=binding.engine_name, lora_path=binding.artifact.path, pinned=True
            )
        )
        if not result.success:
            raise JevError(
                "adapter_load", "SGLang rejected loading: " + str(result.error_message)[:1000], 503
            )
        if result.error_message != "jev_gpu_barrier_v1":
            raise JevError("adapter_barrier", "SGLang GPU load barrier was not confirmed", 503)
        self._adapter_bindings[binding.artifact.reference] = binding

    async def unload_adapter(self, binding: AdapterBinding) -> None:
        from sglang.srt.managers.io_struct import UnloadLoRAAdapterReqInput

        if not self._lora_profile():
            raise JevError("adapter_profile", "Unsupported SGLang managed LoRA profile", 409)
        resident = self.manager.lora_registry.get_all_adapters().get(binding.engine_name)
        if resident is None:
            if binding.engine_name in self._adapter_attempts:
                raise JevError(
                    "adapter_restart_required",
                    "An uncertain SGLang load/unload requires a full engine restart",
                    503,
                )
            # Fresh single-frontend engine session; no operation has dispatched
            # this opaque adapter name. Old database state is not residency.
            return
        if resident.lora_path != binding.artifact.path:
            raise JevError("adapter_collision", "Resident adapter path differs", 409)
        result = await self.manager.unload_lora_adapter(
            UnloadLoRAAdapterReqInput(lora_name=binding.engine_name)
        )
        if not result.success:
            raise JevError(
                "adapter_unload",
                "SGLang rejected removal: " + str(result.error_message)[:1000],
                503,
            )
        if result.error_message != "jev_gpu_barrier_v1":
            raise JevError("adapter_barrier", "SGLang GPU unload barrier was not confirmed", 503)
        self._adapter_bindings.pop(binding.artifact.reference, None)
        self._adapter_attempts.discard(binding.engine_name)

    async def probe(self) -> Capabilities:
        config = self.manager.model_config
        args = self.manager.server_args
        return Capabilities(
            engine="sglang",
            version=self.version,
            model_id=str(config.model_path),
            max_context_tokens=config.context_len,
            lora=self._lora_profile(),
            prefix_cache=not bool(getattr(args, "disable_radix_cache", False)),
            batch_invariant=getattr(args, "enable_deterministic_inference", None),
            model_dtype=reported_dtype(getattr(config, "dtype", None)),
            readout_dtype=(
                "float32"
                if getattr(args, "enable_fp32_lm_head", False)
                else reported_dtype(getattr(config, "dtype", None))
            )
            if getattr(config, "quantization", None) is None
            and getattr(args, "quantization", None) is None
            else None,
        )

    async def score(self, request: ScoreInput) -> ScoreResult:
        from sglang.srt.managers.io_struct import GenerateReqInput

        payload = generate_payload(request)
        if request.adapter_id:
            binding = self._adapter_bindings.get(request.adapter_id)
            if binding is None or not self._lora_profile():
                raise JevError(
                    "adapter_not_ready", "Adapter has not been loaded by this worker", 503
                )
            payload["lora_path"] = binding.engine_name

        async def receive():
            generator = self.manager.generate_request(GenerateReqInput(**payload), None)
            try:
                result = await anext(generator)
                return parse_sglang(result, request)
            finally:
                await generator.aclose()

        # SGLang 0.5.19 discards rid_to_state if its generator is cancelled.
        # Cancelling it before sending abort loses both scheduler routing and
        # the LoRA usage-counter release on the eventual output. Keep the
        # receiver alive until cancel() observes its terminal response.
        if request.request_id in self._scoring:
            raise JevError("duplicate_request", "Native scoring ID is already active", 409)
        task = asyncio.create_task(receive())
        self._scoring[request.request_id] = task
        task.add_done_callback(lambda done: self._score_done(request.request_id, done))
        return await asyncio.shield(task)

    async def cancel(self, request_id: str) -> None:
        result = self.manager.abort_request(rid=request_id, abort_all=False)
        if inspect.isawaitable(result):
            await result
        task = self._scoring.get(request_id)
        if task is not None:
            async with asyncio.timeout(4.5):
                try:
                    await asyncio.shield(task)
                except Exception:
                    # An engine abort is a scoring error, but its terminal response
                    # still confirms drain. Parent cancellation/timeouts propagate.
                    pass

    async def close(self) -> None:
        # The host engine owns its manager and worker lifecycle.
        pass
