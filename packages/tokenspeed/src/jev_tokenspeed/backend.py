"""Pinned TokenSpeed single-label raw readout, with completion-based drain."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass

from jev_runtime.backends.base import Capabilities, ScoreInput, ScoreResult, reported_dtype
from jev_runtime.backends.remote import ScoringHTTP
from jev_runtime.errors import JevError


def create_backend(*, settings, api_key):
    return ScoringHTTP(settings.engine_url, "tokenspeed", api_key)


def generate_payload(request: ScoreInput) -> dict:
    if len(request.label_ids) != 1 or not request.input_ids:
        raise JevError("score_contract", "TokenSpeed requires one label per native request", 413)
    if request.adapter_id:
        raise JevError("lora_unsupported", "TokenSpeed managed LoRA is not supported", 409)
    return {
        "rid": request.request_id,
        "input_ids": list(request.input_ids),
        "return_logprob": True,
        "logprob_format": "sglang",
        "stream": False,
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "ignore_eos": True,
            # Exactly representable in BF16. Never assume it forces the label:
            # the returned token must match, otherwise the request fails.
            "logit_bias": {str(request.label_ids[0]): 16384.0},
        },
    }


def is_terminal(output) -> bool:
    return bool(
        isinstance(output, dict)
        and isinstance(output.get("meta_info"), dict)
        and isinstance(output["meta_info"].get("finish_reason"), dict)
        and output["meta_info"]["finish_reason"].get("type") in {"length", "stop"}
    )


def parse_result(output, request: ScoreInput) -> ScoreResult:
    try:
        meta = output["meta_info"]
        rows = meta["output_token_logprobs"]
        if not is_terminal(output) or meta["id"] != request.request_id:
            raise ValueError("Missing terminal identity")
        if (
            type(meta["completion_tokens"]) is not int
            or meta["completion_tokens"] != 1
            or output["output_ids"] != list(request.label_ids)
            or len(rows) != 1
            or len(rows[0]) != 3
            or type(rows[0][1]) is not int
            or rows[0][1] != request.label_ids[0]
        ):
            raise ValueError("Wrong token or position")
        score = rows[0][0]
        if type(score) not in {int, float} or not math.isfinite(score) or score > 0:
            raise ValueError("Invalid raw logprob")
        prompt = meta["prompt_tokens"]
        cached = meta.get("cached_tokens")
        if type(prompt) is not int or prompt != len(request.input_ids):
            raise ValueError("Wrong prompt usage")
        if cached is not None and (type(cached) is not int or not 0 <= cached <= prompt):
            raise ValueError("Wrong cache usage")
        return ScoreResult(request.request_id, (float(score),), prompt, 1, cached, True)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise JevError(
            "score_contract", "TokenSpeed did not return the requested raw label score", 502
        ) from exc


def validate_profile(args) -> None:
    valid = (
        args.sampling_backend == "triton_full"
        and args.enable_output_logprobs is True
        and args.enforce_eager is True
        and args.quantization is None
        and args.speculative_algorithm is None
        and args.speculative_config is None
        and args.disaggregation_mode == "null"
        and args.pipeline_parallel_size == 1
        and args.mapping.world_size == 1
        and args.rl_control_port is None
        and args.numerics == "auto"
        and args.skip_tokenizer_init is False
    )
    if not valid:
        raise JevError(
            "tokenspeed_profile",
            "TokenSpeed requires triton_full, output logprobs, eager, TP/DP/PP=1; "
            "no quantization, speculation, disaggregation or weight-update control",
            409,
        )


@dataclass
class Attempt:
    task: asyncio.Task | None = None
    terminal: bool = False


class TokenSpeedNative:
    def __init__(self, manager, version, submit):
        self.manager, self.version, self.submit = manager, version, submit
        self._attempts: dict[str, Attempt] = {}
        self._completed: OrderedDict[str, None] = OrderedDict()

    async def probe(self) -> Capabilities:
        args, model = self.manager.server_args, self.manager.model_config
        validate_profile(args)
        dtype = reported_dtype(model.dtype)
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise JevError("tokenspeed_profile", "Unsupported model dtype", 409)
        return Capabilities(
            engine="tokenspeed",
            version=self.version,
            model_id=str(model.model_path),
            max_context_tokens=self.manager.context_len,
            max_label_tokens=1,
            label_scoring="single",
            model_dtype=dtype,
            readout_dtype=dtype,
            batch_invariant=False,
            api_workers=1,
            lora=False,
            prefix_cache=args.enable_prefix_caching,
            verified=False,
        )

    async def score(self, request: ScoreInput) -> ScoreResult:
        from tokenspeed.runtime.engine.io_struct import GenerateReqInput

        payload = generate_payload(request)
        if request.request_id in self._attempts or request.request_id in self._completed:
            raise JevError("duplicate_request", "Native scoring ID is already active", 409)
        attempt = Attempt()
        self._attempts[request.request_id] = attempt

        async def receive():
            generator = self.manager.generate_request(GenerateReqInput(**payload))
            try:
                output = await anext(generator)
                attempt.terminal = (
                    is_terminal(output) and output["meta_info"].get("id") == request.request_id
                )
                return parse_result(output, request)
            finally:
                await generator.aclose()

        async def dispatch():
            # Engine.llm owns AsyncLLM's loop. submit runs receive there; no ZMQ
            # objects or asyncio locks move onto the HTTP server's event loop.
            try:
                return await self.submit(receive())
            finally:
                if attempt.terminal:
                    # Keep a bounded terminal receipt for a bridge whose HTTP
                    # response was lost after native completion. Never accept
                    # reuse while the old receipt could acknowledge a new ID.
                    self._completed[request.request_id] = None
                    if len(self._completed) > 4096:
                        self._completed.popitem(last=False)

        attempt.task = asyncio.create_task(dispatch())
        attempt.task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        result = await asyncio.shield(attempt.task)
        self._attempts.pop(request.request_id, None)
        return result

    async def cancel(self, request_id: str) -> None:
        attempt = self._attempts.get(request_id)
        if attempt is None:
            if request_id in self._completed:
                return
            # Native abort removes frontend state without a scheduler ack.
            # An unknown ID after a gateway/plugin restart is not drain proof.
            raise JevError("cancellation_unconfirmed", "No local terminal receipt for request", 503)
        async with asyncio.timeout(4.5):
            try:
                await asyncio.shield(attempt.task)
            except Exception:
                pass
        if not attempt.terminal:
            raise JevError(
                "cancellation_unconfirmed", "TokenSpeed completion was not observed", 503
            )
        self._attempts.pop(request_id, None)

    async def close(self) -> None:
        for request_id in tuple(self._attempts):
            await self.cancel(request_id)
