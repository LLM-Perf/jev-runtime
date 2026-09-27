from __future__ import annotations

import inspect
from typing import Any

import httpx

from jev_runtime.backends.base import Capabilities, ScoreInput, ScoreResult
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
                lora=bool(config.get("enable_lora", False)),
                prefix_cache=not bool(config.get("disable_radix_cache", False)),
                verified=False,
            )
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            raise JevError(
                "engine_probe_failed", "Cannot inspect SGLang engine capabilities", 503
            ) from exc

    async def score(self, request: ScoreInput) -> ScoreResult:
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

    async def probe(self) -> Capabilities:
        config = self.manager.model_config
        args = self.manager.server_args
        return Capabilities(
            engine="sglang",
            version=self.version,
            model_id=str(config.model_path),
            max_context_tokens=config.context_len,
            lora=bool(getattr(args, "enable_lora", False)),
            prefix_cache=not bool(getattr(args, "disable_radix_cache", False)),
        )

    async def score(self, request: ScoreInput) -> ScoreResult:
        from sglang.srt.managers.io_struct import GenerateReqInput

        generator = self.manager.generate_request(
            GenerateReqInput(**generate_payload(request)), None
        )
        try:
            result = await anext(generator)
            return parse_sglang(result, request)
        finally:
            await generator.aclose()

    async def cancel(self, request_id: str) -> None:
        result = self.manager.abort_request(rid=request_id, abort_all=False)
        if inspect.isawaitable(result):
            await result

    async def close(self) -> None:
        # The host engine owns its manager and worker lifecycle.
        pass
