from types import SimpleNamespace

import httpx
import pytest

from jev_runtime.backends.base import ScoreInput
from jev_runtime.backends.sglang import SGLangHTTP, SGLangNative, parse_sglang
from jev_runtime.backends.vllm import VLLMHTTP, VLLMNative
from jev_runtime.errors import JevError


def scoring_request():
    return ScoreInput("r", "q", (1, 2), (10, 20))


def test_sglang_uses_ids_not_return_order():
    result = parse_sglang(
        {
            "meta_info": {
                "completion_tokens": 0,
                "output_token_ids_logprobs": [[[-2, 20, None], [-1, 10, None]]],
            }
        },
        scoring_request(),
    )
    assert result.logprobs == (-1, -2)
    assert result.cached_tokens is None


@pytest.mark.parametrize(
    "meta",
    [
        {"completion_tokens": 1, "output_token_ids_logprobs": [[[-1, 10, None], [-2, 20, None]]]},
        {"completion_tokens": 0, "output_token_ids_logprobs": [[[-1, 10, None]]]},
        {"completion_tokens": 0, "output_token_ids_logprobs": []},
    ],
)
def test_sglang_rejects_wrong_position_and_missing_labels(meta):
    with pytest.raises(JevError):
        parse_sglang({"meta_info": meta}, scoring_request())


async def test_sglang_transport_contract():
    calls = []

    async def handler(request):
        import json

        calls.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/abort_request":
            return httpx.Response(200)
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "completion_tokens": 0,
                    "output_token_ids_logprobs": [[[-1, 10, None], [-2, 20, None]]],
                    "prompt_tokens": 2,
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fixture")
    backend = SGLangHTTP("http://fixture", "fixture", client=client)
    result = await backend.score(scoring_request())
    await backend.cancel("r")
    assert result.completion_tokens == 0
    assert calls[0][1]["sampling_params"]["max_new_tokens"] == 0
    assert calls[0][1]["input_ids"] == [1, 2]
    assert calls[1][1] == {"rid": "r", "abort_all": False}
    await backend.close()


@pytest.mark.parametrize("workers", [1, 2, None])
async def test_vllm_http_requires_safe_cancellation_routing(workers):
    async def handler(request):
        return httpx.Response(
            200,
            json={
                "engine": "vllm",
                "version": "0.30.0",
                "model_id": "fixture",
                "api_workers": workers,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fixture")
    backend = VLLMHTTP("http://fixture", client=client)
    try:
        if workers == 1:
            assert (await backend.probe()).api_workers == 1
        else:
            with pytest.raises(JevError) as error:
                await backend.probe()
            assert error.value.code == "cancellation_routing_unsupported"
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "mode,limit,expected_raw,expected_limit",
    [
        ("raw_logprobs", 20, True, 20),
        ("raw_logprobs", -1, True, 128),
        ("processed_logprobs", 64, False, 64),
    ],
)
async def test_vllm_reports_configured_scoring_limits(mode, limit, expected_raw, expected_limit):
    engine = SimpleNamespace(
        model_config=SimpleNamespace(logprobs_mode=mode, max_logprobs=limit),
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=False)),
    )
    capabilities = await VLLMNative(engine, "fixture", 2048, "fixture", api_workers=2).probe()
    assert capabilities.raw_logprobs is expected_raw
    assert capabilities.max_label_tokens == expected_limit
    assert capabilities.prefix_cache is False and capabilities.api_workers == 2


def test_sglang_capacity_reserves_base_model_slot_before_dispatch():
    args = SimpleNamespace(max_loaded_loras=4, max_loras_per_batch=2)
    manager = SimpleNamespace(
        server_args=args,
        lora_registry=SimpleNamespace(
            num_registered_loras=1, get_all_adapters=lambda: {"first": SimpleNamespace(pinned=True)}
        ),
    )
    backend = SGLangNative(manager, "0.5.19")
    with pytest.raises(JevError) as error:
        backend._check_adapter_capacity()
    assert error.value.code == "adapter_capacity"
    args.max_loras_per_batch = 3
    backend._check_adapter_capacity()
