import httpx
import pytest

from jev_runtime.backends.base import ScoreInput
from jev_runtime.backends.sglang import SGLangHTTP, parse_sglang
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
