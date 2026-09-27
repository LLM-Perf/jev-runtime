import copy
import json

import httpx
import pytest

from benchmarks.run_case import measure
from jev_runtime.generation_baseline import (
    InvalidBenchmarkResponse,
    generation_request,
    parse_generation,
)
from jev_runtime.schema import Option, Question


def response(engine="vllm"):
    choice = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": '{"choice":"c1"}'},
    }
    data = {
        "choices": [choice],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }
    if engine == "sglang":
        choice.update(prompt_token_ids=[1, 2, 3], response_token_ids=[4, 5])
    else:
        choice["token_ids"] = [4, 5]
        data["prompt_token_ids"] = [1, 2, 3]
    return data


def parse(body, engine="vllm", trace=False):
    reply = httpx.Response(200, json=body, request=httpx.Request("POST", "http://test"))
    return parse_generation(
        engine,
        reply,
        {"c0", "c1"},
        expected_prompt_tokens=3,
        max_tokens=64,
        reference_candidate="c1",
        trace=trace,
    )


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_actual_token_traces_and_generation_cost(engine):
    values, ids = parse(response(engine), engine, trace=True)
    assert values["selected_candidate"] == "c1"
    assert values["selected_candidate_matches_reference"] is True
    assert values["engine_completion_tokens"] == 2
    assert values["cached_prompt_tokens"] == 2
    assert ids == {"prompt_token_ids": [1, 2, 3], "output_token_ids": [4, 5]}


@pytest.mark.parametrize(
    "content",
    [
        '{"choice":"c0","choice":"c1"}',
        '{"choice":"missing"}',
        '{"choice":true}',
        '{"choice":"c1","explanation":"text"}',
        '```json\n{"choice":"c1"}\n```',
        "[]",
        '{"choice":',
    ],
)
def test_schema_violations_keep_observed_output_cost(content):
    data = response()
    data["choices"][0]["message"]["content"] = content
    with pytest.raises(InvalidBenchmarkResponse) as exc:
        parse(data)
    assert exc.value.observations["engine_completion_tokens"] == 2
    assert exc.value.observations["response_content_bytes"] == len(content.encode())


def test_complete_json_at_length_limit_is_still_truncated():
    data = response()
    data["choices"][0]["finish_reason"] = "length"
    with pytest.raises(InvalidBenchmarkResponse) as exc:
        parse(data)
    assert exc.value.code == "generation_truncated"
    assert exc.value.observations["engine_completion_tokens"] == 2


def test_mismatched_token_trace_and_changed_prompt_are_rejected():
    data = response()
    data["prompt_token_ids"] = [1]
    with pytest.raises(InvalidBenchmarkResponse, match="token_trace"):
        parse(data, trace=True)
    data["usage"]["prompt_tokens"] = 4
    with pytest.raises(InvalidBenchmarkResponse, match="prompt_length"):
        parse(data)


def test_missing_usage_is_not_inferred_from_content_or_replaced_with_zero():
    data = response()
    del data["usage"]["completion_tokens"]
    with pytest.raises(InvalidBenchmarkResponse) as exc:
        parse(data)
    assert exc.value.code == "generation_usage_invalid"
    assert "engine_completion_tokens" not in exc.value.observations
    data["usage"]["completion_tokens"] = True
    with pytest.raises(InvalidBenchmarkResponse):
        parse(data)


def test_generation_request_preserves_task_and_options_and_has_no_trace_in_timed_body():
    question = Question(
        id="q",
        type="choice",
        instruction="Choose",
        options=(
            Option(id="c0", description="first"),
            Option(id="c1", description="second"),
        ),
    )
    for engine in ("vllm", "sglang"):
        payload = generation_request(engine, "model", question, "sample", 64)
        assert json.loads(payload["messages"][1]["content"]) == {
            "task": "Choose",
            "candidates": [
                {"id": "c0", "description": "first"},
                {"id": "c1", "description": "second"},
            ],
            "data": "sample",
        }
        assert "return_token_ids" not in payload and "return_prompt_token_ids" not in payload
        schema = payload["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["choice"]["enum"] == ["c0", "c1"]
        assert schema["additionalProperties"] is False


async def test_failed_generation_remains_in_cohort_and_output_token_denominators():
    valid = response()
    truncated = copy.deepcopy(valid)
    truncated["choices"][0]["finish_reason"] = "length"
    calls = 0

    async def call(request_id):
        nonlocal calls
        calls += 1
        return parse(valid if calls % 2 else truncated)[0]

    rows, summary = await measure(call, 0.01, 1, 1, 1, 3)
    assert len(rows) > 1 and summary["failed_requests"] > 0
    assert summary["dispatched_requests"] == calls
    assert summary["errors"]["generation_truncated"] == calls // 2
    cost = summary["output_cost"]
    assert cost["all_attempts"]["full_cohort_completion_tokens"] == 2 * calls
    assert cost["strict_successes"]["full_cohort_completion_tokens"] == 2 * (calls - calls // 2)
    assert all(row.engine_completion_tokens == 2 for row in rows)
