import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from benchmarks.production_traffic import run, strict_response
from jev_runtime.schema import DecisionRequest


async def fixture_response(runtime, question):
    request = DecisionRequest(
        model="model", request_id="test", input={"text": "hello"}, questions=(question,)
    )
    response = await runtime.decide(request)
    body = response.model_dump(mode="json")
    routes = [{key: body[key] for key in ("bundle", "bundle_digest", "generation")}]
    return request, body, routes


@pytest.mark.parametrize("change", ["request", "version", "candidate", "route", "type"])
async def test_strict_response_rejects_valid_200_with_wrong_contract(runtime, question, change):
    request, body, routes = await fixture_response(runtime, question)
    strict_response(body, request, "fixture", "0", routes)
    if change == "request":
        body["request_id"] = "other"
    elif change == "version":
        body["engine"]["version"] = "different"
    elif change == "route":
        body["generation"] += 1
    elif change == "type":
        body["answers"][question.id]["type"] = "score"
    else:
        body["answers"][question.id]["probabilities"] = {"unrelated": 0.8, "other": 0.2}
    with pytest.raises(ValueError):
        strict_response(body, request, "fixture", "0", routes)


@pytest.mark.parametrize("mode", ["success", "invalid", "overload", "http_error"])
async def test_open_loop_keeps_full_denominator_and_excludes_bad_200s(
    runtime, question, tmp_path, monkeypatch, mode
):
    request, body, routes = await fixture_response(runtime, question)
    inputs, routefile = tmp_path / "input.jsonl", tmp_path / "routes.json"
    inputs.write_text(request.model_dump_json() + "\n")
    routefile.write_text(json.dumps(routes))
    monkeypatch.setenv("JEV_TEST_KEY", "not-exported")

    async def respond(http_request):
        if mode == "overload":
            await asyncio.sleep(0.05)
        if mode == "http_error":
            return httpx.Response(429)
        output = {**body, "request_id": json.loads(http_request.content)["request_id"]}
        if mode == "invalid":
            output["generation"] += 1
        return httpx.Response(200, json=output)

    args = SimpleNamespace(
        url="http://fixture/plugins/jev-runtime",
        engine="fixture",
        engine_version="0",
        requests=inputs,
        routes=routefile,
        output=tmp_path / "run",
        api_key_env="JEV_TEST_KEY",
        rate=100.0,
        duration=0.1,
        concurrency=1,
        queue_size=1,
        timeout=1.0,
    )
    result = await run(args, transport=httpx.MockTransport(respond))
    assert result["accounted"] == result["planned"] == 10
    rows = (args.output / "requests.jsonl").read_text()
    assert "not-exported" not in rows and "hello" not in rows
    assert result["passed"] == (mode == "success")
    if mode in {"invalid", "http_error"}:
        assert result["strict_successes"] == 0
    if mode == "overload":
        assert result["outcomes"]["client_overload"] > 0
