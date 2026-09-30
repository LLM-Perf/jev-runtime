import asyncio
import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest

from jev_runtime.schema import TemplateSpec
from tests.integration.hot_switch_traffic import exercise, verify_response


class Target:
    """Explicit CPU HTTP-contract double; never used by the live runner."""

    def __init__(self, bundle, corrupt=None):
        self.bundles = (
            bundle,
            bundle.model_copy(
                update={"version": 2, "template": TemplateSpec(mode="independent-candidate")}
            ),
        )
        self.bundle = self.bundles[1]
        self.generation = 2
        path = Path(__file__).parents[1] / "evidence/dsw/vllm-deepseek15b-ad9b417-abstention.json"
        self.response = json.loads(path.read_text())["response"]
        self.response["answers"]["routing"].update(
            status="answered",
            value=["payments", "sales", "engineering"],
            abstained=False,
            reason=None,
        )
        self.payload = {
            "model": bundle.id,
            "input": {"text": "A billing question"},
            "questions": [],
        }
        for key, answer in self.response["answers"].items():
            question = {"id": key, "type": answer["type"], "instruction": "Test"}
            if answer["type"] != "boolean":
                question["options"] = [
                    {
                        "id": candidate,
                        "description": candidate,
                        **({"value": answer["levels"][candidate]} if answer["levels"] else {}),
                    }
                    for candidate in answer["probabilities"]
                ]
            self.payload["questions"].append(question)
        self.corrupt = corrupt
        self.attempted = 0

    async def call(self, path, request, *, management=False):
        if management:
            assert request["expected_generation"] == self.generation
            self.generation += 1
            self.bundle = next(b for b in self.bundles if b.reference == request["reference"])
            await asyncio.sleep(0.001)
            return {"generation": self.generation}
        self.attempted += 1
        index = self.attempted
        bundle, generation = self.bundle, self.generation
        question = request["questions"][0]
        data = deepcopy(self.response)
        data.update(
            request_id=request["request_id"],
            bundle=bundle.reference,
            bundle_digest=bundle.digest,
            generation=generation,
        )
        answer = data["answers"][question["id"]]
        data["answers"] = {question["id"]: answer}
        independent = bundle.version == 2 and answer["type"] != "boolean"
        if independent:
            answer.update(
                probability_semantics="normalized_support",
                support=answer["probabilities"],
                label_mass=None,
            )
        sequences = len(answer["probabilities"]) if independent else 1
        data["usage"].update(
            questions=1,
            successful_questions=1,
            scoring_sequences=sequences,
            logical_prompt_tokens=20,
            engine_prompt_tokens=20 * sequences,
            engine_completion_tokens=sequences,
            cached_prompt_tokens=0,
        )
        await asyncio.sleep(0.001)
        if self.corrupt:
            self.corrupt(data, index)
        return data


async def run(target, tmp_path, **options):
    checks = {}
    parameters = dict(
        minimum_requests=100,
        minimum_switches=4,
        concurrency=4,
        deadline_seconds=2,
        switch_interval=0.001,
    )
    parameters.update(options)
    task = exercise(
        target.call,
        payload=target.payload,
        bundles=target.bundles,
        generation=2,
        engine="vllm",
        output=tmp_path / "responses.jsonl.gz",
        checks=checks,
        **parameters,
    )
    return task, checks


@pytest.mark.parametrize("requests,switches", [(100, 4), (4, 50)])
async def test_both_minima_required_and_every_attempt_retained(
    bundle, tmp_path, requests, switches
):
    target = Target(bundle)
    task, checks = await run(target, tmp_path, minimum_requests=requests, minimum_switches=switches)
    await task
    report = checks["hot_switch_under_traffic"]
    assert report["passed"] and report["strict_success_requests"] >= requests
    assert report["switches"] >= switches
    assert report["attempted_requests"] == report["strict_success_requests"] == target.attempted
    assert set(report["requests_by_type"]) == {"boolean", "choice", "score", "rank"}
    with gzip.open(tmp_path / "responses.jsonl.gz", "rt") as file:
        rows = [json.loads(line) for line in file]
    requests = [row for row in rows if row["kind"] == "request"]
    assert len(requests) == target.attempted
    assert len({row["request"]["input"]["text"] for row in requests}) == len(requests)
    assert rows[-1]["passed"]


async def test_http_200_invalid_response_fails_without_retry(bundle, tmp_path):
    def corrupt(data, index):
        if index == 3:
            data["usage"]["engine_completion_tokens"] = 0

    target = Target(bundle, corrupt)
    task, checks = await run(target, tmp_path)
    with pytest.raises(AssertionError, match="Strict traffic failed"):
        await task
    report = checks["hot_switch_under_traffic"]
    assert not report["passed"] and report["failed_requests"] == 1 and report["retries"] == 0
    assert report["attempted_requests"] == report["strict_success_requests"] + 1
    assert report["attempted_requests"] < 100


async def test_generation_mismatch_rejected_after_responses_are_drained(bundle, tmp_path):
    def corrupt(data, index):
        if index == 3:
            data["generation"] = 999999

    task, checks = await run(Target(bundle, corrupt), tmp_path, minimum_requests=20)
    with pytest.raises(AssertionError, match="Strict traffic failed"):
        await task
    report = checks["hot_switch_under_traffic"]
    assert report["mixed_bundle_responses"] == report["failed_requests"] == 1
    assert report["strict_success_requests"] == report["attempted_requests"] - 1


async def test_deadline_retains_insufficient_request_count(bundle, tmp_path):
    task, checks = await run(
        Target(bundle), tmp_path, minimum_requests=10000, deadline_seconds=0.01
    )
    with pytest.raises(TimeoutError):
        await task
    report = checks["hot_switch_under_traffic"]
    assert not report["passed"] and report["strict_success_requests"] < 10000
    assert report["unaccounted_requests"] == 0
    assert json.loads((tmp_path / "traffic-progress.json").read_text())["passed"] is False


@pytest.mark.parametrize("field", ["request_id", "bundle_digest", "candidate", "engine", "usage"])
async def test_identity_or_usage_corruption_never_counts_as_strict_success(bundle, field):
    target = Target(bundle)
    request = {
        **target.payload,
        "request_id": "unique",
        "questions": target.payload["questions"][:1],
    }
    data = await target.call("/v1/decisions", request)
    if field == "request_id":
        data["request_id"] = "wrong"
    elif field == "bundle_digest":
        data["bundle_digest"] = "sha256:" + "f" * 64
    elif field == "candidate":
        data["answers"]["intent"]["probabilities"]["wrong"] = 0.0
    elif field == "engine":
        data["engine"]["name"] = "wrong"
    else:
        data["usage"]["engine_prompt_tokens"] = None
    with pytest.raises((AssertionError, ValueError, TypeError)):
        verify_response(data, request, {b.reference: b for b in target.bundles}, "vllm")
