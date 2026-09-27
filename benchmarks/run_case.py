"""Paired native/typed HTTP scoring measurements against a task-owned test service.

Inputs come from /admin/compile in the actual serving worker. This runner does not
start engines, alter their cache settings, or claim release certification. Keep
all attempts and repeats, including failed or short development runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx

from jev_runtime.backends.base import ScoreInput, ScoreResult
from jev_runtime.backends.sglang import generate_payload, parse_sglang
from jev_runtime.compiler import CompiledQuestion
from jev_runtime.config import load_settings
from jev_runtime.performance import Measurement, summarize_cohort
from jev_runtime.schema import (
    Bundle,
    DecisionResponse,
    Policy,
    Question,
    TemplateSpec,
    content_digest,
)
from jev_runtime.scoring import assemble


class InvalidResponse(Exception):
    def __init__(self, http_status: int):
        self.http_status = http_status


def parse_native(engine: str, response: httpx.Response, item: ScoreInput) -> dict:
    response.raise_for_status()
    try:
        if engine == "sglang":
            result = asdict(parse_sglang(response.json(), item))
        else:
            body = response.json()
            assert len(body["choices"]) == 1
            choice = body["choices"][0]
            assert choice["finish_reason"] in {"length", "stop"}
            distributions = choice["logprobs"]["top_logprobs"]
            assert len(distributions) == 1 and body["usage"]["completion_tokens"] == 1
            details = body["usage"].get("prompt_tokens_details") or {}
            result = {
                "logprobs": [distributions[0][f"token_id:{token}"] for token in item.label_ids],
                "prompt_tokens": body["usage"]["prompt_tokens"],
                "cached_tokens": details.get("cached_tokens"),
            }
        assert len(result["logprobs"]) == len(item.label_ids)
        assert all(math.isfinite(value) and value <= 1e-6 for value in result["logprobs"])
        assert math.fsum(math.exp(value) for value in result["logprobs"]) <= 1.0001
        return result
    except Exception as exc:
        raise InvalidResponse(response.status_code) from exc


async def measure(
    call,
    duration: float,
    concurrency: int,
    question_count: int,
    sequence_count: int,
    logical_tokens: int,
):
    rows = []
    started = time.perf_counter()
    deadline = started + duration

    async def worker():
        while time.perf_counter() < deadline:
            begin = time.perf_counter()
            if begin >= deadline:
                break
            request_id = "bench-" + uuid.uuid4().hex
            values = {
                "request_id": request_id,
                "started_ms": (begin - started) * 1000,
                "questions": question_count,
                "successful_questions": 0,
                "scoring_sequences": sequence_count,
                "logical_prompt_tokens": logical_tokens,
                "outcome": "failed",
            }
            try:
                async with asyncio.timeout(120):
                    values.update(await call(request_id))
            except httpx.HTTPStatusError as exc:
                values.update(http_status=exc.response.status_code, error_code="http_error")
            except InvalidResponse as exc:
                values.update(http_status=exc.http_status, error_code="invalid_response_contract")
            except Exception as exc:
                values.update(error_code=type(exc).__name__)
            values["finished_ms"] = (time.perf_counter() - started) * 1000
            rows.append(Measurement.model_validate(values))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    makespan = max(duration, time.perf_counter() - started)
    return rows, summarize_cohort(rows, duration, makespan)


async def run(args):
    if args.output.exists():
        raise ValueError("Output exists; retain previous attempts and choose a new directory")
    args.output.mkdir(parents=True)
    settings = load_settings(args.run_dir / "config.json")
    record = json.loads((args.run_dir / "process.json").read_text())
    keys = json.loads((args.run_dir / "keys.json").read_text())
    url = f"http://127.0.0.1:{record['port']}"
    prefix = "" if record["mode"] == "gateway" else "/plugins/jev-runtime"
    engine_key = keys["api"]
    if args.engine_run_dir:
        engine_key = json.loads((args.engine_run_dir / "keys.json").read_text())["api"]
    report = {
        "schema_version": 1,
        "source_commit": args.source_commit,
        "started_at": time.time(),
        "qualification": "development HTTP measurements; not release performance certification",
        "release_gate_passed": False,
        "engine": settings.backend,
        "mode": record["mode"],
        "scenario": {
            "context_tokens": args.context_tokens,
            "candidates": args.candidates,
            "concurrency": args.concurrency,
            "cache": args.cache,
            "scoring_mode": args.scoring_mode,
        },
        "model": {"id": settings.model_id, "revision": settings.model_revision},
        "duration_seconds": args.duration,
        "repeats": args.repeats,
        "warmup_requests_per_method": args.warmup,
        "service_process_identity": record["identity"],
        "service_launch_command": record["command"],
        "measurements": [],
    }
    alias = "bench-" + uuid.uuid4().hex
    activated = False
    timeout = httpx.Timeout(130, connect=10)
    limits = httpx.Limits(
        max_connections=max(64, args.concurrency * args.candidates), max_keepalive_connections=64
    )
    async with (
        httpx.AsyncClient(
            base_url=url,
            timeout=timeout,
            limits=limits,
            headers={"Authorization": "Bearer " + keys["api"]},
        ) as typed,
        httpx.AsyncClient(
            base_url=url,
            timeout=timeout,
            headers={"Authorization": "Bearer " + keys["admin"]},
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as admin,
        httpx.AsyncClient(
            base_url=settings.engine_url,
            timeout=timeout,
            limits=limits,
            headers={"Authorization": "Bearer " + engine_key},
        ) as native,
    ):

        async def manage(path, body=None):
            response = await (
                admin.get(prefix + path) if body is None else admin.post(prefix + path, json=body)
            )
            response.raise_for_status()
            return response.json()

        async def preview(body):
            return await manage("/admin/compile", body)

        try:
            response = await typed.get(prefix + "/v1/capabilities")
            response.raise_for_status()
            capabilities = response.json()
            report["capabilities"] = capabilities
            expected_cache = args.cache == "hot"
            if capabilities.get("prefix_cache") is not expected_cache:
                raise ValueError(
                    "Cache configuration is unknown or differs from this scenario; "
                    "run on a separately configured matching engine"
                )
            report["cache_definition"] = (
                "repeated exact prompt after warmup"
                if expected_cache
                else "engine prefix cache disabled, including candidate reuse"
            )
            seed = await preview(
                {
                    "model": settings.bootstrap_alias,
                    "input": {"text": "ready"},
                    "questions": [{"id": "q", "type": "boolean", "instruction": "Is it ready?"}],
                }
            )
            model = seed["bundle"]["model"]
            if model["id"] != settings.model_id or model["revision"] != settings.model_revision:
                raise ValueError("Serving model manifest differs from configured checkpoint")
            bundle = Bundle(
                id=alias,
                version=1,
                model=model,
                template=TemplateSpec(mode=args.scoring_mode),
                policy=Policy(max_expanded_tokens=max(262144, args.context_tokens * 64)),
            )
            await manage("/admin/bundles", bundle.model_dump(mode="json"))
            async with asyncio.timeout(180):
                while True:
                    await manage("/admin/bundles/prepare", {"reference": bundle.reference})
                    workers = (await manage("/admin/workers"))["workers"]
                    serving = [
                        w
                        for w in workers
                        if w["state"] == "SERVING" and w["owner_status"] != "dead"
                    ]
                    if serving and all(bundle.reference in w["prepared"] for w in serving):
                        break
            await manage(
                "/admin/bundles/activate",
                {"alias": alias, "reference": bundle.reference, "expected_generation": 0},
            )
            activated = True
            question = {
                "id": "category",
                "type": "choice",
                "instruction": "Choose a category.",
                "options": [{"id": f"c{i}", "description": str(i)} for i in range(args.candidates)],
            }
            body = {
                "model": alias,
                "input": {"text": " x"},
                "questions": [question],
                "execution": {"timeout_ms": 120000},
            }
            # Token lengths are measured by the actual worker, including template,
            # labels and task text. Search padding; never round an 8k request down.
            low, high = 0, args.context_tokens
            fixture = None
            while low <= high:
                size = (low + high) // 2
                body["input"]["text"] = " x" * (size + 1)
                try:
                    candidate = await preview(body)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 413:
                        high = size - 1
                        continue
                    raise
                length = max(len(sequence["input_ids"]) for sequence in candidate["sequences"])
                if length == args.context_tokens:
                    fixture = candidate
                    break
                if length < args.context_tokens:
                    low = size + 1
                else:
                    high = size - 1
            if fixture is None:
                raise ValueError("Cannot construct the exact requested context with this profile")
            fixture["request"] = body
            fixture["sequence_lengths"] = [len(s["input_ids"]) for s in fixture["sequences"]]
            fixture["fixture_digest"] = content_digest(fixture)
            (args.output / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
            report["fixture_digest"] = fixture["fixture_digest"]
            report["sequence_lengths"] = fixture["sequence_lengths"]
            report["bundle_digest"] = bundle.digest
            report["tokenizer_implementation_digest"] = fixture["tokenizer_implementation_digest"]

            async def native_branch(sequence, request_id):
                item = ScoreInput(
                    **{
                        **sequence,
                        "input_ids": tuple(sequence["input_ids"]),
                        "label_ids": tuple(sequence["label_ids"]),
                        "request_id": request_id,
                    }
                )
                if settings.backend == "sglang":
                    response = await native.post("/generate", json=generate_payload(item))
                    return parse_native(settings.backend, response, item)
                response = await native.post(
                    "/v1/completions",
                    json={
                        "model": settings.model_id,
                        "request_id": request_id,
                        "prompt": item.input_ids,
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "logprobs": len(item.label_ids),
                        "logprob_token_ids": item.label_ids,
                        "return_tokens_as_token_ids": True,
                        "stream": False,
                    },
                )
                return parse_native(settings.backend, response, item)

            async def native_call(request_id):
                # A native logical request retains every scoring branch. The
                # independent baseline is deliberately serial within a request;
                # cross-request engine scheduling is unchanged and recorded.
                values = [
                    await native_branch(sequence, f"{request_id}.{i}")
                    for i, sequence in enumerate(fixture["sequences"])
                ]

                def summed(key):
                    return (
                        sum(value[key] for value in values)
                        if all(value.get(key) is not None for value in values)
                        else None
                    )

                return {
                    "outcome": "completed",
                    "successful_questions": 1,
                    "http_status": 200,
                    "engine_prompt_tokens": summed("prompt_tokens"),
                    "cached_prompt_tokens": summed("cached_tokens"),
                }

            async def typed_call(request_id):
                response = await typed.post(
                    prefix + "/v1/decisions", json={**body, "request_id": request_id}
                )
                response.raise_for_status()
                try:
                    result = DecisionResponse.model_validate(response.json())
                    assert result.bundle_digest == bundle.digest and result.generation == 1
                    assert result.usage.logical_prompt_tokens == fixture["logical_prompt_tokens"]
                    probabilities = result.answers["category"].probabilities or {}
                    assert set(probabilities) == {f"c{i}" for i in range(args.candidates)}
                    assert all(0 <= value <= 1 for value in probabilities.values())
                    assert abs(math.fsum(probabilities.values()) - 1) < 1e-6
                except (ValueError, AssertionError, KeyError) as exc:
                    raise InvalidResponse(response.status_code) from exc
                return {
                    "outcome": result.status,
                    "successful_questions": result.usage.successful_questions,
                    "http_status": response.status_code,
                    "engine_prompt_tokens": result.usage.engine_prompt_tokens,
                    "cached_prompt_tokens": result.usage.cached_prompt_tokens,
                }

            reference_sequences, reference_results = [], []
            for sequence in fixture["sequences"]:
                rid = "parity-" + uuid.uuid4().hex
                actual = await native_branch(sequence, rid)
                reference_sequences.append(
                    ScoreInput(
                        **{
                            **sequence,
                            "request_id": rid,
                            "input_ids": tuple(sequence["input_ids"]),
                            "label_ids": tuple(sequence["label_ids"]),
                        }
                    )
                )
                reference_results.append(ScoreResult(rid, tuple(actual["logprobs"])))
            expected = assemble(
                CompiledQuestion(
                    Question.model_validate(question),
                    args.scoring_mode,
                    tuple(fixture["questions"][0]["keys"]),
                    tuple(reference_sequences),
                ),
                reference_results,
                bundle,
            )
            response = await typed.post(prefix + "/v1/decisions", json=body)
            response.raise_for_status()
            actual_answer = DecisionResponse.model_validate(response.json()).answers["category"]
            parity = max(
                abs(expected.probabilities[key] - actual_answer.probabilities[key])
                for key in expected.probabilities
            )
            report["native_typed_probability_parity"] = {
                "max_absolute_error": parity,
                "tolerance": args.parity_tolerance,
                "passed": parity <= args.parity_tolerance,
            }
            assert parity <= args.parity_tolerance, report["native_typed_probability_parity"]

            methods = [("native-label", native_call), (record["mode"], typed_call)]
            for repeat in range(args.repeats):
                for name, call in methods[repeat % 2 :] + methods[: repeat % 2]:
                    for _ in range(args.warmup):
                        result = await call("warmup-" + uuid.uuid4().hex)
                        assert result["outcome"] == "completed"
                    rows, summary = await measure(
                        call,
                        args.duration,
                        args.concurrency,
                        1,
                        len(fixture["sequences"]),
                        fixture["logical_prompt_tokens"],
                    )
                    path = f"{repeat + 1}-{name}.jsonl"
                    (args.output / path).write_text(
                        "\n".join(row.model_dump_json() for row in rows) + "\n"
                    )
                    entry = {"repeat": repeat + 1, "method": name, "rows": path, "summary": summary}
                    report["measurements"].append(entry)
                    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                "repeat": repeat + 1,
                                "method": name,
                                "strict_success": summary["strict_success_requests"],
                                "attempts": summary["dispatched_requests"],
                            }
                        ),
                        flush=True,
                    )
            report["measurement_complete"] = True
            report["all_attempts_successful"] = all(
                item["summary"]["dispatched_requests"] > 0
                and item["summary"]["strict_success_requests"]
                == item["summary"]["dispatched_requests"]
                for item in report["measurements"]
            )
        except BaseException as exc:
            report["measurement_complete"] = False
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            raise
        finally:
            if activated:
                try:
                    await manage(
                        "/admin/bundles/disable", {"alias": alias, "expected_generation": 1}
                    )
                    await manage("/admin/bundles/retire", {"reference": bundle.reference})
                    report["temporary_alias_retired"] = True
                except Exception as exc:
                    report["cleanup_error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc)[:500],
                    }
            report["finished_at"] = time.time()
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine-run-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--candidates", type=int, choices=(2, 8, 32), required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 16), required=True)
    parser.add_argument("--cache", choices=("cold", "hot"), required=True)
    parser.add_argument(
        "--scoring-mode", choices=("joint-label", "independent-candidate"), default="joint-label"
    )
    parser.add_argument("--duration", type=float, default=180)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--parity-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if args.duration <= 0 or args.repeats <= 0 or args.warmup < 1 or args.context_tokens < 16:
        parser.error("Duration, repeats, warmup and context must be positive")
    asyncio.run(run(args))
