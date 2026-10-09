"""Count-bounded live traffic with retained responses and no retry-to-success."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from jev_runtime.schema import Bundle, DecisionResponse


def verify_response(data: dict, request: dict, bundles: dict[str, Bundle], engine: str):
    result = DecisionResponse.model_validate(data)
    assert result.request_id == request["request_id"], "request identity mismatch"
    assert result.engine["name"] == engine, "engine identity mismatch"
    assert result.status == "completed", "incomplete response"
    bundle = bundles[result.bundle]
    assert result.bundle_digest == bundle.digest, "bundle digest mismatch"
    assert result.usage.questions == result.usage.successful_questions == 1
    question = request["questions"][0]
    assert set(result.answers) == {question["id"]}, "question identity mismatch"
    answer = result.answers[question["id"]]
    assert answer.type == question["type"] and answer.status == "answered"
    assert answer.calibration_status == "uncalibrated"
    candidates = (
        {"true", "false"}
        if question["type"] == "boolean"
        else {option["id"] for option in question["options"]}
    )
    assert set(answer.probabilities) == candidates, "candidate identity mismatch"
    independent = bundle.template.mode == "independent-candidate" and question["type"] != "boolean"
    assert answer.probability_semantics == (
        "normalized_support" if independent else "conditional_label_distribution"
    )
    if question["type"] == "score":
        assert answer.levels == {option["id"]: option["value"] for option in question["options"]}
    sequences = len(candidates) if independent else 1
    if engine == "tokenspeed":
        # Each candidate prompt scores true/false separately; a joint prompt
        # scores one native branch per option (two for a Boolean).
        sequences = sequences * 2 if independent else len(candidates)
    assert result.usage.scoring_sequences == sequences, "sequence usage mismatch"
    # SGLang uses max_new_tokens=0 selected-ID readout; vLLM generates one per branch.
    assert engine in {"sglang", "vllm", "tokenspeed"}, "Unknown live-traffic engine"
    completions = 0 if engine == "sglang" else sequences
    assert result.usage.engine_completion_tokens == completions, "completion usage mismatch"
    assert result.usage.logical_prompt_tokens > 0 and result.usage.engine_prompt_tokens > 0
    assert 0 <= result.usage.cached_prompt_tokens <= result.usage.engine_prompt_tokens
    return result


async def exercise(
    call,
    *,
    payload: dict,
    bundles: tuple[Bundle, Bundle],
    generation: int,
    engine: str,
    minimum_requests: int,
    minimum_switches: int,
    concurrency: int,
    deadline_seconds: float,
    output: Path,
    checks: dict,
    switch_interval: float = 0.005,
) -> int:
    if min(minimum_requests, minimum_switches, concurrency, deadline_seconds) <= 0:
        raise ValueError("Traffic count, switches, concurrency and timeout must be positive")
    report = {
        "minimum_requests": minimum_requests,
        "minimum_switches": minimum_switches,
        "concurrency": concurrency,
        "timeout_seconds": deadline_seconds,
        "attempted_requests": 0,
        "response_validated_requests": 0,
        "strict_success_requests": 0,
        "failed_requests": 0,
        "switches": 0,
        "mixed_bundle_responses": 0,
        "retries": 0,
        "passed": False,
        "qualification": "four rotating question types; unique synthetic inputs; functional only",
    }
    checks["hot_switch_under_traffic"] = report
    by_reference = {bundle.reference: bundle for bundle in bundles}
    generations = {generation: bundles[1].reference}
    seen = []
    failures = []
    request_ids = set()
    counts = Counter()
    branches = Counter()
    started = time.monotonic()
    report["started_at"] = time.time()
    stop = asyncio.Event()
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_name("traffic-progress.json")

    def progress():
        value = {**report, "elapsed_seconds": time.monotonic() - started}
        temporary = progress_path.with_suffix(".pending")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(progress_path)

    with gzip.open(output, "xt", encoding="utf-8") as evidence:

        def record(value):
            evidence.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            evidence.flush()

        record(
            {
                "kind": "configuration",
                "starting_generation": generation,
                "starting_bundle": bundles[1].reference,
                "bundles": [bundle.model_dump(mode="json") for bundle in bundles],
                "engine": engine,
                "payload": payload,
                "requirements": dict(report),
            }
        )
        progress()

        async def traffic():
            while not stop.is_set():
                index = report["attempted_requests"]
                report["attempted_requests"] += 1
                question = payload["questions"][index % len(payload["questions"])]
                request = {
                    **payload,
                    "request_id": f"{payload['model']}.traffic.{index:08d}",
                    "questions": [question],
                    "input": {"text": payload["input"]["text"] + f" Case number {index:08d}."},
                }
                row = {"kind": "request", "index": index, "request": request}
                sent = time.monotonic()
                try:
                    data = await call("/v1/decisions", request)
                    row["response"] = data
                    result = verify_response(data, request, by_reference, engine)
                    assert result.request_id not in request_ids, "duplicate response ID"
                    request_ids.add(result.request_id)
                    seen.append((result.generation, result.bundle, result.bundle_digest))
                    report["response_validated_requests"] += 1
                    counts[question["type"]] += 1
                    branches[result.bundle] += 1
                    row["outcome"] = "response_validated_pending_generation_audit"
                except Exception as exc:
                    row["outcome"] = "failed"
                    row["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
                    failures.append(row["failure"])
                    report["failed_requests"] += 1
                    stop.set()
                finally:
                    row["latency_ms"] = (time.monotonic() - sent) * 1000
                    record(row)
                    if (len(seen) + len(failures)) % 100 == 0 or stop.is_set():
                        progress()

        workers = [asyncio.create_task(traffic()) for _ in range(concurrency)]
        controller_failure = None
        try:
            while not stop.is_set():
                if any(worker.done() for worker in workers):
                    raise RuntimeError("A traffic worker exited unexpectedly")
                if report["switches"] >= minimum_switches and len(seen) >= minimum_requests:
                    break
                if time.monotonic() - started >= deadline_seconds:
                    raise TimeoutError("Traffic minimum was not reached within the deadline")
                target = bundles[report["switches"] % 2]
                activated = await call(
                    "/admin/bundles/activate",
                    {
                        "alias": payload["model"],
                        "reference": target.reference,
                        "expected_generation": generation,
                    },
                    management=True,
                )
                assert activated["generation"] == generation + 1, "activation generation mismatch"
                generation = activated["generation"]
                generations[generation] = target.reference
                report["switches"] += 1
                record({"kind": "activation", "generation": generation, "bundle": target.reference})
                await asyncio.sleep(switch_interval)
        except BaseException as exc:
            controller_failure = exc
            report["controller_failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        finally:
            stop.set()
            # Finish every submitted HTTP request; no silent cancellation or retry.
            settled = await asyncio.gather(*workers, return_exceptions=True)
            for value in settled:
                if isinstance(value, BaseException):
                    controller_failure = controller_failure or value
                    report["worker_failure"] = {
                        "type": type(value).__name__,
                        "message": str(value)[:2000],
                    }
            mixed = sum(generations.get(g) != ref for g, ref, _ in seen)
            report.update(
                strict_success_requests=len(seen) - mixed,
                failed_requests=len(failures) + mixed,
                mixed_bundle_responses=mixed,
                versions_seen=sorted(branches),
                requests_by_type=dict(counts),
                requests_by_bundle=dict(branches),
                unaccounted_requests=report["attempted_requests"] - len(seen) - len(failures),
                elapsed_seconds=time.monotonic() - started,
                finished_at=time.time(),
            )
            report["passed"] = bool(
                controller_failure is None
                and not failures
                and not mixed
                and report["unaccounted_requests"] == 0
                and len(seen) >= minimum_requests
                and report["switches"] >= minimum_switches
                and len(branches) == 2
                and set(counts) == {question["type"] for question in payload["questions"]}
            )
            record({"kind": "summary", **report})
            progress()
    report["evidence_file"] = output.name
    report["evidence_sha256"] = hashlib.sha256(
        await asyncio.to_thread(output.read_bytes)
    ).hexdigest()
    if controller_failure is not None:
        raise controller_failure
    if not report["passed"]:
        raise AssertionError(
            "Strict traffic failed; inspect retained request and progress evidence"
        )
    return generation
