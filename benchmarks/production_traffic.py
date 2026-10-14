"""Bounded, open-loop traffic for all three engines; never starts/stops a service.

Writes every planned request, including client overload and invalid HTTP 200s.
The raw log is private, streaming and contains no prompt, response or credential.
This is one input to qualification, not an automatic production certificate.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path

import httpx

from jev_runtime.production import sha256
from jev_runtime.schema import DecisionRequest, DecisionResponse


def strict_response(body, request: DecisionRequest, engine: str, version: str, routes: list[dict]):
    result = DecisionResponse.model_validate(body)
    if (
        result.status != "completed"
        or result.request_id != request.request_id
        or result.engine["name"] != engine
        or result.engine["version"] != version
    ):
        raise ValueError("Incomplete response or wrong request/engine identity")
    route = {
        "bundle": result.bundle,
        "bundle_digest": result.bundle_digest,
        "generation": result.generation,
    }
    if route not in routes:
        raise ValueError("Unqualified bundle or generation")
    expected = {q.id: q for q in request.questions}
    if (
        not expected
        or set(result.answers) != set(expected)
        or result.usage.questions != len(expected)
        or result.usage.successful_questions != len(expected)
    ):
        raise ValueError("Question coverage mismatch")
    for key, answer in result.answers.items():
        question = expected[key]
        labels = (
            {"true", "false"} if question.type == "boolean" else {o.id for o in question.options}
        )
        if answer.type != question.type or set(answer.probabilities or {}) != labels:
            raise ValueError("Answer type or candidate mismatch")
        if question.type == "score" and all(o.value is not None for o in question.options):
            if answer.levels != {o.id: o.value for o in question.options}:
                raise ValueError("Score levels differ from requested levels")
    return result


async def run(args, *, transport=None) -> dict:
    if (
        not math.isfinite(args.rate)
        or args.rate <= 0
        or not math.isfinite(args.duration)
        or args.duration <= 0
        or args.concurrency < 1
        or args.queue_size < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        raise ValueError("Rate, duration, concurrency, queue and timeout must be positive")
    key = os.environ.get(args.api_key_env)
    if not key:
        raise ValueError("Data-plane credential environment variable is missing")
    requests = [
        DecisionRequest.model_validate_json(line)
        for line in args.requests.read_text().splitlines()
        if line.strip()
    ]
    if not requests or any(not request.questions for request in requests):
        raise ValueError("Provide explicit questions so coverage can be independently checked")
    routes = json.loads(args.routes.read_text())
    if not isinstance(routes, list) or not routes:
        raise ValueError("An explicit qualified route list is required")
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    # Disk-backed latency samples keep memory bounded during a 24-hour campaign.
    db_path = args.output / "latencies.sqlite3"
    db_path.touch(mode=0o600)
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE samples (latency_ms REAL NOT NULL)")
    raw_path = args.output / "requests.jsonl"
    raw = os.fdopen(os.open(raw_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")
    counts = collections.Counter()
    queue = asyncio.Queue(maxsize=args.queue_size)
    started_at, start = time.time(), time.monotonic()
    planned = math.ceil(args.duration * args.rate)

    def record(row):
        counts[row["outcome"]] += 1
        if row.get("http_status") is not None:
            counts["http_" + str(row["http_status"])] += 1
        raw.write(json.dumps(row, allow_nan=False) + "\n")
        if row["outcome"] == "strict_success":
            db.execute("INSERT INTO samples VALUES (?)", (row["latency_ms"],))
        if (
            sum(
                counts[k]
                for k in (
                    "strict_success",
                    "http_error",
                    "invalid_response",
                    "transport_error",
                    "client_overload",
                )
            )
            % 1000
            == 0
        ):
            raw.flush()
            db.commit()

    async with httpx.AsyncClient(
        base_url=args.url.rstrip("/") + "/",
        timeout=args.timeout,
        trust_env=False,
        headers={"Authorization": "Bearer " + key},
        transport=transport,
        limits=httpx.Limits(
            max_connections=args.concurrency, max_keepalive_connections=args.concurrency
        ),
    ) as client:

        async def worker():
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    index, scheduled = item
                    request = requests[index % len(requests)].model_copy(
                        update={"request_id": "prod-" + uuid.uuid4().hex}
                    )
                    row = {
                        "sequence": index,
                        "request_id": request.request_id,
                        "scheduled_seconds": scheduled - start,
                        "sent": True,
                    }
                    try:
                        response = await client.post(
                            "v1/decisions", json=request.model_dump(mode="json")
                        )
                        row["http_status"] = response.status_code
                        if response.status_code != 200:
                            row["outcome"] = "http_error"
                        else:
                            try:
                                strict_response(
                                    response.json(),
                                    request,
                                    args.engine,
                                    args.engine_version,
                                    routes,
                                )
                                row["outcome"] = "strict_success"
                            except (ValueError, KeyError, TypeError):
                                row["outcome"] = "invalid_response"
                    except httpx.HTTPError as exc:
                        row.update(outcome="transport_error", error=type(exc).__name__)
                    row["latency_ms"] = (time.monotonic() - scheduled) * 1000
                    record(row)
                finally:
                    queue.task_done()

        async def produce():
            for index in range(planned):
                scheduled = start + index / args.rate
                await asyncio.sleep(max(0, scheduled - time.monotonic()))
                if time.monotonic() - scheduled > max(0.1, 1 / args.rate):
                    counts["late_schedule"] += 1
                try:
                    queue.put_nowait((index, scheduled))
                except asyncio.QueueFull:
                    record(
                        {
                            "sequence": index,
                            "scheduled_seconds": scheduled - start,
                            "sent": False,
                            "outcome": "client_overload",
                        }
                    )
            await queue.join()
            await asyncio.sleep(max(0, start + args.duration - time.monotonic()))
            for _ in range(args.concurrency):
                await queue.put(None)

        completed = False
        try:
            async with asyncio.TaskGroup() as tasks:
                for _ in range(args.concurrency):
                    tasks.create_task(worker())
                tasks.create_task(produce())
            completed = True
        finally:
            elapsed = time.monotonic() - start
            raw.flush()
            os.fsync(raw.fileno())
            raw.close()
            db.commit()
            successes = counts["strict_success"]
            offset = max(0, math.ceil(successes * 0.99) - 1)
            p99 = db.execute(
                "SELECT latency_ms FROM samples ORDER BY latency_ms LIMIT 1 OFFSET ?", (offset,)
            ).fetchone()
            db.close()
            outcomes = (
                "strict_success",
                "http_error",
                "invalid_response",
                "transport_error",
                "client_overload",
            )
            accounted = sum(counts[k] for k in outcomes)
            report = {
                "schema_version": 1,
                "scope": "open-loop traffic; other production gates remain required",
                "completed": completed,
                "engine": args.engine,
                "engine_version": args.engine_version,
                "started_at": started_at,
                "finished_at": time.time(),
                "planned": planned,
                "accounted": accounted,
                "sent": accounted - counts["client_overload"],
                "unaccounted_requests": planned - accounted,
                "strict_successes": successes,
                "strict_success_rate": successes / planned,
                "outcomes": dict(counts),
                "arrival_window_seconds": args.duration,
                "elapsed_seconds": elapsed,
                "strict_requests_per_second": successes / max(args.duration, elapsed),
                "p99_ms": p99[0] if p99 else None,
                "open_loop": True,
                "load_generator_kept_up": counts["client_overload"] == 0
                and counts["late_schedule"] == 0,
                "requests_sha256": sha256(args.requests),
                "routes_sha256": sha256(args.routes),
                "raw_sha256": sha256(raw_path),
                "passed": completed and accounted == planned and successes == planned,
            }
            report_path = args.output / "summary.json"
            with os.fdopen(
                os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
            ) as file:
                json.dump(report, file, indent=2, allow_nan=False)
                file.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", required=True, help="Typed API base, including plugin prefix if applicable"
    )
    parser.add_argument("--engine", required=True, choices=("vllm", "sglang", "tokenspeed"))
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--requests", required=True, type=Path, help="DecisionRequest JSONL")
    parser.add_argument(
        "--routes", required=True, type=Path, help="Accepted bundle/digest/generation tuples"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-key-env", default="JEV_API_KEY")
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120)
    report = asyncio.run(run(parser.parse_args()))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
