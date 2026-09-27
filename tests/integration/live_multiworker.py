"""Validate native-plugin or gateway workers sharing one local registry on Linux."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx

from jev_runtime.config import load_compiler, load_settings, model_identity
from jev_runtime.schema import Bundle, DecisionResponse, Policy


async def run(args):
    settings = load_settings(args.run_dir / "config.json")
    record = json.loads((args.run_dir / "process.json").read_text())
    keys = json.loads((args.run_dir / "keys.json").read_text())
    prefix = "" if record["mode"] == "gateway" else "/plugins/jev-runtime"
    url = f"http://127.0.0.1:{record['port']}"
    report = {
        "source_commit": args.source_commit,
        "engine": settings.backend,
        "mode": record["mode"],
        "started_at": time.time(),
        "qualification": "co-located multi-process functional test; not throughput certification",
        "expected_workers": args.workers,
        "checks": {},
    }
    checks = report["checks"]
    traffic = []
    stop = asyncio.Event()
    async with (
        httpx.AsyncClient(
            base_url=url,
            timeout=180,
            headers={"Authorization": "Bearer " + keys["api"]},
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as data,
        httpx.AsyncClient(
            base_url=url,
            timeout=180,
            headers={"Authorization": "Bearer " + keys["admin"]},
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as admin,
    ):

        async def management(path, payload=None):
            response = await (
                admin.get(prefix + path)
                if payload is None
                else admin.post(prefix + path, json=payload)
            )
            response.raise_for_status()
            return response.json()

        async def prepare_every_worker(bundle, workers):
            seen = set()
            async with asyncio.timeout(180):
                while seen != workers:
                    result = await management(
                        "/admin/bundles/prepare", {"reference": bundle.reference}
                    )
                    seen.add(result["worker_id"])
                    assert seen <= workers, "Unplanned worker joined the test"
            return sorted(seen)

        try:
            seen_ready = {}
            async with asyncio.timeout(240):
                while len(seen_ready) < args.workers:
                    try:
                        response = await data.get(prefix + "/ready")
                        if response.status_code == 200:
                            result = response.json()
                            seen_ready[result["worker_id"]] = result
                    except httpx.TransportError:
                        pass
                    if len(seen_ready) < args.workers:
                        await asyncio.sleep(0.05)
            workers = set(seen_ready)
            snapshot = await management("/admin/workers")
            active = {
                item["worker_id"]
                for item in snapshot["workers"]
                if item["state"] == "SERVING" and item["owner_status"] == "alive"
            }
            assert active == workers and len(workers) == args.workers
            checks["workers_ready"] = seen_ready
            compiler = await asyncio.to_thread(load_compiler, settings)
            first = Bundle(
                id="mw-" + uuid.uuid4().hex, version=1, model=model_identity(settings, compiler)
            )
            second = first.model_copy(update={"version": 2, "policy": Policy(min_probability=0.99)})
            for bundle in (first, second):
                await management("/admin/bundles", bundle.model_dump(mode="json"))
            checks["prepared_v1"] = await prepare_every_worker(first, workers)
            await management(
                "/admin/bundles/activate",
                {"alias": first.id, "reference": first.reference, "expected_generation": 0},
            )
            one = await management("/admin/bundles/prepare", {"reference": second.reference})
            attempt = await admin.post(
                prefix + "/admin/bundles/activate",
                json={"alias": first.id, "reference": second.reference, "expected_generation": 1},
            )
            assert attempt.status_code in (409, 503), attempt.text[:500]
            assert attempt.json()["error"]["code"] in {"replica_not_ready", "replicas_not_ready"}
            state = await management("/admin/bundles")
            route = next(row for row in state["routes"] if row["alias"] == first.id)
            assert route["ref"] == first.reference and route["generation"] == 1
            checks["partial_prepare_preserves_old_route"] = {
                "prepared_worker": one["worker_id"],
                "activation_status": attempt.status_code,
                "error_code": attempt.json()["error"]["code"],
                "route": route,
            }
            checks["prepared_v2"] = await prepare_every_worker(second, workers)
            body = {
                "model": first.id,
                "input": {"text": "Please refund the duplicate charge."},
                "questions": [
                    {
                        "id": "refund",
                        "type": "boolean",
                        "instruction": "Does the buyer ask for a refund?",
                    }
                ],
            }
            generations = {1: first}
            results = []
            errors = []

            async def worker():
                while not stop.is_set():
                    response = await data.post(prefix + "/v1/decisions", json=body)
                    if response.status_code != 200:
                        errors.append({"http": response.status_code, "body": response.text[:500]})
                        stop.set()
                        return
                    try:
                        result = DecisionResponse.model_validate(response.json())
                        assert (
                            result.status == "completed" and result.usage.successful_questions == 1
                        )
                        results.append((result, response.headers.get("x-jev-worker")))
                    except Exception as exc:
                        errors.append({"type": type(exc).__name__, "message": str(exc)[:500]})
                        stop.set()
                        return

            traffic = [asyncio.create_task(worker()) for _ in range(4)]
            for generation in range(1, args.switches + 1):
                target = second if generation % 2 else first
                # Record before publishing: a fast response may arrive before
                # the management HTTP response reaches this load generator.
                generations[generation + 1] = target
                result = await management(
                    "/admin/bundles/activate",
                    {
                        "alias": first.id,
                        "reference": target.reference,
                        "expected_generation": generation,
                    },
                )
                assert result["generation"] == generation + 1
                if errors:
                    break
            stop.set()
            await asyncio.gather(*traffic)
            assert not errors, errors
            assert results, "No completed decisions during switching"
            response_workers = {worker_id for _, worker_id in results}
            assert response_workers == workers, (response_workers, workers)
            versions = set()
            for result, _ in results:
                expected = generations[result.generation]
                assert (
                    result.bundle == expected.reference and result.bundle_digest == expected.digest
                )
                versions.add(result.bundle)
            assert versions == {first.reference, second.reference}
            checks["hot_switches"] = {
                "switches": args.switches,
                "strict_success_requests": len(results),
                "response_workers": sorted(response_workers),
                "versions": sorted(versions),
                "mixed_versions": 0,
                "errors": errors,
            }
            checks["large_joint_readouts"] = []
            for count in (32, 64):
                response = await data.post(
                    prefix + "/v1/decisions",
                    json={
                        "model": first.id,
                        "input": {"text": "A refund"},
                        "questions": [
                            {
                                "id": "category",
                                "type": "choice",
                                "instruction": "Select a category.",
                                "options": [
                                    {"id": f"c{i}", "description": f"Category {i}"}
                                    for i in range(count)
                                ],
                            }
                        ],
                    },
                )
                response.raise_for_status()
                result = DecisionResponse.model_validate(response.json())
                assert result.status == "completed"
                assert len(result.answers["category"].probabilities) == count
                checks["large_joint_readouts"].append(
                    {"candidates": count, "response": result.model_dump(mode="json")}
                )
            state = await management("/admin/bundles")
            assert not state["leases"], state["leases"]
            await management(
                "/admin/bundles/disable",
                {"alias": first.id, "expected_generation": args.switches + 1},
            )
            for bundle in (first, second):
                assert (await management("/admin/bundles/retire", {"reference": bundle.reference}))[
                    "state"
                ] == "RETIRED"
            chat = (
                await data.post(
                    "/v1/chat/completions",
                    json={
                        "model": settings.model_id,
                        "messages": [{"role": "user", "content": "Say hello."}],
                        "max_tokens": 8,
                        "temperature": 0,
                    },
                )
                if prefix
                else None
            )
            if chat is not None:
                chat.raise_for_status()
                assert chat.json()["usage"]["completion_tokens"] > 0
                checks["native_chat_survives"] = True
            report["passed"] = True
        except BaseException as exc:
            report["passed"] = False
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            raise
        finally:
            stop.set()
            await asyncio.gather(*traffic, return_exceptions=True)
            report["finished_at"] = time.time()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "passed": report.get("passed"),
                        "checks": list(checks),
                        "output": str(args.output),
                    }
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--switches", type=int, default=1000)
    args = parser.parse_args()
    if args.workers < 2 or args.switches < 2:
        parser.error("At least two workers and two switches are required")
    asyncio.run(run(args))
