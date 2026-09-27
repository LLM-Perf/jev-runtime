"""Real native-engine LoRA lifecycle/cache test. Does not start/stop services."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path

import httpx

from jev_runtime.schema import Bundle, DecisionResponse, Policy


async def run(args):
    if args.output.exists():
        raise ValueError("Preserve previous attempts; choose a new evidence path")
    root = args.run_dir
    process = json.loads((root / "process.json").read_text())
    config = json.loads((root / "config.json").read_text())
    keys = json.loads((root / "keys.json").read_text())
    base = f"http://127.0.0.1:{process['port']}/plugins/jev-runtime"
    report = {
        "source_commit": args.source_commit,
        "runtime_commit": args.runtime_commit or args.source_commit,
        "started_at": time.time(),
        "qualification": "colocated BF16 TP1 single-frontend untrained-LoRA functional evidence",
        "process": process,
        "fixture": json.loads((args.fixtures / "manifest.json").read_text()),
        "probability_tolerance": 1e-4,
        "checks": {},
        "attempts": [],
    }
    checks = report["checks"]
    alias = "lora-" + uuid.uuid4().hex[:12]
    references, bundles, generation = [], [], 0
    work, rid = None, None
    admin = {"Authorization": "Bearer " + keys["admin"]}
    async with httpx.AsyncClient(
        base_url=base + "/", timeout=180, headers={"Authorization": "Bearer " + keys["api"]}
    ) as client:

        async def call(path, body=None, *, management=False, expected=200):
            headers = admin if management else {}
            response = await (
                client.get(path, headers=headers)
                if body is None
                else client.post(path, json=body, headers=headers)
            )
            report["attempts"].append(
                {"path": path, "status": response.status_code, "expected": expected}
            )
            assert response.status_code == expected, (
                path,
                response.status_code,
                response.text[:2000],
            )
            return response.json()

        def pending(request_id):
            with sqlite3.connect(f"file:{config['registry_path']}?mode=ro", uri=True) as db:
                row = db.execute(
                    "SELECT l.id,w.branches FROM leases l JOIN lease_work w "
                    "ON w.lease_id=l.id WHERE l.request_id=?",
                    (request_id,),
                ).fetchone()
                return {"lease_id": row[0], "branches": len(json.loads(row[1]))} if row else None

        async def activate(index):
            nonlocal generation
            result = await call(
                "admin/bundles/activate",
                {
                    "reference": bundles[index].reference,
                    "alias": alias,
                    "expected_generation": generation,
                },
                management=True,
            )
            generation = result["generation"]

        body = {
            "model": alias,
            "input": {"text": "I was charged twice and would like a refund."},
            "questions": [
                {"id": "q0", "type": "boolean", "instruction": "Is a refund explicitly requested?"}
            ],
        }

        async def decide():
            response = DecisionResponse.model_validate(await call("v1/decisions", body))
            assert response.status == "completed" and response.usage.successful_questions == 1
            return response

        def probabilities(response):
            return response.answers["q0"].probabilities

        def difference(left, right):
            return max(abs(left[key] - right[key]) for key in left)

        async def inflight():
            nonlocal work, rid
            rid = "lora-inflight-" + uuid.uuid4().hex
            payload = {
                **body,
                "request_id": rid,
                "questions": [{**body["questions"][0], "id": f"q{i}"} for i in range(128)],
                "execution": {"timeout_ms": 120000},
            }
            work = asyncio.create_task(client.post("v1/decisions", json=payload))
            async with asyncio.timeout(10):
                while True:
                    row = pending(rid)
                    if row and row["branches"] == 128:
                        assert not work.done()
                        return row
                    if work.done():
                        raise AssertionError(
                            f"Request ended before observation: {(await work).text[:1000]}"
                        )
                    await asyncio.sleep(0.005)

        try:
            checks["ready"] = await call("ready")
            profile = await call("admin/profile", management=True)
            assert profile["capabilities"]["lora"]
            model = profile["model"]
            identities = [model]
            for name in ("positive", "negative"):
                registered = await call(
                    "admin/adapters/register",
                    {"id": alias + "-" + name, "source": str(args.fixtures / name)},
                    management=True,
                )
                references.append(registered["reference"])
                loaded = await call(
                    "admin/adapters/load", {"reference": references[-1]}, management=True
                )
                assert loaded["state"] == "READY"
                artifact = loaded["binding"]["artifact"]
                identities.append(
                    {
                        **model,
                        "adapter_id": artifact["id"],
                        "adapter_revision": artifact["revision"],
                    }
                )
            checks["immutable_bindings"] = await call("admin/adapters", management=True)
            for version, identity in enumerate(identities, 1):
                bundle = Bundle(
                    id=alias,
                    version=version,
                    model=identity,
                    policy=Policy(
                        max_questions=128, max_scoring_sequences=128, max_parallel_branches=1
                    ),
                )
                await call("admin/bundles", bundle.model_dump(mode="json"), management=True)
                bundles.append(bundle)
                await call(
                    "admin/bundles/prepare", {"reference": bundle.reference}, management=True
                )
            expected = []
            for index in range(3):
                await activate(index)
                await decide()
                response = await decide()
                expected.append(probabilities(response))
            assert all(
                difference(expected[i], expected[j]) > 1e-3 for i, j in ((0, 1), (0, 2), (1, 2))
            ), expected
            checks["distinct_nonzero_adapter_outputs"] = expected
            errors, cache_observations = [], []
            for iteration in range(24):
                index = iteration % 3
                await activate(index)
                response = await decide()
                assert response.bundle == bundles[index].reference
                errors.append(difference(probabilities(response), expected[index]))
                cache_observations.append(
                    {
                        "bundle": response.bundle,
                        "cached_prompt_tokens": response.usage.cached_prompt_tokens,
                        "logical_prompt_tokens": response.usage.logical_prompt_tokens,
                    }
                )
            assert max(errors) <= report["probability_tolerance"], errors
            assert all(
                row["cached_prompt_tokens"] is not None and row["cached_prompt_tokens"] > 0
                for row in cache_observations
            ), cache_observations
            checks["alternating_cache_isolation"] = {
                "switches": 24,
                "max_probability_error": max(errors),
                "cache_observations": cache_observations,
            }

            await activate(1)
            checks["old_request_journal"] = await inflight()
            await activate(2)
            refused = await call(
                "admin/adapters/unload", {"reference": references[0]}, management=True, expected=409
            )
            assert refused["error"]["code"] == "adapter_in_use"
            old_http = await work
            old_http.raise_for_status()
            old = DecisionResponse.model_validate(old_http.json())
            assert (
                old.bundle == bundles[1].reference
                and len(old.answers) == 128
                and old.status == "completed"
            )
            assert all(
                difference(answer.probabilities, expected[1]) <= report["probability_tolerance"]
                for answer in old.answers.values()
            )
            assert pending(rid) is None
            checks["inflight_switch_and_drain"] = {
                "old_bundle": old.bundle,
                "questions": len(old.answers),
                "new_bundle": (await decide()).bundle,
                "premature_unload_status": 409,
            }
            assert (
                await call("admin/adapters/unload", {"reference": references[0]}, management=True)
            )["state"] == "UNLOADED"

            checks["cancel_journal"] = await inflight()
            await activate(0)
            cancelled = await call(f"v1/requests/{rid}/cancel", {})
            assert cancelled["cancelled"]
            assert (await work).status_code == 499 and pending(rid) is None
            assert (
                await call("admin/adapters/unload", {"reference": references[1]}, management=True)
            )["state"] == "UNLOADED"
            checks["cancel_then_gpu_unload"] = True
            reload_errors = []
            for _ in range(3):
                for index, reference in enumerate(references, 1):
                    await call("admin/adapters/load", {"reference": reference}, management=True)
                    await call(
                        "admin/bundles/prepare",
                        {"reference": bundles[index].reference},
                        management=True,
                    )
                    await activate(index)
                    await decide()
                    reload_errors.append(difference(probabilities(await decide()), expected[index]))
                    await activate(0)
                    await call("admin/adapters/unload", {"reference": reference}, management=True)
            assert max(reload_errors) <= report["probability_tolerance"], reload_errors
            checks["reload_and_rollback"] = {
                "cycles": 6,
                "max_probability_error": max(reload_errors),
            }
            preview = await call("admin/compile", body, management=True)
            raw = {**preview["sequences"][0], "adapter_id": references[0]}
            refused = await call("v1/scores", raw, expected=409)
            assert refused["error"]["code"] == "lora_unsupported"
            checks["raw_adapter_bypass_refused"] = True
            report["passed"] = True
        except BaseException as exc:
            report["passed"] = False
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
            raise
        finally:
            try:
                if work is not None and not work.done():
                    await call(f"v1/requests/{rid}/cancel", {})
                    await work
                if generation:
                    await call(
                        "admin/bundles/disable",
                        {"alias": alias, "expected_generation": generation},
                        management=True,
                    )
                for reference in references:
                    await call("admin/adapters/unload", {"reference": reference}, management=True)
                checks["cleanup_completed"] = True
            except Exception as exc:
                report["cleanup_error"] = str(exc)[:1000]
                report["passed"] = False
            report["finished_at"] = time.time()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"passed": report.get("passed"), "checks": list(checks)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-commit")
    arguments = parser.parse_args()
    asyncio.run(run(arguments))
