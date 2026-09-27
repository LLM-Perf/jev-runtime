"""Fault an isolated native LoRA process group and verify explicit reconciliation.

Only accepts a single-worker dsw_service.py run. Kills the recorded API/GPU group,
checks exit and restored GPU memory, then restarts its exact engine configuration.
A successful run leaves the restarted service alive for independent inspection.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import httpx
from live_crash_recovery import LAUNCHER, gpu_snapshot, identity, process_group_members, ready

from jev_runtime.config import load_settings
from jev_runtime.registry import Registry
from jev_runtime.schema import Bundle, DecisionResponse, Policy


async def run(args):
    if args.output.exists():
        raise ValueError("Preserve previous attempts; choose a new evidence path")
    root = args.run_dir.resolve()
    settings = load_settings(root / "config.json")
    record = json.loads((root / "process.json").read_text())
    assert record["mode"] == "native-plugin" and settings.workers == 1
    assert record.get("tensor_parallel_size", 1) == 1, "Managed LoRA crash harness covers TP1 only"
    assert record.get("readout_dtype", "bfloat16") == "bfloat16", (
        "Managed LoRA crash harness covers the frozen BF16 readout profile only"
    )
    assert settings.adapters.enabled
    assert settings.adapters.allowed_roots == (str(args.fixtures.resolve()),)
    assert settings.admission.max_requests >= 2
    assert settings.admission.max_tenant_requests >= 2
    assert settings.admission.max_branches >= 129
    assert settings.admission.max_tenant_branches >= 129
    assert settings.engine_url == f"http://127.0.0.1:{record['port']}"
    original = record["identity"]
    assert identity(original["pid"]) == original and os.getpgid(original["pid"]) == original["pid"]
    config_before, keys_before = (
        (root / "config.json").read_bytes(),
        (root / "keys.json").read_bytes(),
    )
    keys = json.loads(keys_before)
    fixture = json.loads((args.fixtures / "manifest.json").read_text())
    for name, files in fixture["artifacts"].items():
        for filename, digest in files.items():
            assert (
                "sha256:"
                + hashlib.sha256((args.fixtures / name / filename).read_bytes()).hexdigest()
                == digest
            )
    assert fixture["base_model"]["revision"] == settings.model_revision
    assert fixture["base_model"]["model_id"] == settings.model_id
    url = settings.engine_url + "/plugins/jev-runtime"
    report = {
        "source_commit": args.source_commit,
        "runtime_source_commit": args.runtime_source_commit or args.source_commit,
        "qualification": "Colocated SmolLM2 BF16 TP1 API1 native process-group SIGKILL, "
        "LoRA quarantine/reconciliation; not hardware, worker-only or load/quality certification",
        "started_at": time.time(),
        "engine": record["engine"],
        "process_record_before": record,
        "fixture": fixture,
        "probability_tolerance": 1e-4,
        "checks": {},
        "attempts": [],
    }
    checks = report["checks"]
    namespace = "lora-crash-" + uuid.uuid4().hex[:12]
    aliases = [namespace + "-base", namespace + "-active", namespace + "-idle"]
    references, bindings, bundles = [], [], []
    work = None
    registry = None

    def tickets():
        with sqlite3.connect(f"file:{settings.registry_path}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in db.execute(
                    "SELECT l.request_id,l.owner,l.ref,t.lease_id,t.tenant,t.tokens,t.branches,"
                    "t.state FROM admission_tickets t JOIN leases l ON l.id=t.lease_id "
                    "ORDER BY t.id"
                )
            ]

    def probabilities(response):
        return response.answers["q0"].probabilities

    def difference(left, right):
        assert set(left) == set(right)
        return max(abs(left[key] - right[key]) for key in left)

    body = {
        "model": aliases[0],
        "input": {"text": "I was charged twice and would like a refund."},
        "questions": [
            {"id": "q0", "type": "boolean", "instruction": "Is a refund explicitly requested?"}
        ],
    }
    async with httpx.AsyncClient(
        base_url=url + "/", timeout=150, headers={"Authorization": "Bearer " + keys["api"]}
    ) as client:

        async def call(path, payload=None, *, admin=True, expected=200, error=None):
            headers = {"Authorization": "Bearer " + keys["admin"]} if admin else {}
            response = await (
                client.get(path, headers=headers)
                if payload is None
                else client.post(path, json=payload, headers=headers)
            )
            report["attempts"].append(
                {"path": path, "status": response.status_code, "expected": expected}
            )
            assert response.status_code == expected, (
                path,
                response.status_code,
                response.text[:1500],
            )
            value = response.json()
            if error is not None:
                assert value["error"]["code"] == error, value
            return value

        async def decide(alias):
            result = DecisionResponse.model_validate(
                await call("v1/decisions", {**body, "model": alias}, admin=False)
            )
            assert result.status == "completed" and result.usage.successful_questions == 1
            assert set(result.answers) == {"q0"} and result.answers["q0"].type == "boolean"
            return result

        async def activate(alias, index, generation):
            return await call(
                "admin/bundles/activate",
                {
                    "alias": alias,
                    "reference": bundles[index].reference,
                    "expected_generation": generation,
                },
            )

        try:
            checks["ready_before"] = await ready(url, keys["api"])
            profile = await call("admin/profile")
            assert profile["capabilities"]["lora"] and tickets() == []
            checks["profile_before"] = profile
            models = [profile["model"]]
            for name in ("positive", "negative"):
                registered = await call(
                    "admin/adapters/register",
                    {"id": namespace + "-" + name, "source": str(args.fixtures / name)},
                )
                references.append(registered["reference"])
                loaded = await call("admin/adapters/load", {"reference": references[-1]})
                assert loaded["state"] == "READY"
                bindings.append(loaded["binding"])
                artifact = loaded["binding"]["artifact"]
                assert artifact["files"] == fixture["artifacts"][name]
                models.append(
                    {
                        **models[0],
                        "adapter_id": artifact["id"],
                        "adapter_revision": artifact["revision"],
                    }
                )
            for index, model in enumerate(models):
                bundle = Bundle(
                    id=namespace,
                    version=index + 1,
                    model=model,
                    policy=Policy(
                        max_questions=128, max_scoring_sequences=128, max_parallel_branches=1
                    ),
                )
                bundles.append(bundle)
                await call("admin/bundles", bundle.model_dump(mode="json"))
                await call("admin/bundles/prepare", {"reference": bundle.reference})
                await activate(aliases[index], index, 0)
            baseline, baseline_responses = [], []
            for alias in aliases:
                first, warm = await decide(alias), await decide(alias)
                baseline.append(probabilities(warm))
                baseline_responses.append(
                    {"first": first.model_dump(mode="json"), "warm": warm.model_dump(mode="json")}
                )
            assert all(
                difference(baseline[i], baseline[j]) > 1e-3 for i, j in ((0, 1), (0, 2), (1, 2))
            )
            checks["distinct_nonzero_baselines"] = baseline_responses
            checks["bindings_before"] = bindings
            checks["routes_before"] = (await call("admin/bundles"))["routes"]
            rid = namespace + "-unfinished"
            long_body = {
                **body,
                "model": aliases[1],
                "request_id": rid,
                "input": {
                    "text": "The buyer reports a duplicated charge and asks for a refund. " * 80
                },
                "questions": [{**body["questions"][0], "id": f"q{i}"} for i in range(128)],
                "execution": {"timeout_ms": 120000},
            }
            work = asyncio.create_task(client.post("v1/decisions", json=long_body))
            async with asyncio.timeout(15):
                while True:
                    pending = (await call("admin/requests/recovery"))["requests"]
                    orphan = next((row for row in pending if row["request_id"] == rid), None)
                    rows = tickets()
                    if (
                        orphan
                        and len(orphan["engine_request_ids"]) == 128
                        and len(rows) == 1
                        and rows[0]["state"] == "ADMITTED"
                    ):
                        break
                    assert not work.done(), "Request finished before fault observation"
                    await asyncio.sleep(0.005)
            assert orphan["owner_status"] == "alive" and not orphan["recoverable"]
            assert rows[0]["branches"] == 128 and rows[0]["ref"] == bundles[1].reference
            checks["inflight_before_kill"] = orphan
            checks["admission_before_kill"] = rows
            await call(
                f"admin/requests/{rid}/recover", {}, expected=409, error="recovery_not_confirmed"
            )
            assert not work.done() and tickets() == rows
            assert (
                identity(original["pid"]) == original
                and os.getpgid(original["pid"]) == original["pid"]
            )
            checks["group_before_kill"] = process_group_members(original["pid"])
            os.killpg(original["pid"], signal.SIGKILL)
            checks["signal"] = "SIGKILL"
            async with asyncio.timeout(45):
                while process_group_members(original["pid"]):  # noqa: ASYNC110 - external OS processes
                    await asyncio.sleep(0.1)
            checks["old_group_exited"] = True
            assert identity(original["pid"]) is None
            async with asyncio.timeout(30):
                while True:
                    gpu = await asyncio.to_thread(gpu_snapshot, record["gpu_before"]["uuid"])
                    if int(gpu["free_mib"]) >= int(record["gpu_before"]["free_mib"]):
                        break
                    await asyncio.sleep(0.25)
            checks["gpu_memory_after_kill"] = gpu
            result = (await asyncio.gather(work, return_exceptions=True))[0]
            assert isinstance(result, httpx.TransportError), type(result).__name__
            checks["client_disconnect_error"] = type(result).__name__
            assert tickets() == rows
            registry = Registry(settings.registry_path)
            orphan_after = next(r for r in registry.recovery_candidates() if r["request_id"] == rid)
            assert orphan_after["owner_status"] == "dead" and orphan_after["recoverable"]
            assert orphan_after["engine_request_ids"] == orphan["engine_request_ids"]
            checks["durable_dead_owner"] = orphan_after
            assert all(registry.inspect_adapter(ref)["state"] == "READY" for ref in references)
            admission_path = root / "restart-admission.json"
            admission_path.write_text(json.dumps(settings.admission.model_dump(), indent=2) + "\n")
            memory_flag = (
                "--gpu-memory-utilization"
                if settings.backend == "vllm"
                else "--mem-fraction-static"
            )
            command = [
                sys.executable,
                str(LAUNCHER),
                "launch",
                "--run-dir",
                str(root),
                "--engine",
                settings.backend,
                "--model-path",
                str(settings.tokenizer),
                "--port",
                str(settings.port),
                "--gpu",
                gpu["index"],
                "--memory-fraction",
                record["command"][record["command"].index(memory_flag) + 1],
                "--reserve-mib",
                str(args.reserve_mib),
                "--api-workers",
                "1",
                "--adapters-root",
                str(args.fixtures.resolve()),
                "--admission-config",
                str(admission_path),
            ]
            for tenant in settings.tenant_key_envs:
                command.extend(["--tenant", tenant])
            child = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await child.communicate()
            assert child.returncode == 0, stderr.decode()[-2000:]
            checks["restart"] = json.loads(stdout)
            new_record = json.loads((root / "process.json").read_text())
            assert (
                new_record["identity"] != original
                and identity(new_record["identity"]["pid"]) == new_record["identity"]
            )
            assert new_record["command"] == record["command"]
            assert new_record["gpu_before"]["uuid"] == record["gpu_before"]["uuid"]
            checks["ready_after_restart"] = await ready(url, keys["api"])
            assert load_settings(root / "config.json") == settings
            assert (root / "config.json").read_bytes() == config_before
            assert (root / "keys.json").read_bytes() == keys_before
            checks["same_engine_config_gpu_credentials"] = True
            assert tickets() == rows
            checks["admission_after_restart"] = tickets()
            adapters = {a["reference"]: a for a in await call("admin/adapters")}
            listing = await call("admin/bundles")
            routes = {r["alias"]: r for r in listing["routes"]}
            bundle_states = {r["ref"]: r["state"] for r in listing["bundles"]}
            for index, ref in enumerate(references, 1):
                assert (
                    adapters[ref]["state"] == "UNKNOWN"
                    and adapters[ref]["error"] == "engine_session_changed"
                )
                assert adapters[ref]["binding"] == bindings[index - 1]
                assert (
                    routes[aliases[index]]["ref"] is None
                    and routes[aliases[index]]["generation"] == 2
                )
                assert bundle_states[bundles[index].reference] == "VALIDATED"
                assert (
                    bundles[index].reference
                    not in checks["ready_after_restart"]["prepared_bundles"]
                )
                await call(
                    "v1/decisions",
                    {**body, "model": aliases[index]},
                    admin=False,
                    expected=503,
                    error="route_unavailable",
                )
                await call(
                    "admin/adapters/load", {"reference": ref}, expected=409, error="adapter_state"
                )
                await call(
                    "admin/bundles/prepare",
                    {"reference": bundles[index].reference},
                    expected=503,
                    error="adapter_not_ready",
                )
                await call(
                    "admin/bundles/activate",
                    {
                        "alias": aliases[index],
                        "reference": bundles[index].reference,
                        "expected_generation": 2,
                    },
                    expected=503,
                    error="replica_not_ready",
                )
            assert (
                routes[aliases[0]]["ref"] == bundles[0].reference
                and routes[aliases[0]]["generation"] == 1
            )
            checks["quarantined_adapters"] = [adapters[ref] for ref in references]
            checks["paused_routes"] = [routes[alias] for alias in aliases]
            checks["base_serving_with_retained_adapter_lease"] = (
                await decide(aliases[0])
            ).model_dump(mode="json")
            assert tickets() == rows
            await call(
                "admin/adapters/unload",
                {"reference": references[0], "recover": True},
                expected=409,
                error="adapter_in_use",
            )
            checks["recover_flag_cannot_bypass_lease"] = True
            # Reconcile the idle adapter without settling the busy adapter's old lease.
            idle = await call("admin/adapters/unload", {"reference": references[1]})
            assert idle["state"] == "UNLOADED" and tickets() == rows
            checks["idle_reconciliation_preserves_active_lease"] = True
            recovered = await call(f"admin/requests/{rid}/recover", {})
            assert recovered == {"recovered": True} and tickets() == []
            assert not registry.list()["leases"]
            checks["old_request_recovered_no_quota"] = True
            assert (await call("admin/adapters/unload", {"reference": references[0]}))[
                "state"
            ] == "UNLOADED"
            after_responses = []
            for index, ref in enumerate(references, 1):
                loaded = await call("admin/adapters/load", {"reference": ref})
                assert loaded["state"] == "READY" and loaded["binding"] == bindings[index - 1]
                await call(
                    "admin/bundles/activate",
                    {
                        "alias": aliases[index],
                        "reference": bundles[index].reference,
                        "expected_generation": 2,
                    },
                    expected=503,
                    error="replica_not_ready",
                )
                await call("admin/bundles/prepare", {"reference": bundles[index].reference})
                await call(
                    "admin/bundles/activate",
                    {
                        "alias": aliases[index],
                        "reference": bundles[index].reference,
                        "expected_generation": 1,
                    },
                    expected=409,
                    error="generation_conflict",
                )
                activation = await activate(aliases[index], index, 2)
                assert activation["generation"] == 3
                first, warm = await decide(aliases[index]), await decide(aliases[index])
                assert first.bundle == warm.bundle == bundles[index].reference
                first_error = difference(
                    probabilities(first),
                    probabilities(
                        DecisionResponse.model_validate(baseline_responses[index]["first"])
                    ),
                )
                warm_error = difference(probabilities(warm), baseline[index])
                after_responses.append(
                    {
                        "first": first.model_dump(mode="json"),
                        "warm": warm.model_dump(mode="json"),
                        "first_probability_error": first_error,
                        "warm_probability_error": warm_error,
                    }
                )
                checks["reloaded_outputs"] = after_responses
                assert max(first_error, warm_error) <= report["probability_tolerance"]
            checks["new_canaries_and_generation_required"] = True
            generation = 3
            switches = []
            for iteration in range(24):
                index = iteration % 3
                activation = await activate(aliases[1], index, generation)
                generation = activation["generation"]
                response = await decide(aliases[1])
                error = difference(probabilities(response), baseline[index])
                switches.append(
                    {
                        "bundle": response.bundle,
                        "generation": response.generation,
                        "probability_error": error,
                        "cached_prompt_tokens": response.usage.cached_prompt_tokens,
                        "logical_prompt_tokens": response.usage.logical_prompt_tokens,
                    }
                )
                checks["post_restart_switches"] = switches
                assert (
                    response.bundle == bundles[index].reference
                    and response.generation == generation
                )
                assert error <= report["probability_tolerance"]
                assert (
                    response.usage.cached_prompt_tokens is not None
                    and response.usage.cached_prompt_tokens > 0
                )
            for alias, expected in zip(aliases, [1, generation, 3], strict=True):
                await call(
                    "admin/bundles/disable", {"alias": alias, "expected_generation": expected}
                )
            for ref in references:
                assert (await call("admin/adapters/unload", {"reference": ref}))[
                    "state"
                ] == "UNLOADED"
            for bundle in bundles:
                await call("admin/bundles/retire", {"reference": bundle.reference})
            assert tickets() == [] and not registry.list()["leases"]
            checks["all_test_routes_disabled_adapters_unloaded_bundles_retired"] = True
            async with httpx.AsyncClient(base_url=settings.engine_url, timeout=30) as native:
                response = await native.post(
                    "/v1/chat/completions",
                    json={
                        "model": settings.model_id,
                        "messages": [{"role": "user", "content": "Say hello."}],
                        "max_tokens": 8,
                        "temperature": 0,
                    },
                )
                response.raise_for_status()
                assert response.json()["usage"]["completion_tokens"] > 0
            checks["native_chat_after_cleanup"] = True
            assert identity(new_record["identity"]["pid"]) == new_record["identity"]
            report["passed"] = True
        except BaseException as exc:
            report["passed"] = False
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
            raise
        finally:
            if work is not None and not work.done():
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
            if registry is not None:
                registry.close()
            report["finished_at"] = time.time()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
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
    parser = argparse.ArgumentParser(description="Managed native TP1 LoRA crash/restart checks")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-source-commit")
    parser.add_argument("--reserve-mib", type=int, default=3072)
    args = parser.parse_args()
    if args.reserve_mib < 0:
        parser.error("reserve-mib must be non-negative")
    asyncio.run(run(args))
