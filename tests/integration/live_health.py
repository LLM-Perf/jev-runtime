"""Pause only a recorded task-owned engine; check its independent gateway health."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import httpx
from live_crash_recovery import identity, process_group_members, ready

from jev_runtime.config import load_settings
from jev_runtime.schema import DecisionResponse


def group_states(pid):
    return {
        member: Path(f"/proc/{member}/stat").read_text().rsplit(")", 1)[1].split()[0]
        for member in process_group_members(pid)
    }


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def run(args):
    if args.output.exists():
        raise ValueError("Preserve previous evidence; choose a fresh output")
    settings = load_settings(args.gateway_dir / "config.json")
    engine = json.loads((args.engine_dir / "process.json").read_text())
    gateway = json.loads((args.gateway_dir / "process.json").read_text())
    assert engine["mode"] == "native-plugin" and gateway["mode"] == "gateway"
    assert settings.workers == 1 and not settings.adapters.enabled
    assert engine["engine"] == gateway["engine"] == settings.backend
    assert settings.engine_url == f"http://127.0.0.1:{engine['port']}"
    for record in (engine, gateway):
        assert identity(record["identity"]["pid"]) == record["identity"]
        assert os.getpgid(record["identity"]["pid"]) == record["identity"]["pid"]
    engine_keys = json.loads((args.engine_dir / "keys.json").read_text())
    keys = json.loads((args.gateway_dir / "keys.json").read_text())
    pid = engine["identity"]["pid"]
    url = f"http://127.0.0.1:{gateway['port']}"
    report = {
        "source_commit": args.source_commit,
        "qualification": (
            "Colocated single-worker gateway/native-engine pause and recovery; "
            "not failover, performance or soak certification"
        ),
        "engine": engine["engine"],
        "model": engine["model"],
        "engine_process": engine,
        "gateway_process": gateway,
        "health_settings": settings.health.model_dump(),
        "checks": {},
    }
    checks = report["checks"]
    paused = False
    hashes = {
        str(directory / name): hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for directory in (args.engine_dir, args.gateway_dir)
        for name in ("config.json", "keys.json")
    }
    payload = {
        "model": settings.bootstrap_alias,
        "input": {"text": "I was charged twice and would like a refund."},
        "questions": [{"id": "refund", "type": "boolean", "instruction": "Is a refund requested?"}],
    }
    try:
        checks["gateway_ready_before"] = await ready(url, keys["api"])
        checks["native_ready_before"] = await ready(
            settings.engine_url + "/plugins/jev-runtime", engine_keys["api"]
        )
        async with (
            httpx.AsyncClient(
                base_url=url, headers={"Authorization": "Bearer " + keys["api"]}, timeout=5
            ) as client,
            httpx.AsyncClient(
                base_url=url, headers={"Authorization": "Bearer " + keys["admin"]}, timeout=5
            ) as admin,
        ):

            async def get(client, path):
                response = await client.get(path)
                response.raise_for_status()
                return response.json()

            response = await client.post("/v1/decisions", json=payload)
            response.raise_for_status()
            baseline = DecisionResponse.model_validate(response.json())
            assert baseline.status == "completed"
            checks["baseline"] = baseline.model_dump(mode="json")
            routes_before = (await get(admin, "/admin/bundles"))["routes"]
            checks["routes_before"] = routes_before
            checks["profile_before"] = await get(admin, "/admin/profile")
            assert checks["profile_before"]["health"]["monitor_running"]
            members = process_group_members(pid)
            assert members and pid in members
            assert identity(pid) == engine["identity"]
            os.killpg(pid, signal.SIGSTOP)
            paused = True
            checks["pause"] = {"signal": "SIGSTOP", "at": time.time(), "group_members": members}
            async with asyncio.timeout(5):
                while True:
                    states = await asyncio.to_thread(group_states, pid)
                    if states and all(state in {"T", "t"} for state in states.values()):
                        break
                    await asyncio.sleep(0.02)
            checks["paused_group_states"] = states
            async with asyncio.timeout(
                settings.health.interval_seconds + settings.health.timeout_seconds + 10
            ):
                while True:
                    response = await client.get("/ready")
                    if response.status_code == 503:
                        assert response.json()["error"]["code"] == "engine_unavailable"
                        break
                    assert response.status_code == 200
                    await asyncio.sleep(0.05)
            checks["readiness_withdrawn_after_seconds"] = time.time() - checks["pause"]["at"]
            checks["profile_paused"] = await get(admin, "/admin/profile")
            assert all(
                not value["ready"]
                for value in checks["profile_paused"]["health"]["bundles"].values()
            )
            rejected = []
            for _ in range(10):
                start = time.perf_counter()
                response = await client.post("/v1/decisions", json=payload)
                assert response.status_code == 503
                assert response.json()["error"]["code"] == "engine_unavailable"
                rejected.append(
                    {
                        "status": response.status_code,
                        "elapsed_ms": (time.perf_counter() - start) * 1000,
                    }
                )
            checks["rejected_while_paused"] = rejected
            listing = await get(admin, "/admin/bundles")
            assert not listing["leases"] and listing["routes"] == routes_before
            checks["no_dispatched_leases_or_route_changes_while_paused"] = True
            assert identity(pid) == engine["identity"]
            os.killpg(pid, signal.SIGCONT)
            paused = False
            checks["resume"] = {"signal": "SIGCONT", "at": time.time()}
            checks["ready_after"] = await ready(url, keys["api"])
            assert identity(pid) == engine["identity"]
            assert identity(gateway["identity"]["pid"]) == gateway["identity"]
            checks["same_processes_recovered"] = True
            checks["recovered_outputs"] = []
            for _ in range(2):
                response = await client.post("/v1/decisions", json=payload)
                response.raise_for_status()
                decision = DecisionResponse.model_validate(response.json())
                assert decision.status == "completed"
                assert decision.bundle_digest == baseline.bundle_digest
                assert decision.generation == baseline.generation
                error = max(
                    abs(value - decision.answers["refund"].probabilities[key])
                    for key, value in baseline.answers["refund"].probabilities.items()
                )
                assert error <= 1e-4
                checks["recovered_outputs"].append(
                    {"response": decision.model_dump(mode="json"), "max_probability_error": error}
                )
            async with asyncio.timeout(5):
                while True:
                    listing = await get(admin, "/admin/bundles")
                    if not listing["leases"]:
                        break
                    await asyncio.sleep(0.05)
            assert listing["routes"] == routes_before
            profile = await get(admin, "/admin/profile")
            assert all(
                profile["admission"][key] == 0
                for key in ("requests", "expanded_tokens", "expanded_branches", "queued_requests")
            )
            checks["final_profile"] = profile
            checks["zero_final_leases_and_admission"] = True
            assert all(file_hash(path) == value for path, value in hashes.items())
            checks["config_and_credentials_unchanged"] = True
            async with httpx.AsyncClient(base_url=settings.engine_url, timeout=30) as native:
                response = await native.post(
                    "/v1/chat/completions",
                    json={
                        "model": settings.model_id,
                        "messages": [{"role": "user", "content": "Say hello"}],
                        "max_tokens": 8,
                        "temperature": 0,
                    },
                )
                response.raise_for_status()
                assert response.json()["choices"][0]["message"]["content"]
                checks["native_chat_after_recovery"] = response.json()
            report["passed"] = True
    except BaseException as exc:
        report["passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if paused:
            assert identity(pid) == engine["identity"], "Cannot safely resume a changed identity"
            os.killpg(pid, signal.SIGCONT)
            checks["finally_resumed_owned_engine"] = True
        report["finished_at"] = time.time()
        with args.output.open("x") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
        print(json.dumps({"output": str(args.output), "passed": report.get("passed", False)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--gateway-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
