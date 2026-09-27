"""Kill only a recorded, task-owned gateway; validate its durable dispatch recovery.

The engine stays alive. Run on Linux after dsw_service.py created both run dirs.
Reports prove journal/owner recovery, not a GPU LoRA unload barrier.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import httpx

from jev_runtime.config import load_compiler, load_settings, model_identity
from jev_runtime.registry import Registry
from jev_runtime.schema import Bundle, Policy

LAUNCHER = Path(__file__).resolve().parents[2] / "deployment" / "dsw_service.py"


def identity(pid: int) -> dict | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except FileNotFoundError:
        return None


async def ready(url: str, key: str) -> dict:
    async with httpx.AsyncClient(
        base_url=url, headers={"Authorization": "Bearer " + key}, timeout=2
    ) as client:
        async with asyncio.timeout(120):
            while True:
                try:
                    response = await client.get("/ready")
                    if response.status_code == 200:
                        return response.json()
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.25)


async def run(args):
    gateway_dir = args.gateway_run_dir.resolve()
    engine_dir = args.engine_run_dir.resolve()
    settings = load_settings(gateway_dir / "config.json")
    gateway_record = json.loads((gateway_dir / "process.json").read_text())
    engine_record = json.loads((engine_dir / "process.json").read_text())
    assert gateway_record["mode"] == "gateway"
    assert engine_record["mode"] == "native-plugin"
    assert identity(engine_record["identity"]["pid"]) == engine_record["identity"]
    assert identity(gateway_record["identity"]["pid"]) == gateway_record["identity"]
    assert settings.engine_url == f"http://127.0.0.1:{engine_record['port']}"
    # The launcher creates a fresh process group. Never kill arbitrary children
    # or infer process ownership from a port or a substring in ps output.
    assert os.getpgid(gateway_record["identity"]["pid"]) == gateway_record["identity"]["pid"]
    keys = json.loads((gateway_dir / "keys.json").read_text())
    url = f"http://127.0.0.1:{settings.port}"
    report = {
        "source_commit": args.source_commit,
        "qualification": "co-located gateway process-kill and journal recovery",
        "started_at": time.time(),
        "engine": engine_record["engine"],
        "engine_identity": engine_record["identity"],
        "gateway_before": gateway_record["identity"],
        "checks": {},
    }
    checks = report["checks"]
    request_task = None
    try:
        checks["ready_before"] = await ready(url, keys["api"])
        compiler = await asyncio.to_thread(load_compiler, settings)
        bundle = Bundle(
            id="crash-" + uuid.uuid4().hex,
            version=1,
            model=model_identity(settings, compiler),
            policy=Policy(max_questions=128, max_parallel_branches=4),
        )
        rid = "crash-request-" + uuid.uuid4().hex
        body = {
            "model": bundle.id,
            "request_id": rid,
            "input": {"text": "The buyer reports a duplicated charge and asks for a refund. " * 80},
            "questions": [
                {
                    "id": f"q{i}",
                    "type": "boolean",
                    "instruction": f"Case {i}: Does the buyer request a payment correction?",
                }
                for i in range(128)
            ],
            "execution": {"timeout_ms": 120000},
        }
        async with (
            httpx.AsyncClient(
                base_url=url, headers={"Authorization": "Bearer " + keys["admin"]}, timeout=120
            ) as admin,
            httpx.AsyncClient(
                base_url=url, headers={"Authorization": "Bearer " + keys["api"]}, timeout=150
            ) as data,
        ):
            for path, payload in (
                ("/admin/bundles", bundle.model_dump(mode="json")),
                ("/admin/bundles/prepare", {"reference": bundle.reference}),
                (
                    "/admin/bundles/activate",
                    {"alias": bundle.id, "reference": bundle.reference, "expected_generation": 0},
                ),
            ):
                response = await admin.post(path, json=payload)
                response.raise_for_status()
            request_task = asyncio.create_task(data.post("/v1/decisions", json=body))
            observed = None
            async with asyncio.timeout(30):
                while observed is None:
                    response = await admin.get("/admin/requests/recovery")
                    response.raise_for_status()
                    observed = next(
                        (
                            item
                            for item in response.json()["requests"]
                            if item["request_id"] == rid and item["engine_request_ids"]
                        ),
                        None,
                    )
                    if request_task.done() and observed is None:
                        result = await request_task
                        raise AssertionError(f"Request completed before kill: {result.status_code}")
                    if observed is None:
                        await asyncio.sleep(0.01)
            checks["observed_before_kill"] = observed
            assert observed["owner_status"] == "alive" and not observed["recoverable"]
            assert len(observed["engine_request_ids"]) == 128
            # Final identity comparison immediately before the fault injection.
            original = gateway_record["identity"]
            assert identity(original["pid"]) == original
            os.killpg(original["pid"], signal.SIGKILL)
            checks["signal"] = "SIGKILL"
            for _ in range(300):
                if identity(original["pid"]) is None:
                    break
                await asyncio.sleep(0.05)
            else:
                raise TimeoutError("Killed gateway did not exit within 15 seconds")
            result = (await asyncio.gather(request_task, return_exceptions=True))[0]
            assert isinstance(result, httpx.TransportError), type(result).__name__
            checks["client_disconnect_error"] = type(result).__name__

        registry = Registry(settings.registry_path)
        orphan = next(item for item in registry.recovery_candidates() if item["request_id"] == rid)
        assert orphan["owner_status"] == "dead" and orphan["recoverable"]
        assert orphan["engine_request_ids"] == observed["engine_request_ids"]
        checks["durable_orphan_after_kill"] = orphan
        command = [
            sys.executable,
            str(LAUNCHER),
            "launch",
            "--run-dir",
            str(gateway_dir),
            "--engine",
            settings.backend,
            "--model-path",
            str(settings.tokenizer),
            "--port",
            str(settings.port),
            "--gateway",
            "--engine-url",
            settings.engine_url,
            "--engine-run-dir",
            str(engine_dir),
        ]
        child = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await child.communicate()
        assert child.returncode == 0, stderr.decode()[-2000:]
        report["gateway_restart"] = json.loads(stdout)
        checks["ready_after_restart"] = await ready(url, keys["api"])
        assert bundle.reference in checks["ready_after_restart"]["prepared_bundles"]
        async with httpx.AsyncClient(base_url=url, timeout=120) as client:
            response = await client.post(
                f"/admin/requests/{rid}/recover",
                headers={"Authorization": "Bearer " + keys["admin"]},
            )
            response.raise_for_status()
            assert response.json() == {"recovered": True}
            assert not registry.list()["leases"]
            checks["recovered_no_leases"] = True
            response = await client.post(
                "/v1/decisions",
                headers={"Authorization": "Bearer " + keys["api"]},
                json={
                    "model": bundle.id,
                    "input": {"text": "Please refund me."},
                    "questions": [body["questions"][0]],
                },
            )
            response.raise_for_status()
            assert response.json()["status"] == "completed"
            checks["decision_after_recovery"] = response.json()
            # Remove only this harness's temporary alias after checking recovery.
            response = await client.post(
                "/admin/bundles/disable",
                headers={"Authorization": "Bearer " + keys["admin"]},
                json={"alias": bundle.id, "expected_generation": 1},
            )
            response.raise_for_status()
            response = await client.post(
                "/admin/bundles/retire",
                headers={"Authorization": "Bearer " + keys["admin"]},
                json={"reference": bundle.reference},
            )
            response.raise_for_status()
        async with httpx.AsyncClient(base_url=settings.engine_url, timeout=120) as native:
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
        assert identity(engine_record["identity"]["pid"]) == engine_record["identity"]
        checks["original_engine_survives_native_chat"] = True
        report["passed"] = True
    except BaseException as exc:
        report["passed"] = False
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        raise
    finally:
        if request_task is not None and not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
        report["finished_at"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {"passed": report.get("passed"), "checks": list(checks), "output": str(args.output)}
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-run-dir", type=Path, required=True)
    parser.add_argument("--engine-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    asyncio.run(run(parser.parse_args()))
