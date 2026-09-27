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
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import httpx

from jev_runtime.config import load_settings
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
    if args.output.exists():
        raise ValueError("Keep previous evidence; choose a new output")
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
    config_before = (gateway_dir / "config.json").read_bytes()
    keys_before = (gateway_dir / "keys.json").read_bytes()
    if args.verify_admission:
        assert settings.workers == 1, "Fault injection targets one gateway API process"
        assert settings.admission.max_requests == 1
        assert settings.admission.max_queue >= 2 and settings.admission.max_tenant_queue >= 2
    tenant_key = keys["tenants"][args.tenant] if args.tenant else keys["api"]
    url = f"http://127.0.0.1:{settings.port}"
    report = {
        "source_commit": args.source_commit,
        "runtime_source_commit": args.runtime_source_commit or args.source_commit,
        "qualification": "co-located gateway process-kill and journal recovery",
        "started_at": time.time(),
        "engine": engine_record["engine"],
        "engine_identity": engine_record["identity"],
        "gateway_before": gateway_record["identity"],
        "admission_limits": settings.admission.model_dump(),
        "verify_admission": args.verify_admission,
        "tenant": args.tenant or "default",
        "checks": {},
    }
    checks = report["checks"]
    request_task = None
    queued_task = None
    blocked = None
    registry = None

    def admission_rows():
        with sqlite3.connect(f"file:{settings.registry_path}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in db.execute(
                    "SELECT l.request_id,l.owner,t.lease_id,t.tenant,t.tokens,t.branches,t.state "
                    "FROM admission_tickets t JOIN leases l ON l.id=t.lease_id ORDER BY t.id"
                )
            ]

    async def wait_admission(rid, state, task):
        async with asyncio.timeout(15):
            while True:
                row = next((r for r in admission_rows() if r["request_id"] == rid), None)
                if row and row["state"] == state:
                    return row
                if task.done():
                    response = await task
                    raise AssertionError(f"Request finished before {state}: {response.status_code}")
                await asyncio.sleep(0.005)

    try:
        checks["ready_before"] = await ready(url, keys["api"])
        async with httpx.AsyncClient(base_url=url, timeout=30) as client:
            profile_response = await client.get(
                "/admin/profile", headers={"Authorization": "Bearer " + keys["admin"]}
            )
            profile_response.raise_for_status()
            profile = profile_response.json()
        assert profile["model"]["id"] == settings.model_id
        assert profile["model"]["revision"] == settings.model_revision
        if args.verify_admission:
            assert profile["admission"]["scope"] == "shared_registry_engine"
            assert admission_rows() == []
        bundle = Bundle(
            id="crash-" + uuid.uuid4().hex,
            version=1,
            model=profile["model"],
            policy=Policy(max_questions=128, max_parallel_branches=4),
        )
        rid = "crash-request-" + uuid.uuid4().hex
        queued_rid = "crash-queued-" + uuid.uuid4().hex
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
                base_url=url, headers={"Authorization": "Bearer " + tenant_key}, timeout=150
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
            if args.verify_admission:
                checks["admitted_before_kill"] = await wait_admission(rid, "ADMITTED", request_task)
                queued_body = {**body, "request_id": queued_rid, "questions": body["questions"][:1]}
                queued_task = asyncio.create_task(data.post("/v1/decisions", json=queued_body))
                checks["queued_before_kill"] = await wait_admission(
                    queued_rid, "QUEUED", queued_task
                )
                alive_recovery = await admin.post(f"/admin/requests/{rid}/recover")
                assert alive_recovery.status_code == 409
                assert alive_recovery.json()["error"]["code"] == "recovery_not_confirmed"
                checks["alive_owner_recovery_rejected"] = 409
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
            if args.verify_admission:
                before = admission_rows()
                assert [row["state"] for row in before] == ["ADMITTED", "QUEUED"]
                assert not request_task.done() and not queued_task.done()
                checks["admission_before_kill"] = before
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
            if queued_task is not None:
                result = (await asyncio.gather(queued_task, return_exceptions=True))[0]
                assert isinstance(result, httpx.TransportError), type(result).__name__
                checks["queued_client_disconnect_error"] = type(result).__name__

        registry = Registry(settings.registry_path)
        orphan = next(item for item in registry.recovery_candidates() if item["request_id"] == rid)
        assert orphan["owner_status"] == "dead" and orphan["recoverable"]
        assert orphan["engine_request_ids"] == observed["engine_request_ids"]
        checks["durable_orphan_after_kill"] = orphan
        if args.verify_admission:
            assert admission_rows() == before
            checks["admission_retained_after_kill"] = admission_rows()
            queued_orphan = next(
                item for item in registry.recovery_candidates() if item["request_id"] == queued_rid
            )
            assert queued_orphan["owner_status"] == "dead" and queued_orphan["recoverable"]
            assert len(queued_orphan["engine_request_ids"]) == 1
            checks["queued_orphan_after_kill"] = queued_orphan
        # Preserve the exact admission policy and tenant environment mapping.
        # Launcher defaults could otherwise change limits while old leases remain.
        admission_config = gateway_dir / "restart-admission.json"
        admission_config.write_text(json.dumps(settings.admission.model_dump(), indent=2) + "\n")
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
            "--api-workers",
            str(settings.workers),
            "--admission-config",
            str(admission_config),
        ]
        for tenant in settings.tenant_key_envs:
            command.extend(["--tenant", tenant])
        child = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await child.communicate()
        assert child.returncode == 0, stderr.decode()[-2000:]
        report["gateway_restart"] = json.loads(stdout)
        checks["ready_after_restart"] = await ready(url, keys["api"])
        assert bundle.reference in checks["ready_after_restart"]["prepared_bundles"]
        # Validate normalized settings because old launchers omitted default
        # fields; credential bytes must remain unchanged.
        assert load_settings(gateway_dir / "config.json") == settings
        assert (gateway_dir / "keys.json").read_bytes() == keys_before
        checks["restart_preserved_settings_and_credentials"] = True
        checks["config_bytes_identical"] = (
            gateway_dir / "config.json"
        ).read_bytes() == config_before
        async with httpx.AsyncClient(base_url=url, timeout=120) as client:
            if args.verify_admission:
                assert admission_rows() == before
                checks["admission_retained_after_restart"] = admission_rows()
                response = await client.get(
                    "/admin/profile", headers={"Authorization": "Bearer " + keys["admin"]}
                )
                response.raise_for_status()
                state = response.json()["admission"]
                assert state["limits"] == settings.admission.model_dump()
                assert state["requests"] == 1 and state["queued_requests"] == 1
                assert (
                    state["expanded_tokens"] == before[0]["tokens"]
                    and state["expanded_branches"] == 128
                )
                checks["profile_after_restart"] = state
                blocked_id = "blocked-after-restart-" + uuid.uuid4().hex
                blocked_body = {
                    **body,
                    "request_id": blocked_id,
                    "questions": body["questions"][:1],
                    "execution": {"timeout_ms": 150},
                }
                blocked = asyncio.create_task(
                    client.post(
                        "/v1/decisions",
                        headers={"Authorization": "Bearer " + tenant_key},
                        json=blocked_body,
                    )
                )
                await wait_admission(blocked_id, "QUEUED", blocked)
                response = await blocked
                assert (
                    response.status_code == 504
                    and response.json()["error"]["code"] == "deadline_exceeded"
                )
                assert admission_rows() == before
                checks["restart_did_not_reset_capacity"] = {
                    "http_status": 504,
                    "original_reservations_unchanged": True,
                }
                response = await client.post(
                    f"/admin/requests/{rid}/recover",
                    headers={"Authorization": "Bearer " + tenant_key},
                )
                assert response.status_code == 401 and admission_rows() == before
                checks["recovery_requires_admin"] = 401
                response = await client.post(
                    f"/admin/requests/{queued_rid}/recover",
                    headers={"Authorization": "Bearer " + keys["admin"]},
                )
                response.raise_for_status()
                assert response.json() == {"recovered": True}
                assert admission_rows() == before[:1]
                checks["queued_recovery_preserves_active_reservation"] = admission_rows()
            response = await client.post(
                f"/admin/requests/{rid}/recover",
                headers={"Authorization": "Bearer " + keys["admin"]},
            )
            response.raise_for_status()
            assert response.json() == {"recovered": True}
            assert not registry.list()["leases"]
            checks["recovered_no_leases"] = True
            if args.verify_admission:
                assert admission_rows() == []
                checks["recovered_no_admission_tickets"] = True
            response = await client.post(
                "/v1/decisions",
                headers={"Authorization": "Bearer " + tenant_key},
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
        if queued_task is not None and not queued_task.done():
            queued_task.cancel()
            await asyncio.gather(queued_task, return_exceptions=True)
        if blocked is not None and not blocked.done():
            blocked.cancel()
            await asyncio.gather(blocked, return_exceptions=True)
        if registry is not None:
            registry.close()
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
    parser.add_argument("--runtime-source-commit")
    parser.add_argument("--verify-admission", action="store_true")
    parser.add_argument("--tenant")
    asyncio.run(run(parser.parse_args()))
