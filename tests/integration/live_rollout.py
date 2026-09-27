"""Real HAProxy and two multi-worker gateways; optional existing native engine.

CPU fixture results establish HTTP/control semantics only. Native mode uses an
already owned isolated engine and never stops it. All created process groups are
recorded and cleaned up with PID/start-tick/boot-ID checks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx

from deployment.dsw_service import process_identity
from jev_runtime.rollout import GatewayProbe, Rollout, RolloutError, write_json
from jev_runtime.schema import DecisionResponse
from tests.integration.run_native_validation import group_members


def main(args):
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "source_commit": args.source_commit,
        "started_at": time.time(),
        "fixture": args.engine_run_dir is None,
        "checks": {},
        "processes": [],
        "qualification": "Colocated rollout functional test; no performance certification",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    keys = {"api": secrets.token_urlsafe(32), "admin": secrets.token_urlsafe(32)}
    write_json(root / "keys.json", keys)
    env = {**os.environ, "JEV_API_KEY": keys["api"], "JEV_ADMIN_KEY": keys["admin"]}
    template = {
        "backend": "sglang",
        "model_id": "fixture",
        "model_revision": "a" * 40,
        "bootstrap_alias": "decision-model",
    }
    if args.engine_run_dir:
        template = json.loads((args.engine_run_dir / "config.json").read_text())
        engine_record = json.loads((args.engine_run_dir / "process.json").read_text())
        assert process_identity(engine_record["identity"]["pid"]) == engine_record["identity"]
        report["native_engine_identity"] = engine_record["identity"]
        env["JEV_ENGINE_API_KEY"] = json.loads((args.engine_run_dir / "keys.json").read_text())[
            "api"
        ]
    records, stop_traffic, traffic, tasks = [], threading.Event(), [], []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    checks = report["checks"]
    front, blue, green = args.port, args.port + 1, args.port + 2
    probe = GatewayProbe(keys["api"], keys["admin"], timeout=30)
    controller = Rollout(root / "proxy", probe)
    body = {
        "model": template["bootstrap_alias"],
        "input": {"text": "Please refund my payment."},
        "questions": [{"id": "q", "type": "boolean", "instruction": "Is a refund requested?"}],
    }
    headers = {"Authorization": "Bearer " + keys["api"]}
    admin_headers = {"Authorization": "Bearer " + keys["admin"]}
    source = Path(__file__).resolve().parents[2]

    def save():
        write_json(root / "report.json", report)

    def launch(name, command, environment, port):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
        with (root / f"{name}.log").open("x") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        identity = process_identity(process.pid)
        assert identity is not None
        record = {"name": name, "identity": identity, "command": command, "port": port}
        records.append((record, process))
        report["processes"].append(record)
        save()
        return process

    def post(payload, client=None):
        if client is None:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{front}", timeout=120, headers=headers, trust_env=False
            ) as own:
                return own.post("/v1/decisions", json=payload)
        return client.post("/v1/decisions", json=payload)

    def strict(response):
        response.raise_for_status()
        decision = DecisionResponse.model_validate(response.json())
        assert decision.status == "completed"
        assert response.headers["X-Jev-Deployment"] in {"blue", "green"}
        return decision

    def switch(slot):
        state = controller.status()["state"]
        return controller.switch(
            slot, state["generation"], uuid.uuid4().hex, slot + "-test", args.source_commit, body
        )

    def traffic_loop():
        with httpx.Client(
            base_url=f"http://127.0.0.1:{front}", headers=headers, timeout=120, trust_env=False
        ) as client:
            while not stop_traffic.is_set():
                start = time.monotonic()
                try:
                    response = post(body, client)
                    decision = strict(response)
                    traffic.append(
                        {
                            "status": response.status_code,
                            "bundle": decision.bundle,
                            "generation": decision.generation,
                            "slot": response.headers["X-Jev-Deployment"],
                            "latency_ms": (time.monotonic() - start) * 1000,
                        }
                    )
                except Exception as exc:
                    traffic.append({"error": type(exc).__name__, "message": str(exc)[:300]})

    try:
        for name, port in (("blue", blue), ("green", green)):
            config = {
                **template,
                "host": "127.0.0.1",
                "port": port,
                "workers": 2,
                "registry_path": str(root / "registry.db"),
                "deployment_id": name + "-test",
                "release_id": args.source_commit,
                "tenant_key_envs": {},
            }
            path = root / f"{name}.json"
            write_json(path, config)
            factory = (
                "tests.integration.rollout_fixture:create"
                if args.engine_run_dir is None
                else ("jev_runtime.api:create_app_from_env")
            )
            process = launch(
                name,
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    factory,
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--workers",
                    "2",
                ],
                {**env, "JEV_CONFIG": str(path)},
                port,
            )
            deadline = time.monotonic() + 180
            while True:
                assert process.poll() is None, f"{name} exited"
                try:
                    checks[name + "_qualified"] = probe.snapshot(port)
                    break
                except (httpx.HTTPError, RolloutError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.5)
        controller.initialize(blue, green, front)
        subprocess.run([str(args.haproxy), "-c", "-f", str(root / "proxy/haproxy.cfg")], check=True)
        launch(
            "haproxy", [str(args.haproxy), "-db", "-f", str(root / "proxy/haproxy.cfg")], env, front
        )
        for _ in range(100):
            if (root / "proxy/control.sock").exists():
                break
            time.sleep(0.05)
        identity = controller.proxy.identity()
        report["proxy_identity"] = identity
        for _ in range(2):
            tasks.append(pool.submit(traffic_loop))
        with httpx.Client(
            base_url=f"http://127.0.0.1:{front}", headers=headers, timeout=120, trust_env=False
        ) as persistent:
            assert post(body, persistent).headers["X-Jev-Deployment"] == "blue"
            # Headers are complete but JSON parsing has not finished. The proxy
            # must retain this old stream even though no registry lease exists.
            partial = socket.create_connection(("127.0.0.1", front), timeout=30)
            payload = json.dumps(body).encode()
            partial.sendall(
                (
                    f"POST /v1/decisions HTTP/1.1\r\nHost: localhost\r\n"
                    f"Authorization: Bearer {keys['api']}\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
                ).encode()
                + payload[:5]
            )
            time.sleep(0.2)
            first = switch("green")
            assert not controller.drain(first["id"], timeout=0.1)["drained"]
            partial.sendall(payload[5:])
            parsed = http.client.HTTPResponse(partial)
            parsed.begin()
            assert parsed.status == 200 and parsed.getheader("X-Jev-Deployment") == "blue"
            DecisionResponse.model_validate_json(parsed.read())
            partial.close()
            checks["partial_body_survives_switch_and_prevents_early_drain"] = True
            assert post(body, persistent).headers["X-Jev-Deployment"] == "green"
            checks["keepalive_next_request_uses_new_slot"] = True
        checks["old_slot_drain"] = controller.drain(first["id"], timeout=30)
        assert checks["old_slot_drain"]["drained"]
        switch("blue")
        # CPU fixture holds a real gateway request and tests peer cancellation
        # across the proxy switch. GPU cancellation has separate native tests.
        if args.engine_run_dir is None:
            rid = "rollout-cancel-" + uuid.uuid4().hex
            work = pool.submit(
                post,
                {**body, "request_id": rid, "questions": [{**body["questions"][0], "id": "slow"}]},
            )
            deadline = time.monotonic() + 5
            while True:
                inventory = httpx.get(
                    f"http://127.0.0.1:{blue}/admin/bundles", headers=admin_headers
                ).json()
                if any(row["request_id"] == rid for row in inventory["leases"]):
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
            switched = switch("green")
            result = httpx.post(
                f"http://127.0.0.1:{front}/v1/requests/{rid}/cancel", headers=headers, timeout=30
            )
            assert result.json() == {"cancelled": True}
            assert work.result(timeout=30).status_code == 499
            assert controller.drain(switched["id"], timeout=30)["drained"]
            checks["peer_cancellation_across_switch"] = True
        for _ in range(args.switches):
            target = "green" if controller.proxy.active() == "blue" else "blue"
            receipt = switch(target)
            assert controller.drain(receipt["id"], timeout=30)["drained"]
        checks["continuous_switches"] = args.switches
        state = controller.status()["state"]
        target = "green" if state["active"] == "blue" else "blue"
        try:
            controller.switch(
                target,
                state["generation"],
                "bad-release",
                target + "-test",
                "not-the-running-source",
                body,
            )
        except RolloutError:
            pass
        else:
            raise AssertionError("Wrong release accepted")
        assert controller.proxy.active() == state["active"]
        checks["incorrect_release_rejected"] = True

        def interrupt(stage):
            if stage == "runtime":
                raise RuntimeError("injected lost controller after runtime switch")

        try:
            controller.switch(
                target,
                state["generation"],
                "interrupted",
                target + "-test",
                args.source_commit,
                body,
                checkpoint=interrupt,
            )
        except RuntimeError as exc:
            assert "injected lost" in str(exc)
        assert controller.status()["state"]["pending"] is not None
        checks["reconciled"] = controller.reconcile()
        assert checks["reconciled"]["observed"] == target
        assert controller.proxy.identity() == identity
        checks["same_proxy_pid"] = True
        report["passed"] = True
    except BaseException as exc:
        report["passed"] = False
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        raise
    finally:
        stop_traffic.set()
        for task in tasks:
            task.result(timeout=150)
        pool.shutdown(wait=True)
        report["traffic"] = traffic
        report["traffic_successes"] = sum("error" not in row for row in traffic)
        report["traffic_errors"] = sum("error" in row for row in traffic)
        report["cleanup"] = []
        for record, process in reversed(records):
            identity = record["identity"]
            if process_identity(identity["pid"]) == identity:
                os.killpg(identity["pid"], signal.SIGTERM)
            process.wait(timeout=60)
            remaining = group_members(identity["pid"])
            report["cleanup"].append(
                {"name": record["name"], "remaining": remaining, "returncode": process.returncode}
            )
        report["passed"] = bool(
            report.get("passed")
            and traffic
            and not report["traffic_errors"]
            and all(not row["remaining"] for row in report["cleanup"])
        )
        report["finished_at"] = time.time()
        save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--haproxy", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--engine-run-dir", type=Path)
    parser.add_argument("--port", type=int, default=18800)
    parser.add_argument("--switches", type=int, default=10)
    raise SystemExit(main(parser.parse_args()))
