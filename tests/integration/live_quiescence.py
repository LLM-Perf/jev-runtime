"""Real two-worker Jev drain, admission rejection and preserved-route evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from pathlib import Path

import httpx

from jev_runtime.schema import Bundle, DecisionResponse, Policy


async def run(args):
    if args.output.exists():
        raise ValueError("Preserve earlier evidence; choose a fresh output")
    record = json.loads((args.run_dir / "process.json").read_text())
    config = json.loads((args.run_dir / "config.json").read_text())
    keys = json.loads((args.run_dir / "keys.json").read_text())
    base = f"http://127.0.0.1:{record['port']}/plugins/jev-runtime/"
    admin = {"Authorization": "Bearer " + keys["admin"]}
    data = {"Authorization": "Bearer " + keys["api"]}
    report = {
        "source_commit": args.source_commit,
        "engine": record["engine"],
        "process": record["identity"],
        "started_at": time.time(),
        "checks": {},
        "qualification": (
            "Two-worker colocated GPU lifecycle check; not release or performance certification"
        ),
    }
    checks, connections, tasks = report["checks"], {}, []

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def work():
        with sqlite3.connect(f"file:{config['registry_path']}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            return {
                table: [dict(row) for row in db.execute("SELECT * FROM " + table)]
                for table in ("leases", "raw_work", "recovery_claims")
            }

    async def call(client, path, body=None, *, auth=admin):
        response = await (
            client.get(path, headers=auth)
            if body is None
            else client.post(path, headers=auth, json=body)
        )
        response.raise_for_status()
        return response.json()

    try:
        async with asyncio.timeout(60):
            while len(connections) != 2 or any(len(items) != 2 for items in connections.values()):
                client = httpx.AsyncClient(
                    base_url=base,
                    timeout=130,
                    trust_env=False,
                    limits=httpx.Limits(
                        max_connections=1, max_keepalive_connections=1, keepalive_expiry=None
                    ),
                )
                profile = await call(client, "admin/profile")
                owner = profile["worker_id"]
                if len(connections.setdefault(owner, [])) < 2:
                    connections[owner].append(client)
                else:
                    await client.aclose()
        checks["workers"] = sorted(connections)
        (owner, (first, control)), (peer, (second, peer_control)) = connections.items()
        profile = await call(control, "admin/profile")
        bundle = Bundle(
            id="quiescence",
            version=1,
            model=profile["model"],
            policy=Policy(max_questions=128, max_scoring_sequences=128, max_parallel_branches=1),
        )
        await call(control, "admin/bundles", bundle.model_dump(mode="json"))
        for client in (control, peer_control):
            prepared = await call(client, "admin/bundles/prepare", {"reference": bundle.reference})
            assert prepared["worker_id"] in connections
        await call(
            control,
            "admin/bundles/activate",
            {
                "alias": bundle.id,
                "reference": bundle.reference,
                "expected_generation": 0,
            },
        )
        before = await call(control, "admin/bundles")
        checks["routes_before"] = before["routes"]
        body = {
            "model": bundle.id,
            "request_id": "drain-long",
            "input": {"text": "A customer asks for a refund. " * 80},
            "questions": [
                {"id": f"q{i}", "type": "boolean", "instruction": "Refund requested?"}
                for i in range(128)
            ],
            "execution": {"timeout_ms": 120000},
        }
        short = {**body, "request_id": "short", "questions": body["questions"][:1]}
        checks["warm"] = []
        for client in (first, second):
            result = DecisionResponse.model_validate(
                await call(client, "v1/decisions", short, auth=data)
            )
            assert result.status == "completed"
            checks["warm"].append(result.model_dump(mode="json"))
        preview = await call(control, "admin/compile", short)
        raw_body = {**preview["sequences"][0], "request_id": "drain-raw"}
        # Observe two real periodic health iterations before testing their stop.
        await asyncio.sleep(2.2)
        checks["health_before"] = [
            (await call(client, "admin/profile"))["health"] for client in (control, peer_control)
        ]
        typed = asyncio.create_task(first.post("v1/decisions", headers=data, json=body))
        tasks.append(typed)
        async with asyncio.timeout(10):
            while not any(row["request_id"] == body["request_id"] for row in work()["leases"]):
                assert not typed.done(), "Long request ended before the drain observation"
                await asyncio.sleep(0.002)
        raw = asyncio.create_task(second.post("v1/scores", headers=data, json=raw_body))
        tasks.append(raw)
        async with asyncio.timeout(10):
            while not work()["raw_work"]:
                assert not raw.done(), "Raw request ended before its journal observation"
                await asyncio.sleep(0.001)
        checks["work_before_gate"] = work()
        initial = await call(control, "admin/quiescence")
        closed = await call(
            peer_control,
            "admin/quiescence",
            {
                "expected_generation": initial["generation"],
                "timeout_seconds": 0,
            },
        )
        checks["gate_closed"] = closed
        assert not closed["drained"] and closed["outstanding"]["leases"] > 0
        checks["rejections"] = []
        for client in (control, peer_control):
            for path, payload in (
                ("v1/decisions", {**short, "request_id": "rejected"}),
                ("v1/scores", {**raw_body, "request_id": "rejected-raw"}),
                ("admin/bundles/prepare", {"reference": bundle.reference}),
                (
                    "admin/bundles/activate",
                    {"alias": "late", "reference": bundle.reference, "expected_generation": 0},
                ),
            ):
                response = await client.post(
                    path, headers=admin if path.startswith("admin") else data, json=payload
                )
                assert (
                    response.status_code == 503
                    and response.json()["error"]["code"] == "backend_quiescing"
                )
                checks["rejections"].append(
                    {"path": path, "status": response.status_code, "body": response.json()}
                )
            ready = await client.get("ready", headers=data)
            assert ready.status_code == 503
        typed_response, raw_response = await asyncio.gather(typed, raw)
        typed_response.raise_for_status()
        result = DecisionResponse.model_validate(typed_response.json())
        assert result.status == "completed" and result.usage.successful_questions == 128
        assert typed_response.headers["x-jev-worker"] == owner
        raw_response.raise_for_status()
        assert raw_response.json()["request_id"] == "drain-raw"
        checks["typed_completed"] = result.model_dump(mode="json")
        checks["raw_completed"] = raw_response.json()
        checks["drained"] = await call(
            control,
            "admin/quiescence",
            {
                "expected_generation": closed["generation"],
                "timeout_seconds": 30,
            },
        )
        assert checks["drained"]["drained"] and all(
            row["state"] == "QUIESCED" for row in checks["drained"]["workers"]
        )
        await asyncio.sleep(2.2)
        checks["health_after"] = [
            (await call(client, "admin/profile"))["health"] for client in (control, peer_control)
        ]
        assert all(not row["monitor_running"] for row in checks["health_after"])
        checks["work_after"] = work()
        assert not any(checks["work_after"].values())
        checks["routes_after"] = (await call(control, "admin/bundles"))["routes"]
        assert checks["routes_before"] == checks["routes_after"]
        report["passed"] = True
    except BaseException as exc:
        report["passed"] = False
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        # Keep connections until accepted work finishes; do not discard its evidence.
        if tasks:
            report["terminal_http"] = [
                r.status_code if isinstance(r, httpx.Response) else type(r).__name__
                for r in await asyncio.gather(*tasks, return_exceptions=True)
            ]
        for items in connections.values():
            for client in items:
                await client.aclose()
        report["finished_at"] = time.time()
        save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
