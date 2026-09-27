"""Prove that another local API worker routes cancellation to the request owner."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path

import httpx

from jev_runtime.schema import Bundle, Policy


async def run(args):
    if args.output.exists():
        raise ValueError("Keep previous evidence; choose a new output")
    root = args.run_dir
    config = json.loads((root / "config.json").read_text())
    process = json.loads((root / "process.json").read_text())
    keys = json.loads((root / "keys.json").read_text())
    prefix = "" if process["mode"] == "gateway" else "/plugins/jev-runtime"
    base = f"http://127.0.0.1:{process['port']}" + prefix
    report = {
        "source_commit": args.source_commit,
        "qualification": "colocated local multi-worker functional test, not performance evidence",
        "started_at": time.time(),
        "process_identity": process["identity"],
        "checks": {},
    }
    checks = report["checks"]
    connections = {}
    work = None
    alias = "cancel-" + uuid.uuid4().hex
    activated = False
    admin = {"Authorization": "Bearer " + keys["admin"]}

    def pending(rid):
        with sqlite3.connect(f"file:{config['registry_path']}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT l.id,l.owner,l.request_id,w.branches FROM leases l "
                "JOIN lease_work w ON w.lease_id=l.id WHERE request_id=?",
                (rid,),
            ).fetchone()
            return dict(row) if row else None

    try:
        # A persistent HTTP/1 connection stays on its accepting worker. Capture
        # two different workers and verify both identities again after cancel.
        async with asyncio.timeout(90):
            while len(connections) < 2:
                client = httpx.AsyncClient(
                    base_url=base + "/",
                    timeout=120,
                    headers={"Authorization": "Bearer " + keys["api"]},
                    limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                )
                try:
                    response = await client.get("ready")
                    response.raise_for_status()
                    worker = response.json()["worker_id"]
                    if worker not in connections:
                        connections[worker] = client
                    else:
                        await client.aclose()
                except httpx.HTTPError:
                    await client.aclose()
                    await asyncio.sleep(0.2)
        (owner, first), (peer, second) = connections.items()
        checks["distinct_workers"] = [owner, peer]
        response = await first.get("admin/profile", headers=admin)
        response.raise_for_status()
        bundle = Bundle(
            id=alias,
            version=1,
            model=response.json()["model"],
            policy=Policy(max_questions=128, max_scoring_sequences=128, max_parallel_branches=1),
        )
        response = await first.post(
            "admin/bundles", headers=admin, json=bundle.model_dump(mode="json")
        )
        response.raise_for_status()
        for client in (first, second):
            response = await client.post(
                "admin/bundles/prepare", headers=admin, json={"reference": bundle.reference}
            )
            response.raise_for_status()
        response = await first.post(
            "admin/bundles/activate",
            headers=admin,
            json={"alias": alias, "reference": bundle.reference, "expected_generation": 0},
        )
        response.raise_for_status()
        activated = True
        rid = "cancel-proof-" + uuid.uuid4().hex
        body = {
            "model": alias,
            "request_id": rid,
            "input": {"text": "A customer asks for a refund. " * 80},
            "questions": [
                {"id": f"q{i}", "type": "boolean", "instruction": "Is a refund requested?"}
                for i in range(128)
            ],
            "execution": {"timeout_ms": 120000},
        }
        work = asyncio.create_task(first.post("v1/decisions", json=body))
        async with asyncio.timeout(10):
            while True:
                row = pending(rid)
                if row and len(json.loads(row["branches"])) == 128:
                    break
                if work.done():
                    raise AssertionError(
                        f"Request ended before cancellation: {(await work).status_code}"
                    )
                await asyncio.sleep(0.005)
        assert row["owner"] == owner
        checks["request_owner_and_journal"] = {
            "owner": owner,
            "request_id": rid,
            "branches": 128,
            "lease_id": row["id"],
        }
        response = await second.post(f"v1/requests/{rid}/cancel")
        response.raise_for_status()
        assert response.json() == {"cancelled": True}
        result = await work
        assert result.status_code == 499 and result.json()["error"]["code"] == "request_cancelled"
        assert pending(rid) is None
        checks["peer_cancelled_owner_request"] = {"cancelling_worker": peer, "http_status": 499}
        for worker, client in connections.items():
            response = await client.get("ready")
            response.raise_for_status()
            assert response.json()["worker_id"] == worker
        response = await second.post(
            "v1/decisions",
            json={**body, "request_id": rid + "-next", "questions": body["questions"][:1]},
        )
        response.raise_for_status()
        assert response.json()["status"] == "completed"
        checks["serving_after_cancellation"] = True
        report["passed"] = True
    except BaseException as exc:
        report["passed"] = False
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        raise
    finally:
        if work is not None and not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
        if activated:
            try:
                response = await first.post(
                    "admin/bundles/disable",
                    headers=admin,
                    json={"alias": alias, "expected_generation": 1},
                )
                response.raise_for_status()
                response = await first.post(
                    "admin/bundles/retire", headers=admin, json={"reference": bundle.reference}
                )
                response.raise_for_status()
                checks["retired_after_drain"] = True
            except Exception as exc:
                report["cleanup_error"] = type(exc).__name__
                report["passed"] = False
        for client in connections.values():
            await client.aclose()
        report["finished_at"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": report.get("passed"), "checks": list(checks)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    asyncio.run(run(parser.parse_args()))
