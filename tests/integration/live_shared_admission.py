"""Exercise durable shared quotas through two actual native API workers on DSW."""

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
        raise ValueError("Keep prior evidence; choose a new output")
    root = args.run_dir
    config = json.loads((root / "config.json").read_text())
    process = json.loads((root / "process.json").read_text())
    keys = json.loads((root / "keys.json").read_text())
    assert config["workers"] == 2
    expected = {
        "max_requests": 2,
        "max_tokens": 524288,
        "max_queue": 2,
        "max_tenant_requests": 1,
        "max_tenant_tokens": 262144,
        "max_tenant_queue": 1,
        "max_branches": 129,
        "max_tenant_branches": 128,
    }
    assert config["admission"] == expected
    assert set(keys["tenants"]) == {"alpha", "beta"}
    prefix = "" if process["mode"] == "gateway" else "/plugins/jev-runtime"
    base = f"http://127.0.0.1:{process['port']}" + prefix + "/"
    report = {
        "source_commit": args.source_commit,
        "runtime_source_commit": args.source_commit,
        "engine": process["engine"],
        "mode": process["mode"],
        "started_at": time.time(),
        "process_identity": process["identity"],
        "service_launch_command": process["command"],
        "model": process["model"],
        "limits": expected,
        "qualification": (
            "Colocated two-worker GPU functional check; not throughput or multi-node certification"
        ),
        "checks": {},
    }
    checks = report["checks"]
    connections = {}
    tasks = []
    active_requests = {}
    alias = "quotas-" + uuid.uuid4().hex
    activated = False

    def headers(tenant=None, admin=False):
        key = keys["admin"] if admin else keys["tenants"][tenant] if tenant else keys["api"]
        return {"Authorization": "Bearer " + key}

    def tickets():
        with sqlite3.connect(f"file:{config['registry_path']}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            return [
                dict(r)
                for r in db.execute(
                    "SELECT l.request_id,l.owner,t.tenant,t.tokens,t.branches,t.state "
                    "FROM admission_tickets t JOIN leases l ON l.id=t.lease_id ORDER BY t.id"
                )
            ]

    async def wait_state(rid, state, work):
        async with asyncio.timeout(10):
            while True:
                rows = tickets()
                row = next((r for r in rows if r["request_id"] == rid), None)
                if row and row["state"] == state:
                    return row
                if work.done():
                    result = await work
                    raise AssertionError(
                        f"Request ended before {state}: {result.status_code} {result.text[:500]}"
                    )
                await asyncio.sleep(0.005)

    async def call(client, path, body=None, tenant=None, admin=False):
        response = await (
            client.get(path, headers=headers(tenant, admin))
            if body is None
            else client.post(path, json=body, headers=headers(tenant, admin))
        )
        response.raise_for_status()
        return response.json()

    def body(rid, count=1, timeout=120000):
        return {
            "model": alias,
            "request_id": rid,
            "input": {"text": "A customer asks for a refund. " * 80},
            "questions": [
                {"id": f"q{i}", "type": "boolean", "instruction": "Is a refund requested?"}
                for i in range(count)
            ],
            "execution": {"timeout_ms": timeout},
        }

    def dispatch(client, tenant, rid, count=1, timeout=120000):
        work = asyncio.create_task(
            client.post("v1/decisions", headers=headers(tenant), json=body(rid, count, timeout))
        )
        tasks.append(work)
        active_requests[rid] = tenant
        return work

    try:
        # Two independent persistent connections per worker let control traffic
        # reach that worker while its data connection is awaiting a decision.
        async with asyncio.timeout(90):
            while len(connections) != 2 or any(len(v) != 2 for v in connections.values()):
                c = httpx.AsyncClient(
                    base_url=base,
                    timeout=130,
                    limits=httpx.Limits(
                        max_connections=1, max_keepalive_connections=1, keepalive_expiry=None
                    ),
                )
                try:
                    ready = await call(c, "ready")
                    worker = ready["worker_id"]
                    if len(connections.setdefault(worker, [])) < 2:
                        connections[worker].append(c)
                    else:
                        await c.aclose()
                except httpx.HTTPError:
                    await c.aclose()
                    await asyncio.sleep(0.1)
        (owner, (first, control)), (peer, (second, peer_control)) = connections.items()
        checks["distinct_worker_connections"] = {owner: 2, peer: 2}
        profile = await call(control, "admin/profile", admin=True)
        assert profile["admission"]["scope"] == "shared_registry_engine"
        assert profile["admission"]["limits"] == expected
        bundle = Bundle(
            id=alias,
            version=1,
            model=profile["model"],
            policy=Policy(max_questions=128, max_scoring_sequences=128, max_parallel_branches=1),
        )
        await call(control, "admin/bundles", bundle.model_dump(mode="json"), admin=True)
        for c in (control, peer_control):
            await call(c, "admin/bundles/prepare", {"reference": bundle.reference}, admin=True)
        await call(
            control,
            "admin/bundles/activate",
            {"alias": alias, "reference": bundle.reference, "expected_generation": 0},
            admin=True,
        )
        activated = True
        long_id = "long-" + uuid.uuid4().hex
        long = dispatch(first, "alpha", long_id, 128)
        row = await wait_state(long_id, "ADMITTED", long)
        assert row["owner"] == owner and row["branches"] == 128
        checks["long_request_admitted"] = row
        waiting_id = "queued-" + uuid.uuid4().hex
        waiting = dispatch(second, "alpha", waiting_id)
        row = await wait_state(waiting_id, "QUEUED", waiting)
        assert row["owner"] == peer
        checks["same_tenant_queued_on_peer"] = row
        rejected = await control.post(
            "v1/decisions", headers=headers("alpha"), json=body("full-" + uuid.uuid4().hex)
        )
        assert rejected.status_code == 429 and rejected.json()["error"]["code"] == "queue_full"
        checks["shared_tenant_queue_limit"] = {"http_status": 429}
        wrong = await call(peer_control, f"v1/requests/{waiting_id}/cancel", {}, tenant="beta")
        assert wrong == {"cancelled": False} and not waiting.done()
        checks["tenant_cannot_cancel_peer_tenant"] = True
        result = await call(
            control, "v1/decisions", body("beta-" + uuid.uuid4().hex), tenant="beta"
        )
        assert result["status"] == "completed" and not long.done() and not waiting.done()
        checks["eligible_other_tenant_served_during_long_request"] = {
            "status": result["status"],
            "usage": result["usage"],
        }
        assert await call(peer_control, f"v1/requests/{waiting_id}/cancel", {}, tenant="alpha") == {
            "cancelled": True
        }
        cancelled = await waiting
        assert cancelled.status_code == 499
        assert all(r["request_id"] != waiting_id for r in tickets())
        checks["queued_cancel_releases_ticket"] = {"http_status": 499}
        timeout_id = "timeout-" + uuid.uuid4().hex
        expired = await dispatch(second, "alpha", timeout_id, timeout=100)
        assert expired.status_code == 504 and expired.json()["error"]["code"] == "deadline_exceeded"
        assert not long.done() and all(r["request_id"] != timeout_id for r in tickets())
        checks["queued_deadline_releases_ticket"] = {"http_status": 504}
        branch_id = "branches-" + uuid.uuid4().hex
        branch_wait = dispatch(second, "beta", branch_id, count=2)
        row = await wait_state(branch_id, "QUEUED", branch_wait)
        assert row["owner"] == peer and row["branches"] == 2
        state = await call(control, "admin/profile", admin=True)
        assert (
            state["admission"]["requests"] == 1 and state["admission"]["expanded_branches"] == 128
        )
        checks["expanded_branch_budget_shared_across_workers"] = {
            "queued": row,
            "admission": state["admission"],
        }
        for c in (control, peer_control):
            text = (await c.get("metrics", headers=headers())).text
            assert 'jev_shared_admission{quantity="requests"} 1.0' in text
            assert 'jev_shared_admission{quantity="expanded_branches"} 128.0' in text
        checks["shared_gauge_identical_on_both_workers"] = True
        assert await call(peer_control, f"v1/requests/{branch_id}/cancel", {}, tenant="beta") == {
            "cancelled": True
        }
        assert (await branch_wait).status_code == 499
        assert await call(peer_control, f"v1/requests/{long_id}/cancel", {}, tenant="alpha") == {
            "cancelled": True
        }
        assert (await long).status_code == 499 and tickets() == []
        checks["peer_cancel_active_owner_and_drain"] = {"http_status": 499, "remaining_tickets": 0}
        for worker, (data, ctrl) in connections.items():
            assert (await call(ctrl, "ready"))["worker_id"] == worker
            result = await call(
                data, "v1/decisions", body("after-" + uuid.uuid4().hex), tenant="alpha"
            )
            assert result["status"] == "completed"
        checks["serving_after_drain"] = True
        report["passed"] = True
    except BaseException as exc:
        report["passed"] = False
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        raise
    finally:
        # An independent control connection remains available even when a data
        # connection is waiting. Ask the owning runtime to fence GPU cancellation.
        async with httpx.AsyncClient(base_url=base, timeout=20) as cleanup:
            for rid, tenant in active_requests.items():
                try:
                    await call(cleanup, f"v1/requests/{rid}/cancel", {}, tenant=tenant)
                except Exception as exc:
                    report.setdefault("cleanup_errors", []).append(type(exc).__name__)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if activated:
                try:
                    await call(
                        cleanup,
                        "admin/bundles/disable",
                        {"alias": alias, "expected_generation": 1},
                        admin=True,
                    )
                    await call(
                        cleanup, "admin/bundles/retire", {"reference": bundle.reference}, admin=True
                    )
                    checks["retired_after_drain"] = True
                except Exception as exc:
                    report.setdefault("cleanup_errors", []).append(type(exc).__name__)
        for clients in connections.values():
            for c in clients:
                await c.aclose()
        if report.get("cleanup_errors"):
            report["passed"] = False
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
