"""Observe real completed/in-flight scoring RPCs across a gateway version switch.

No engine calls are mocked, delayed, or replaced. Counters establish adapter RPC
progress, not a claim that a specific CUDA kernel was resident at the sample time.
"""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

import httpx

from jev_runtime.schema import Bundle, DecisionResponse, Policy


def validate_progress(before: dict, after: dict, planned: int) -> None:
    for item in (before, after):
        assert item["scope"] == "local_worker" and item["request"] is not None
        request = item["request"]
        assert request["stage"] == "execute" and request["scoring_sequences"] == planned
        engine = request["work"]["engine_call"]
        assert 0 < engine["succeeded"] < planned
        assert engine["active"] == 1 and engine["failed_or_cancelled"] == 0
        assert engine["started"] == engine["succeeded"] + 1
    assert before["worker_id"] == after["worker_id"]
    for field in ("request_id", "bundle", "bundle_digest", "generation", "lease_id"):
        assert before["request"][field] == after["request"][field]
    assert (
        after["request"]["work"]["engine_call"]["succeeded"]
        >= (before["request"]["work"]["engine_call"]["succeeded"])
    )


def run_case(
    controller,
    root: Path,
    front: int,
    ports: dict,
    releases: dict,
    keys: dict,
    canary: dict,
    source_slot: str,
    iterations: int = 3,
) -> dict:
    report = {
        "qualification": __doc__,
        "source_slot": source_slot,
        "planned_branches_per_request": 128,
        "attempts": [],
        "passed": False,
    }
    output = root / "native-cancellation.json"
    assert not output.exists()
    alias = "rollout-cancel-" + uuid.uuid4().hex
    headers = {"Authorization": "Bearer " + keys["api"]}
    admin = {"Authorization": "Bearer " + keys["admin"]}
    published, future, rid = False, None, None

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    def journal(request_id):
        with sqlite3.connect(f"file:{root}/registry.db?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT l.id,l.owner,l.request_id,l.ref,w.phase,w.branches FROM leases l "
                "JOIN lease_work w ON w.lease_id=l.id WHERE l.request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "lease_id": row["id"],
                "worker_id": row["owner"],
                "bundle": row["ref"],
                "phase": row["phase"],
                "branches": len(json.loads(row["branches"])),
            }

    def post(payload):
        with httpx.Client(
            base_url=f"http://127.0.0.1:{front}", headers=headers, timeout=180, trust_env=False
        ) as client:
            return client.post("/v1/decisions", json=payload)

    def change(slot):
        state = controller.status()["state"]
        return controller.switch(
            slot, state["generation"], uuid.uuid4().hex, slot + "-test", releases[slot], canary
        )

    with ExitStack() as stack, concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        clients, rosters = {}, {}
        for slot, port in ports.items():
            expected = set(controller.probe.snapshot(port)["workers"])
            rosters[slot] = expected
            deadline = time.monotonic() + 30
            while not expected <= clients.keys():
                assert time.monotonic() < deadline, "Cannot pin an admin connection to every worker"
                client = httpx.Client(
                    base_url=f"http://127.0.0.1:{port}",
                    headers=headers,
                    timeout=30,
                    trust_env=False,
                    limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                )
                response = client.get("/ready")
                response.raise_for_status()
                owner = response.json()["worker_id"]
                assert owner in expected
                if owner in clients:
                    client.close()
                else:
                    clients[owner] = stack.enter_context(client)
        first = clients[next(iter(rosters[source_slot]))]

        def management(client, path, payload=None):
            result = (
                client.get(path, headers=admin)
                if payload is None
                else client.post(path, headers=admin, json=payload)
            )
            result.raise_for_status()
            return result.json()

        try:
            profile = management(first, "/admin/profile")
            # Ask for an unsupported ID before changing state to establish that
            # the source release implements the observation contract.
            missing = management(first, "/admin/requests/not-active/progress")
            assert missing["scope"] == "local_worker" and missing["request"] is None
            bundle = Bundle(
                id=alias,
                version=1,
                model=profile["model"],
                policy=Policy(
                    max_questions=128,
                    max_scoring_sequences=128,
                    max_parallel_branches=1,
                ),
            )
            management(first, "/admin/bundles", bundle.model_dump(mode="json"))
            for client in clients.values():
                management(client, "/admin/bundles/prepare", {"reference": bundle.reference})
            management(
                first,
                "/admin/bundles/activate",
                {
                    "alias": alias,
                    "reference": bundle.reference,
                    "expected_generation": 0,
                },
            )
            published = True
            report["bundle"] = bundle.model_dump(mode="json")
            report["bundle_digest"] = bundle.digest
            for _ in range(iterations):
                if controller.proxy.active() != source_slot:
                    change(source_slot)
                rid = "gpu-rollout-cancel-" + uuid.uuid4().hex
                payload = {
                    "model": alias,
                    "request_id": rid,
                    "input": {
                        "text": "A customer requests a refund after a duplicate charge. " * 120
                    },
                    "questions": [
                        {
                            "id": f"q{i}",
                            "type": "boolean",
                            "instruction": f"Question {i}: Is a refund requested?",
                        }
                        for i in range(128)
                    ],
                    "execution": {"timeout_ms": 120000},
                }
                preview = management(first, "/admin/compile", payload)
                assert len(preview["sequences"]) == 128 and preview["engine_dispatched"] is False
                attempt = {
                    "request_id": rid,
                    "compiled_input_digest": preview["input_digest"],
                    "logical_prompt_tokens": preview["logical_prompt_tokens"],
                }
                report["attempts"].append(attempt)
                save()
                future = pool.submit(post, payload)
                deadline = time.monotonic() + 15
                while True:
                    assert not future.done(), "Request completed before the observed switch window"
                    row = journal(rid)
                    if row and row["branches"] == 128:
                        assert row["worker_id"] in rosters[source_slot]
                        owner = clients[row["worker_id"]]
                        before = management(owner, f"/admin/requests/{rid}/progress")
                        current = before.get("request")
                        if (
                            current
                            and current["work"]["engine_call"]["succeeded"] >= 2
                            and current["work"]["engine_call"]["active"] == 1
                        ):
                            break
                    assert time.monotonic() < deadline, "No completed native scoring RPC observed"
                    time.sleep(0.01)
                attempt.update(journal_before=row, progress_before=before)
                save()
                target = "green" if source_slot == "blue" else "blue"
                receipt = change(target)
                deadline = time.monotonic() + 3
                while True:
                    assert not future.done(), "Request completed before post-switch observation"
                    after = management(owner, f"/admin/requests/{rid}/progress")
                    current = after.get("request")
                    if current and current["work"]["engine_call"]["active"] == 1:
                        break
                    assert time.monotonic() < deadline, "No post-switch active scoring RPC"
                    time.sleep(0.005)
                attempt.update(switch=receipt, progress_after=after)
                save()
                validate_progress(before, after, 128)
                assert not future.done()
                assert journal(rid) == row
                held = controller.drain(receipt["id"], timeout=0.05)
                assert held["drained"] is False
                attempt["drain_while_active"] = held
                with httpx.Client(
                    base_url=f"http://127.0.0.1:{front}",
                    headers=headers,
                    timeout=30,
                    trust_env=False,
                ) as client:
                    cancelled = client.post(f"/v1/requests/{rid}/cancel")
                cancelled.raise_for_status()
                assert cancelled.json() == {"cancelled": True}
                assert cancelled.headers["X-Jev-Deployment"] == target
                cancel_worker = cancelled.headers.get("X-Jev-Worker")
                if cancel_worker is not None:
                    assert cancel_worker in rosters[target]
                result = future.result(timeout=30)
                assert (
                    result.status_code == 499
                    and result.json()["error"]["code"] == "request_cancelled"
                )
                assert result.headers["X-Jev-Deployment"] == source_slot
                assert journal(rid) is None
                assert management(owner, f"/admin/requests/{rid}/progress")["request"] is None
                attempt["cancellation"] = {
                    "slot": target,
                    "worker": cancel_worker,
                    "request_http_status": 499,
                    "lease_released": True,
                }
                drained = controller.drain(receipt["id"], timeout=30)
                assert drained["drained"]
                attempt["drained_after_cancel"] = drained
                recovered = post(
                    {**payload, "request_id": rid + "-after", "questions": payload["questions"][:1]}
                )
                recovered.raise_for_status()
                assert DecisionResponse.model_validate(recovered.json()).status == "completed"
                attempt["serving_after_cancel"] = True
                save()
            report["passed"] = True
        except BaseException as exc:
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            raise
        finally:
            if future is not None and not future.done():
                try:
                    with httpx.Client(timeout=30, trust_env=False) as client:
                        cancelled = client.post(
                            f"http://127.0.0.1:{front}/v1/requests/{rid}/cancel", headers=headers
                        )
                    cancelled.raise_for_status()
                    future.result(timeout=30)
                except Exception as exc:
                    report["cleanup_failure"] = str(exc)
                    report["passed"] = False
            if published:
                try:
                    management(
                        first, "/admin/bundles/disable", {"alias": alias, "expected_generation": 1}
                    )
                    management(first, "/admin/bundles/retire", {"reference": bundle.reference})
                    report["bundle_retired"] = True
                except Exception as exc:
                    report["cleanup_failure"] = str(exc)
                    report["passed"] = False
            save()
    return report
