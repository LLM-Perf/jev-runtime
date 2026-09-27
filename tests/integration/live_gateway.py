"""Real standalone gateway checks using the Python SDK and a live engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx

from jev_runtime.config import load_settings
from jev_runtime.errors import JevError
from jev_runtime.sdk import AsyncJevClient


async def run(run_dir: Path, output: Path, source_commit: str):
    settings = load_settings(run_dir / "config.json")
    keys = json.loads((run_dir / "keys.json").read_text())
    url = f"http://127.0.0.1:{settings.port}"
    report = {
        "source_commit": source_commit,
        "started_at": time.time(),
        "qualification": "co-located gateway functional test",
        "checks": {},
    }
    checks = report["checks"]
    body = {
        "model": settings.bootstrap_alias,
        "input": {"text": "I need a refund."},
        "questions": [{"id": "refund", "type": "boolean", "instruction": "Is a refund requested?"}],
    }
    async with (
        AsyncJevClient(url, keys["api"]) as sdk,
        httpx.AsyncClient(
            base_url=url, headers={"Authorization": "Bearer " + keys["admin"]}, timeout=120
        ) as admin,
    ):

        async def management(path, payload=None):
            result = await (admin.get(path) if payload is None else admin.post(path, json=payload))
            result.raise_for_status()
            return result.json()

        try:
            response = await sdk.decide(body)
            assert response.status == "completed" and response.usage.successful_questions == 1
            assert response.engine["name"] == settings.backend
            assert response.answers["refund"].probabilities is not None
            checks["python_sdk_real_decision"] = response.model_dump(mode="json")

            rid = "gateway-cancel-" + uuid.uuid4().hex
            large = {
                **body,
                "request_id": rid,
                "input": {"text": "The buyer reports a duplicated charge. " * 128},
                "questions": [
                    {
                        "id": f"q{i}",
                        "type": "boolean",
                        "instruction": "Does the buyer report a payment problem?",
                    }
                    for i in range(16)
                ],
            }
            work = asyncio.create_task(sdk.decide(large))
            try:
                confirmed = False
                for _ in range(100):
                    await asyncio.sleep(0.01)
                    if await sdk.cancel(rid):
                        confirmed = True
                        break
                    if work.done():
                        break
                assert confirmed, "Request finished before cancellation could be tested"
                try:
                    await work
                except JevError as exc:
                    assert exc.code == "request_cancelled" and exc.status_code == 499
                else:
                    raise AssertionError("Cancelled request incorrectly returned success")
            finally:
                if not work.done():
                    await sdk.cancel(rid)
                await asyncio.gather(work, return_exceptions=True)
            snapshot = await management("/admin/bundles")
            assert not snapshot["leases"]
            assert (await sdk.decide(body)).status == "completed"
            checks["cancellation_http499_no_gateway_leases_then_recovery"] = True

            route = next(
                row for row in snapshot["routes"] if row["alias"] == settings.bootstrap_alias
            )
            disabled = await management(
                "/admin/bundles/disable",
                {"alias": settings.bootstrap_alias, "expected_generation": route["generation"]},
            )
            try:
                try:
                    await sdk.decide(body)
                except JevError as exc:
                    assert exc.code == "route_unavailable" and exc.status_code == 503
                else:
                    raise AssertionError("Disabled gateway alias accepted new traffic")
                async with httpx.AsyncClient(base_url=settings.engine_url, timeout=120) as native:
                    chat = await native.post(
                        "/v1/chat/completions",
                        json={
                            "model": settings.model_id,
                            "messages": [{"role": "user", "content": "Say hello."}],
                            "max_tokens": 8,
                            "temperature": 0,
                        },
                    )
                    assert chat.status_code == 200, chat.text[:500]
                    assert chat.json()["usage"]["completion_tokens"] > 0
            finally:
                await management(
                    "/admin/bundles/activate",
                    {
                        "alias": settings.bootstrap_alias,
                        "reference": route["ref"],
                        "expected_generation": disabled["generation"],
                    },
                )
            assert (await sdk.decide(body)).status == "completed"
            checks["gateway_disable_native_chat_survives_and_rollback"] = True
            report["passed"] = True
        except BaseException as exc:
            report["passed"] = False
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            raise
        finally:
            report["finished_at"] = time.time()
            await asyncio.to_thread(output.write_text, json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {"passed": report.get("passed"), "checks": list(checks), "output": str(output)}
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.run_dir, args.output, args.source_commit))
