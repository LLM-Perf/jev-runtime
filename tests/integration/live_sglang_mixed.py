"""Overlap a native generation stream with typed selected-label scoring."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from jev_runtime.schema import DecisionResponse


async def run(args):
    if args.output.exists():
        raise ValueError("Preserve previous attempts")
    record = json.loads((args.run_dir / "process.json").read_text())
    config = json.loads((args.run_dir / "config.json").read_text())
    keys = json.loads((args.run_dir / "keys.json").read_text())
    assert record["mode"] == "native-plugin" and record["engine"] == "sglang"
    url = f"http://127.0.0.1:{record['port']}"
    report = {
        "source_commit": args.source_commit,
        "qualification": "Concurrent native/typed request lifetimes; "
        "not batch-composition or performance certification",
        "process": record,
        "started_at": time.time(),
        "decisions": [],
    }
    first = asyncio.Event()
    work = None
    async with httpx.AsyncClient(base_url=url, timeout=120) as client:

        async def native():
            last = None
            done = False
            async with client.stream(
                "POST",
                "/generate",
                json={
                    "text": "Explain the sequence of positive integers:",
                    "stream": True,
                    "sampling_params": {
                        "max_new_tokens": 512,
                        "ignore_eos": True,
                        "temperature": 0,
                    },
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        continue
                    last = json.loads(payload)
                    count = last.get("meta_info", {}).get("completion_tokens", 0)
                    if count > 0 and not first.is_set():
                        report["native_first_token_at"] = time.time()
                        first.set()
            assert done and last and last["meta_info"]["completion_tokens"] == 512
            report["native_finished_at"] = time.time()
            report["native_final_meta"] = last["meta_info"]
            return last

        try:
            work = asyncio.create_task(native())
            async with asyncio.timeout(60):
                while not first.is_set():
                    if work.done():
                        await work
                        raise AssertionError("Native request ended without a token")
                    await asyncio.sleep(0.01)
            for index in range(12):
                pending_before = not work.done()
                started = time.time()
                response = await client.post(
                    "/plugins/jev-runtime/v1/decisions",
                    headers={"Authorization": "Bearer " + keys["api"]},
                    json={
                        "model": config["bootstrap_alias"],
                        "input": {"text": f"Ticket {index}: refund a duplicated payment."},
                        "questions": [
                            {
                                "id": "q0",
                                "type": "boolean",
                                "instruction": "Does the buyer ask for a refund?",
                            }
                        ],
                    },
                )
                response.raise_for_status()
                parsed = DecisionResponse.model_validate(response.json())
                assert parsed.status == "completed" and parsed.usage.successful_questions == 1
                report["decisions"].append(
                    {
                        "started_at": started,
                        "finished_at": time.time(),
                        "native_pending_before": pending_before,
                        "native_pending_after": not work.done(),
                        "response": parsed.model_dump(mode="json"),
                    }
                )
            await work
            overlapping = sum(
                row["native_pending_before"] and row["native_pending_after"]
                for row in report["decisions"]
            )
            assert overlapping >= 8, overlapping
            report["decisions_completed_during_native_stream"] = overlapping
            report["passed"] = True
        except BaseException as error:
            report["passed"] = False
            report["failure"] = {"type": type(error).__name__, "message": str(error)[:2000]}
            raise
        finally:
            if work is not None and not work.done():
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
            report["finished_at"] = time.time()
            with args.output.open("x") as file:
                json.dump(report, file, indent=2)
                file.write("\n")
            print(
                json.dumps(
                    {
                        "passed": report.get("passed"),
                        "decisions": len(report["decisions"]),
                        "overlap": report.get("decisions_completed_during_native_stream"),
                    }
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
