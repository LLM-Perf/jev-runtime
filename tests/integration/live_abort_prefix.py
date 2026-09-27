"""Exercise SGLang prefix cancellation with IDs exported by the actual serving compiler.

Only a live task-owned native service launched by dsw_service.py is accepted.
Scoring prompts are extended to 128 generated tokens solely to create a bounded,
observable in-flight abort window. This is not decision quality or latency evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx

from jev_runtime.config import load_settings
from jev_runtime.schema import Bundle, TemplateSpec


def validate_owner(record):
    identity = record["identity"]
    stat = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()
    assert stat[0] != "Z" and stat[19] == identity["start_ticks"]
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == identity["boot_id"]
    assert record["engine"] == "sglang" and record["mode"] == "native-plugin"


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


async def run(args):
    if args.output.exists():
        raise ValueError("Choose a new evidence path")
    record = json.loads((args.run_dir / "process.json").read_text())
    await asyncio.to_thread(validate_owner, record)
    settings = load_settings(args.run_dir / "config.json")
    assert settings.backend == "sglang"
    url = f"http://127.0.0.1:{record['port']}"
    assert settings.engine_url == url
    keys = json.loads((args.run_dir / "keys.json").read_text())
    report = {
        "source_commit": args.source_commit,
        "runtime_source_commit": args.runtime_source_commit,
        "service_process_identity": record["identity"],
        "qualification": __doc__,
        "started_at": time.time(),
        "passed": False,
    }
    alias = "abort-check-" + uuid.uuid4().hex
    uploaded = activated = False
    ids, tasks, last = [], [], {}
    events = [asyncio.Event(), asyncio.Event()]
    async with (
        httpx.AsyncClient(base_url=url, timeout=30) as native,
        httpx.AsyncClient(
            base_url=url + "/plugins/jev-runtime",
            headers={"Authorization": "Bearer " + keys["admin"]},
            timeout=30,
        ) as admin,
    ):

        async def manage(path, body=None):
            response = await (admin.get(path) if body is None else admin.post(path, json=body))
            response.raise_for_status()
            return response.json()

        try:
            profile = await manage("/admin/profile")
            report["serving_profile"] = profile
            bundle = Bundle(
                id=alias,
                version=1,
                model=profile["model"],
                template=TemplateSpec(mode="independent-candidate"),
            )
            await manage("/admin/bundles", bundle.model_dump(mode="json"))
            uploaded = True
            await manage("/admin/bundles/prepare", {"reference": bundle.reference})
            await manage(
                "/admin/bundles/activate",
                {
                    "alias": alias,
                    "reference": bundle.reference,
                    "expected_generation": 0,
                },
            )
            activated = True
            preview = await manage(
                "/admin/compile",
                {
                    "model": alias,
                    "input": {"text": "A short example."},
                    "questions": [
                        {
                            "id": "a",
                            "type": "choice",
                            "instruction": "Choose a category.",
                            "options": [
                                {"id": "x", "description": "X"},
                                {"id": "y", "description": "Y"},
                            ],
                        },
                        {"id": "a.1", "type": "boolean", "instruction": "Is this an example?"},
                    ],
                },
            )
            victim = next(
                s
                for s in preview["sequences"]
                if s["question_id"] == "a" and s["candidate_id"] == "y"
            )
            sibling = next(s for s in preview["sequences"] if s["question_id"] == "a.1")
            branches = [victim, sibling]
            # Prefix both exported IDs with the same unique namespace. Their
            # relative prefix relationship is preserved without guessing format.
            nonce = "isolation-" + uuid.uuid4().hex
            ids = [nonce + "." + s["request_id"] for s in branches]
            report["compiled_branches"] = branches
            report["submitted_ids"] = ids
            report["sibling_matches_victim_prefix"] = ids[1].startswith(ids[0])

            async def consume(index):
                async with native.stream(
                    "POST",
                    "/generate",
                    json={
                        "input_ids": branches[index]["input_ids"],
                        "rid": ids[index],
                        "sampling_params": {
                            "max_new_tokens": 128,
                            "temperature": 0,
                            "ignore_eos": True,
                        },
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        meta = json.loads(payload).get("meta_info", {})
                        last[str(index)] = {
                            key: meta.get(key)
                            for key in ("id", "finish_reason", "completion_tokens")
                        }
                        if meta.get("completion_tokens", 0) > 0:
                            events[index].set()

            tasks = [asyncio.create_task(consume(i)) for i in range(2)]
            async with asyncio.timeout(30):
                await asyncio.gather(*(event.wait() for event in events))
                assert not any(task.done() for task in tasks), "Abort window was not observed"
                response = await native.post(
                    "/abort_request", json={"rid": ids[0], "abort_all": False}
                )
                response.raise_for_status()
                report["abort_http_status"] = response.status_code
                await asyncio.gather(*tasks)
            assert last["0"]["finish_reason"]["type"] == "abort"
            assert last["1"]["finish_reason"]["type"] == "length"
            assert last["1"]["completion_tokens"] == 128
            assert not report["sibling_matches_victim_prefix"]
            report["passed"] = True
        except BaseException as exc:
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            raise
        finally:
            report["responses"] = last
            for rid in ids:
                try:
                    response = await native.post(
                        "/abort_request", json={"rid": rid, "abort_all": False}
                    )
                    response.raise_for_status()
                except Exception as exc:
                    report.setdefault("cleanup_errors", []).append(type(exc).__name__)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                if activated:
                    await manage(
                        "/admin/bundles/disable", {"alias": alias, "expected_generation": 1}
                    )
                if uploaded:
                    retired = await manage("/admin/bundles/retire", {"reference": bundle.reference})
                    assert retired["state"] == "RETIRED" and retired["inflight"] == 0
                    report["temporary_bundle_retired"] = True
                assert not (await manage("/admin/bundles"))["leases"]
                report["registry_leases_after"] = 0
            except Exception as exc:
                report.setdefault("cleanup_errors", []).append(type(exc).__name__)
            if report.get("cleanup_errors"):
                report["passed"] = False
            report["finished_at"] = time.time()
            await asyncio.to_thread(save, args.output, report)
            print(json.dumps({"passed": report["passed"], "responses": last}), flush=True)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-source-commit", required=True)
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
