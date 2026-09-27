"""Live mode binding and multi-case plugin/native readout corroboration."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx

from jev_runtime.backends.base import ScoreInput
from jev_runtime.backends.sglang import SGLangHTTP, generate_payload, parse_sglang
from jev_runtime.backends.vllm import VLLMHTTP
from jev_runtime.config import build_runtime, load_settings
from jev_runtime.errors import JevError
from jev_runtime.schema import Bundle, ModelIdentity
from tests.integration.numerical_suite import save, scores, sha


async def verify(run: Path, engine_report: Path, output: Path):
    if await asyncio.to_thread(output.exists):
        raise ValueError("Choose a fresh output")
    settings = load_settings(run / "config.json")
    keys = json.loads((run / "keys.json").read_text())
    captured = json.loads(await asyncio.to_thread(engine_report.read_text))
    assert captured["complete"] and len(captured["cases"]) == 32
    report = {
        "started_at": time.time(),
        "script_sha256": sha(Path(__file__)),
        "engine_report_sha256": sha(engine_report),
        "cases": [],
        "complete": False,
    }
    try:
        async with httpx.AsyncClient(
            base_url=settings.engine_url,
            timeout=60,
            trust_env=False,
            headers={"Authorization": "Bearer " + keys["api"]},
        ) as client:
            admin = {"Authorization": "Bearer " + keys["admin"]}
            response = await client.get("/plugins/jev-runtime/admin/profile", headers=admin)
            response.raise_for_status()
            profile = response.json()
            model = ModelIdentity.model_validate(profile["model"])
            assert profile["model"] == captured["model"]
            assert profile["capabilities"]["batch_invariant"] is settings.batch_invariant
            assert bool(model.batch_invariant) is settings.batch_invariant
            report["model"] = profile["model"]
            report["capabilities"] = profile["capabilities"]
            wrong = Bundle(
                id="mode-reject-" + uuid.uuid4().hex,
                version=1,
                model=model.model_copy(
                    update={"batch_invariant": None if settings.batch_invariant else True}
                ),
            )
            response = await client.post(
                "/plugins/jev-runtime/admin/bundles",
                headers=admin,
                json=wrong.model_dump(mode="json"),
            )
            response.raise_for_status()
            response = await client.post(
                "/plugins/jev-runtime/admin/bundles/prepare",
                headers=admin,
                json={"reference": wrong.reference},
            )
            report["wrong_bundle"] = {"status": response.status_code, "body": response.json()}
            assert (
                response.status_code == 409 and response.json()["error"]["code"] == "model_mismatch"
            )
            response = await client.post(
                "/plugins/jev-runtime/admin/bundles/retire",
                headers=admin,
                json={"reference": wrong.reference},
            )
            response.raise_for_status()
            for case in captured["cases"]:
                wire = {**case["sequence"], "request_id": "mode-" + uuid.uuid4().hex}
                response = await client.post("/plugins/jev-runtime/v1/scores", json=wire)
                response.raise_for_status()
                plugin = response.json()
                assert plugin["request_id"] == wire["request_id"] and plugin["raw_logprobs"]
                scores(plugin["logprobs"], len(wire["label_ids"]))
                if settings.backend == "vllm":
                    response = await client.post(
                        "/v1/completions",
                        json={
                            "model": settings.model_id,
                            "prompt": wire["input_ids"],
                            "max_tokens": 1,
                            "temperature": 0,
                            "logprobs": len(wire["label_ids"]),
                            "logprob_token_ids": wire["label_ids"],
                            "return_tokens_as_token_ids": True,
                        },
                    )
                    response.raise_for_status()
                    value = response.json()
                    assert value["usage"]["completion_tokens"] == 1
                    positions = value["choices"][0]["logprobs"]["top_logprobs"]
                    assert len(positions) == 1
                    native = [positions[0][f"token_id:{token}"] for token in wire["label_ids"]]
                else:
                    seq = ScoreInput(**{**wire, "request_id": "native-" + uuid.uuid4().hex})
                    response = await client.post("/generate", json=generate_payload(seq))
                    response.raise_for_status()
                    native = list(parse_sglang(response.json(), seq).logprobs)
                scores(native, len(wire["label_ids"]))
                error = max(abs(a - b) for a, b in zip(plugin["logprobs"], native, strict=True))
                report["cases"].append(
                    {
                        "id": case["id"],
                        "plugin": plugin["logprobs"],
                        "native": native,
                        "max_absolute_error": error,
                        "passed": error < 1e-4,
                    }
                )
        backend = (
            SGLangHTTP(settings.engine_url, settings.model_id, keys["api"])
            if settings.backend == "sglang"
            else VLLMHTTP(settings.engine_url, keys["api"])
        )
        wrong_settings = settings.model_copy(
            update={
                "batch_invariant": not settings.batch_invariant,
                "bootstrap_alias": None,
                "registry_path": str(run / "mode-reject.db"),
            }
        )
        runtime = await build_runtime(wrong_settings, native_backend=backend)
        try:
            try:
                await runtime.start()
            except JevError as exc:
                report["wrong_attached_runtime"] = {"code": exc.code}
                assert exc.code == "engine_execution_mismatch"
            else:
                raise AssertionError("Mismatched execution mode reached serving")
        finally:
            await runtime.close()
        report["complete"] = True
        report["parity_passed"] = all(x["passed"] for x in report["cases"])
    except BaseException as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["finished_at"] = time.time()
        save(output, report)
    print(json.dumps({"complete": report["complete"], "parity_passed": report["parity_passed"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(verify(args.run_dir, args.engine_report, args.output))
