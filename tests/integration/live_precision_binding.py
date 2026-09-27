"""Check precision binding against an owned live engine without changing its precision."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx

from jev_runtime.backends.sglang import SGLangHTTP
from jev_runtime.backends.vllm import VLLMHTTP
from jev_runtime.config import build_runtime, load_settings
from jev_runtime.errors import JevError
from jev_runtime.schema import Bundle, ModelIdentity


async def verify(args):
    if args.output.exists():
        raise ValueError("Choose a new output path to preserve attempts")
    settings = load_settings(args.run_dir / "config.json")
    keys = json.loads((args.run_dir / "keys.json").read_text())
    report = {
        "harness_source_commit": args.source_commit,
        "runtime_source_commit": args.runtime_source_commit,
        "started_at": time.time(),
        "qualification": "Targeted precision identity and rejection checks; not numerical accuracy",
        "checks": {},
    }
    checks = report["checks"]
    try:
        async with httpx.AsyncClient(
            base_url=settings.engine_url + "/plugins/jev-runtime",
            headers={"Authorization": "Bearer " + keys["admin"]},
            timeout=30,
        ) as client:
            response = await client.get("/admin/profile")
            response.raise_for_status()
            profile = response.json()
            model = ModelIdentity.model_validate(profile["model"])
            assert model.readout_dtype in {"bfloat16", "float32"}
            assert profile["capabilities"]["readout_dtype"] == model.readout_dtype
            assert profile["capabilities"]["model_dtype"] == model.dtype == "bfloat16"
            report["model"] = model.model_dump(mode="json")
            report["capabilities"] = profile["capabilities"]
            other = "float32" if model.readout_dtype == "bfloat16" else "bfloat16"
            for name, readout in (("legacy", None), ("wrong", other)):
                bundle = Bundle(
                    id="precision-" + uuid.uuid4().hex,
                    version=1,
                    model=model.model_copy(update={"readout_dtype": readout}),
                )
                response = await client.post("/admin/bundles", json=bundle.model_dump(mode="json"))
                response.raise_for_status()
                rejected = await client.post(
                    "/admin/bundles/prepare", json={"reference": bundle.reference}
                )
                checks[name + "_bundle"] = {
                    "status": rejected.status_code,
                    "body": rejected.json(),
                    "reference": bundle.reference,
                }
                assert (
                    rejected.status_code == 409
                    and rejected.json()["error"]["code"] == "model_mismatch"
                )
                retired = await client.post(
                    "/admin/bundles/retire", json={"reference": bundle.reference}
                )
                retired.raise_for_status()
                assert retired.json()["inflight"] == 0
            listing = await client.get("/admin/bundles")
            listing.raise_for_status()
            assert not listing.json()["leases"]
            checks["zero_leases_after_rejections"] = True
        # A fresh HTTP-attached runtime must compare its declaration with the
        # real engine before it can publish or score. Never mutate the host.
        bad_settings = settings.model_copy(
            update={
                "readout_dtype": other,
                "bootstrap_alias": None,
                "registry_path": str(
                    args.run_dir / ("precision-reject-" + uuid.uuid4().hex + ".db")
                ),
            }
        )
        backend = (
            SGLangHTTP(settings.engine_url, settings.model_id, keys["api"])
            if settings.backend == "sglang"
            else VLLMHTTP(settings.engine_url, keys["api"])
        )
        runtime = await build_runtime(bad_settings, native_backend=backend)
        try:
            try:
                await runtime.start()
            except JevError as exc:
                checks["mismatched_attached_runtime"] = {"code": exc.code, "message": str(exc)}
                assert exc.code == "engine_precision_mismatch"
            else:
                raise AssertionError("Mismatched readout precision reached serving")
        finally:
            await runtime.close()
        report["passed"] = True
    except BaseException as exc:
        report["passed"] = False
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:3000]}
        raise
    finally:
        report["finished_at"] = time.time()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": report.get("passed"), "output": str(args.output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-source-commit", required=True)
    asyncio.run(verify(parser.parse_args()))
