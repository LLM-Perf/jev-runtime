"""Opt-in real-engine certification. Never starts or stops an engine.

Run with the target environment's Python and a dsw_service.py run directory.
Credentials are read in memory and excluded from the report. Assertions remain
strict: a response with HTTP 200 but invalid semantics fails the run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx

from jev_runtime.backends.vllm import VLLMHTTP
from jev_runtime.config import load_compiler, load_settings, model_identity
from jev_runtime.schema import Bundle, DecisionRequest, DecisionResponse, TemplateSpec


async def certify(run_dir: Path, output: Path, switches: int) -> dict:
    settings = load_settings(run_dir / "config.json")
    keys = json.loads((run_dir / "keys.json").read_text())
    compiler = await asyncio.to_thread(load_compiler, settings)
    prefix = "/plugins/jev-runtime"
    report = {
        "schema_version": 1,
        "started_at": time.time(),
        "qualification": "colocated-functional-test; not a performance or quality benchmark",
        "model": model_identity(settings, compiler).model_dump(mode="json"),
        "checks": {},
    }
    checks = report["checks"]
    case_id = "cert-" + uuid.uuid4().hex[:12]
    payload = {
        "model": settings.bootstrap_alias,
        "input": {"text": "I was charged twice and would like a refund."},
        "questions": [
            {
                "id": "intent",
                "type": "choice",
                "instruction": "Select the customer's primary intent.",
                "options": [
                    {"id": "billing", "description": "Payments, charges, or refunds"},
                    {"id": "technical", "description": "A technical fault in the product"},
                    {"id": "other", "description": "A different topic"},
                ],
            },
            {"id": "refund", "type": "boolean", "instruction": "Is a refund explicitly requested?"},
            {
                "id": "urgency",
                "type": "score",
                "instruction": "Rate the urgency of the message.",
                "options": [
                    {"id": "low", "description": "Low urgency", "value": 0},
                    {"id": "medium", "description": "Medium urgency", "value": 5},
                    {"id": "high", "description": "High urgency", "value": 10},
                ],
            },
            {
                "id": "routing",
                "type": "rank",
                "instruction": "Rank the teams by relevance.",
                "options": [
                    {"id": "payments", "description": "Payment support"},
                    {"id": "engineering", "description": "Engineering support"},
                    {"id": "sales", "description": "Sales"},
                ],
            },
        ],
    }
    timeout = httpx.Timeout(120, connect=10)
    async with (
        httpx.AsyncClient(
            base_url=settings.engine_url,
            timeout=timeout,
            headers={"Authorization": "Bearer " + keys["api"]},
        ) as client,
        httpx.AsyncClient(
            base_url=settings.engine_url,
            timeout=timeout,
            headers={"Authorization": "Bearer " + keys["admin"]},
        ) as admin,
    ):

        async def call(path, body=None, *, management=False, expected=200):
            transport = admin if management else client
            response = await (
                transport.get(prefix + path)
                if body is None
                else transport.post(prefix + path, json=body)
            )
            assert response.status_code == expected, (
                path,
                response.status_code,
                response.text[:2000],
            )
            return response.json()

        try:
            checks["ready"] = await call("/ready")
            report["capabilities"] = await call("/v1/capabilities")
            denied = await client.get(
                prefix + "/ready", headers={"Authorization": "Bearer invalid"}
            )
            assert denied.status_code == 401
            denied = await client.get(prefix + "/admin/bundles")
            assert denied.status_code == 401
            checks["auth_separation"] = True

            response = DecisionResponse.model_validate(await call("/v1/decisions", payload))
            assert response.status == "completed" and len(response.answers) == 4
            for answer in response.answers.values():
                assert answer.status in {"answered", "abstained"}
                assert answer.calibration_status == "uncalibrated"
                assert all(math.isfinite(p) and 0 <= p <= 1 for p in answer.probabilities.values())
                assert abs(sum(answer.probabilities.values()) - 1) < 1e-6
            assert isinstance(response.answers["refund"].value, bool)
            assert 0 <= response.answers["urgency"].value <= 10
            assert set(response.answers["routing"].value) == {"payments", "engineering", "sales"}
            assert response.usage.successful_questions == 4
            checks["four_types"] = response.model_dump(mode="json")

            chat = await client.post(
                "/v1/chat/completions",
                json={
                    "model": settings.model_id,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 8,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            assert chat.status_code == 200, chat.text[:2000]
            chat_data = chat.json()
            assert chat_data["choices"][0]["finish_reason"] in {"stop", "length"}
            assert chat_data["usage"]["completion_tokens"] > 0
            checks["native_chat_coexists"] = {"usage": chat_data["usage"]}

            invalid = {**payload, "bundle": "nonexistent@999"}
            error = await call("/v1/decisions", invalid, expected=409)
            assert error["error"]["code"] == "bundle_not_active"
            checks["invalid_bundle_explicit"] = True

            bundle = Bundle(id=case_id, version=1, model=model_identity(settings, compiler))
            request = DecisionRequest.model_validate(payload)
            compiled = compiler.compile(request.input.text, request.questions[0], bundle, case_id)
            seq = compiled.sequences[0]
            raw = await call("/v1/scores", asdict(seq))
            assert len(raw["logprobs"]) == len(seq.label_ids)
            if settings.backend == "vllm":
                baseline = await client.post(
                    "/v1/completions",
                    json={
                        "model": settings.model_id,
                        "prompt": list(seq.input_ids),
                        "max_tokens": 1,
                        "temperature": 0,
                        "logprobs": len(seq.label_ids),
                        "logprob_token_ids": list(seq.label_ids),
                        "return_tokens_as_token_ids": True,
                    },
                )
                assert baseline.status_code == 200, baseline.text[:2000]
                data = baseline.json()
                probs = data["choices"][0]["logprobs"]["top_logprobs"][0]
                expected = [probs[f"token_id:{token}"] for token in seq.label_ids]
                differences = [abs(a - b) for a, b in zip(raw["logprobs"], expected, strict=True)]
                assert max(differences) < 1e-4, differences
                adapter = VLLMHTTP(settings.engine_url, keys["api"])
                try:
                    attached = await adapter.score(seq)
                    assert (
                        max(abs(a - b) for a, b in zip(attached.logprobs, expected, strict=True))
                        < 1e-4
                    )
                finally:
                    await adapter.close()
                checks["native_attach_logprob_parity"] = {"max_abs_error": max(differences)}
            else:
                from jev_runtime.backends.sglang import SGLangHTTP

                adapter = SGLangHTTP(settings.engine_url, settings.model_id)
                try:
                    attached = await adapter.score(seq)
                    differences = [
                        abs(a - b) for a, b in zip(raw["logprobs"], attached.logprobs, strict=True)
                    ]
                    assert max(differences) < 1e-4, differences
                    checks["native_attach_logprob_parity"] = {"max_abs_error": max(differences)}
                finally:
                    await adapter.close()
            report["reference_logits"] = {
                "input_ids": seq.input_ids,
                "label_ids": seq.label_ids,
                "logprobs": raw["logprobs"],
            }

            independent = bundle.model_copy(
                update={"version": 2, "template": TemplateSpec(mode="independent-candidate")}
            )
            for item in (bundle, independent):
                await call("/admin/bundles", item.model_dump(mode="json"), management=True)
                prepared = await call(
                    "/admin/bundles/prepare", {"reference": item.reference}, management=True
                )
                assert prepared["state"] == "READY"
            activated = await call(
                "/admin/bundles/activate",
                {
                    "alias": case_id,
                    "reference": independent.reference,
                    "expected_generation": 0,
                },
                management=True,
            )
            indep_result = await call("/v1/decisions", {**payload, "model": case_id})
            assert indep_result["answers"]["intent"]["support"] is not None
            assert indep_result["usage"]["scoring_sequences"] == 10
            checks["independent_candidate"] = indep_result

            generation_map = {activated["generation"]: independent.reference}
            generation = activated["generation"]
            seen = []
            stop = asyncio.Event()
            traffic_payload = {**payload, "model": case_id, "questions": payload["questions"][:1]}

            async def traffic():
                while not stop.is_set():
                    result = await call("/v1/decisions", traffic_payload)
                    assert result["status"] == "completed"
                    seen.append((result["generation"], result["bundle"], result["bundle_digest"]))

            workers = [asyncio.create_task(traffic()) for _ in range(4)]
            try:
                for index in range(switches):
                    target = bundle if index % 2 == 0 else independent
                    activated = await call(
                        "/admin/bundles/activate",
                        {
                            "alias": case_id,
                            "reference": target.reference,
                            "expected_generation": generation,
                        },
                        management=True,
                    )
                    generation = activated["generation"]
                    generation_map[generation] = target.reference
                    await asyncio.sleep(0.005)
            finally:
                stop.set()
                await asyncio.gather(*workers)
            digests = {item.reference: item.digest for item in (bundle, independent)}
            assert seen and all(
                generation_map[g] == ref and digests[ref] == digest for g, ref, digest in seen
            )
            assert len({ref for _, ref, _ in seen}) == 2 or switches < 2
            checks["hot_switch_under_traffic"] = {
                "switches": switches,
                "strict_success_requests": len(seen),
                "versions_seen": sorted({ref for _, ref, _ in seen}),
                "mixed_bundle_responses": 0,
            }
            conflict = await call(
                "/admin/bundles/activate",
                {
                    "alias": case_id,
                    "reference": bundle.reference,
                    "expected_generation": 0,
                },
                management=True,
                expected=409,
            )
            assert conflict["error"]["code"] == "generation_conflict"
            await call(
                "/admin/bundles/disable",
                {"alias": case_id, "expected_generation": generation},
                management=True,
            )
            await call("/v1/decisions", traffic_payload, expected=503)
            for item in (bundle, independent):
                retired = await call(
                    "/admin/bundles/retire", {"reference": item.reference}, management=True
                )
                assert retired["state"] == "RETIRED" and retired["inflight"] == 0
            checks["disable_drain_retire_and_cas"] = True
            report["passed"] = True
        except BaseException as exc:
            report["passed"] = False
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
            raise
        finally:
            report["finished_at"] = time.time()
            output.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                output.write_text, json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {"passed": report.get("passed"), "checks": list(checks), "output": str(output)}
                )
            )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--switches", type=int, default=1000)
    args = parser.parse_args()
    asyncio.run(certify(args.run_dir, args.output, args.switches))
