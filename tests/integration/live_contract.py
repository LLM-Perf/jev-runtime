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
import os
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import httpx

from jev_runtime.backends.base import ScoreInput
from jev_runtime.backends.vllm import VLLMHTTP
from jev_runtime.config import load_settings
from jev_runtime.schema import Bundle, DecisionResponse, Policy, TemplateSpec
from tests.integration.hot_switch_traffic import exercise


def verify_four_types(response: DecisionResponse, *, require_selected: bool = False) -> None:
    """A valid default-policy abstention is not an unsupported output type."""
    assert response.status == "completed" and response.usage.successful_questions == 4
    expected = {"intent": "choice", "refund": "boolean", "urgency": "score", "routing": "rank"}
    assert {key: answer.type for key, answer in response.answers.items()} == expected
    for answer in response.answers.values():
        assert answer.status in {"answered", "abstained"}
        assert answer.calibration_status == "uncalibrated"
        if require_selected:
            assert answer.status == "answered"
        if answer.status == "abstained":
            assert answer.abstained and answer.reason and answer.value is None
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in answer.probabilities.values())
        assert abs(sum(answer.probabilities.values()) - 1) < 1e-6
    if response.answers["refund"].status == "answered":
        assert type(response.answers["refund"].value) is bool
    if response.answers["urgency"].status == "answered":
        assert 0 <= response.answers["urgency"].value <= 10
    if response.answers["routing"].status == "answered":
        assert set(response.answers["routing"].value) == {"payments", "engineering", "sales"}


async def certify(
    run_dir: Path,
    output: Path,
    switches: int,
    source_commit: str,
    runtime_source_commit: str,
    *,
    minimum_requests: int = 10000,
    concurrency: int = 4,
    traffic_timeout: float = 1200,
) -> dict:
    if await asyncio.to_thread(output.exists):
        raise ValueError("Choose a new output path to retain previous attempts")
    settings = load_settings(run_dir / "config.json")
    keys = (
        json.loads((run_dir / "keys.json").read_text())
        if (run_dir / "keys.json").exists()
        else {"api": os.environ[settings.api_key_env], "admin": os.environ[settings.admin_key_env]}
    )
    prefix = "/plugins/jev-runtime"
    report = {
        "schema_version": 1,
        "source_commit": source_commit,
        "runtime_source_commit": runtime_source_commit,
        "started_at": time.time(),
        "qualification": "colocated-functional-test; not a performance or quality benchmark",
        "configured_model": {"id": settings.model_id, "revision": settings.model_revision},
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
            profile = await call("/admin/profile", management=True)
            report["serving_profile"] = profile
            assert profile["model"]["id"] == settings.model_id
            assert profile["model"]["revision"] == settings.model_revision
            report["model"] = profile["model"]

            response = DecisionResponse.model_validate(await call("/v1/decisions", payload))
            checks["four_types"] = response.model_dump(mode="json")
            verify_four_types(response)

            if settings.backend == "tokenspeed":
                checks["native_chat_coexists"] = {
                    "status": "not_applicable",
                    "reason": "standalone typed-scoring launcher",
                }
            else:
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

            # Exercise selected values independently of legitimate default-policy
            # ties. This is an explicit test bundle, never a production fallback.
            bundle = Bundle(
                id=case_id,
                version=1,
                model=profile["model"],
                policy=Policy(tie="first"),
            )
            preview = await call(
                "/admin/compile",
                {**payload, "questions": payload["questions"][:1]},
                management=True,
            )
            assert preview["bundle"]["model"] == profile["model"]
            assert len(preview["sequences"]) == (3 if settings.backend == "tokenspeed" else 1)
            sequence = preview["sequences"][0]
            seq = ScoreInput(
                **{
                    **sequence,
                    "request_id": uuid.uuid4().hex,
                    "input_ids": tuple(sequence["input_ids"]),
                    "label_ids": tuple(sequence["label_ids"]),
                }
            )
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
            elif settings.backend == "tokenspeed":
                from jev_runtime.backends.remote import ScoringHTTP

                adapter = ScoringHTTP(settings.engine_url, "tokenspeed", keys["api"])
                try:
                    capabilities = await adapter.probe()
                    assert capabilities.label_scoring == "single"
                    attached = await adapter.score(replace(seq, request_id=uuid.uuid4().hex))
                    differences = [
                        abs(a - b) for a, b in zip(raw["logprobs"], attached.logprobs, strict=True)
                    ]
                    assert max(differences) < 1e-4, differences
                    await adapter.cancel(attached.request_id)
                    checks["bridge_contract_parity"] = {
                        "max_abs_error": max(differences),
                        "qualification": "same backend, two clients; not an independent reference",
                    }
                finally:
                    await adapter.close()
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
                    "reference": bundle.reference,
                    "expected_generation": 0,
                },
                management=True,
            )
            selected = DecisionResponse.model_validate(
                await call("/v1/decisions", {**payload, "model": case_id})
            )
            checks["four_types_explicit_first_policy"] = selected.model_dump(mode="json")
            verify_four_types(selected, require_selected=True)
            activated = await call(
                "/admin/bundles/activate",
                {
                    "alias": case_id,
                    "reference": independent.reference,
                    "expected_generation": activated["generation"],
                },
                management=True,
            )
            indep_result = DecisionResponse.model_validate(
                await call("/v1/decisions", {**payload, "model": case_id})
            ).model_dump(mode="json")
            assert indep_result["answers"]["intent"]["support"] is not None
            assert indep_result["usage"]["scoring_sequences"] == (
                20 if settings.backend == "tokenspeed" else 10
            )
            checks["independent_candidate"] = indep_result

            traffic_payload = {**payload, "model": case_id, "questions": payload["questions"][:1]}
            generation = await exercise(
                call,
                payload={**payload, "model": case_id},
                bundles=(bundle, independent),
                generation=activated["generation"],
                engine=settings.backend,
                minimum_requests=minimum_requests,
                minimum_switches=switches,
                concurrency=concurrency,
                deadline_seconds=traffic_timeout,
                output=output.with_name("traffic-responses.jsonl.gz"),
                checks=checks,
            )
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
    parser.add_argument("--minimum-requests", type=int, default=10000)
    parser.add_argument("--traffic-concurrency", type=int, default=4)
    parser.add_argument("--traffic-timeout", type=float, default=1200)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-source-commit", required=True)
    args = parser.parse_args()
    if (
        min(args.switches, args.minimum_requests, args.traffic_concurrency, args.traffic_timeout)
        <= 0
    ):
        parser.error("Switches, request minimum, concurrency and timeout must be positive")
    asyncio.run(
        certify(
            args.run_dir,
            args.output,
            args.switches,
            args.source_commit,
            args.runtime_source_commit,
            minimum_requests=args.minimum_requests,
            concurrency=args.traffic_concurrency,
            traffic_timeout=args.traffic_timeout,
        )
    )
