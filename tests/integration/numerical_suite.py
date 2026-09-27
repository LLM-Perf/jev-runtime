"""Multi-input numerical diagnostics; never changes a release tolerance.

Collection requires an explicitly owned, isolated dsw_service run: it flushes
that engine's local prefix cache. Reference forwards run after serving stops.
Synthetic inputs test numerical behavior, not business accuracy or calibration.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
import uuid
from pathlib import Path

from jev_runtime.schema import DecisionRequest, content_digest

STATES = ("reset_serial", "repeat_serial", "concurrent_reversed")
ATOL = 0.15  # Existing development tolerance, not a newly certified error budget.


def corpus() -> list[dict]:
    texts = [
        "I paid twice. Please refund the duplicate payment.",
        "The application crashes at startup. My payment was correct.",
        "Thanks, everything works. I am only asking about next year's price.",
        "Refund or repair? Both would help; I have not decided.",
        "我被重复扣款了，请退回多收的费用。",
        "应用无法打开，付款正常，不需要退款。",
        "谢谢，问题解决了。我想了解明年的价格。",
        "退款和修复都可以，我还没有决定。",
    ]
    cases = []
    for index, text in enumerate(texts):
        for kind in ("boolean", "choice4", "choice8", "ambiguous4"):
            question = {
                "id": "q",
                "type": "boolean" if kind == "boolean" else "choice",
                "instruction": (
                    "Is a refund explicitly requested?"
                    if kind == "boolean"
                    else "Select the best matching category from the supplied evidence."
                ),
            }
            if kind != "boolean":
                labels = ["Billing and refunds", "Technical repair", "Prices and sales", "Other"]
                if kind == "choice8":
                    labels += ["Account access", "Delivery", "Cancellation", "Positive feedback"]
                if kind == "ambiguous4":
                    # Identical descriptions challenge label bias; an actual near
                    # tie is determined from reference scores, never assumed.
                    labels = ["An equally applicable general support category"] * 4
                question["options"] = [
                    {"id": f"c{i}", "description": label} for i, label in enumerate(labels)
                ]
            payload = {
                "model": "decision-model",
                "input": {"text": text},
                "questions": [question],
            }
            DecisionRequest.model_validate(payload)
            cases.append({"id": f"text{index}-{kind}", "kind": kind, "request": payload})
    return cases


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    with path.open("x") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def scores(value: list, size: int) -> list[float]:
    if len(value) != size or size < 2:
        raise ValueError("Incomplete label distribution")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x > 1e-6 for x in value):
        raise ValueError("Expected finite, nonpositive full-vocabulary logprobs")
    return list(value)


def compare(actual: list, expected: list, atol: float = ATOL) -> dict:
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("Invalid tolerance")
    expected = scores(expected, len(expected))
    actual = scores(actual, len(expected))

    def normalize(values):
        unnormalized = [math.exp(x - max(values)) for x in values]
        return [x / sum(unnormalized) for x in unnormalized]

    left, right = normalize(actual), normalize(expected)
    order = sorted(range(len(expected)), key=expected.__getitem__, reverse=True)
    actual_top = max(range(len(actual)), key=actual.__getitem__)
    error = max(abs(a - b) for a, b in zip(actual, expected, strict=True))
    gap = expected[order[0]] - expected[order[1]]
    return {
        "max_absolute_logprob_error": error,
        "max_absolute_conditional_probability_error": max(
            abs(a - b) for a, b in zip(left, right, strict=True)
        ),
        "conditional_probability_l1": sum(abs(a - b) for a, b in zip(left, right, strict=True)),
        "actual_label_mass": sum(math.exp(x) for x in actual),
        "reference_label_mass": sum(math.exp(x) for x in expected),
        "reference_top_gap": gap,
        "near_tie": gap <= 2 * atol,
        "top_label_equal": actual_top == order[0],
        "passed": error <= atol and actual_top == order[0],
    }


def summarize(engine: dict, reference: dict) -> dict:
    """Reject incomplete/reordered/tampered inputs before counting comparisons."""
    if engine["corpus_digest"] != content_digest(corpus()):
        raise ValueError("Unexpected corpus")
    if not engine.get("complete") or not reference.get("complete"):
        raise ValueError("Incomplete attempt")
    if engine["model"] != reference["model"]:
        raise ValueError("Model identity mismatch")
    if content_digest(engine["cases"]) != reference["cases_digest"]:
        raise ValueError("Reference input or label ordering mismatch")
    expected_ids = [x["id"] for x in corpus()]
    if [x["id"] for x in engine["cases"]] != expected_ids:
        raise ValueError("Missing, duplicate or reordered cases")
    if set(engine["observations"]) != set(STATES):
        raise ValueError("Incomplete execution states")
    ref_rows = reference["rows"]
    if [x["id"] for x in ref_rows] != expected_ids:
        raise ValueError("Missing, duplicate or reordered reference rows")
    reference_by_id = {x["id"]: x for x in ref_rows}
    rows = []
    for state in STATES:
        observed = engine["observations"][state]
        if [x["id"] for x in observed] != expected_ids:
            raise ValueError("Incomplete execution-state denominator")
        for case, observation in zip(engine["cases"], observed, strict=True):
            ref = reference_by_id[case["id"]]
            if ref["sequence_digest"] != content_digest(case["sequence"]):
                raise ValueError("Reference sequence mismatch")
            scores(observation["logprobs"], len(case["sequence"]["label_ids"]))
            rows.append(
                {
                    "id": case["id"],
                    "state": state,
                    **compare(observation["logprobs"], ref["logprobs"]),
                }
            )
    state_differences = []
    for state in STATES[1:]:
        for base, other in zip(
            engine["observations"][STATES[0]], engine["observations"][state], strict=True
        ):
            state_differences.append(
                {"id": base["id"], "state": state, **compare(other["logprobs"], base["logprobs"])}
            )
    return {
        "scope": "Synthetic multi-input development diagnostics; not release certification",
        "atol": ATOL,
        "case_count": len(expected_ids),
        "comparisons": len(rows),
        "passed_comparisons": sum(row["passed"] for row in rows),
        "near_tie_cases": sum(row["near_tie"] for row in rows if row["state"] == STATES[0]),
        "max_absolute_logprob_error": max(row["max_absolute_logprob_error"] for row in rows),
        "max_absolute_conditional_probability_error": max(
            row["max_absolute_conditional_probability_error"] for row in rows
        ),
        "top_label_mismatches": sum(not row["top_label_equal"] for row in rows),
        "all_comparisons_passed": all(row["passed"] for row in rows),
        "rows": rows,
        "state_differences": state_differences,
        "full_release_gate_passed": False,
    }


async def collect(run: Path, output: Path) -> None:
    import httpx

    from deployment.dsw_service import process_identity

    if await asyncio.to_thread(output.exists):
        raise ValueError("Choose a fresh output to retain attempts")
    record = json.loads((run / "process.json").read_text())
    if record["mode"] != "native-plugin" or record["tensor_parallel_size"] != 1:
        raise ValueError("This diagnostic requires the declared TP1 native profile")
    if process_identity(record["identity"]["pid"]) != record["identity"]:
        raise ValueError("Owned engine identity is not alive")
    config = json.loads((run / "config.json").read_text())
    keys = json.loads((run / "keys.json").read_text())
    prefix = "/plugins/jev-runtime"
    report = {
        "schema_version": 1,
        "corpus_digest": content_digest(corpus()),
        "script_sha256": sha(Path(__file__)),
        "engine": record["engine"],
        "process_identity": record["identity"],
        "started_at": time.time(),
        "cases": [],
        "observations": {},
        "cache_resets": [],
        "sglang_flush_idle_timeout_seconds": 10,
        "concurrency": 4,
        "complete": False,
    }
    async with httpx.AsyncClient(
        base_url=config["engine_url"],
        timeout=60,
        trust_env=False,
        headers={"Authorization": "Bearer " + keys["api"]},
    ) as client:
        try:
            profile = await client.get(
                prefix + "/admin/profile", headers={"Authorization": "Bearer " + keys["admin"]}
            )
            profile.raise_for_status()
            report["profile"] = profile.json()
            report["model"] = report["profile"]["model"]
            for case in corpus():
                response = await client.post(
                    prefix + "/admin/compile",
                    json=case["request"],
                    headers={"Authorization": "Bearer " + keys["admin"]},
                )
                response.raise_for_status()
                preview = response.json()
                if len(preview["sequences"]) != 1:
                    raise ValueError("Expected one joint-label sequence per corpus case")
                sequence = {**preview["sequences"][0], "request_id": "numerical-fixture"}
                report["cases"].append({**case, "sequence": sequence})

            async def reset():
                if process_identity(record["identity"]["pid"]) != record["identity"]:
                    raise ValueError("Owned engine identity changed before cache reset")
                response = await (
                    client.post("/reset_prefix_cache")
                    if record["engine"] == "vllm"
                    # A completed response can precede scheduler quiescence.
                    # Use the engine's bounded idle barrier; never flush live
                    # work forcibly or label a rejected reset as cold input.
                    else client.get("/flush_cache", params={"timeout": 10})
                )
                response.raise_for_status()
                if record["engine"] == "vllm" and response.json().get("success") is not True:
                    raise ValueError("Prefix cache reset was not confirmed")
                report["cache_resets"].append(
                    {"status": response.status_code, "body": response.text}
                )

            async def score(case):
                wire = {**case["sequence"], "request_id": "num-" + uuid.uuid4().hex}
                response = await client.post(prefix + "/v1/scores", json=wire)
                response.raise_for_status()
                data = response.json()
                expected_output = 1 if record["engine"] == "vllm" else 0
                if (
                    data["request_id"] != wire["request_id"]
                    or not data["raw_logprobs"]
                    or data["prompt_tokens"] != len(wire["input_ids"])
                    or data["completion_tokens"] != expected_output
                ):
                    raise ValueError("Scoring response violated position/usage/identity contract")
                scores(data["logprobs"], len(wire["label_ids"]))
                return {"id": case["id"], **data}

            report["observations"][STATES[0]] = []
            report["observations"][STATES[1]] = []
            for case in report["cases"]:
                await reset()
                report["observations"][STATES[0]].append(await score(case))
                report["observations"][STATES[1]].append(await score(case))
            await reset()
            # Reverse order and bounded concurrent submissions. This is client
            # concurrency, not an assertion about the engine's realized batch.
            concurrent = []
            cases = list(reversed(report["cases"]))
            for offset in range(0, len(cases), 4):
                concurrent += await asyncio.gather(*(score(c) for c in cases[offset : offset + 4]))
            by_id = {row["id"]: row for row in concurrent}
            report["observations"][STATES[2]] = [by_id[c["id"]] for c in report["cases"]]
            if process_identity(record["identity"]["pid"]) != record["identity"]:
                raise ValueError("Engine identity changed during collection")
            report["complete"] = True
        except BaseException as exc:
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            report["finished_at"] = time.time()
            save(output, report)


def reference_readout(source: dict, engine: dict) -> str:
    """Bind the reference to the saved identity before loading any model weights."""
    model = engine["model"]
    if (
        not engine["complete"]
        or source["model_id"] != model["id"]
        or source["revision"] != model["revision"]
        or model["dtype"] != "bfloat16"
        or model.get("readout_dtype") not in ("bfloat16", "float32")
        or model["quantization"] is not None
    ):
        raise ValueError("Expected matching unquantized BF16 backbone and explicit BF16/FP32 head")
    return model["readout_dtype"]


def reference(model_path: Path, engine_path: Path, output: Path, attention: str) -> None:
    import os

    import torch
    import transformers

    from tests.integration.reference_device import cuda_budget

    if output.exists():
        raise ValueError("Choose a fresh output to retain attempts")
    engine = json.loads(engine_path.read_text())
    source = json.loads((model_path / "jev-source.json").read_text())
    readout_dtype = reference_readout(source, engine)
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one allocated GPU")
    free, total = torch.cuda.mem_get_info()
    budget = cuda_budget(free, total, 6144, 3072)
    weights = sum(p.stat().st_size for p in model_path.glob("*.safetensors"))
    if weights * 2 > budget:
        raise ValueError("Insufficient bounded resident reference capacity")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(budget / total)
    torch.cuda.reset_peak_memory_stats()
    report = {
        "model": engine["model"],
        "cases_digest": content_digest(engine["cases"]),
        "engine_report_sha256": sha(engine_path),
        "script_sha256": sha(Path(__file__)),
        "reference_contract": {
            "device": "cuda",
            "backbone_dtype": "bfloat16",
            "head_dtype": readout_dtype,
            "head_projection": (
                "float32_linear_at_module_boundary"
                if readout_dtype == "float32"
                else "unchanged_model_linear"
            ),
            "log_softmax_dtype": "float32",
            "attention": attention,
            "batch_size": 1,
            "use_cache": False,
            "placement": "resident",
        },
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "cuda": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "budget_bytes": budget,
            "free_before": free,
            "reserve_mib": 3072,
        },
        "math_settings": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        },
        "rows": [],
        "started_at": time.time(),
        "complete": False,
    }
    try:
        model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                str(model_path),
                dtype=torch.bfloat16,
                trust_remote_code=False,
                local_files_only=True,
                attn_implementation=attention,
            )
            .eval()
            .to("cuda")
        )
        head, embedding = model.get_output_embeddings(), model.get_input_embeddings()
        if not isinstance(head, torch.nn.Linear):
            raise ValueError("Reference requires an unquantized Linear output head")
        head_weight, embedding_weight = head.weight, embedding.weight
        pointers = (head_weight.data_ptr(), embedding_weight.data_ptr())
        if head_weight.dtype != torch.bfloat16 or embedding_weight.dtype != torch.bfloat16:
            raise ValueError("Unexpected backbone/embedding parameter dtype")
        observed = []

        def project(module, inputs, output):
            if readout_dtype == "float32":
                # Recompute without casting a shared Parameter: the input
                # embedding must remain BF16 even when it is tied to the head.
                output = torch.nn.functional.linear(
                    inputs[0].float(),
                    module.weight.float(),
                    module.bias.float() if module.bias is not None else None,
                )
            observed.append(
                {
                    "input_dtype": str(inputs[0].dtype).removeprefix("torch."),
                    "weight_dtype": str(module.weight.dtype).removeprefix("torch."),
                    "output_dtype": str(output.dtype).removeprefix("torch."),
                }
            )
            return output

        hook = head.register_forward_hook(project)
        try:
            with torch.inference_mode():
                for case in engine["cases"]:
                    sequence = case["sequence"]
                    ids = torch.tensor([sequence["input_ids"]], dtype=torch.long, device="cuda")
                    returned = model(ids, use_cache=False).logits
                    if returned.dtype != getattr(torch, readout_dtype):
                        raise ValueError("Model returned a different readout precision")
                    logits = returned[0, -1].float()
                    values = torch.log_softmax(logits, dim=-1)[sequence["label_ids"]].tolist()
                    scores(values, len(sequence["label_ids"]))
                    report["rows"].append(
                        {
                            "id": case["id"],
                            "sequence_digest": content_digest(sequence),
                            "logprobs": values,
                        }
                    )
        finally:
            hook.remove()
        if observed != [
            {
                "input_dtype": "bfloat16",
                "weight_dtype": "bfloat16",
                "output_dtype": readout_dtype,
            }
        ] * len(engine["cases"]):
            raise ValueError("Observed head precision did not match")
        if (
            head.weight is not head_weight
            or embedding.weight is not embedding_weight
            or (head.weight.data_ptr(), embedding.weight.data_ptr()) != pointers
            or head.weight.dtype != torch.bfloat16
            or embedding.weight.dtype != torch.bfloat16
        ):
            raise ValueError("Reference changed the model's shared parameter identity or dtype")
        report["readout_observations"] = observed
        report["parameters"] = {
            "tied_input_output_weights": pointers[0] == pointers[1],
            "input_embedding_dtype": "bfloat16",
            "output_weight_dtype": "bfloat16",
            "identities_and_dtypes_preserved": True,
        }
        torch.cuda.synchronize()
        report["cuda"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["complete"] = True
    except BaseException as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["finished_at"] = time.time()
        save(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    live = sub.add_parser("collect")
    live.add_argument("--run-dir", type=Path, required=True)
    live.add_argument("--output", type=Path, required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--engine-report", type=Path, required=True)
    ref.add_argument("--model-path", type=Path, required=True)
    ref.add_argument("--output", type=Path, required=True)
    ref.add_argument("--attention", choices=("eager", "sdpa"), required=True)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--engine-report", type=Path, required=True)
    compare_parser.add_argument("--reference-report", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        asyncio.run(collect(args.run_dir, args.output))
    elif args.command == "reference":
        reference(args.model_path, args.engine_report, args.output, args.attention)
    else:
        ref = json.loads(args.reference_report.read_text())
        if ref["engine_report_sha256"] != sha(args.engine_report):
            raise ValueError("Engine report changed after reference execution")
        result = summarize(json.loads(args.engine_report.read_text()), ref)
        save(args.output, result)
        print(
            json.dumps({k: v for k, v in result.items() if k not in {"rows", "state_differences"}})
        )
        raise SystemExit(0 if result["all_comparisons_passed"] else 1)


if __name__ == "__main__":
    main()
