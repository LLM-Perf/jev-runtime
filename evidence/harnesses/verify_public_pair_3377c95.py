"""Validate four combinations with retained tokenizer and precision follow-ups."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw"
SOURCE = "3377c95ca9955ae96ce7c7deb2c4d97b41b1d9c1"
TOKENIZER_HARNESS = "b28bb4611b6a8142ffc37b1b12af84514a713b5f"
EXPECTED = {
    "qwen25-7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "r1-llama8b": (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "6a6f4aa4197940add57724a7707d069478df56b1",
    ),
    "qwen25-7b-f32": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "r1-llama8b-bytelevel": (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "6a6f4aa4197940add57724a7707d069478df56b1",
    ),
}


def read(path):
    return json.loads(path.read_text())


def main():
    manifest = read(BASE / "public-pair-export-manifest-3377c95.json")
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest, name
    components = read(BASE / "tokenizer-component-manifest-b28bb46.json")
    assert len(components["artifacts"]) == 2
    for name, digest in components["artifacts"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        audit = read(ROOT / name)
        assert not audit["profiles"]["default_auto"]["matches_checkpoint"]
        assert audit["profiles"]["explicit_generic"]["matches_checkpoint"]
        assert (
            audit["harness_sha256"]
            == hashlib.sha256(
                (ROOT / "evidence/harnesses/public-pair/tokenizer_components.py.txt").read_bytes()
            ).hexdigest()
        )
    source = read(BASE / "source-revalidation-public-pair-3377c95.json")
    assert source["source_commit"] == SOURCE
    for name, digest in source["files"].items():
        actual = subprocess.check_output(["git", "show", SOURCE + ":" + name], cwd=ROOT)
        assert hashlib.sha256(actual).hexdigest() == digest, name
    assert source["verified_files"] == len(source["files"])
    tokenizer_source = read(BASE / "source-upload-b28bb46.json")
    assert tokenizer_source["source_commit"] == TOKENIZER_HARNESS
    for name, digest in tokenizer_source["files"].items():
        actual = subprocess.check_output(["git", "show", TOKENIZER_HARNESS + ":" + name], cwd=ROOT)
        assert hashlib.sha256(actual).hexdigest() == digest, name
    fidelity = read(BASE / "r1-llama-tokenizer-fidelity-3377c95.json")
    qwen_fidelity = read(BASE / "qwen25-tokenizer-fidelity-3377c95.json")
    for audit, script in (
        (fidelity, "tokenizer_fidelity.py.txt"),
        (qwen_fidelity, "qwen_tokenizer_fidelity.py.txt"),
    ):
        assert (
            audit["harness_sha256"]
            == hashlib.sha256(
                (ROOT / "evidence/harnesses/public-pair" / script).read_bytes()
            ).hexdigest()
        )
    assert all(
        c["matches_checkpoint_tokenizer"] for c in qwen_fidelity["saved_serving_inputs"].values()
    )
    assert not fidelity["saved_serving_inputs"]["vllm"]["matches_checkpoint_tokenizer"]
    assert fidelity["saved_serving_inputs"]["sglang"]["matches_checkpoint_tokenizer"]
    assert all(c["matches_checkpoint_tokenizer"] for c in fidelity["explicit_profile"]["checks"])
    assert (
        fidelity["explicit_profile"]["files"]["tokenizer.json"]
        == fidelity["explicit_profile"]["source_files"]["tokenizer.json"]
    )
    tokenizer_followup = read(BASE / "tokenizer-followup-public-pair-b28bb46.json")
    assert (
        tokenizer_followup["harness_sha256"]
        == hashlib.sha256(
            (ROOT / "evidence/harnesses/public-pair/run_tokenizer_followup.py.txt").read_bytes()
        ).hexdigest()
    )
    assert tokenizer_followup["completed"] and len(tokenizer_followup["attempts"]) == 2
    for attempt in tokenizer_followup["attempts"]:
        assert attempt["saved_input_matches_checkpoint_tokenizer"]
        assert attempt["validation"]["functional_passed"]
    assets = read(BASE / "model-verification-public-pair-3377c95.json")
    assert assets["passed"] and len(assets["models"]) == 2
    assert (
        assets["harness_sha256"]
        == hashlib.sha256(
            (ROOT / "evidence/harnesses/public-pair/download_verify.py.txt").read_bytes()
        ).hexdigest()
    )
    campaign = read(BASE / "public-pair-campaign-3377c95.json")
    assert campaign["completed"] and campaign["all_functional_passed"]
    assert len(campaign["attempts"]) == 4
    assert (
        campaign["harness_sha256"]
        == hashlib.sha256(
            (ROOT / "evidence/harnesses/public-pair/run_campaign.py.txt").read_bytes()
        ).hexdigest()
    )
    followup = read(BASE / "precision-followup-public-pair-3377c95.json")
    assert followup["completed"] and len(followup["attempts"]) == 2
    assert (
        followup["harness_sha256"]
        == hashlib.sha256(
            (ROOT / "evidence/harnesses/public-pair/run_precision_followup.py.txt").read_bytes()
        ).hexdigest()
    )
    result = {
        "runtime_source_commit": SOURCE,
        "harness_source_commit": SOURCE,
        "tokenizer_harness_source_commit": TOKENIZER_HARNESS,
        "release_gate_passed": False,
        "qualification": (
            "Colocated native BF16-backbone TP4/API1/eager checks with separate "
            "BF16/FP32 heads; CPU references cover one "
            "saved answer position each. No full numerical, quality, performance "
            "or LoRA certification."
        ),
        "profiles": [],
        "cross_engine_single_position_checks": [],
        "verified_uploaded_files": source["verified_files"],
        "verified_artifact_hashes": len(manifest["artifacts"]),
        "verified_additional_component_hashes": len(components["artifacts"]),
        "model_artifact_files_verified": 0,
    }
    verified_models = set()
    for tag, (model, revision) in EXPECTED.items():
        readout_dtype = "float32" if tag.endswith("-f32") else "bfloat16"
        harness = TOKENIZER_HARNESS if tag.endswith("-bytelevel") else SOURCE
        asset = next(m for m in assets["models"] if m["model_id"] == model)
        assert asset["revision"] == revision and asset["verified"]
        assert not asset["gated"] and not asset["private"]
        assert asset["verified_files"] == asset["expected_files"] == len(asset["files"])
        assert all(f["verified"] for f in asset["files"])
        assert asset["weight_bytes"] == sum(
            f["size_bytes"] for f in asset["files"] if f["name"].endswith(".safetensors")
        )
        if model not in verified_models:
            result["model_artifact_files_verified"] += len(asset["files"])
            verified_models.add(model)
        positions = []
        for engine in ("vllm", "sglang"):
            run = BASE / f"{engine}-{tag}-tp4-{harness[:7]}"
            c = read(run / "contract.json")
            v = read(run / "validation.json")
            ref = read(run / "reference-cpu.json")
            cleanup = read(run / "cleanup.json")
            precision = read(run / "precision-binding.json")
            topology = read(run / "topology.json")
            env = read(run / "environment.json")
            post = read(run / "postcheck.json")
            process = read(run / "process.json")
            assert c["passed"] and v["functional_passed"] and v["cleanup_passed"]
            assert precision["passed"] and cleanup["passed"] and "failure" not in v
            assert c["source_commit"] == v["harness_source_commit"] == harness
            assert c["runtime_source_commit"] == v["runtime_source_commit"] == SOURCE
            assert env["runtime_source_commit"] == precision["runtime_source_commit"] == SOURCE
            assert precision["harness_source_commit"] == harness
            assert c["model"]["id"] == model and c["model"]["revision"] == revision
            assert c["model"]["dtype"] == "bfloat16"
            assert c["model"]["readout_dtype"] == readout_dtype
            assert c["checks"]["four_types"]["usage"]["successful_questions"] == 4
            hot = c["checks"]["hot_switch_under_traffic"]
            assert hot["switches"] == 1000 and hot["mixed_bundle_responses"] == 0
            assert hot["strict_success_requests"] > 0 and len(hot["versions_seen"]) == 2
            assert c["checks"]["native_attach_logprob_parity"]["max_abs_error"] == 0
            assert all(
                post[k] for k in ("zero_leases", "zero_admission", "all_active_bundles_healthy")
            )
            assert topology["tensor_parallel_size"] == process["tensor_parallel_size"] == 4
            assert topology["parent_cuda_visible_devices"] == "3,4,5,6"
            assert not cleanup["remaining_process_group_members"] and cleanup["gpu_uuids_match"]
            assert [g["index"] for g in cleanup["gpus_after"]] == [3, 4, 5, 6]
            for before, after in zip(process["gpus_before"], cleanup["gpus_after"], strict=True):
                assert before["uuid"] == after["uuid"] and before["free_mib"] == after["free_mib"]
            assert ref["atol"] == 0.15
            assert ref["model_id"] == model and ref["revision"] == revision
            assert ref["readout_dtype"] == ref["observed_readout_dtype"] == readout_dtype
            assert ref["passed"] == v["numerical_position_passed"] == v["passed"]
            errors = [
                abs(a - b)
                for a, b in zip(ref["engine_logprobs"], ref["reference_logprobs"], strict=True)
            ]
            assert errors == ref["absolute_errors"]
            assert max(errors) == v["stages"]["cpu_reference"]["max_absolute_error"]
            assert ref["passed"] == (max(errors) <= 0.15 and ref["top_label_equal"])
            saved = c["reference_logits"]
            audit = fidelity if tag.startswith("r1-llama8b") else qwen_fidelity
            artifact_fidelity = (
                saved["input_ids"] == audit["expected_input_ids"]
                and saved["label_ids"] == audit["expected_label_ids"]
            )
            if tag.startswith("r1-llama8b"):
                assert artifact_fidelity == (engine == "sglang" or tag.endswith("-bytelevel"))
                if tag.endswith("-bytelevel"):
                    assert process["tokenizer_path"] == fidelity["explicit_profile"]["path"]
            else:
                assert artifact_fidelity
            result["profiles"].append(
                {
                    "model_id": model,
                    "revision": revision,
                    "engine": engine,
                    "engine_version": env["versions"][engine],
                    "run": str(run.relative_to(ROOT)),
                    "profile_tag": tag,
                    "harness_source_commit": harness,
                    "tokenizer_artifact_fidelity": artifact_fidelity,
                    "original_tokenizer_failure_preserved": tag == "r1-llama8b"
                    and engine == "vllm",
                    "functional_harness_passed": True,
                    "functional_profile_qualified": artifact_fidelity,
                    "tensor_parallel_size": 4,
                    "api_workers": 1,
                    "readout_dtype": readout_dtype,
                    "switches": hot["switches"],
                    "strict_success_requests_during_switches": hot["strict_success_requests"],
                    "mixed_bundle_responses": 0,
                    "native_attach_max_logprob_error": 0,
                    "reference_cpu_single_position_passed": ref["passed"],
                    "reference_cpu_max_logprob_error": max(errors),
                    "reference_cpu_atol": ref["atol"],
                    "top_label_equal": ref["top_label_equal"],
                    "full_numerical_gate_passed": False,
                    "precision_binding_passed": True,
                    "zero_final_leases_admission": True,
                    "all_owned_group_members_exited": True,
                    "gpus_after": cleanup["gpus_after"],
                }
            )
            positions.append(c["reference_logits"])
        a, b = positions
        same_input = a["input_ids"] == b["input_ids"]
        same_labels = a["label_ids"] == b["label_ids"]
        result["cross_engine_single_position_checks"].append(
            {
                "model_id": model,
                "readout_dtype": readout_dtype,
                "same_input_ids": same_input,
                "same_label_ids": same_labels,
                "max_logprob_error": max(
                    abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"], strict=True)
                )
                if same_input and same_labels
                else None,
                "qualification": "One saved position; not broad cross-engine numerical parity",
            }
        )
    preflight = read(BASE / "preflight-public-pair-3377c95.json")
    assert not preflight["matching_live_owned_records"]
    result["owned_records_checked_after_cleanup"] = preflight["owned_records_checked"]
    result["distinct_functional_combinations_added"] = len(
        {
            (p["model_id"], p["engine"])
            for p in result["profiles"]
            if p["functional_profile_qualified"]
        }
    )
    assert result["distinct_functional_combinations_added"] == 4
    path = BASE / "public-pair-validation-3377c95.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
