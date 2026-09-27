"""Recompute the four functional rows without hiding failed numerical checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw"
HARNESSES = ROOT / "evidence/harnesses/glm-r1"
SOURCE = "c48f18dfc9715dae09588e0c592c38e25c9e4b80"
R1_HARNESS = "d1ef50e8420cc0e0af82000c462499e8180929b1"
R1_RUNTIME = "3377c95ca9955ae96ce7c7deb2c4d97b41b1d9c1"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(BASE / "glm-r1-c48f18d-export-manifest.json")
    for name, expected in manifest["artifacts"].items():
        path = BASE / name
        assert sha(path) == expected["sha256"] and path.stat().st_size == expected["size_bytes"], (
            name
        )
    audit = read(BASE / "glm-r1-c48f18d-final-audit.json")
    assert audit["passed"] and audit["owned_records_checked"] == len(audit["records"]) == 91
    assert not audit["matching_live_owned_records"] and not audit["live_owned_group_members"]
    assert all(not r["matching_live_process"] for r in audit["records"])
    selected_gpu_rows = [
        [v.strip() for v in line.split(",")]
        for line in audit["gpu_snapshot"].splitlines()
        if int(line.split(",")[0]) in (3, 4, 5, 6)
    ]
    assert len(selected_gpu_rows) == 4
    assert all(int(row[2]) == 10792 for row in selected_gpu_rows)
    assert (
        audit["script_sha256"]
        == manifest["script_sha256"]
        == sha(HARNESSES / "export_evidence.py.txt")
    )
    for commit, files in audit["source_files"].items():
        for name, digest in files.items():
            actual = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
            assert hashlib.sha256(actual).hexdigest() == digest, name
    assets = read(BASE / "model-verification-glm-r1-d1ef50e.json")
    assert assets["passed"] and assets["harness_sha256"] == sha(
        HARNESSES / "download_verify.py.txt"
    )
    assert len(assets["models"]) == 2 and len(audit["model_files"]) == 30
    expected_files = {
        (m["model_id"], m["revision"], f["name"]): f for m in assets["models"] for f in m["files"]
    }
    assert len(expected_files) == 30
    for f in audit["model_files"]:
        old = expected_files[(f["model_id"], f["revision"], f["name"])]
        assert old["verified"] and (f["sha256"], f["size_bytes"]) == (
            old["sha256"],
            old["size_bytes"],
        )
        if old["remote_digest_kind"] == "lfs_sha256":
            assert old["remote_digest"] == f["sha256"]
    profile = read(BASE / "glm4-tokenizer-profile-c48f18d.json")
    source_bytes = subprocess.check_output(
        ["git", "show", SOURCE + ":src/jev_runtime/tokenizer_profiles.py"], cwd=ROOT
    )
    assert profile["implementation_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert profile["validation"]["passed"] and profile["validation"]["cases"] == 588
    original_code = {
        f["name"]: f["sha256"] for f in assets["models"][1]["files"] if f["name"].endswith(".py")
    }
    for engine in ("vllm", "sglang"):
        eq = read(BASE / f"glm4-profile-equivalence-{engine}-c48f18d.json")
        assert eq["script_sha256"] == sha(HARNESSES / "profile_equivalence.py.txt")
        assert eq["profile_manifest_sha256"] == sha(BASE / "glm4-tokenizer-profile-c48f18d.json")
        assert eq["original_tokenizer_code_sha256"] == original_code["tokenization_chatglm.py"]
        assert eq["passed"] and eq["template_equal"] and len(eq["encoding_checks"]) == 630
        assert len(eq["compiler_cases"]) == 16
        for check in eq["encoding_checks"]:
            assert (
                check["equal"]
                and check["default_decode_equal"]
                and check["expected_ids"] == check["actual_ids"]
            )
        for case in eq["compiler_cases"]:
            assert case["passed"] and all(s["original_wrapper_equal"] for s in case["sequences"])
    glm_reference = read(BASE / "glm4-reference-tf444-c48f18d.json")
    assert glm_reference["script_sha256"] == sha(HARNESSES / "glm_reference.py.txt")
    assert glm_reference["completed"] and len(glm_reference["positions"]) == 2
    assert glm_reference["local_code_sha256"] == original_code
    loader_failure = read(BASE / "glm-reference-loader-d1ef50e.json")
    assert not loader_failure["passed"] and loader_failure["failure"]["type"] == "AttributeError"
    assert loader_failure["local_code_sha256"] == original_code
    prototype = read(BASE / "glm-tokenizer-conversion-probe2-d1ef50e.json")
    assert prototype["script_sha256"] == sha(HARNESSES / "probe_conversion2.py.txt")
    assert all(c["equal"] and c["expected_ids"] == c["actual_ids"] for c in prototype["checks"])
    rows = []
    for tag, family, campaign_name, script, commit, runtime in (
        (
            "r1-qwen7b-tp4-d1ef50e",
            "r1",
            "r1-qwen7b-campaign-d1ef50e.json",
            "r1_campaign.py.txt",
            R1_HARNESS,
            R1_RUNTIME,
        ),
        (
            "glm4-fast-tp4-c48f18d",
            "glm",
            "glm4-campaign-c48f18d.json",
            "glm_campaign.py.txt",
            SOURCE,
            SOURCE,
        ),
    ):
        campaign = read(BASE / campaign_name)
        assert (
            campaign["completed"]
            and campaign["all_functional_passed"]
            and len(campaign["attempts"]) == 2
        )
        assert campaign["harness_sha256"] == sha(HARNESSES / script)
        fidelity_name = (
            "r1-qwen7b-tokenizer-fidelity-d1ef50e.json"
            if family == "r1"
            else "glm4-tokenizer-fidelity-c48f18d.json"
        )
        fidelity = read(BASE / fidelity_name)
        assert fidelity["harness_sha256"] == sha(HARNESSES / f"{family}_fidelity.py.txt")
        contracts = []
        for engine in ("vllm", "sglang"):
            run = BASE / f"{engine}-{tag}"
            contract = read(run / "contract.json")
            contracts.append(contract)
            validation = read(run / "validation.json")
            assert (
                contract["passed"]
                and validation["functional_passed"]
                and validation["cleanup_passed"]
            )
            assert (
                validation["harness_source_commit"] == commit
                and validation["runtime_source_commit"] == runtime
            )
            assert (
                contract["source_commit"] == commit and contract["runtime_source_commit"] == runtime
            )
            for filename in ("precision-binding.json", "cleanup.json"):
                assert read(run / filename)["passed"], filename
            postcheck = read(run / "postcheck.json")
            assert all(
                postcheck[key]
                for key in ("zero_leases", "zero_admission", "all_active_bundles_healthy")
            )
            topology = read(run / "topology.json")
            assert (
                topology["tensor_parallel_size"] == 4
                and topology["parent_cuda_visible_devices"] == "3,4,5,6"
            )
            assert [g["uuid"] for g in topology["selected_gpus"]] == [
                row[1] for row in selected_gpu_rows
            ]
            environment = read(run / "environment.json")
            assert environment["runtime_source_commit"] == runtime
            assert all(
                "/releases/" + runtime + "/" in path
                for path in environment["import_paths"].values()
            )
            model = contract["model"]
            saved = contract["reference_logits"]
            assert (model["id"], model["revision"]) == (fidelity["model_id"], fidelity["revision"])
            assert (
                model["dtype"] == model["readout_dtype"] == "bfloat16"
                and model["quantization"] is None
            )
            assert (
                saved["input_ids"] == fidelity["expected_input_ids"]
                and saved["label_ids"] == fidelity["expected_label_ids"]
            )
            assert fidelity["saved_serving_inputs"][engine]["matches_checkpoint_tokenizer"]
            switches = contract["checks"]["hot_switch_under_traffic"]
            assert (
                switches["switches"] == 1000
                and switches["mixed_bundle_responses"] == 0
                and switches["strict_success_requests"] > 0
            )
            assert contract["checks"]["native_attach_logprob_parity"]["max_abs_error"] == 0
            if family == "r1":
                reference = read(run / "reference-cpu.json")
            else:
                reference = next(p for p in glm_reference["positions"] if p["engine"] == engine)
                assert (
                    validation["failure"]["stage"] == "cpu_reference" and not validation["passed"]
                )
                assert "trust_remote_code=True" in (run / "reference-cpu.log").read_text()
                for field in (
                    "tokenizer_digest",
                    "tokenizer_implementation_digest",
                    "template_digest",
                ):
                    assert model[field] == profile[field]
            assert reference["contract_sha256"] == sha(run / "contract.json")
            assert reference["engine_logprobs"] == saved["logprobs"]
            errors = [
                abs(a - b)
                for a, b in zip(reference["reference_logprobs"], saved["logprobs"], strict=True)
            ]
            assert errors == reference["absolute_errors"] and reference["atol"] == 0.15
            top_equal = max(
                range(len(errors)), key=reference["reference_logprobs"].__getitem__
            ) == max(range(len(errors)), key=saved["logprobs"].__getitem__)
            assert top_equal == reference["top_label_equal"]
            assert reference["passed"] == (max(errors) <= 0.15 and top_equal)
            rows.append(
                {
                    "model_id": model["id"],
                    "revision": model["revision"],
                    "engine": engine,
                    "run": run.name,
                    "harness_source_commit": commit,
                    "runtime_source_commit": runtime,
                    "functional_passed": True,
                    "switches": switches["switches"],
                    "strict_success_requests": switches["strict_success_requests"],
                    "max_absolute_logprob_error": max(errors),
                    "single_position_passed": reference["passed"],
                    "top_label_equal": top_equal,
                    "full_numerical_gate_passed": False,
                }
            )
        for key in ("input_ids", "label_ids"):
            assert contracts[0]["reference_logits"][key] == contracts[1]["reference_logits"][key]
    matrix = read(ROOT / "profiles/certification-matrix.json")
    assert len(matrix["combinations"]) == matrix["combination_denominator"] == 40
    for engine in ("vllm", "sglang"):
        assert (
            sum(
                c["functional"] == "passed" for c in matrix["combinations"] if c["engine"] == engine
            )
            == matrix["functional_passes"][engine]
            == 12
        )
    assert matrix["required_functional_passes_per_engine"] == 18
    assert not matrix["release_gate_passed"]
    result = {
        "qualification": (
            "Four colocated native functional combinations; one numerical position "
            "per engine/model; no release certificate"
        ),
        "source_commit": SOURCE,
        "new_functional_combinations": len(rows),
        "rows": rows,
        "verified_model_files": 30,
        "owned_process_records_checked": 91,
        "release_gate_passed": False,
        "all_numerical_positions_passed": all(r["single_position_passed"] for r in rows),
    }
    (BASE / "glm-r1-validation-c48f18d.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
