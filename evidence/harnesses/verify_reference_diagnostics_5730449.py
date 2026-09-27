"""Recompute the paired reference evidence without promoting any release gate."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence/dsw/reference-diagnostics-5730449"
SOURCE = "5730449e7936e9de46941a2e43c7574b889c8905"


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditional(values):
    weights = [math.exp(value - max(values)) for value in values]
    return [value / sum(weights) for value in weights]


def main():
    campaign = load(DATA / "campaign.json")
    assert campaign["completed"] and campaign["source_commit"] == SOURCE
    assert len(campaign["attempts"]) == 19
    for name, expected in campaign["source_files"].items():
        blob = subprocess.check_output(
            ["git", "show", f"{SOURCE}:tests/integration/{name}"], cwd=ROOT
        )
        assert hashlib.sha256(blob).hexdigest() == expected
    assert campaign["controller_sha256"] == digest(
        ROOT / "evidence/harnesses/reference-diagnostics/campaign.py.txt"
    )
    assert campaign["vllm_helper_check_sha256"] == digest(DATA / "vllm-helper-equality.json")
    helper_comparisons = 0
    for engine in ("vllm", "sglang"):
        helper = load(DATA / f"{engine}-helper-equality.json")
        assert helper["passed"] and len(helper["models"]) == 3
        assert helper["helper_sha256"] == campaign["source_files"]["reference_device.py"]
        assert helper["script_sha256"] == campaign["source_files"]["check_reference_offload.py"]
        for model in helper["models"]:
            assert model["state_unchanged"] and model["exception_cleanup_passed"]
            assert len(model["comparisons"]) == 4
            assert {(r["attention"], r["readout_dtype"]) for r in model["comparisons"]} == {
                (a, d) for a in ("eager", "sdpa") for d in ("bfloat16", "float32")
            }
            for row in model["comparisons"]:
                assert row["bitwise_equal"] and row["max_absolute_error"] == 0
                assert not row["placement"]["unvisited_modules"]
                helper_comparisons += 1
    summaries = []
    for attempt in campaign["attempts"]:
        path = DATA / (attempt["name"] + ".json")
        assert digest(path) == attempt["artifact_sha256"]
        assert not attempt["same_process_live_after"]
        assert attempt["identity"]["pid"] == attempt["identity"]["pgid"]
        assert attempt["gpus_before"]["uuid"] == attempt["gpus_after"]["uuid"]
        assert attempt["gpus_after"]["free_mib"] >= attempt["gpus_before"]["free_mib"]
        result = load(path)
        assert attempt["returncode"] == (0 if result["passed"] else 1)
        if attempt["name"] == "sglang-helper-equality":
            continue
        command = attempt["command"]
        original = Path(command[command.index("--contract-report") + 1])
        contract_path = ROOT / "evidence/dsw" / original.parent.name / original.name
        contract = load(contract_path)
        assert result["contract_sha256"] == digest(contract_path)
        assert result["reference_script_sha256"] == campaign["source_files"]["reference_logits.py"]
        assert result["device_helper_sha256"] == campaign["source_files"]["reference_device.py"]
        assert result["revision"] == contract["model"]["revision"]
        assert result["readout_dtype"] == contract["model"]["readout_dtype"]
        assert result["observed_readout_dtype"] == result["readout_dtype"]
        assert result["engine_logprobs"] == contract["reference_logits"]["logprobs"]
        assert result["atol"] == 0.15
        errors = [
            abs(a - b)
            for a, b in zip(result["engine_logprobs"], result["reference_logprobs"], strict=True)
        ]
        assert errors == result["absolute_errors"] and all(map(math.isfinite, errors))
        left, right = result["engine_logprobs"], result["reference_logprobs"]
        top_equal = max(range(len(left)), key=left.__getitem__) == max(
            range(len(right)), key=right.__getitem__
        )
        assert result["top_label_equal"] == top_equal
        assert result["passed"] == (max(errors) <= 0.15 and top_equal)
        if result["device"] == "cuda":
            cuda = result["cuda"]
            assert cuda["peak_allocated_bytes"] <= cuda["peak_reserved_bytes"]
            assert cuda["peak_reserved_bytes"] <= cuda["allocator_budget_bytes"]
            assert not result["placement"]["unvisited_modules"]
            assert "GPU-" + cuda["device_uuid"] == attempt["gpus_before"]["uuid"]
        else:
            assert result["cuda"] is None and result["placement"]["strategy"] == "resident"
        conditional_error = max(
            abs(a - b) for a, b in zip(conditional(left), conditional(right), strict=True)
        )
        summaries.append(
            {
                "attempt": attempt["name"],
                "model_id": result["model_id"],
                "engine": attempt["name"].split("-")[0],
                "readout_dtype": result["readout_dtype"],
                "input_and_labels_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            key: contract["reference_logits"][key]
                            for key in ("input_ids", "label_ids")
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                "reference_device": result["device"],
                "reference_attention": result["attention"],
                "maximum_logprob_error": max(errors),
                "maximum_conditional_probability_error": conditional_error,
                "top_label_equal": top_equal,
                "single_position_passed": result["passed"],
                "artifact": str(path.relative_to(ROOT)),
            }
        )
    assert len(summaries) == 18
    expected = {
        (engine, model, dtype, device, attention)
        for engine in ("vllm", "sglang")
        for model, dtype in (
            ("Qwen/Qwen2.5-7B-Instruct", "bfloat16"),
            ("Qwen/Qwen2.5-7B-Instruct", "float32"),
            ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "bfloat16"),
        )
        for device, attention in (("cpu", "sdpa"), ("cuda", "eager"), ("cuda", "sdpa"))
    }
    assert {
        (
            x["engine"],
            x["model_id"],
            x["readout_dtype"],
            x["reference_device"],
            x["reference_attention"],
        )
        for x in summaries
    } == expected
    exploratory = load(DATA / "exploratory-qwen25.json")
    assert exploratory["completed"] and len(exploratory["profiles"]) == 8
    assert exploratory["script_sha256"] == digest(
        ROOT / "evidence/harnesses/reference-diagnostics/exploratory-probe.py.txt"
    )
    audit = load(DATA / "final-audit.json")
    assert audit["passed"] and not audit["live_owned_identities"]
    assert not audit["live_owned_group_members"]
    assert audit["script_sha256"] == digest(
        ROOT / "evidence/harnesses/reference-diagnostics/audit-export.py.txt"
    )
    assert audit["verified_model_files"] == 22
    prior_models = load(ROOT / "evidence/dsw/model-verification-public-pair-3377c95.json")
    prior_files = {
        (model["model_id"], model["revision"], file["name"]): (file["size_bytes"], file["sha256"])
        for model in prior_models["models"]
        for file in model["files"]
    }
    assert {
        (file["model_id"], file["revision"], file["name"]): (file["size_bytes"], file["sha256"])
        for file in audit["model_files"]
    } == prior_files
    assert audit["gpu"]["free_mib"] == 11990
    export = load(DATA / "export-manifest.json")
    for name, expected_file in export["files"].items():
        assert digest(DATA / name) == expected_file["sha256"]
        assert (DATA / name).stat().st_size == expected_file["size_bytes"]
    summary = {
        "source_commit": SOURCE,
        "scope": "saved answer positions only; earlier failures and release gates unchanged",
        "helper_full_logit_equalities": helper_comparisons,
        "reference_profile_comparisons": len(summaries),
        "distinct_compiled_inputs": len({r["input_and_labels_sha256"] for r in summaries}),
        "single_position_passes": sum(x["single_position_passed"] for x in summaries),
        "single_position_failures": sum(not x["single_position_passed"] for x in summaries),
        "positions": summaries,
        "evidence_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in sorted(DATA.glob("*.json"))
        },
    }
    output = ROOT / "evidence/dsw/reference-diagnostics-5730449.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("positions", "evidence_sha256")})
    )


if __name__ == "__main__":
    main()
