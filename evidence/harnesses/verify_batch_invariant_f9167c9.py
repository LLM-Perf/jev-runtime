"""Recompute mode A/B results from retained raw vectors, without CUDA."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

from verify_numerical_suite_877f319 import digest, metrics, read, sha

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence/dsw/batch-invariant-f9167c9"
COMMIT = "f9167c9cb3447e14ab7f10abee3226240e14e1be"
STATES = ("reset_serial", "repeat_serial", "concurrent_reversed")


def main():
    manifest = read(DATA / "export-manifest.json")
    for name, meta in manifest["artifacts"].items():
        p = DATA / name
        assert p.stat().st_size == meta["size_bytes"] and sha(p) == meta["sha256"]
    audit, campaign = (read(DATA / p) for p in ("audit.json", "campaign.json"))
    assert campaign["source_commit"] == audit["source_commit"] == COMMIT
    assert campaign["complete"] and not campaign["execution_complete"] and audit["passed"]
    retest = read(DATA / "retest-campaign.json")
    assert retest["complete"] and retest["execution_complete"]
    assert campaign["attempts"][3]["failure"]["message"] == "('collection failed', 1)"
    assert len(retest["attempts"]) == 1
    attempts = campaign["attempts"][:3] + retest["attempts"]
    assert audit["credential_values_absent_from_export"]
    assert audit["owned_records_checked"] == len(audit["records"])
    assert all(not x["matching_live_process"] and not x["live_group"] for x in audit["records"])
    assert audit["gpu7"]["free_mib"] == "11990"
    assert audit["gpu7"]["uuid"] == "GPU-b57fb933-0e5a-dd28-7041-a177da03405e"
    assert len(audit["model_files"]) == 7 and all(
        x["hub_digest_verified"] for x in audit["model_files"]
    )
    assert len(audit["profiles"]) == 2 and all(x["payload_verified"] for x in audit["profiles"])
    harness = ROOT / "evidence/harnesses/batch-invariant"
    assert sha(harness / "campaign.py.txt") == campaign["script_sha256"]
    assert sha(harness / "audit_export.py.txt") == audit["script_sha256"]
    assert sha(harness / "sglang_retest.py.txt") == retest["script_sha256"]
    for commit, files in audit["source_files_by_commit"].items():
        for path, expected in files.items():
            source = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
            assert hashlib.sha256(source).hexdigest() == expected
    for path, value in audit["source_files_by_commit"][COMMIT].items():
        if path.startswith(("src/", "packages/", "deployment/")):
            assert audit["source_files_by_commit"][retest["source_commit"]][path] == value
    summary = {
        "source_commit": COMMIT,
        "retest_harness_commit": retest["source_commit"],
        "profiles": {},
        "full_release_gate_passed": False,
        "scoring_responses": 0,
        "native_parity_pairs": 0,
        "reference_forwards": 0,
        "reference_comparisons": 0,
        "owned_records_terminal": audit["owned_records_checked"],
    }
    all_inputs, all_references = {}, {}
    assert [a["name"] for a in campaign["attempts"]] == [
        "vllm-ordinary",
        "vllm-invariant",
        "sglang-ordinary",
        "sglang-invariant",
    ]
    for attempt in attempts:
        commit = retest["source_commit"] if attempt["name"] == "sglang-invariant" else COMMIT
        source_files = audit["source_files_by_commit"][commit]
        name, engine, enabled = attempt["name"], attempt["engine"], attempt["batch_invariant"]
        assert attempt["execution_complete"] and "failure" not in attempt
        assert (
            attempt["cleanup"]["passed"]
            and not attempt["cleanup"]["remaining_process_group_members"]
        )
        assert all(
            not c["remaining_group"] and c["returncode"] in (0, 1) for c in attempt["children"]
        )
        directory = DATA / name
        record, raw, guard = (
            read(directory / p) for p in ("process.json", "engine-scores.json", "mode-check.json")
        )
        assert record["batch_invariant"] is enabled and record["tensor_parallel_size"] == 1
        assert raw["complete"] and raw["process_identity"] == record["identity"]
        assert raw["script_sha256"] == source_files["tests/integration/numerical_suite.py"]
        assert raw["profile"]["capabilities"]["batch_invariant"] is enabled
        assert raw["model"].get("batch_invariant") is (True if enabled else None)
        assert raw["model"]["dtype"] == raw["model"]["readout_dtype"] == "bfloat16"
        assert not raw["profile"]["capabilities"]["lora"]
        assert len(raw["cache_resets"]) == 33 and all(
            x["status"] == 200 for x in raw["cache_resets"]
        )
        ids = [c["id"] for c in raw["cases"]]
        assert len(ids) == len(set(ids)) == 32
        all_inputs[name] = [
            (c["sequence"]["input_ids"], c["sequence"]["label_ids"]) for c in raw["cases"]
        ]
        actual = {}
        for state in STATES:
            rows = raw["observations"][state]
            assert [r["id"] for r in rows] == ids
            for case, row in zip(raw["cases"], rows, strict=True):
                assert row["prompt_tokens"] == len(case["sequence"]["input_ids"])
                assert len(row["logprobs"]) == len(case["sequence"]["label_ids"])
                assert row["raw_logprobs"] and row["completion_tokens"] == (
                    1 if engine == "vllm" else 0
                )
            actual[state] = [r["logprobs"] for r in rows]
        summary["scoring_responses"] += 96
        assert all(r["cached_tokens"] == 0 for r in raw["observations"][STATES[0]])
        assert guard["complete"] and guard["model"] == raw["model"]
        assert guard["script_sha256"] == source_files["tests/integration/live_execution_mode.py"]
        assert guard["engine_report_sha256"] == sha(directory / "engine-scores.json")
        assert guard["wrong_bundle"]["status"] == 409
        assert guard["wrong_bundle"]["body"]["error"]["code"] == "model_mismatch"
        assert guard["wrong_attached_runtime"]["code"] == "engine_execution_mismatch"
        assert [c["id"] for c in guard["cases"]] == ids
        for c in guard["cases"]:
            err = max(abs(a - b) for a, b in zip(c["plugin"], c["native"], strict=True))
            assert err == c["max_absolute_error"] and c["passed"] == (err < 1e-4)
        summary["native_parity_pairs"] += 32
        output = {
            "batch_invariant": enabled,
            "native_parity_passed": sum(c["passed"] for c in guard["cases"]),
            "native_parity_max_error": max(c["max_absolute_error"] for c in guard["cases"]),
            "states": {},
            "references": {},
            "cache_counts": {
                state: sorted(set(r["cached_tokens"] for r in raw["observations"][state]))
                for state in STATES
            },
        }
        for state in STATES[1:]:
            values = [metrics(a, b) for a, b in zip(actual[state], actual[STATES[0]], strict=True)]
            output["states"][state] = {
                "comparisons": 32,
                "exact_vector_matches": sum(
                    a == b for a, b in zip(actual[state], actual[STATES[0]], strict=True)
                ),
                "max_logprob_change": max(x[0] for x in values),
                "max_probability_change": max(x[1] for x in values),
                "argmax_changes": sum(not x[2] for x in values),
            }
        for attention in ("eager", "sdpa"):
            ref, published = (
                read(directory / f"{kind}-{attention}.json") for kind in ("reference", "comparison")
            )
            assert ref["complete"] and ref["model"] == raw["model"]
            assert ref["cases_digest"] == digest(raw["cases"]) and ref[
                "engine_report_sha256"
            ] == sha(directory / "engine-scores.json")
            assert ref["script_sha256"] == raw["script_sha256"]
            assert ref["cuda"]["peak_reserved_bytes"] <= ref["cuda"]["budget_bytes"]
            assert [r["id"] for r in ref["rows"]] == ids
            for case, r in zip(raw["cases"], ref["rows"], strict=True):
                assert r["sequence_digest"] == digest(case["sequence"])
            computed = [
                metrics(a, r["logprobs"])
                for state in STATES
                for a, r in zip(actual[state], ref["rows"], strict=True)
            ]
            assert len(computed) == len(published["rows"]) == 96
            for values, row in zip(computed, published["rows"], strict=True):
                assert math.isclose(values[0], row["max_absolute_logprob_error"], abs_tol=1e-12)
                assert math.isclose(
                    values[1], row["max_absolute_conditional_probability_error"], abs_tol=1e-12
                )
                assert values[2] == row["top_label_equal"] and values[4] == row["passed"]
            output["references"][attention] = {
                "passed": sum(x[4] for x in computed),
                "total": 96,
                "max_logprob_error": max(x[0] for x in computed),
                "max_probability_error": max(x[1] for x in computed),
                "argmax_mismatches": sum(not x[2] for x in computed),
            }
            summary["reference_forwards"] += 32
            summary["reference_comparisons"] += 96
            all_references[(name, attention)] = [r["logprobs"] for r in ref["rows"]]
        assert attempt["postcheck"]["zero_admission"] and attempt["postcheck"]["zero_leases"]
        summary["profiles"][name] = output
    assert all(v == all_inputs["vllm-ordinary"] for v in all_inputs.values())
    for engine in ("vllm", "sglang"):
        for attention in ("eager", "sdpa"):
            assert (
                all_references[(engine + "-ordinary", attention)]
                == all_references[(engine + "-invariant", attention)]
            )
    summary["input_ids_equal_across_profiles"] = True
    summary["reference_vectors_equal_between_modes"] = True
    (DATA / "verified-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
