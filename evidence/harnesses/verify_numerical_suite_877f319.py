"""Recompute the numerical campaign from raw retained scores without Torch/GPU."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence/dsw/numerical-suite-877f319"
COMMIT = "877f319cf03932193db362a9475bd6a5aebc723d"
STATES = ("reset_serial", "repeat_serial", "concurrent_reversed")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def metrics(actual, expected):
    assert len(actual) == len(expected) >= 2
    assert all(type(x) in (int, float) and math.isfinite(x) and x <= 0 for x in actual + expected)
    order = sorted(range(len(expected)), key=lambda i: expected[i], reverse=True)
    difference = max(abs(actual[i] - expected[i]) for i in range(len(expected)))
    probability = []
    for values in (actual, expected):
        norm = math.fsum(math.exp(x - max(values)) for x in values)
        probability.append([math.exp(x - max(values)) / norm for x in values])
    probability_error = max(abs(a - b) for a, b in zip(*probability, strict=True))
    equal = max(range(len(actual)), key=lambda i: actual[i]) == order[0]
    gap = expected[order[0]] - expected[order[1]]
    return difference, probability_error, equal, gap, difference <= 0.15 and equal


def main():
    export = read(DATA / "export-manifest.json")
    for name, meta in export["artifacts"].items():
        path = DATA / name
        assert path.stat().st_size == meta["size_bytes"] and sha(path) == meta["sha256"]
    audit, campaign, finish = (
        read(DATA / name) for name in ("audit.json", "campaign.json", "finish.json")
    )
    assert audit["passed"] and finish["complete"] and campaign["complete"]
    assert not campaign["diagnostic_execution_complete"]  # Retain initial metadata failure.
    assert finish["initial_campaign_sha256"] == sha(DATA / "campaign.json")
    assert audit["owned_records_checked"] == len(audit["records"]) == 171
    assert all(not r["matching_live_process"] and not r["live_group"] for r in audit["records"])
    assert audit["gpu7"]["free_mib"] == "11990"
    assert audit["gpu7"]["uuid"] == "GPU-b57fb933-0e5a-dd28-7041-a177da03405e"
    assert len(audit["model_files"]) == 7 and all(
        x["hub_digest_verified"] for x in audit["model_files"]
    )
    assert len(audit["profiles"]) == 2 and all(x["payload_verified"] for x in audit["profiles"])
    assert audit["registry_final_blockers"] == {"vllm": {}, "sglang": {}}
    for name, expected in audit["source_files"].items():
        payload = subprocess.check_output(["git", "show", f"{COMMIT}:{name}"], cwd=ROOT)
        assert hashlib.sha256(payload).hexdigest() == expected, name
    harness = ROOT / "evidence/harnesses/numerical-suite"
    assert campaign["script_sha256"] == sha(harness / "campaign.py.txt")
    assert finish["script_sha256"] == sha(harness / "finish_references.py.txt")
    assert audit["script_sha256"] == sha(harness / "audit_export.py.txt")
    source_hash = audit["source_files"]["tests/integration/numerical_suite.py"]
    summary = {
        "source_commit": COMMIT,
        "cases_per_engine": 32,
        "scoring_responses": 192,
        "reference_forwards": 128,
        "comparisons": 384,
        "engines": {},
        "full_release_gate_passed": False,
        "owned_records_terminal": 171,
    }
    inputs = {}
    for attempt in campaign["attempts"]:
        engine = attempt["engine"]
        data = read(DATA / engine / "engine-scores.json")
        assert data["complete"] and data["script_sha256"] == source_hash
        assert data["process_identity"] == attempt["cleanup"]["process_record"]["identity"]
        assert (
            attempt["cleanup"]["passed"]
            and not attempt["cleanup"]["remaining_process_group_members"]
        )
        assert attempt["topology"]["tensor_parallel_size"] == 1
        if engine == "sglang":
            assert attempt["failure"]["type"] == "PackageNotFoundError"
        else:
            assert attempt["diagnostic_execution_complete"] and attempt["postcheck"]["zero_leases"]
        for package, imported in attempt["imports"].items():
            base = (
                "src/jev_runtime"
                if package == "jev_runtime"
                else f"packages/{engine}/src/{package}"
            )
            assert COMMIT in imported["path"]
            for path, expected in imported["files"].items():
                assert audit["source_files"][f"{base}/{path}"] == expected
        assert len(data["cases"]) == 32 and len(data["cache_resets"]) == 33
        assert all(x["status"] == 200 for x in data["cache_resets"])
        assert all(x["cached_tokens"] == 0 for x in data["observations"]["reset_serial"])
        assert all(x["cached_tokens"] > 0 for x in data["observations"]["repeat_serial"])
        health = data["profile"]["health"]
        assert health["settings"]["interval_seconds"] == 300
        assert (
            data["finished_at"]
            - data["started_at"]
            + health["bundles"]["default@1"]["last_success_age_seconds"]
        ) < 300
        assert finish["live_postcheck_profiles"][engine]["zero_admission"]
        inputs[engine] = [
            (c["sequence"]["input_ids"], c["sequence"]["label_ids"]) for c in data["cases"]
        ]
        ids = [c["id"] for c in data["cases"]]
        assert len(set(ids)) == 32
        actual_by_state = {}
        for state in STATES:
            rows = data["observations"][state]
            assert [r["id"] for r in rows] == ids
            actual_by_state[state] = [r["logprobs"] for r in rows]
            for row, case in zip(rows, data["cases"], strict=True):
                assert row["prompt_tokens"] == len(case["sequence"]["input_ids"])
                assert len(row["logprobs"]) == len(case["sequence"]["label_ids"])
                assert row["raw_logprobs"] and row["completion_tokens"] == (
                    1 if engine == "vllm" else 0
                )
        output = {
            "references": {},
            "state_changes": {},
            "versions": finish["environments"][engine]["versions"],
        }
        for attention in ("eager", "sdpa"):
            reference = read(DATA / engine / f"reference-{attention}.json")
            published = read(DATA / engine / f"comparison-{attention}.json")
            assert reference["complete"] and reference["script_sha256"] == source_hash
            assert reference["engine_report_sha256"] == sha(DATA / engine / "engine-scores.json")
            assert reference["cases_digest"] == digest(data["cases"])
            assert reference["model"] == data["model"]
            assert reference["reference_contract"]["attention"] == attention
            assert reference["cuda"]["peak_reserved_bytes"] <= reference["cuda"]["budget_bytes"]
            assert [r["id"] for r in reference["rows"]] == ids
            for case, ref in zip(data["cases"], reference["rows"], strict=True):
                assert ref["sequence_digest"] == digest(case["sequence"])
            computed = [
                metrics(actual, ref["logprobs"])
                for state in STATES
                for actual, ref in zip(actual_by_state[state], reference["rows"], strict=True)
            ]
            assert len(published["rows"]) == len(computed) == 96
            for row, values in zip(published["rows"], computed, strict=True):
                assert math.isclose(row["max_absolute_logprob_error"], values[0], abs_tol=1e-12)
                assert math.isclose(
                    row["max_absolute_conditional_probability_error"], values[1], abs_tol=1e-12
                )
                assert row["top_label_equal"] == values[2] and row["passed"] == values[4]
            values = {
                "passed": sum(x[4] for x in computed),
                "total": 96,
                "max_logprob_error": max(x[0] for x in computed),
                "max_probability_error": max(x[1] for x in computed),
                "top_label_mismatches": sum(not x[2] for x in computed),
                "near_tie_cases": sum(x[3] <= 0.3 for x in computed[:32]),
            }
            assert values["passed"] == published["passed_comparisons"]
            assert not published["full_release_gate_passed"]
            output["references"][attention] = values
        for state in STATES[1:]:
            computed = [
                metrics(a, b)
                for a, b in zip(actual_by_state[state], actual_by_state[STATES[0]], strict=True)
            ]
            output["state_changes"][state] = {
                "total": 32,
                "max_logprob_error": max(x[0] for x in computed),
                "max_probability_error": max(x[1] for x in computed),
                "top_label_mismatches": sum(not x[2] for x in computed),
                "all_top_mismatches_near_tie": all(x[3] <= 0.3 for x in computed if not x[2]),
            }
        summary["engines"][engine] = output
    assert inputs["vllm"] == inputs["sglang"]
    summary["cross_engine_input_label_ids_equal"] = True
    (DATA / "verified-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
