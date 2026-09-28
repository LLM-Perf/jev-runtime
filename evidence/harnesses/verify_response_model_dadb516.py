"""Recompute unprofiled cohorts and verify separate diagnostic call profiles."""

import hashlib
import json
import pstats
import sqlite3
import subprocess
from pathlib import Path

from verify_combined_reservation_7391ed3 import case

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw/response-model-dadb516"
BEFORE = "7391ed362117ab39a32bfa1dd1554356ee7a6178"
AFTER = "dadb516b38816b32254c710b8f33cba6dd4a6af3"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path, workers):
    value = read(path)
    assert value["timer"] == "cProfile default monotonic wall clock"
    assert value["timing_sanity_passed"] and value["profiled_requests"] == 128
    assert value["statuses"] == {"200": 128} and value["failed_requests"] == 0
    assert value["nested_or_concurrent_skips"] == 0
    assert value["pid"] in {json.loads(w["identity"])["pid"] for w in workers}
    assert value["hook_sha256"] == sha(BASE / "bootstrap/sitecustomize.py")
    stats = pstats.Stats(str(path.with_suffix(".prof")))
    assert stats.total_tt == value["profiler_total_seconds"]
    assert 0 < stats.total_tt <= value["window_wall_seconds"] * 1.05
    assert len(stats.stats) == len(value["functions"])
    for f in value["functions"]:
        primitive, calls, own, cumulative, _ = stats.stats[(f["file"], f["line"], f["name"])]
        assert (primitive, calls, own, cumulative) == (
            f["primitive_calls"],
            f["calls"],
            f["own_seconds"],
            f["cumulative_seconds"],
        )
        assert own >= 0 and cumulative >= 0
    return {
        "lane": value["lane"],
        "profiled_requests": 128,
        "jsonable_encoder_calls": sum(
            f["calls"] for f in value["functions"] if f["name"] == "jsonable_encoder"
        ),
        "jsonable_encoder_cumulative_seconds": sum(
            f["cumulative_seconds"] for f in value["functions"] if f["name"] == "jsonable_encoder"
        ),
        "decision_response_validation_calls": sum(
            f["calls"]
            for f in value["functions"]
            if f["name"] == "verify_outcomes" and f["file"].endswith("/jev_runtime/schema.py")
        ),
        "window_wall_seconds": value["window_wall_seconds"],
        "profiler_total_seconds": stats.total_tt,
    }


def main():
    manifest = read(BASE / "export-manifest.json")
    for name, info in manifest["artifacts"].items():
        p = BASE / name
        assert sha(p) == info["sha256"] and p.stat().st_size == info["size_bytes"], name
    audit = read(BASE / "audit.json")
    campaign = read(BASE / "campaign.json")
    assert audit["passed"] and campaign["complete"] and len(campaign["attempts"]) == 4
    assert audit["credential_values_absent_from_export"]
    assert len(audit["records"]) == audit["owned_records_checked"]
    assert all(not r["matching_live_process"] and not r["live_group"] for r in audit["records"])
    assert int(audit["gpu7"]["free_mib"]) == 11990
    assert campaign["script_sha256"] == sha(BASE / "campaign.py")
    assert audit["script_sha256"] == sha(BASE / "export.py")
    assert campaign["modified_runner_sha256"] == sha(BASE / "run_case_profile.py")
    prior = ROOT / "evidence/dsw/combined-reservation-7391ed3/audit.json"
    assert sha(prior) == audit["prior_Hub_verified_audit_sha256"]
    assert len(audit["model_files"]) == 8
    assert all(
        f["hub_digest_verified"] and f["unchanged_since_Hub_verified_audit"]
        for f in audit["model_files"]
    )
    count = 0
    for tag, commit in [("before", BEFORE), ("after", AFTER)]:
        source = audit["sources"][tag]
        assert source["commit"] == commit
        for path, digest in source["python_files"].items():
            data = subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT)
            assert hashlib.sha256(data).hexdigest() == digest
            count += 1
    initial = read(BASE / "initial/campaign.json")
    recovery = read(BASE / "initial/recovery.json")
    assert not initial["complete"] and recovery["complete"] and recovery["cleanup"]["passed"]
    assert not read(BASE / "initial/sglang/cleanup.json")["passed"]
    remediation = read(BASE / "initial/sglang/cleanup-remediation.json")
    assert not remediation["remaining_group"] and len(remediation["pending_recovery"]) == 2
    assert remediation["signal"] == "SIGKILL"
    assert all(c["recoverable"] and c["owner_status"] == "dead" for c in recovery["candidates"])
    assert len(recovery["candidates"]) == len(recovery["recovered"]) == 2
    assert all(c["response"] == {"recovered": True} for c in recovery["recovered"])
    for which, expected in [("before", 2), ("after", 0)]:
        p = BASE / "initial" / ("recovery-" + which + ".sqlite")
        assert sha(p) == recovery[which + "_snapshot"]["sha256"]
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            for table in ["leases", "lease_work"]:
                assert db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0] == expected
    invalid = [read(p) for p in (BASE / "initial/sglang/cpu-profiles").glob("*.json")]
    assert len(invalid) == 2 and all(
        p["profiler_total_cpu_seconds"] > p["wall_seconds"] for p in invalid
    )
    assert any(f["cumulative_cpu_seconds"] < 0 for p in invalid for f in p["functions"])
    contracts = read(BASE / "contracts.json")
    assert contracts["complete"] and contracts["source_commit"] == AFTER
    assert contracts["script_sha256"] == sha(BASE / "contracts.py")
    for engine in ["vllm", "sglang"]:
        assert contracts["engines"][engine]["returncode"] == 0
        assert "40 passed" in (BASE / ("contracts-" + engine + ".log")).read_text()
    result = {
        "before_source": BEFORE,
        "after_source": AFTER,
        "release_gate_passed": False,
        "qualification": (
            "Matched colocated C1 samples with an inactive diagnostic header gate; "
            "no causal or release certificate."
        ),
        "profiling_qualification": (
            "Separate continuous default-clock windows include idle/background "
            "callbacks. The copied runner retains stale thread-CPU wording; collector metadata "
            "and this qualification supersede it."
        ),
        "comparisons": [],
        "call_profiles": {},
        "artifact_hashes_verified": len(manifest["artifacts"]),
        "source_python_files_verified": count,
        "owned_records_checked": audit["owned_records_checked"],
    }
    for engine in ["vllm", "sglang"]:
        pair = []
        for tag, commit in [("before", BEFORE), ("after", AFTER)]:
            root = BASE / (engine + "-" + tag)
            attempt = next(
                x for x in campaign["attempts"] if x["engine"] == engine and x["tag"] == tag
            )
            assert (
                attempt["complete"]
                and attempt["profile_complete"]
                and attempt["measurement_complete"]
            )
            assert attempt["cleanup"]["passed"] and not attempt["quiesce"]["leases"]
            assert attempt["quiesce"]["disabled_routes"]
            workers = read(root / "worker-registry.json")
            assert len(workers["workers"]) == 2 and set(workers["retained_counts"].values()) == {0}
            profiles = [
                profile(p, workers["workers"]) for p in (root / "call-profiles").glob("*.json")
            ]
            assert len(profiles) == 2 and {p["lane"] for p in profiles} == {
                "native-label",
                "native-plugin",
            }
            result["call_profiles"][engine + "-" + tag] = profiles
            for p in profiles:
                expected = 10240 if tag == "before" and p["lane"] == "native-plugin" else 0
                assert p["jsonable_encoder_calls"] == expected
            report, summary = case(root, "measurement")
            assert report["provided_runtime_source_commit"] == commit
            assert report["source_commit"] == BEFORE
            pair.append((report, summary, read(root / "measurement/fixture.json")))
        (b, bs, bf), (a, ast, af) = pair
        for key in [
            "scenario",
            "model",
            "service_launch_command",
            "capabilities",
            "compiler_profile",
            "tokenizer_implementation_digest",
            "duration_seconds",
            "repeats",
            "warmup_requests_per_method",
            "source_commit",
        ]:
            assert a[key] == b[key], key

        def ids(f):
            return [
                {k: s[k] for k in ["input_ids", "label_ids", "candidate_id", "adapter_id"]}
                for s in f["sequences"]
            ]

        assert ids(bf) == ids(af)
        bp, ap = [read(BASE / (engine + "-" + tag) / "profile.json") for tag in ["before", "after"]]
        assert bp["model"] == ap["model"] and bp["admission"]["limits"] == ap["admission"]["limits"]
        assert (
            bp["health"]["settings"]
            == ap["health"]["settings"]
            == {"interval_seconds": 300.0, "timeout_seconds": 10.0, "max_age_seconds": 900.0}
        )
        result["comparisons"].append(
            {"engine": engine, "concurrency": 1, "before": bs, "after": ast}
        )
    result["total_unprofiled_strict_success_attempts"] = sum(
        c[s]["strict_success_attempts"] for c in result["comparisons"] for s in ["before", "after"]
    )
    result["initial_invalid_timing_and_shutdown_failure_preserved"] = True
    result["recovered_health_probe_leases"] = 2
    result["engine_environment_contract_tests"] = {"vllm": 40, "sglang": 40}
    (BASE / "verified-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
