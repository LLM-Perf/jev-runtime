"""Verify artifacts and recompute every matched performance cohort from raw rows."""

import hashlib
import json
import subprocess
from pathlib import Path

from jev_runtime.performance import Measurement, summarize_cohort

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw/combined-reservation-7391ed3"
BEFORE = "d8a2649339da391e506dd9cab17570c659267716"
AFTER = "7391ed362117ab39a32bfa1dd1554356ee7a6178"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case(root, name):
    report = read(root / name / "report.json")
    assert report["measurement_complete"] and report["all_attempts_successful"]
    assert report["temporary_alias_retired"] and not report.get("cleanup_error")
    assert report["native_typed_probability_parity"]["passed"]
    assert report["native_typed_probability_parity"]["max_absolute_error"] == 0
    assert not report["release_gate_passed"]
    assert len(report["measurements"]) == 6
    stages, native, plugin = [], [], []
    attempts = 0
    for item in report["measurements"]:
        summary = item["summary"]
        rows = [
            Measurement.model_validate_json(line)
            for line in (root / name / item["rows"]).read_text().splitlines()
        ]
        actual = summarize_cohort(rows, summary["window_seconds"], summary["makespan_seconds"])
        assert actual == summary, (root.name, name, item["rows"])
        assert summary["dispatched_requests"] == summary["strict_success_requests"] > 0
        assert summary["latency_successful_requests"]["p99_ms"] is None
        attempts += len(rows)
        assert item["method"] in ["native-label", "native-plugin"]
        (native if item["method"] == "native-label" else plugin).append(item)
        if item["method"] == "native-plugin":
            stages.append(
                {
                    "repeat": item["repeat"],
                    "requests": len(rows),
                    "pin_journal_queue_mean_ms": sum(
                        sum(row.runtime_timings_ms[k] for k in ["pin", "journal", "queue"])
                        for row in rows
                    )
                    / len(rows),
                    "runtime_total_mean_ms": sum(row.runtime_timings_ms["total"] for row in rows)
                    / len(rows),
                    "compile_mean_ms": sum(row.runtime_timings_ms["compile"] for row in rows)
                    / len(rows),
                    "release_mean_ms": sum(row.runtime_timings_ms["release"] for row in rows)
                    / len(rows),
                    "client_mean_ms": sum(row.finished_ms - row.started_ms for row in rows)
                    / len(rows),
                }
            )
    ratios = []
    for i in (1, 2, 3):
        n = next(m["summary"] for m in native if m["repeat"] == i)
        p = next(m["summary"] for m in plugin if m["repeat"] == i)
        ratios.append(
            p["strict_success_rps_over_full_cohort"] / n["strict_success_rps_over_full_cohort"]
        )
    return report, {
        "strict_success_attempts": attempts,
        "plugin_native_rps_ratios": ratios,
        "plugin_stages": stages,
        "native_rps": [m["summary"]["strict_success_rps_over_full_cohort"] for m in native],
        "plugin_rps": [m["summary"]["strict_success_rps_over_full_cohort"] for m in plugin],
    }


def main():
    manifest = read(BASE / "export-manifest.json")
    for name, expected in manifest["artifacts"].items():
        p = BASE / name
        assert sha(p) == expected["sha256"] and p.stat().st_size == expected["size_bytes"], name
    campaign = read(BASE / "campaign.json")
    quota = read(BASE / "quota-campaign.json")
    audit = read(BASE / "audit.json")
    assert not campaign["complete"] and quota["complete"] and audit["passed"]
    assert len(campaign["attempts"]) == 4 and len(quota["attempts"]) == 2
    assert campaign["script_sha256"] == sha(BASE / "campaign.py")
    assert quota["script_sha256"] == sha(BASE / "quota_campaign.py")
    assert audit["script_sha256"] == sha(BASE / "export.py")
    assert campaign["linux_contracts_returncode"] == 1
    assert "No module named pytest" in (BASE / "linux-contracts.log").read_text()
    assert quota["linux_contracts_returncode"] == 0
    assert "49 passed" in (BASE / "linux-contracts-isolated.log").read_text()
    assert len(audit["records"]) == audit["owned_records_checked"]
    assert all(not r["matching_live_process"] and not r["live_group"] for r in audit["records"])
    assert audit["credential_values_absent_from_export"]
    assert int(audit["gpu7"]["free_mib"]) == 11990
    assert len(audit["model_files"]) == 8 and all(
        x["hub_digest_verified"] for x in audit["model_files"]
    )
    source_files = 0
    for tag, commit in [("before", BEFORE), ("after", AFTER)]:
        entry = audit["sources"][tag]
        assert entry["commit"] == commit
        assert entry["archive_sha256"] == campaign["sources"][tag]["sha256"]
        for path, digest in entry["python_files"].items():
            data = subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT)
            assert hashlib.sha256(data).hexdigest() == digest, (commit, path)
            source_files += 1
    result = {
        "before_source": BEFORE,
        "after_source": AFTER,
        "release_gate_passed": False,
        "qualification": (
            "Matched colocated development samples; not controlled causal or release certification"
        ),
        "comparisons": [],
        "shared_quota_checks": {},
        "evidence_artifact_hashes_verified": len(manifest["artifacts"]),
        "source_python_files_verified": source_files,
        "owned_records_checked_after_cleanup": audit["owned_records_checked"],
        "linux_contract_tests_passed": 49,
        "initial_linux_runner_missing_pytest_preserved": True,
    }
    for engine in ["vllm", "sglang"]:
        before, after = (BASE / (engine + "-" + tag) for tag in ["before", "after"])
        for tag in ["before", "after", "quota"]:
            root = BASE / (engine + "-" + tag)
            attempt = read(
                root / ("quota-attempt.json" if tag == "quota" else "performance-attempt.json")
            )
            assert attempt["passed"]
            assert attempt["wrapper_sha256"] == sha(
                BASE / ("run_quota.py" if tag == "quota" else "run_perf.py")
            )
            assert all(
                attempt["stages"]["postcheck"][key]
                for key in ["zero_leases", "zero_admission", "all_active_bundles_healthy"]
            )
            workers = read(root / "worker-registry.json")
            assert len(workers["workers"]) == 2
            assert len({json.loads(w["identity"])["pid"] for w in workers["workers"]}) == 2
            assert len({w["owner"] for w in workers["prepared"] if w["ref"] == "default@1"}) == 2
            assert set(workers["retained_counts"].values()) == {0}
            cleanup = read(root / "cleanup.json")
            assert cleanup["passed"] and not cleanup["remaining_process_group_members"]
            assert int(cleanup["gpus_after"][0]["free_mib"]) == 11990
        bp, ap = (read(p / "profile.json") for p in [before, after])
        assert bp["model"] == ap["model"]
        assert bp["admission"]["limits"] == ap["admission"]["limits"]
        assert bp["health"]["settings"] == ap["health"]["settings"]
        assert (
            read(before / "environment.json")["versions"]
            == read(after / "environment.json")["versions"]
        )
        for name in ["c1", "c16"]:
            b, bs = case(before, name)
            a, ast = case(after, name)
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
            assert b["provided_runtime_source_commit"] == BEFORE
            assert a["provided_runtime_source_commit"] == AFTER
            fixtures = [read(p / name / "fixture.json") for p in [before, after]]

            def ids(f):
                return [
                    {k: s[k] for k in ["input_ids", "label_ids", "candidate_id", "adapter_id"]}
                    for s in f["sequences"]
                ]

            assert ids(fixtures[0]) == ids(fixtures[1])
            digest = hashlib.sha256(
                json.dumps(ids(fixtures[1]), sort_keys=True).encode()
            ).hexdigest()
            result["comparisons"].append(
                {
                    "engine": engine,
                    "concurrency": int(name[1:]),
                    "actual_scoring_ids_sha256": digest,
                    "before": bs,
                    "after": ast,
                }
            )
        q = read(BASE / (engine + "-quota") / "quota.json")
        assert q["passed"]
        result["shared_quota_checks"][engine] = q["checks"]
    result["all_cohort_summaries_recomputed"] = True
    result["total_strict_success_attempts"] = sum(
        c[s]["strict_success_attempts"] for c in result["comparisons"] for s in ["before", "after"]
    )
    (BASE / "verified-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
