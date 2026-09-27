"""Recompute retained cohorts and compare only matched source/profile pairs."""

import hashlib
import json
from pathlib import Path

from jev_runtime.performance import Measurement, summarize_cohort

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw"
BEFORE = "75649bfb633ab5d6219e4b1c6bc4d7e9b9780bd5"
AFTER = "3377c95ca9955ae96ce7c7deb2c4d97b41b1d9c1"


def read(path):
    return json.loads(path.read_text())


def case(root, name):
    report = read(root / name / "report.json")
    assert report["measurement_complete"] and report["all_attempts_successful"]
    assert report["temporary_alias_retired"] and not report.get("cleanup_error")
    assert report["native_typed_probability_parity"]["passed"]
    assert report["native_typed_probability_parity"]["max_absolute_error"] == 0
    assert not report["release_gate_passed"]
    if report["engine"] == "vllm":
        assert report["capabilities"]["api_workers"] == 2
    else:
        assert report["capabilities"]["api_workers"] is None
    assert len(report["measurements"]) == 6
    stages = []
    native, plugin = [], []
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
        (native if item["method"] == "native-label" else plugin).append(item)
        if item["method"] == "native-plugin":
            stages.append(
                {
                    "repeat": item["repeat"],
                    "requests": len(rows),
                    "journal_plus_queue_mean_ms": sum(
                        row.runtime_timings_ms["journal"] + row.runtime_timings_ms["queue"]
                        for row in rows
                    )
                    / len(rows),
                    "runtime_total_mean_ms": sum(row.runtime_timings_ms["total"] for row in rows)
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
    manifest = read(BASE / "atomic-export-manifest-3377c95.json")
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest, name
    harnesses = ROOT / "evidence/harnesses/atomic-admission"

    def wrapper_hash(name):
        return hashlib.sha256((harnesses / name).read_bytes()).hexdigest()

    preflight = read(BASE / "preflight-atomic-3377c95.json")
    assert preflight["owned_records_checked"] == 79
    assert not preflight["matching_live_owned_records"]
    result = {
        "before_source": BEFORE,
        "after_source": AFTER,
        "release_gate_passed": False,
        "qualification": (
            "Same-profile colocated development samples; "
            "not controlled causal or release certification"
        ),
        "comparisons": [],
        "shared_quota_checks": {},
        "failed_setup_preserved": True,
        "evidence_artifact_hashes_verified": len(manifest["artifacts"]),
        "owned_records_checked_after_cleanup": preflight["owned_records_checked"],
    }
    for engine in ("vllm", "sglang"):
        before = BASE / f"{engine}-atomic-before-75649bf-r2"
        after = BASE / f"{engine}-atomic-after-3377c95"
        for root in (before, after):
            attempt = read(root / "performance-attempt.json")
            assert attempt["passed"]
            assert attempt["wrapper_sha256"] == wrapper_hash("run_perf_attempt.py.txt")
            assert all(
                attempt["stages"]["postcheck"][key]
                for key in ("zero_leases", "zero_admission", "all_active_bundles_healthy")
            )
            workers = read(root / "worker-registry.json")
            assert len(workers["workers"]) == 2
            assert len({json.loads(w["identity"])["pid"] for w in workers["workers"]}) == 2
            assert len({w["owner"] for w in workers["prepared"] if w["ref"] == "default@1"}) == 2
            assert workers["retained_counts"] == {"leases": 0, "admission_tickets": 0}
            cleanup = read(root / "cleanup.json")
            assert cleanup["passed"]
            assert not cleanup["remaining_process_group_members"]
            for gpu in cleanup["gpus_after"]:
                prior = next(
                    g for g in cleanup["process_record"]["gpus_before"] if g["uuid"] == gpu["uuid"]
                )
                assert prior["free_mib"] == gpu["free_mib"]
        bp, ap = (read(p / "profile.json") for p in (before, after))
        assert bp["model"] == ap["model"]
        assert bp["admission"]["limits"] == ap["admission"]["limits"]
        assert bp["health"]["settings"] == ap["health"]["settings"]
        assert (
            read(before / "environment.json")["versions"]
            == read(after / "environment.json")["versions"]
        )
        assert (
            read(before / "process.json")["gpus_before"]
            == read(after / "process.json")["gpus_before"]
        )
        for name in ("c1", "c16"):
            b, bs = case(before, name)
            a, ast = case(after, name)
            for key in (
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
            ):
                assert a[key] == b[key], key
            assert b["provided_runtime_source_commit"] == BEFORE
            assert a["provided_runtime_source_commit"] == AFTER
            bf, af = (read(p / name / "fixture.json") for p in (before, after))

            def ids(fixture):
                return [
                    {k: s[k] for k in ("input_ids", "label_ids", "candidate_id", "adapter_id")}
                    for s in fixture["sequences"]
                ]

            assert ids(bf) == ids(af)
            digest = hashlib.sha256(json.dumps(ids(af), sort_keys=True).encode()).hexdigest()
            result["comparisons"].append(
                {
                    "engine": engine,
                    "concurrency": int(name[1:]),
                    "actual_scoring_ids_sha256": digest,
                    "before": bs,
                    "after": ast,
                }
            )
        qr = BASE / f"{engine}-atomic-quota-3377c95"
        q = read(qr / "quota.json")
        assert q["passed"] and read(qr / "quota-attempt.json")["passed"]
        assert read(qr / "quota-attempt.json")["wrapper_sha256"] == wrapper_hash(
            "run_quota_attempt.py.txt"
        )
        assert read(qr / "cleanup.json")["passed"]
        result["shared_quota_checks"][engine] = q["checks"]
    failed = read(BASE / "vllm-atomic-before-75649bf/performance-attempt.json")
    assert not failed["passed"] and failed["stages"]["cleanup"]["passed"]
    assert "c1" not in failed["stages"]
    assert failed["wrapper_sha256"] == wrapper_hash("run_perf_attempt_failed_auth.py.txt")
    result["all_cohort_summaries_recomputed"] = True
    result["total_strict_success_attempts"] = sum(
        c[s]["strict_success_attempts"] for c in result["comparisons"] for s in ("before", "after")
    )
    output = BASE / "atomic-admission-comparison-3377c95.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "total_strict_success_attempts": result["total_strict_success_attempts"],
                "comparisons": result["comparisons"],
            }
        )
    )


if __name__ == "__main__":
    main()
