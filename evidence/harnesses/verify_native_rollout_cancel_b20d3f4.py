"""Verify failure-inclusive native scoring/cancellation and installed-wheel evidence."""

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw/native-rollout-cancel-b20d3f4"
SCRIPTS = ROOT / "evidence/harnesses/native-rollout-cancel"
SOURCE = "b20d3f43d91c2e13ed2b044a77d35cc3037eb62b"
GREEN = "6f0cda85f72619193be7a1d1cc89d75142899bf3"
BLUE = "773914587b4fab287ca1e4b17f60ff66eb20d261"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha(commit, path):
    return hashlib.sha256(
        subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    ).hexdigest()


def progress(case, bundle_digest):
    before, after = case["progress_before"], case["progress_after"]
    journal = case["journal_before"]
    assert journal["phase"] == "inflight" and journal["branches"] == 128
    assert before["worker_id"] == after["worker_id"] == journal["worker_id"]
    for item in (before, after):
        request = item["request"]
        assert item["scope"] == "local_worker" and request["stage"] == "execute"
        assert request["request_id"] == case["request_id"]
        assert request["lease_id"] == journal["lease_id"]
        assert request["bundle"] == journal["bundle"]
        assert request["bundle_digest"] == bundle_digest
        assert request["scoring_sequences"] == 128
        calls = request["work"]["engine_call"]
        assert 2 <= calls["succeeded"] < 128
        assert calls["started"] == calls["succeeded"] + 1
        assert calls["active"] == 1 and calls["failed_or_cancelled"] == 0
    assert before["request"]["generation"] == after["request"]["generation"]
    assert after["request"]["elapsed_seconds"] >= before["request"]["elapsed_seconds"]
    completed = [item["request"]["work"]["engine_call"]["succeeded"] for item in (before, after)]
    assert completed[1] >= completed[0]
    switch = case["switch"]
    assert switch["source"] == "green" and switch["target"] == switch["observed"] == "blue"
    assert switch["outcome"] == "switched"
    assert switch["source_snapshot"]["deployment"]["release"] == GREEN
    assert switch["target_snapshot"]["deployment"]["release"] == BLUE
    assert journal["worker_id"] in switch["source_snapshot"]["workers"]
    assert not case["drain_while_active"]["drained"]
    assert case["drain_while_active"]["remaining_leases"] >= 1
    cancellation = case["cancellation"]
    assert cancellation["slot"] == "blue" and cancellation["request_http_status"] == 499
    assert cancellation["lease_released"] and case["serving_after_cancel"]
    # The old blue release does not emit X-Jev-Worker; its deployment is observable.
    if cancellation["worker"] is not None:
        assert cancellation["worker"] in switch["target_snapshot"]["workers"]
    drained = case["drained_after_cancel"]
    assert drained["drained"] and drained["remaining_leases"] == 0
    assert all(row["scur"] == row["qcur"] == 0 for row in drained["stats"].values())
    return {"completed_rpc_before": completed[0], "completed_rpc_after": completed[1]}


def main():
    export = read(BASE / "export-manifest.json")
    for name, metadata in export["artifacts"].items():
        path = BASE / name
        assert sha(path) == metadata["sha256"] and path.stat().st_size == metadata["size_bytes"]
    audit = read(BASE / "audit.json")
    assert audit["passed"] and audit["source_commit"] == SOURCE
    assert audit["script_sha256"] == export["script_sha256"] == sha(SCRIPTS / "audit_export.py.txt")
    assert audit["prior_audit_sha256"] == sha(
        ROOT / "evidence/dsw/gateway-rollout-7739145/audit.json"
    )
    identities = {
        (row["identity"]["boot_id"], row["identity"]["pid"], row["identity"]["start_ticks"])
        for row in audit["records"]
    }
    assert audit["owned_records_checked"] == len(audit["records"]) == len(identities) == 145
    assert all(
        not row["matching_live_process"] and not row["live_group"] for row in audit["records"]
    )
    assert int(audit["gpu7"]["free_mib"]) == 11990
    assert len(audit["model_files"]) == 6 and all(
        row["hub_digest_verified"] for row in audit["model_files"]
    )
    assert len(audit["profiles"]) == 2 and all(row["payload_verified"] for row in audit["profiles"])
    for commit, paths in audit["source_files"].items():
        for path, expected in paths.items():
            assert git_sha(commit, path) == expected
    core = {
        commit: {
            path.removeprefix("src/jev_runtime/"): value
            for path, value in paths.items()
            if path.startswith("src/jev_runtime/")
        }
        for commit, paths in audit["source_files"].items()
    }
    assert core[SOURCE] == core[GREEN] and len(core[GREEN]) == 30
    first, retest = read(BASE / "initial-attempt.json"), read(BASE / "retest.json")
    assert first["completed"] and not first["passed"] and first["source_commit"] == GREEN
    assert retest["completed"] and retest["passed"] and retest["source_commit"] == SOURCE
    assert retest["candidate_wheel_source"] == GREEN and retest["previous_source"] == BLUE
    assert first["script_sha256"] == sha(SCRIPTS / "campaign.py.txt")
    assert retest["script_sha256"] == sha(SCRIPTS / "retest_campaign.py.txt")
    assert retest["initial_report_sha256"] == sha(BASE / "initial-attempt.json")
    assert retest["installation"] == first["installation"]
    install = retest["installation"]
    assert install["wheel_manifest_sha256"] == sha(BASE / "wheels-6f0cda8-manifest.json")
    assert install["gateway_manifest_sha256"] == sha(BASE / "gateway-6f0cda8-manifest.json")
    assert install["receipt"]["source_commit"] == GREEN
    assert install["receipt"]["manifest_sha256"] == install["gateway_manifest_sha256"]
    assert install["receipt"]["verified_core_files"] == 30
    assert len(read(BASE / "gateway-6f0cda8-manifest.json")["distributions"]) == 44
    rows, failures = [], []
    for campaign, commit, successful in ((first, GREEN, False), (retest, SOURCE, True)):
        assert len(campaign["attempts"]) == 2
        assert {a["engine"] for a in campaign["attempts"]} == {"vllm", "sglang"}
        for attempt in campaign["attempts"]:
            engine = attempt["engine"]
            report = attempt["gateway"]
            assert attempt["passed"] is successful and report["passed"] is successful
            assert attempt["same_native_engine_identity"] and attempt["cleanup"]["passed"]
            assert report["source_commit"] == commit and report["fixture"] is False
            assert report["script_sha256"] == git_sha(commit, "tests/integration/live_rollout.py")
            process = read(BASE / f"native-{engine}-{commit[:7]}/process.json")
            cleanup = read(BASE / f"native-{engine}-{commit[:7]}/cleanup.json")
            assert cleanup == attempt["cleanup"] and cleanup["passed"]
            assert report["native_engine_identity"] == process["identity"]
            for slot, release in (("blue", BLUE), ("green", GREEN)):
                code = report["installed_code"][slot]
                assert "/site-packages/jev_runtime" in code["path"]
                assert code["files"] == core[release]
                assert report["slot_releases"][slot] == release
            assert len(report["processes"]) == len(report["cleanup"]) == 3
            assert all(not row["remaining"] for row in report["cleanup"])
            traffic = report["traffic"]
            assert len(traffic) == report["traffic_successes"] and report["traffic_errors"] == 0
            assert all(
                row["status"] == 200 and row["bundle"] == "default@1" and row["generation"] == 1
                for row in traffic
            )
            if not successful:
                assert attempt["returncode"] != 0 and "cancellation" not in attempt
                assert report["failure"] == {
                    "type": "RuntimeError",
                    "message": "Cannot open a client instance more than once.",
                }
                failures.append(
                    {
                        "engine": engine,
                        "phase": "admin connection setup",
                        "continuous_requests_before_failure": len(traffic),
                    }
                )
                continue
            assert attempt["returncode"] == 0
            checks = report["checks"]
            for key in (
                "partial_body_survives_switch_and_prevents_early_drain",
                "keepalive_next_request_uses_new_slot",
                "incorrect_release_rejected",
                "same_proxy_pid",
            ):
                assert checks[key]
            assert checks["continuous_switches"] == 10
            assert checks["reconciled"]["generation"] == 19
            assert checks["reconciled"]["outcome"] == "reconciled_observation"
            for slot in ("blue", "green"):
                qualified = checks[slot + "_qualified"]
                assert len(qualified["workers"]) == qualified["deployment"]["expected_workers"] == 2
            cancellation = attempt["cancellation"]
            assert cancellation == checks["native_cancellation_across_switch"]
            assert cancellation["passed"] and cancellation["bundle_retired"]
            assert (
                cancellation["source_slot"] == "green"
                and cancellation["planned_branches_per_request"] == 128
            )
            assert len(cancellation["attempts"]) == 3
            counts = Counter(row["slot"] for row in traffic)
            assert set(counts) == {"blue", "green"}
            rows.append(
                {
                    "engine": engine,
                    "continuous_requests": len(traffic),
                    "continuous_errors": 0,
                    "by_slot": dict(counts),
                    "proxy_map_generations": 19,
                    "cancellation_cases": [
                        progress(case, cancellation["bundle_digest"])
                        for case in cancellation["attempts"]
                    ],
                }
            )
    summary = {
        "verified": True,
        "harness_source": SOURCE,
        "candidate_wheel_source": GREEN,
        "previous_wheel_source": BLUE,
        "initial_failures_preserved": failures,
        "results": rows,
        "owned_records_terminal": 145,
        "gpu7_free_mib": 11990,
        "model_files_rehashed": 6,
        "qualification": (
            "One colocated BF16 SmolLM2 TP1 profile, two gateway workers per slot; "
            "native scoring RPC progress and cancellation, not CUDA kernel occupancy, "
            "quality or performance certification."
        ),
    }
    (BASE / "verified-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
