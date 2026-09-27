"""Verify rollout traffic accounting, installed source bytes and terminal ownership."""

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw/gateway-rollout-7739145"
SCRIPTS = ROOT / "evidence/harnesses/gateway-rollout"
PREVIOUS = "accef862f11bcc8ebb666225f328f8f088270713"
SOURCE = "773914587b4fab287ca1e4b17f60ff66eb20d261"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_bytes(commit, name):
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)


def check_report(report, source, fixture=False):
    assert report["passed"] and report["source_commit"] == source
    assert report["fixture"] is fixture
    assert (
        report["script_sha256"]
        == hashlib.sha256(git_bytes(source, "tests/integration/live_rollout.py")).hexdigest()
    )
    rows = report["traffic"]
    assert rows and len(rows) == report["traffic_successes"] and report["traffic_errors"] == 0
    assert all(
        row["status"] == 200
        and row["generation"] == 1
        and row["bundle"] == "default@1"
        and row["latency_ms"] >= 0
        for row in rows
    )
    counts = Counter(row["slot"] for row in rows)
    assert set(counts) == {"blue", "green"}
    checks = report["checks"]
    for name in (
        "partial_body_survives_switch_and_prevents_early_drain",
        "keepalive_next_request_uses_new_slot",
        "incorrect_release_rejected",
        "same_proxy_pid",
    ):
        assert checks[name] is True
    assert checks["old_slot_drain"]["drained"]
    assert checks["continuous_switches"] == 10
    assert checks["reconciled"]["generation"] == (14 if fixture else 13)
    assert checks["reconciled"]["outcome"] == "reconciled_observation"
    if fixture:
        assert checks["peer_cancellation_across_switch"]
    for slot in ("blue", "green"):
        qualified = checks[slot + "_qualified"]
        assert len(qualified["workers"]) == qualified["deployment"]["expected_workers"] == 2
    assert len(report["processes"]) == len(report["cleanup"]) == 3
    assert all(not item["remaining"] for item in report["cleanup"])
    return {
        "source": source,
        "fixture": fixture,
        "requests": len(rows),
        "errors": 0,
        "by_slot": dict(counts),
        "switches": checks["reconciled"]["generation"],
    }


def main():
    export = read(BASE / "export-manifest.json")
    for name, item in export["artifacts"].items():
        p = BASE / name
        assert sha(p) == item["sha256"] and p.stat().st_size == item["size_bytes"]
    audit = read(BASE / "audit.json")
    assert audit["passed"] and audit["source_commit"] == SOURCE
    assert (
        audit["script_sha256"]
        == export["script_sha256"]
        == sha(SCRIPTS / "audit_export-7739145.py.txt")
    )
    assert audit["owned_records_checked"] == len(audit["records"])
    assert all(not r["matching_live_process"] and not r["live_group"] for r in audit["records"])
    assert int(audit["gpu7"]["free_mib"]) == 11990
    assert len(audit["model_files"]) == 6 and all(
        f["hub_digest_verified"] for f in audit["model_files"]
    )
    assert len(audit["profiles"]) == 2 and all(p["payload_verified"] for p in audit["profiles"])
    assert (
        audit["proxy"]["source_sha256"]
        == "f765638cc4819f25e20d974a4a1bc24ed54342467c88363d7fc34a1fa95b725a"
    )
    for commit, paths in audit["source_files"].items():
        for path, expected in paths.items():
            assert hashlib.sha256(git_bytes(commit, path)).hexdigest() == expected
    rows = [check_report(read(BASE / "same-code-fixture.json"), PREVIOUS, True)]
    for name, source, script in (
        ("same-code-native.json", PREVIOUS, "native_campaign-accef86.py.txt"),
        ("installed-wheel-native.json", SOURCE, "wheel_native_campaign-7739145.py.txt"),
    ):
        campaign = read(BASE / name)
        assert campaign["completed"] and campaign["passed"] and campaign["source_commit"] == source
        assert campaign["script_sha256"] == sha(SCRIPTS / script)
        assert {a["engine"] for a in campaign["attempts"]} == {"sglang", "vllm"}
        for attempt in campaign["attempts"]:
            assert attempt["passed"] and attempt["returncode"] == 0 and attempt["cleanup"]["passed"]
            report = attempt["gateway"]
            row = check_report(report, source)
            row.update(engine=attempt["engine"], installed_wheels=source == SOURCE)
            if source == SOURCE:
                assert attempt["same_native_engine_identity"]
                assert (
                    report["proxy_identity"]["boot_id"] and report["proxy_identity"]["start_ticks"]
                )
                for slot, commit in (("blue", PREVIOUS), ("green", SOURCE)):
                    assert report["slot_releases"][slot] == commit
                    code = report["installed_code"][slot]
                    assert "/site-packages/jev_runtime" in code["path"]
                    expected = {
                        path.removeprefix("src/jev_runtime/"): value
                        for path, value in audit["source_files"][commit].items()
                        if path.startswith("src/jev_runtime/")
                    }
                    assert len(expected) == 30 and code["files"] == expected
            rows.append(row)
    package = read(BASE / "package-install.json")
    assert package["passed"] and package["source_commit"] == SOURCE
    assert package["script_sha256"] == sha(SCRIPTS / "package_campaign-7739145.py.txt")
    assert len(package["profiles"]) == 2
    for row in package["profiles"]:
        short = row["source"]
        assert row["wheel_manifest_sha256"] == sha(BASE / f"wheels-{short}-manifest.json")
        assert row["gateway_manifest_sha256"] == sha(BASE / f"gateway-{short}-manifest.json")
        assert read(BASE / f"gateway-{short}-manifest.json")["source_commit"] in {SOURCE, PREVIOUS}
    summary = {
        "verified": True,
        "source_commit": SOURCE,
        "rows": rows,
        "owned_records_terminal": len(audit["records"]),
        "gpu7_free_mib": 11990,
        "model_files_rehashed": 6,
        "installed_core_files_per_slot": 30,
        "limitation": (
            "Same-host compatible gateway releases only; not engine failover, multi-node HA, "
            "performance, quality, or full release acceptance"
        ),
    }
    (BASE / "verified-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
