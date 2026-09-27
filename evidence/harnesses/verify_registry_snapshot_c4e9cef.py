"""Verify scoped snapshot/restore metadata, source identities and real serving evidence."""

import hashlib
import json
import subprocess
from pathlib import Path

from jev_runtime.schema import DecisionResponse

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw/registry-snapshot-c4e9cef"
SCRIPTS = ROOT / "evidence/harnesses/registry-snapshot"
SOURCE = "c4e9cefa6f6d97fc283513b04836f4d0079bcfd8"
NATIVE = "b20d3f43d91c2e13ed2b044a77d35cc3037eb62b"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    exported = read(BASE / "export-manifest.json")
    for name, item in exported["artifacts"].items():
        path = BASE / name
        assert sha(path) == item["sha256"] and path.stat().st_size == item["size_bytes"]
        assert not name.endswith((".db", ".sqlite3", ".log"))
    audit = read(BASE / "audit.json")
    assert audit["passed"] and audit["source_commit"] == SOURCE
    assert (
        audit["script_sha256"] == exported["script_sha256"] == sha(SCRIPTS / "audit_export.py.txt")
    )
    assert audit["prior_audit_sha256"] == sha(
        ROOT / "evidence/dsw/image-entrypoint-49fea1c/audit.json"
    )
    identities = {
        (r["identity"]["boot_id"], r["identity"]["pid"], r["identity"]["start_ticks"])
        for r in audit["records"]
    }
    assert len(identities) == len(audit["records"]) == audit["owned_records_checked"] == 159
    assert all(not r["matching_live_process"] and not r["live_group"] for r in audit["records"])
    assert int(audit["gpu7"]["free_mib"]) == 11990
    assert len(audit["model_files"]) == 6 and all(
        r["hub_digest_verified"] for r in audit["model_files"]
    )
    assert len(audit["profiles"]) == 2 and all(r["payload_verified"] for r in audit["profiles"])
    for path, expected in audit["source_files"].items():
        contents = subprocess.check_output(["git", "show", f"{SOURCE}:{path}"], cwd=ROOT)
        assert hashlib.sha256(contents).hexdigest() == expected
    core = {
        path.removeprefix("src/jev_runtime/"): value
        for path, value in audit["source_files"].items()
        if path.startswith("src/jev_runtime/")
    }
    assert len(core) == 31
    for env in audit["environments"].values():
        assert all("/releases/" + NATIVE + "/" in path for path in env["imports"].values())
    checks = {(row["engine"], row["label"]): row for row in audit["snapshot_payload_checks"]}
    assert len(checks) == 6 and all(row["payload_verified"] for row in checks.values())
    campaign = read(BASE / "campaign.json")
    assert campaign["completed"] and campaign["passed"] and campaign["source_commit"] == SOURCE
    assert campaign["native_source"] == NATIVE and campaign["script_sha256"] == sha(
        SCRIPTS / "campaign.py.txt"
    )
    assert len(campaign["attempts"]) == 2
    assert {row["engine"] for row in campaign["attempts"]} == {"sglang", "vllm"}
    results = []
    for attempt in campaign["attempts"]:
        engine = attempt["engine"]
        assert attempt["passed"] and attempt["same_native_engine_identity"]
        assert attempt["gateway_imports"]["files"] == core
        assert SOURCE + "/src" in attempt["gateway_imports"]["path"]
        assert (
            attempt["live_restore_rejection"]
            == 'Source is not safely stopped/drained: {"owners_alive": 2}'
        )
        assert "changed since snapshot" in attempt["stale_restore_rejection"]
        assert (
            attempt["stale_generation_rejected"]
            and attempt["original_unchanged_after_serving_restored"]
        )
        assert not attempt["restored_blockers"] and attempt["native_cleanup"]["passed"]
        assert read(BASE / f"native-{engine}/cleanup.json") == attempt["native_cleanup"]
        assert len(attempt["gateways"]) == 2
        for gateway in attempt["gateways"]:
            assert (
                gateway["cleanup"]["returncode"] == 0 and not gateway["cleanup"]["remaining_group"]
            )
            assert gateway["qualified"]["deployment"]["release"] == SOURCE
            assert len(gateway["qualified"]["workers"]) == 2
        for label, key in [
            ("historical", "historical_snapshot"),
            ("live", "live_snapshot"),
            ("final", "final_snapshot"),
        ]:
            path = BASE / engine / f"{label}-snapshot-manifest.json"
            assert (
                sha(path)
                == checks[(engine, label)]["manifest_sha256"]
                == attempt[key]["manifest_sha256"]
            )
            manifest = read(path)
            assert manifest["format"] == "jev-registry-snapshot-v1"
            assert manifest["summary"] == attempt[key]["summary"]
        final = read(BASE / engine / "final-snapshot-manifest.json")
        assert not final["restore_blockers_at_snapshot"]
        staged = read(BASE / engine / "restored-restore.json")
        assert staged["format"] == "jev-registry-staged-restore-v1" and staged["activated"] is False
        assert staged["snapshot_manifest_sha256"] == attempt["final_snapshot"]["manifest_sha256"]
        assert (
            staged["database_sha256"]
            == final["database_sha256"]
            == attempt["restore"]["database_sha256"]
        )
        assert sha(BASE / engine / "restored-restore.json") == attempt["restore"]["receipt_sha256"]
        assert (
            staged["summary"] == final["summary"] == attempt["original_state_before_restored_start"]
        )
        assert attempt["restore"]["summary"] == final["summary"]
        assert (
            attempt["restored_final_summary"]["tables"]["routes"]
            == final["summary"]["tables"]["routes"]
        )
        for table in ["leases", "lease_work", "lease_tenants", "admission_tickets"]:
            assert final["summary"]["tables"][table]["rows"] == 0
            assert attempt["restored_final_summary"]["tables"][table]["rows"] == 0
        rows = [(row, 1) for row in attempt["before_decisions"]]
        rows += [(attempt["generation_three_decision"], 3)]
        rows += [(row, 3) for row in attempt["after_decisions"]]
        assert len(attempt["before_decisions"]) == len(attempt["after_decisions"]) == 10
        for payload, generation in rows:
            response = DecisionResponse.model_validate(payload)
            assert response.status == "completed" and response.usage.successful_questions == 1
            assert response.bundle == "default@1" and response.generation == generation
            assert response.engine["name"] == engine
        assert len({row[0]["bundle_digest"] for row in rows}) == 1
        results.append(
            {
                "engine": engine,
                "before_requests": 10,
                "after_requests": 10,
                "generation_three_control_requests": 1,
                "restored_generation": 3,
                "live_and_stale_restore_rejected": True,
                "old_generation_rejected": True,
                "original_state_preserved": True,
                "zero_final_leases": True,
            }
        )
    result = {
        "verified": True,
        "source_commit": SOURCE,
        "native_source": NATIVE,
        "results": results,
        "owned_records_terminal": 159,
        "gpu7_free_mib": 11990,
        "snapshot_payloads_verified_remotely": 6,
        "qualification": (
            "Same-host staged copy of identical stopped state; no schema migration, "
            "lost-source recovery, container replacement or performance gate."
        ),
    }
    (BASE / "verified-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
