"""Verify retained recovery/blocked-write/resume evidence and SQLite snapshots."""

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

from jev_runtime.registry_backup import verify_snapshot
from jev_runtime.schema import Bundle, DecisionResponse

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence/dsw/recovery-mode-95de3d5"
COMMIT = "95de3d5aa88641031962aee59ad5f8136dff7d82"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(DATA / "export-manifest.json")
    for name, meta in manifest["artifacts"].items():
        path = DATA / name
        assert path.stat().st_size == meta["size_bytes"] and sha(path) == meta["sha256"]
    campaign, audit = read(DATA / "campaign.json"), read(DATA / "audit.json")
    assert campaign["complete"] and "failure" not in campaign
    assert campaign["source_commit"] == audit["source_commit"] == COMMIT
    assert audit["passed"] and audit["registries_drained"]
    assert audit["credential_values_absent_from_export"]
    prior = ROOT / "evidence/dsw/full-fp32-f2ccc2a"
    assert audit["prior_failed_audit_sha256"] == sha(prior / "audit.json")
    assert not read(prior / "audit.json")["passed"]
    harness = ROOT / "evidence/harnesses/recovery-mode"
    assert campaign["script_sha256"] == sha(harness / "campaign.py.txt")
    assert audit["script_sha256"] == sha(harness / "audit_export.py.txt")
    for path, digest in audit["source_files"].items():
        data = subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == digest
    assert audit["owned_records_checked"] == len(audit["records"]) == 276
    assert (
        len(
            {
                (r["identity"]["pid"], r["identity"]["start_ticks"], r["identity"]["boot_id"])
                for r in audit["records"]
            }
        )
        == 276
    )
    assert all(not r["matching_live_process"] and not r["live_group"] for r in audit["records"])
    assert audit["gpu7"]["free_mib"] == "11990"
    assert audit["gpu7"]["uuid"] == "GPU-b57fb933-0e5a-dd28-7041-a177da03405e"
    assert len(audit["model_files"]) == 7 and all(
        x["hub_digest_verified"] for x in audit["model_files"]
    )
    assert len(audit["profiles"]) == 2 and all(p["payload_verified"] for p in audit["profiles"])
    assert [a["engine"] for a in campaign["attempts"]] == ["sglang", "vllm"]
    results = {
        "source_commit": COMMIT,
        "engines": {},
        "strict_responses": 0,
        "full_release_gate_passed": False,
        "owned_records_terminal": 276,
        "registries_drained": True,
    }
    for attempt in campaign["attempts"]:
        engine = attempt["engine"]
        assert attempt["execution_complete"] and len(attempt["stages"]) == 3
        before, after = attempt["before"], attempt["after"]
        assert before["blockers"] == {"leases": 1, "lease_work": 1, "preparing_bundles": 1}
        assert not after["blockers"] and not after["leases"] and not after["lease_work"]
        original = before["bundles"][0]
        assert len(before["bundles"]) == 1 and original["state"] == "PREPARING"
        retained = next(b for b in after["bundles"] if b["ref"] == original["ref"])
        assert retained["state"] == "FAILED" and retained["error"] == "preparation owner exited"
        for key in ("ref", "digest", "manifest", "backend", "created"):
            assert retained[key] == original[key]
        rid, lease = before["leases"][0]["request_id"], before["leases"][0]["id"]
        assert before["lease_work"][0]["lease_id"] == lease
        assert before["lease_work"][0]["phase"] == "inflight"
        branch_ids = json.loads(before["lease_work"][0]["branches"])
        assert len(branch_ids) == 1
        recovered = [e for e in after["events"] if e["action"] == "recovered"]
        assert len(recovered) == 1
        assert json.loads(recovered[0]["details"]) == {"lease_id": lease, "request_id": rid}
        for which in ("before", "after"):
            directory = DATA / f"{engine}-{which}-snapshot"
            snap = verify_snapshot(directory, attempt[which + "_snapshot"]["manifest_sha256"])
            assert snap["summary"] == attempt[which]["summary"]
            assert snap["restore_blockers_at_snapshot"] == attempt[which]["blockers"]
            with sqlite3.connect(f"file:{directory / 'registry.sqlite3'}?mode=ro", uri=True) as db:
                db.row_factory = sqlite3.Row
                for table in (
                    "bundles",
                    "leases",
                    "lease_work",
                    "routes",
                    "events",
                    "owners",
                    "workers",
                ):
                    assert [dict(r) for r in db.execute(f"SELECT * FROM {table}")] == attempt[
                        which
                    ][table]
        if engine == "sglang":
            assert (
                before["summary"]
                == read(prior / "failed-registry-snapshot/manifest.json")["summary"]
            )
            assert attempt["original_process_record"] == read(
                prior / "sglang-ordinary-full-fp32-initial/process.json"
            )
        else:
            assert (
                attempt["seed_child"]["returncode"] == -9
                and not attempt["seed_child"]["remaining_group"]
            )
            assert attempt["seed_journal"]["dispatched_to_engine"] is False
            assert (
                sha(DATA / "vllm-seed/create_journal.py") == attempt["seed_child"]["script_sha256"]
            )
            assert read(DATA / "vllm-seed/journal.json") == attempt["seed_journal"]
            assert (
                attempt["seed_child"]["identity"]["pid"]
                == json.loads(before["owners"][0]["identity"])["pid"]
            )
        maintenance, serve, restarted = attempt["stages"]
        assert [s["stage"] for s in attempt["stages"]] == ["maintenance", "serve", "restart"]
        identities = []
        for stage in attempt["stages"]:
            assert stage["execution_complete"] and "failure" not in stage
            directory = DATA / (engine + "-" + stage["stage"])
            record = read(directory / "process.json")
            assert record == stage["cleanup"]["process_record"]
            assert read(directory / "cleanup.json") == stage["cleanup"]
            assert (
                stage["cleanup"]["passed"]
                and not stage["cleanup"]["remaining_process_group_members"]
            )
            assert record["registry_path"] == attempt["registry_path"]
            assert record["tensor_parallel_size"] == 1
            assert (
                record["dtype"]
                == record["readout_dtype"]
                == ("float32" if engine == "sglang" else "bfloat16")
            )
            assert record["batch_invariant"] is (engine == "sglang")
            assert record["recovery_only"] is (stage["stage"] == "maintenance")
            identities.append((record["identity"]["pid"], record["identity"]["start_ticks"]))
            assert stage["profile"]["recovery_only"] is record["recovery_only"]
            for module in stage["imports"].values():
                assert COMMIT in module["path"]
                prefix = (
                    "src/jev_runtime/"
                    if module["path"].endswith("jev_runtime/__init__.py")
                    else f"packages/{engine}/src/jev_{engine}/"
                )
                for path, digest in module["files"].items():
                    assert audit["source_files"][prefix + path] == digest
        assert len(set(identities)) == 3
        pending = maintenance["pending_before"]
        assert (
            len(pending) == 1 and pending[0]["owner_status"] == "dead" and pending[0]["recoverable"]
        )
        assert pending[0]["request_id"] == rid and pending[0]["engine_request_ids"] == branch_ids
        assert pending[0]["backend"] == original["backend"]
        assert maintenance["ready_before"]["status"] == 503
        assert maintenance["ready_before"]["body"]["error"]["code"] == "recovery_only"
        assert len(maintenance["blocked_writes"]) == 6
        assert all(
            r["status"] == 503 and r["body"]["error"]["code"] == "recovery_only"
            for r in maintenance["blocked_writes"]
        )
        assert maintenance["admin_auth_rejection"]["error"]["code"] == "admin_unauthorized"
        assert maintenance["recover_response"] == {"recovered": True}
        assert maintenance["repeat_recover_response"] == {"recovered": False}
        assert maintenance["recovery_worker_not_admitted"]
        assert not maintenance["after_recover_registry"]["leases"]
        assert maintenance["listing_after"]["routes"] == before["routes"]
        assert not attempt["after_recovery_stopped"]["blockers"]
        bundle = Bundle.model_validate(serve["new_bundle"])
        assert (
            bundle.model
            == Bundle.model_validate_json(
                next(b for b in after["bundles"] if b["ref"] == bundle.reference)["manifest"]
            ).model
        )
        for stage in (serve, restarted):
            assert stage["ready"]["ready"] and stage["ready"]["prepared_bundles"] == [
                bundle.reference
            ]
            assert len(stage["responses"]) == 10
            for data in stage["responses"]:
                response = DecisionResponse.model_validate(data)
                assert response.status == "completed" and response.bundle == bundle.reference
                assert response.bundle_digest == bundle.digest and response.generation == 1
                assert response.usage.questions == response.usage.successful_questions == 1
                assert all(answer.status == "answered" for answer in response.answers.values())
                assert response.engine["name"] == engine
            assert not stage["listing_after"]["leases"]
            assert not any(
                stage["profile_after"]["admission"][k]
                for k in ("requests", "expanded_tokens", "expanded_branches", "queued_requests")
            )
            results["strict_responses"] += 10
        assert (
            serve["listing_after"]["routes"]
            == restarted["listing_after"]["routes"]
            == after["routes"]
        )
        results["engines"][engine] = {
            "fault_scope": attempt["fault_scope"],
            "recovered_requests": 1,
            "confirmed_branch_ids": len(branch_ids),
            "blocked_write_checks": 6,
            "strict_responses": 20,
            "normal_restarts_after_recovery": 2,
            "old_bundle_manifest_preserved": True,
            "registry_drained": True,
        }
    assert results["strict_responses"] == 40
    (DATA / "verified-summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
