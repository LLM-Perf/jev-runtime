"""Verify exact-source migration, preserved state, GPU responses and crash publication."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from contextlib import closing
from pathlib import Path

from jev_runtime.registry import Registry
from jev_runtime.registry_backup import connect, summary, verify_snapshot
from jev_runtime.registry_migration import transform, verify_migration
from jev_runtime.registry_schema import identify
from jev_runtime.schema import DecisionResponse

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "evidence/dsw/registry-migration-0ac5935"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(ROOT / "export-manifest.json")["artifacts"]
    for name, expected in manifest.items():
        p = ROOT / name
        assert p.stat().st_size == expected["size_bytes"] and sha(p) == expected["sha256"]
    audit = read(ROOT / "audit.json")
    assert audit["passed"] and audit["credential_values_absent_from_export"]
    assert audit["owned_records_checked"] == len(audit["records"])
    assert all(not r["matching_live_process"] and not r["live_group"] for r in audit["records"])
    assert int(audit["gpu7"]["free_mib"]) == 11990
    previous = REPO / "evidence/dsw/vllm-shutdown-e3f82d7/audit.json"
    assert sha(previous) == audit["prior_audit_sha256"]
    assert read(previous)["model_files"] == audit["model_files"]
    assert read(previous)["model"] == audit["model"]
    source_files = 0
    for source in audit["sources"].values():
        for name, expected in source["python_files"].items():
            blob = subprocess.check_output(["git", "show", source["commit"] + ":" + name], cwd=REPO)
            assert hashlib.sha256(blob).hexdigest() == expected
            source_files += 1
    campaign = read(ROOT / "campaign.json")
    assert campaign["complete"] and set(campaign["engines"]) == {"sglang", "vllm"}
    assert read(ROOT / "contracts.json")["complete"]
    typed = questions = stale = live = constructor = transformations = 0
    native_warnings = {}
    for engine, entry in campaign["engines"].items():
        assert entry["original_state_unchanged"]
        assert [a["phase"] for a in entry["attempts"]] == ["baseline", "candidate", "rollback"]
        assert "140 passed" in (ROOT / ("contracts-" + engine + ".log")).read_text()
        assert read(ROOT / "contracts.json")["engines"][engine]["returncode"] == 0
        seen_workers = set()
        bundle_digests = set()
        for a, generation in zip(entry["attempts"], [3, 5, 5], strict=True):
            directory = ROOT / (engine + "-" + a["phase"])
            assert a["complete"] and a["cleanup"]["passed"]
            assert not a["cleanup"]["remaining_process_group_members"]
            source = "candidate" if a["phase"] == "candidate" else "baseline"
            assert a["source_commit"] == audit["sources"][source]["commit"]
            record = read(directory / "process.json")
            assert record["engine"] == engine and record["tensor_parallel_size"] == 1
            assert record["dtype"] == record["readout_dtype"] == "bfloat16"
            checks = a["serving"]
            assert len(checks["profiles"]) == 2 and len(checks["responses"]) == 6
            assert not seen_workers & set(checks["profiles"])
            seen_workers.update(checks["profiles"])
            assert checks["routes_after"][0]["generation"] == generation
            assert checks["stale_generation_rejection"]["status"] == 409
            stale += 1
            for response in checks["responses"]:
                parsed = DecisionResponse.model_validate(response)
                assert parsed.status == "completed" and parsed.generation == generation
                assert parsed.engine["name"] == engine
                assert parsed.bundle == checks["routes_after"][0]["ref"]
                assert set(parsed.answers) == {"boolean", "choice", "score", "rank"}
                assert parsed.usage.successful_questions == len(parsed.answers) == 4
                bundle_digests.add(parsed.bundle_digest)
                typed += 1
                questions += 4
            stop = read(directory / "quiescence-stop.json")
            assert stop["drained"] and not any(stop["outstanding"].values())
            saved = read(directory / "registry-final.json")
            assert not saved["blockers"]
            with closing(connect(directory / "registry-final.sqlite")) as db:
                assert summary(db) == saved["summary"] and identify(db) == saved["format"]
                assert db.execute("SELECT generation FROM routes").fetchall() == [(generation,)]
                assert db.execute("SELECT state FROM backend_controls").fetchall() == [
                    ("QUIESCING",)
                ]
                for table in (
                    "leases",
                    "lease_work",
                    "admission_tickets",
                    "raw_work",
                    "recovery_claims",
                ):
                    assert db.execute("SELECT count(*) FROM " + table).fetchone()[0] == 0
            log = (directory / "engine.log").read_text()
            native_warnings[engine + "-" + a["phase"]] = {
                pattern: log.count(pattern)
                for pattern in (
                    "force killing",
                    "leaked semaphore",
                    "Traceback (most recent call last)",
                )
            }
            if a["phase"] == "candidate":
                assert checks["routes_before"][0]["generation"] == 3
                assert "owners_alive" in checks["live_migration_rejection"]["message"]
                assert checks["live_quiesced"]["drained"]
                live += 1
                rejection = checks["old_constructor_rejected"]
                assert rejection["returncode"] != 0 and rejection["state_unchanged"]
                assert "Registry protocol required" in rejection["stderr"]
                constructor += 1
                snap = checks["live_migration_rejection"]["snapshot"]
                checked = verify_snapshot(directory / "live-snapshot", snap["manifest_sha256"])
                assert checked["restore_blockers_at_snapshot"]["owners_alive"] == 2
        assert len(bundle_digests) == 1
        for key in ("new_constructor_requires_migration", "new_constructor_rejects_downgrade"):
            rejected = entry[key]
            assert rejected["returncode"] != 0 and rejected["state_unchanged"]
            assert "migration required" in rejected["stderr"]
            constructor += 1
        for phase, key, source_phase in (
            ("upgrade", "upgrade", "baseline"),
            ("rollback", "downgrade", "candidate"),
        ):
            action = entry[key]
            directory = ROOT / (engine + "-" + phase)
            snapdir = ROOT / (engine + "-" + phase + "-snapshot")
            snapshot = verify_snapshot(snapdir, action["snapshot"]["manifest_sha256"])
            receipt = read(directory / "migration.json")
            assert sha(directory / "migration.json") == action["migration"]["receipt_sha256"]
            assert sha(directory / "migration-intent.json") == receipt["intent_sha256"]
            assert receipt == action["verified"]
            assert receipt["source_summary"] == snapshot["summary"]
            assert (
                receipt["source_summary"]
                == read(ROOT / (engine + "-" + source_phase) / "registry-final.json")["summary"]
            )
            # Initial staged bytes were verified on DSW before activation. Replay
            # the exact transformation from immutable input and compare logical
            # output, since a different local SQLite can lay out pages differently.
            with tempfile.TemporaryDirectory(prefix="jev-migration-verify-") as temporary:
                copy = Path(temporary) / "registry.sqlite3"
                shutil.copyfile(snapdir / "registry.sqlite3", copy)
                with closing(connect(copy, writable=True)) as db:
                    result = transform(db, receipt["target_format"])
                    for name, value in result.items():
                        assert receipt[name] == value
            transformations += 1
    fault = read(ROOT / "faults.json")
    assert fault["passed"] and fault["returncode"] == -9 and fault["source_unchanged"]
    interrupted = ROOT / "fault-interrupted"
    assert (interrupted / "migration.json").is_file()
    assert (interrupted / "registry.pending.sqlite3").is_file()
    assert not (interrupted / "registry.sqlite3").exists()
    try:
        Registry(interrupted / "registry.sqlite3")
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("Incomplete migration was accepted")
    assert not (interrupted / "registry.sqlite3").exists()
    verify_snapshot(ROOT / "fault-snapshot", fault["snapshot"]["manifest_sha256"])
    verify_migration(ROOT / "fault-retry", fault["retry"]["receipt_sha256"])
    assert (typed, questions, stale, live, constructor, transformations) == (36, 144, 6, 2, 6, 4)
    print(
        json.dumps(
            {
                "passed": True,
                "verified_artifacts": len(manifest),
                "verified_python_sources": source_files,
                "owned_groups_terminal": audit["owned_records_checked"],
                "typed_responses": typed,
                "answered_questions": questions,
                "stale_generation_rejections": stale,
                "live_migration_rejections": live,
                "incompatible_constructor_rejections": constructor,
                "replayed_logical_migrations": transformations,
                "stopped_registry_snapshots": 6,
                "migration_process_sigkill_checks": 1,
                "native_log_counts": native_warnings,
                "release_gate_passed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
