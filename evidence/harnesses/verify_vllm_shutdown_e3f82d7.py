"""Verify shutdown budgets, preserved routes, raw logs and stopped DSW artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
from pathlib import Path

from jev_runtime.schema import DecisionResponse

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "evidence/dsw/vllm-shutdown-e3f82d7"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(ROOT / "export-manifest.json")["artifacts"]
    for name, expected in manifest.items():
        path = ROOT / name
        assert path.stat().st_size == expected["size_bytes"] and sha(path) == expected["sha256"]
    audit = read(ROOT / "audit.json")
    assert audit["passed"] and audit["credential_values_absent_from_export"]
    assert audit["owned_records_checked"] == len(audit["records"])
    assert all(not r["matching_live_process"] and not r["live_group"] for r in audit["records"])
    assert int(audit["gpu7"]["free_mib"]) == 11990
    assert audit["gpu7"]["uuid"] == "GPU-b57fb933-0e5a-dd28-7041-a177da03405e"
    for name, expected in audit["python_files"].items():
        blob = subprocess.check_output(
            ["git", "show", audit["source_commit"] + ":" + name], cwd=REPO
        )
        assert hashlib.sha256(blob).hexdigest() == expected
    previous = REPO / "evidence/dsw/quiescence-7fe8e72/audit.json"
    assert sha(previous) == audit["prior_audit_sha256"]
    assert read(previous)["model_files"] == audit["model_files"]
    assert read(previous)["model"] == audit["model"]
    campaign = read(ROOT / "campaign.json")
    assert campaign["complete"] and campaign["sources"]["commit"] == audit["source_commit"]
    assert campaign["native_source_files"] == campaign["native_source_files_after"]
    for name, value in read(ROOT / "native-shutdown-excerpts.json").items():
        assert value["sha256"] == campaign["native_source_files"][name]
    assert read(ROOT / "contracts.json")["engines"]["vllm"]["returncode"] == 0
    assert "114 passed" in (ROOT / "contracts-vllm.log").read_text()
    attempts = campaign["attempts"]
    assert [(r["phase"], r["api_workers"], r["native_grace_seconds"]) for r in attempts] == [
        ("single", 1, 30),
        ("dual", 2, 30),
        ("restart", 2, 45),
    ]
    typed = questions = raws = rejections = 0
    registries = set()
    dual = read(ROOT / "dual/quiescence.json")["checks"]
    for attempt in attempts:
        path = ROOT / attempt["phase"]
        assert attempt["complete"] and attempt["cleanup"]["passed"]
        assert not attempt["cleanup"]["remaining_process_group_members"]
        record = read(path / "process.json")
        registries.add(record["registry_path"])
        assert record == attempt["cleanup"]["process_record"]
        grace = attempt["native_grace_seconds"]
        assert record["native_shutdown_timeout_seconds"] == grace
        assert record["command"][record["command"].index("--shutdown-timeout") + 1] == str(grace)
        if attempt["phase"] != "restart":
            assert "--vllm-shutdown-timeout" not in attempt["command"]
        else:
            assert "--no-bootstrap" in attempt["command"]
            assert attempt["resume"]["state"] == "OPEN"
        stopped = read(path / "quiescence-stop.json")
        assert stopped["drained"] and not stopped["waiting_workers"]
        assert not any(stopped["outstanding"].values())
        state = read(path / "worker-registry.json")
        assert not any(state["retained_counts"].values())
        with sqlite3.connect(
            (path / "registry-final.sqlite").as_uri() + "?mode=ro", uri=True
        ) as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert not db.execute("PRAGMA foreign_key_check").fetchall()
            for name, count in state["retained_counts"].items():
                assert db.execute("SELECT count(*) FROM " + name).fetchone()[0] == count == 0
            assert set(db.execute("SELECT state FROM backend_controls").fetchall()) == {
                ("QUIESCING",)
            }
        log = (path / "engine.log").read_text()
        for pattern, count in attempt["native_log_counts"].items():
            assert log.count(pattern) == count
        for pattern in (
            "force killing",
            "leaked semaphore",
            "Traceback (most recent call last)",
            "mode=abort",
        ):
            assert attempt["native_log_counts"][pattern] == 0
        assert attempt["native_log_counts"]["mode=drain"] >= attempt["api_workers"] + 1
        assert (
            attempt["native_log_counts"]["request processing complete; starting resource teardown"]
            == 1
        )
        if attempt["phase"] == "dual":
            assert read(path / "quiescence.json")["passed"]
            assert len(dual["workers"]) == 2 and dual["work_before_gate"]["raw_work"]
            assert not dual["gate_closed"]["drained"]
            assert dual["gate_closed"]["outstanding"]["leases"] > 0
            assert dual["routes_before"] == dual["routes_after"]
            assert dual["drained"]["drained"] and not any(dual["work_after"].values())
            assert all(not row["monitor_running"] for row in dual["health_after"])
            assert len(dual["rejections"]) == 8
            assert all(
                r["status"] == 503 and r["body"]["error"]["code"] == "backend_quiescing"
                for r in dual["rejections"]
            )
            rejections += len(dual["rejections"])
            responses = dual["warm"] + [dual["typed_completed"]]
            raw = dual["raw_completed"]
        else:
            checks = attempt["serving_checks"]
            assert len(checks["profiles"]) == attempt["api_workers"]
            responses = checks["responses"]
            raw = attempt.get("raw_response")
            if attempt["phase"] == "restart":
                assert checks["routes"] == dual["routes_before"]
                assert not set(checks["profiles"]) & set(dual["workers"])
                assert len(responses) == 4
            else:
                assert len(responses) == 1
        if raw:
            assert raw["raw_logprobs"] and len(raw["logprobs"]) == 2
            assert all(math.isfinite(value) and value <= 0 for value in raw["logprobs"])
            raws += 1
        for response in responses:
            parsed = DecisionResponse.model_validate(response)
            assert parsed.status == "completed"
            assert parsed.usage.successful_questions == len(parsed.answers)
            typed += 1
            questions += parsed.usage.successful_questions
    assert len(registries) == 2
    assert (typed, questions, raws, rejections) == (8, 135, 2, 8)
    result = {
        "passed": True,
        "source_commit": audit["source_commit"],
        "verified_artifacts": len(manifest),
        "verified_python_sources": len(audit["python_files"]),
        "owned_records_terminal": audit["owned_records_checked"],
        "typed_responses": typed,
        "answered_questions": questions,
        "raw_responses": raws,
        "expected_gate_rejections": rejections,
        "snapshots": 3,
        "distinct_registries": len(registries),
        "cleanup_elapsed_seconds": {r["phase"]: r["cleanup_elapsed_seconds"] for r in attempts},
        "exit_codes_captured": False,
        "release_gate_passed": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
