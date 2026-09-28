"""Verify the original failure, corrected GPU runs, source identity and stopped registries."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
from pathlib import Path

from jev_runtime.schema import DecisionResponse

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "evidence/dsw/quiescence-7fe8e72"


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
    assert int(audit["gpu7"]["free_mib"]) == 11990
    assert audit["gpu7"]["uuid"] == "GPU-b57fb933-0e5a-dd28-7041-a177da03405e"
    assert audit["owned_records_checked"] == len(audit["records"])
    assert all(
        not row["matching_live_process"] and not row["live_group"] for row in audit["records"]
    )
    source_files = 0
    for source in audit["sources"].values():
        for name, expected in source["python_files"].items():
            blob = subprocess.check_output(["git", "show", source["commit"] + ":" + name], cwd=REPO)
            assert hashlib.sha256(blob).hexdigest() == expected
            source_files += 1
    corrected, original = read(ROOT / "campaign.json"), read(ROOT / "initial/campaign.json")
    assert corrected["complete"] and not original["complete"]
    assert [(row["engine"], row["phase"]) for row in corrected["attempts"]] == [
        ("vllm", "initial"),
        ("vllm", "restart"),
        ("sglang", "initial"),
        ("sglang", "restart"),
        ("sglang", "restart-2"),
    ]
    assert [row["cleanup"]["passed"] for row in original["attempts"]] == [True, True, True, False]
    remediation = read(ROOT / "initial/sglang-restart/cleanup-remediation.json")
    assert remediation["cleanup_confirmed"] and not remediation["remaining_after"]
    assert remediation["signal"] == "SIGKILL" and not any(
        remediation["retained_counts_before"].values()
    )
    assert remediation["original_cleanup_sha256"] == sha(
        ROOT / "initial/sglang-restart/cleanup.json"
    )
    typed_requests = raw_requests = questions = rejections = 0
    for prefix, campaign in (("", corrected), ("initial/", original)):
        for attempt in campaign["attempts"]:
            directory = ROOT / (prefix + attempt["engine"] + "-" + attempt["phase"])
            assert attempt["complete"]
            if not prefix:
                assert attempt["cleanup"]["passed"]
                assert not attempt["cleanup"]["remaining_process_group_members"]
            stopped = read(directory / "quiescence-stop.json")
            assert stopped["drained"] and not stopped["waiting_workers"]
            assert not any(stopped["outstanding"].values())
            state = read(directory / "worker-registry.json")
            assert not any(state["retained_counts"].values())
            with sqlite3.connect(
                (directory / "registry-final.sqlite").as_uri() + "?mode=ro", uri=True
            ) as db:
                assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert not db.execute("PRAGMA foreign_key_check").fetchall()
                for name, count in state["retained_counts"].items():
                    assert db.execute("SELECT count(*) FROM " + name).fetchone()[0] == count == 0
                assert set(db.execute("SELECT state FROM backend_controls").fetchall()) == {
                    ("QUIESCING",)
                }
            if attempt["phase"] == "initial":
                check = read(directory / "quiescence.json")
                assert check["passed"]
                c = check["checks"]
                assert len(c["workers"]) == 2
                assert c["work_before_gate"]["raw_work"]
                assert (
                    not c["gate_closed"]["drained"]
                    and c["gate_closed"]["outstanding"]["leases"] > 0
                )
                assert c["routes_before"] == c["routes_after"]
                assert c["drained"]["drained"] and not any(c["work_after"].values())
                assert all(not row["monitor_running"] for row in c["health_after"])
                assert len(c["rejections"]) == 8
                assert all(
                    row["status"] == 503 and row["body"]["error"]["code"] == "backend_quiescing"
                    for row in c["rejections"]
                )
                responses = c["warm"] + [c["typed_completed"]]
                assert len(c["typed_completed"]["answers"]) == 128
                assert c["raw_completed"]["request_id"] == "drain-raw"
                raw = c["raw_completed"]
                assert raw["raw_logprobs"] is True and len(raw["logprobs"]) == 2
                assert all(math.isfinite(value) and value <= 0 for value in raw["logprobs"])
                if not prefix:
                    raw_requests += 1
                    rejections += 8
            else:
                responses = attempt["restart_checks"]["responses"]
                assert len(responses) == 4 and len(attempt["restart_checks"]["profiles"]) == 2
                assert attempt["resume"]["state"] == "OPEN"
                initial = read(ROOT / (prefix + attempt["engine"] + "-initial/quiescence.json"))
                assert (
                    attempt["restart_checks"]["routes_unchanged"]
                    == initial["checks"]["routes_before"]
                )
            for response in responses:
                parsed = DecisionResponse.model_validate(response)
                assert parsed.status == "completed"
                assert parsed.usage.questions == parsed.usage.successful_questions
                if not prefix:
                    typed_requests += 1
                    questions += parsed.usage.successful_questions
        contracts = read(ROOT / (prefix + "contracts.json"))
        assert contracts["complete"]
        assert all(row["returncode"] == 0 for row in contracts["engines"].values())
    print(
        json.dumps(
            {
                "passed": True,
                "artifacts": len(manifest),
                "source_files": source_files,
                "owned_records_terminal": audit["owned_records_checked"],
                "corrected_runs": len(corrected["attempts"]),
                "corrected_typed_responses": typed_requests,
                "corrected_answered_questions": questions,
                "corrected_raw_responses": raw_requests,
                "corrected_saved_rejections": rejections,
                "original_native_shutdown_failure_preserved": True,
                "release_gate_passed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
