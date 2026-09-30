"""Verify immutable sources, failure retention and every request in the expanded run."""

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.integration.verify_traffic_evidence import verify  # noqa: E402

DIRECTORY = ROOT / "evidence/dsw/strict-traffic-9da1813"


def main():
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    actual = {str(p.relative_to(DIRECTORY)) for p in DIRECTORY.rglob("*") if p.is_file()}
    assert actual == set(manifest["files"]) | {"manifest.json"}
    for name, entry in manifest["files"].items():
        data = (DIRECTORY / name).read_bytes()
        assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
    sources = json.loads((DIRECTORY / "source-files.json").read_text())
    for commit, files in sources.items():
        for name, digest in files.items():
            raw = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
            assert hashlib.sha256(raw).hexdigest() == digest
    results = {}
    for short in ("4ee78ca", "9da1813"):
        campaign = json.loads((DIRECTORY / f"strict-traffic-{short}/campaign.json").read_text())
        assert campaign["passed"] == (short == "9da1813")
        assert campaign["protected_service_unchanged"]
        assert all(row["returncode"] == 0 for row in campaign["contracts"].values())
        assert [g["free_mib"] for g in campaign["gpus_before"]] == [
            g["free_mib"] for g in campaign["gpus_after"]
        ]
        for engine in ("vllm", "sglang"):
            run = DIRECTORY / "runs" / f"{engine}-strict-traffic-{short}"
            validation = json.loads((run / "validation.json").read_text())
            assert validation["runtime_source_commit"] == campaign["source_commit"]
            assert validation["passed"] == (short == "9da1813")
            assert validation["cleanup_passed"]
            cleanup = json.loads((run / "cleanup.json").read_text())
            assert cleanup["passed"] and not cleanup["remaining_process_group_members"]
            assert not any(json.loads((run / "final-journals.json").read_text()).values())
            if short == "9da1813":
                result = verify(run / "traffic-responses.jsonl.gz", run / "contract.json")
                assert result == json.loads((run / "recount.json").read_text())
                assert result["strict_success_requests"] >= 10000 and result["switches"] >= 1000
                results[engine] = result
            else:
                with gzip.open(run / "traffic-responses.jsonl.gz", "rt") as stream:
                    failed = [
                        row
                        for line in stream
                        if (row := json.loads(line)).get("outcome") == "failed"
                    ]
                if engine == "vllm":
                    assert (
                        len(failed) == 1 and "engine_unavailable" in failed[0]["failure"]["message"]
                    )
                else:
                    assert len(failed) == 4
                    assert all(
                        row["failure"]["message"] == "completion usage mismatch" for row in failed
                    )
    print(
        json.dumps(
            {
                "artifact_hashes_verified": len(manifest["files"]),
                "git_source_files_verified": {k: len(v) for k, v in sources.items()},
                "final_cohorts": results,
                "failed_attempts_retained": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
