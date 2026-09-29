"""Verify saved multi-engine artifacts against their manifest and immutable Git source."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "62340feddb4d73274c0902d68cead1012da94624"
EVIDENCE = ROOT / "evidence/dsw/multi-engine-62340fe"
FINAL_SOURCE = "55674ecbe63850dc9c46da14893ffcad59d2506f"


def verify_final_cpu():
    directory = ROOT / "evidence/dsw/multi-engine-55674ec"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["source_commit"] == FINAL_SOURCE
    assert {p.name for p in directory.iterdir()} == set(manifest["files"]) | {"manifest.json"}
    for name, entry in manifest["files"].items():
        data = (directory / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert len(data) == entry["bytes"]
    report = json.loads((directory / "contracts.json").read_text())
    assert report["source_commit"] == FINAL_SOURCE and report["passed"]
    assert (
        hashlib.sha256((directory / "contracts.py").read_bytes()).hexdigest()
        == report["script_sha256"]
    )
    for engine in ("vllm", "sglang"):
        assert report["engines"][engine]["returncode"] == 0
        assert "151 passed in" in (directory / f"{engine}.log").read_text()
    assert not subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            SOURCE,
            FINAL_SOURCE,
            "--",
            "src",
            "packages/vllm",
            "packages/sglang",
        ],
        cwd=ROOT,
    ).strip()
    return {"artifact_hashes_verified": len(manifest["files"]), "cpu_tests_per_engine": 151}


def main():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    actual = {str(p.relative_to(EVIDENCE)) for p in EVIDENCE.rglob("*") if p.is_file()}
    assert actual == set(manifest["files"]) | {"manifest.json"}
    for name, entry in manifest["files"].items():
        data = (EVIDENCE / name).read_bytes()
        assert (
            hashlib.sha256(data).hexdigest() == entry["sha256"] and len(data) == entry["bytes"]
        ), name
    sources = json.loads((EVIDENCE / "source-files.json").read_text())
    for name, digest in sources.items():
        data = subprocess.check_output(["git", "show", f"{SOURCE}:{name}"], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == digest, name
    engines = {}
    for engine in ("vllm", "sglang"):
        for suffix in ("", "-r3"):
            run = EVIDENCE / "runs" / f"{engine}-multi-engine-62340fe{suffix}"
            validation = json.loads((run / "validation.json").read_text())
            assert validation["runtime_source_commit"] == SOURCE
            assert validation["cleanup_passed"]
            cleanup = json.loads((run / "cleanup.json").read_text())
            assert cleanup["passed"] and not cleanup["remaining_process_group_members"]
            assert all(
                n == 0 for n in json.loads((run / "final-journals.json").read_text()).values()
            )
            contract = json.loads((run / "contract.json").read_text())
            switch = contract["checks"]["hot_switch_under_traffic"]
            assert (
                contract["passed"]
                and switch["switches"] == 1000
                and switch["mixed_bundle_responses"] == 0
            )
            assert contract["checks"]["native_attach_logprob_parity"]["max_abs_error"] == 0
            if suffix:
                assert validation["passed"] and validation["numerical_position_passed"]
                engines[engine] = {
                    "traffic_requests": switch["strict_success_requests"],
                    "switches": 1000,
                    "single_position_max_absolute_error": validation["stages"]["cpu_reference"][
                        "max_absolute_error"
                    ],
                }
            else:
                assert not validation["passed"] and validation["failure"]["stage"] == "postcheck"
    readout = json.loads(
        (EVIDENCE / "multi-engine-62340fe-r2/tokenspeed-source-readout.json").read_text()
    )
    assert readout["passed"] and len(readout["cases"]) == 6
    assert not readout["native_tokenspeed_engine_executed"] and not readout["gpu_executed"]
    print(
        json.dumps(
            {
                "artifact_hashes_verified": len(manifest["files"]),
                "git_source_files_verified": len(sources),
                "engines": engines,
                "tokenspeed_gpu_passed": False,
                "final_checkpoint": verify_final_cpu(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
