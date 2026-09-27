"""Verify complete tokenizer inventory, native reports and retained initial failures."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw/preserve-fast-cdd0caf"
SCRIPTS = ROOT / "evidence/harnesses/preserve-fast"
SOURCE = "cdd0caf330a645f15279cd346f2ffa8ba62190f9"
FIRST = "fbd5b361a561bd951a4e5295885de30b2618e972"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    export = read(BASE / "export-manifest.json")
    for name, entry in export["artifacts"].items():
        path = BASE / name
        assert sha(path) == entry["sha256"] and path.stat().st_size == entry["size_bytes"]
    audit = read(BASE / "preserve-fast-audit-cdd0caf.json")
    assert audit["passed"]
    assert (
        export["script_sha256"]
        == audit["script_sha256"]
        == sha(SCRIPTS / "audit_export-cdd0caf.py.txt")
    )
    for commit, paths in audit["source_files"].items():
        for name, expected in paths.items():
            actual = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
            assert hashlib.sha256(actual).hexdigest() == expected, (commit, name)
    assert audit["owned_records_checked"] == len(audit["records"])
    assert all(not r["matching_live_process"] and not r["live_group"] for r in audit["records"])
    assert int(audit["gpu7"]["free_mib"]) == 11990
    assert len(audit["model_files"]) == 6
    for file in audit["model_files"]:
        if file["remote_digest_kind"] == "lfs_sha256":
            assert file["sha256"] == file["remote_digest"]
    frozen = read(ROOT / "profiles/model-inventory.json")["models"]
    inventory = {(r["id"], r["revision"]) for r in frozen}
    profiles = []
    corpus_checks = compiler_passes = compiler_attempts = 0
    expected_initial_rejections = {
        "microsoft/Phi-4-mini-instruct",
        "allenai/OLMo-2-1124-7B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "zai-org/glm-4-9b-chat",
    }
    for tag, commit in (("fbd5b36", FIRST), ("cdd0caf", SOURCE)):
        for engine in ("vllm", "sglang"):
            report = read(BASE / f"preserve-fast-{engine}-{tag}.json")
            assert report["completed"] and report["source_commit"] == commit
            assert report["script_sha256"] == sha(SCRIPTS / f"tokenizer_campaign-{tag}.py.txt")
            rows = report["models"]
            assert len(rows) == 20 and {(r["model_id"], r["revision"]) for r in rows} == inventory
            rejected = {r["model_id"] for r in rows if r["status"] == "profile_rejected"}
            assert rejected == (
                expected_initial_rejections if tag == "fbd5b36" else {"zai-org/glm-4-9b-chat"}
            )
            assert sum(r["status"] == "local_checkpoint_unavailable" for r in rows) == 8
            for row in rows:
                if row["status"] != "profile_verified":
                    continue
                profile = row["profile"]
                assert row["cli_returncode"] == 0 and profile["validation"]["passed"]
                assert row["independent_corpus"]["inputs"] == 1005
                assert row["independent_corpus"]["all_equal"]
                assert (
                    profile["source_files"]["tokenizer.json"] == profile["files"]["tokenizer.json"]
                )
                assert (
                    profile["implementation_sha256"]
                    == hashlib.sha256(
                        subprocess.check_output(
                            ["git", "show", f"{commit}:src/jev_runtime/tokenizer_profiles.py"],
                            cwd=ROOT,
                        )
                    ).hexdigest()
                )
                profiles.append(row["profile_manifest_sha256"])
                if tag != "cdd0caf":
                    continue
                corpus_checks += 1005
                cases = row["compiler"]["cases"]
                assert len(cases) == 16
                failed = [c for c in cases if not c["passed"]]
                assert len(failed) == (2 if "Phi-3-mini" in row["model_id"] else 0)
                assert all(c["mode"] == "joint-label" and c["candidates"] == 64 for c in failed)
                compiler_attempts += len(cases)
                compiler_passes += sum(c["passed"] for c in cases)
                expected = (
                    999
                    if "Distill-Llama" in row["model_id"]
                    else 232
                    if "SmolLM2" in row["model_id"]
                    else 557
                    if "OLMo" in row["model_id"] and engine == "sglang"
                    else 0
                )
                assert row["default_auto_diagnostic"]["mismatches"] == expected
    assert corpus_checks == 22110 and (compiler_passes, compiler_attempts) == (348, 352)
    assert sorted(profiles) == sorted(p["manifest_sha256"] for p in audit["profiles"])
    diagnosis = read(BASE / "preserve-fast-backend-diagnosis-fbd5b36.json")
    assert diagnosis["script_sha256"] == sha(SCRIPTS / "diagnose_backend-fbd5b36.py.txt")
    assert len(diagnosis["rows"]) == 3
    assert all(
        [d["path"] for d in r["differences"]] == ["/post_processor"] for r in diagnosis["rows"]
    )
    olmo = read(BASE / "preserve-fast-olmo-native-diagnosis-cdd0caf.json")
    assert olmo["script_sha256"] == sha(SCRIPTS / "olmo_native_diagnosis-cdd0caf.py.txt")
    assert olmo["inputs"] == 1005 and olmo["mismatches"] == 0
    assert olmo["native_implementation_digest"] == olmo["historical_implementation_digest"]
    failed = read(BASE / "preserve-fast-native-cdd0caf.json")
    assert failed["script_sha256"] == sha(SCRIPTS / "native_campaign-cdd0caf.py.txt")
    assert not failed["completed"] and not failed["attempts"]
    assert (
        "Cannot import 'hatchling.build'"
        in (BASE / "preserve-fast-install-vllm-cdd0caf.log").read_text()
    )
    final = read(BASE / "preserve-fast-native-cdd0caf-r2.json")
    assert final["completed"] and len(final["attempts"]) == 2
    assert final["script_sha256"] == sha(SCRIPTS / "native_campaign-cdd0caf-r2.py.txt")
    results = []
    for attempt in final["attempts"]:
        engine = attempt["engine"]
        run = BASE / "runs" / Path(attempt["run"]).name
        validation = read(run / "validation.json")
        assert validation == attempt["validation"]
        assert validation["runtime_source_commit"] == validation["harness_source_commit"] == SOURCE
        assert validation["functional_passed"] and validation["cleanup_passed"]
        contract = read(run / "contract.json")
        switches = contract["checks"]["hot_switch_under_traffic"]
        assert contract["passed"] and switches["switches"] == 1000
        assert switches["mixed_bundle_responses"] == 0
        assert contract["checks"]["native_attach_logprob_parity"]["max_abs_error"] == 0
        reference = read(run / "reference-cpu.json")
        assert reference["contract_sha256"] == sha(run / "contract.json")
        assert reference["atol"] == 0.15
        assert reference["device"] == "cpu" and reference["attention"] == "eager"
        assert (
            reference["reference_script_sha256"]
            == audit["source_files"][SOURCE]["tests/integration/reference_logits.py"]
        )
        assert reference["passed"] == validation["numerical_position_passed"]
        assert attempt["returncode"] == (0 if validation["passed"] else 1)
        profile = read(run / "profile.json")["model"]
        cpu = read(BASE / f"preserve-fast-{engine}-cdd0caf.json")
        row = next(r for r in cpu["models"] if "SmolLM2" in r["model_id"])
        assert (
            profile["tokenizer_implementation_digest"]
            == row["profile"]["tokenizer_implementation_digest"]
        )
        results.append(
            {
                "engine": engine,
                "switches": switches["switches"],
                "strict_success_requests": switches["strict_success_requests"],
                "max_absolute_logprob_error": max(reference["absolute_errors"]),
                "single_position_passed": reference["passed"],
                "full_numerical_gate_passed": False,
            }
        )
    result = {
        "integrity_and_accounting_passed": True,
        "source_commit": SOURCE,
        "release_gate_passed": False,
        "profiles_verified_at_final_source": 22,
        "independent_corpus_checks": corpus_checks,
        "compiler_passes": compiler_passes,
        "compiler_attempts": compiler_attempts,
        "native_rows": results,
        "functional_model_engine_denominator_unchanged": 40,
        "owned_records_checked": audit["owned_records_checked"],
    }
    (BASE / "verified-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
