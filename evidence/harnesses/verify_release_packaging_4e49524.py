"""Verify release artifacts, installed-package evidence and failure-inclusive rollout."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw/release-packaging-4e49524"
SCRIPTS = ROOT / "evidence/harnesses/release-packaging"
TOOL = "4e49524b54aade12988a8249faad6a90c736a173"
CANDIDATE = "861ed3ad99df9028ef5d3df2d1dfc4127ab0a946"
PREVIOUS = "3377c95ca9955ae96ce7c7deb2c4d97b41b1d9c1"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha(commit, path):
    return hashlib.sha256(
        subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    ).hexdigest()


def main():
    manifest = read(BASE / "export-manifest.json")
    for name, expected in manifest["artifacts"].items():
        path = BASE / name
        assert sha(path) == expected["sha256"] and path.stat().st_size == expected["size_bytes"]
    audit = read(BASE / "final-audit.json")
    assert (
        manifest["script_sha256"] == audit["script_sha256"] == sha(SCRIPTS / "audit_export.py.txt")
    )
    assert audit["passed"] and audit["owned_records_checked"] == len(audit["records"]) == 108
    assert all(not r["matching_live_process"] and not r["live_group"] for r in audit["records"])
    assert int(audit["gpu7"]["free_mib"]) == 11990
    assert audit["artifacts"]["release.py"]["sha256"] == git_sha(TOOL, "deployment/release.py")
    rebuild = read(BASE / "wheel-rebuild-equality.json")
    assert rebuild["original_source_commit"] == CANDIDATE
    assert rebuild["current_source_commit"] == TOOL
    candidate_wheels = read(BASE / "wheels-861ed3a/manifest.json")
    assert len(rebuild["wheels"]) == 3
    assert {"wheels/" + w["name"] for w in rebuild["wheels"]} == {
        d["wheel"] for d in candidate_wheels["distributions"].values()
    }
    for wheel in rebuild["wheels"]:
        assert wheel["byte_identical"]
        assert wheel["sha256"] == candidate_wheels["files"]["wheels/" + wheel["name"]]["sha256"]
    for commit, paths in audit["source_files"].items():
        for path, expected in paths.items():
            assert git_sha(commit, path) == expected, path
    assert audit["model"]["revision"] == "31b70e2e869a7173562077fd711b654946d38674"
    assert len(audit["model_files"]) == 6
    for entry in audit["model_files"]:
        if entry["remote_digest_kind"] == "lfs_sha256":
            assert entry["sha256"] == entry["remote_digest"]
    for tag, source, count, core_files in (
        ("3377c95", PREVIOUS, 40, 28),
        ("861ed3a", CANDIDATE, 44, 29),
    ):
        built = read(BASE / f"wheels-{tag}/manifest.json")
        local = read(BASE / f"local-build-{tag}.json")
        assert sha(BASE / f"wheels-{tag}/manifest.json") == local["manifest_sha256"]
        assert built["source_commit"] == local["source_commit"] == source
        assert built["builder_sha256"] == git_sha(CANDIDATE, "deployment/release.py")
        assert set(built["distributions"]) == {"jev-runtime-core", "jev-vllm", "jev-sglang"}
        directory = "gateway-861ed3a-r2" if tag == "861ed3a" else "gateway-3377c95"
        locked = read(BASE / directory / "manifest.json")
        installed = read(BASE / f"install-{tag}.json")
        assert locked["source_commit"] == installed["source_commit"] == source
        assert locked["source_manifest_sha256"] == local["manifest_sha256"]
        assert sha(BASE / directory / "manifest.json") == installed["manifest_sha256"]
        assert locked["resolver_sha256"] == git_sha(TOOL, "deployment/release.py")
        assert len(locked["distributions"]) == len(installed["packages"]) == count
        assert installed == audit["installations"][tag]
        assert installed["verified_core_files"] == core_files
        assert installed["prefix"] != installed["base_prefix"]
        assert installed["core_path"].startswith(installed["prefix"] + "/")
        for name in ("requirements.txt", "constraints.txt"):
            assert sha(BASE / directory / name) == locked["files"][name]["sha256"]
        lines = (BASE / directory / "requirements.txt").read_text().splitlines()
        expected = [
            f"{n}=={d['version']} --hash=sha256:{d['sha256']}"
            for n, d in sorted(locked["distributions"].items())
        ]
        assert lines == expected
        core = built["distributions"]["jev-runtime-core"]["wheel"]
        assert built["files"][core]["sha256"] == locked["source_core_sha256"]
    diagnosis = read(BASE / "smol-tokenizer-diagnosis.json")
    profile = read(BASE / "smol-explicit-profile.json")
    equality = read(BASE / "smol-explicit-native-equality.json")
    assert diagnosis["auto_digest"] != diagnosis["native_digest"]
    assert any(d["path"] == "/pre_tokenizer/type" for d in diagnosis["differences"])
    checks = profile["checks"]
    assert profile["attempted"] == len(checks) == 1005
    differences = [r for r in checks if r["auto_ids"] != r["profile_ids"]]
    assert len(differences) == profile["differing_inputs"] == 232
    assert profile["differences"] == differences
    assert equality["passed"] and len(equality["checks"]) == len(checks)
    for a, b in zip(checks, equality["checks"], strict=True):
        assert a["text"] == b["text"]
        assert a["profile_ids"] == b["checkpoint_ids"] == b["generic_ids"] == b["sglang_ids"]
    digest = profile["profile_digest"]
    assert digest == equality["sglang_profile_digest"] == equality["generic_profile_digest"]
    assert profile["checkpoint_tokenizer_sha256"] == profile["files"]["tokenizer.json"]
    assert audit["profile_files"] == profile["files"]
    assert (
        next(f for f in audit["model_files"] if f["name"] == "tokenizer.json")["sha256"]
        == profile["checkpoint_tokenizer_sha256"]
    )
    initial = read(BASE / "rollout-campaign.json")
    rejected = read(BASE / "rollout-campaign-sglang-followup.json")
    final = read(BASE / "rollout-campaign-explicit-profile.json")
    for campaign, script in (
        (initial, "campaign"),
        (rejected, "campaign_sglang_followup"),
        (final, "campaign_explicit_profile"),
    ):
        assert campaign["script_sha256"] == sha(SCRIPTS / f"{script}.py.txt")
    assert initial.get("completed") is None
    assert "FileExistsError" in (BASE / "rollout-campaign.log").read_text()
    assert not rejected["completed"] and not rejected["attempts"][0]["passed"]
    wrong = rejected["attempts"][0]
    assert (
        wrong["native_profile"]["model"]["tokenizer_implementation_digest"]
        != wrong["gateways"][0]["profile"]["model"]["tokenizer_implementation_digest"]
    )
    assert wrong["native_cleanup"]["passed"] and wrong["gateways"][0]["cleanup"]["passed"]
    assert "one metadata record" in (BASE / "resolve-861ed3a.stderr").read_text()
    for tag in ("861ed3a", "3377c95"):
        assert (
            "setuptools==80.10.2"
            in (BASE / f"gateway-sglang-{tag}/failure-resolve.log").read_text()
        )
        assert (
            "setuptools==84.0.0" in (BASE / f"gateway-sglang-{tag}/failure-resolve.log").read_text()
        )
    assert final["completed"] and len(final["attempts"]) == 2
    rows = []
    for attempt in final["attempts"]:
        assert attempt["passed"] and attempt["native_cleanup"]["passed"]
        assert attempt["native_identity_unchanged"] and attempt["overlap_canary_passed"]
        assert attempt["native_profile"]["model"]["tokenizer_implementation_digest"] == digest
        native = attempt["native-contract.json"]
        assert native["passed"] and attempt["native-precision.json"]["passed"]
        switches = native["checks"]["hot_switch_under_traffic"]
        assert switches["switches"] == 1000 and switches["mixed_bundle_responses"] == 0
        assert native["checks"]["native_attach_logprob_parity"]["max_abs_error"] == 0
        count = suites = 0
        assert [g["label"] for g in attempt["gateways"]] == [
            "previous",
            "candidate-canary",
            "candidate-active",
            "rollback",
        ]
        for gateway in attempt["gateways"]:
            assert gateway["ready"]["ready"] and gateway["cleanup"]["passed"]
            assert (
                gateway["installation"]["manifest_sha256"]
                == audit["installations"][gateway["record"]["version"]]["manifest_sha256"]
            )
            assert gateway["profile"]["model"]["tokenizer_implementation_digest"] == digest
            parity = gateway["native_input_parity"]
            assert parity["passed"]
            assert parity["gateway"]["bundle"]["model"] == parity["native"]["bundle"]["model"]
            for a, b in zip(
                parity["gateway"]["sequences"], parity["native"]["sequences"], strict=True
            ):
                assert a["input_ids"] == b["input_ids"] and a["label_ids"] == b["label_ids"]
            assert gateway["persisted_route"]["ref"] == "release-round-trip@1"
            assert gateway["persisted_route"]["generation"] == 1
            assert len(gateway["persistent_route_responses"]) == 8
            for response in gateway["persistent_route_responses"]:
                assert (
                    response["status"] == "completed"
                    and response["usage"]["successful_questions"] == 1
                )
                assert response["bundle"] == "release-round-trip@1"
                count += 1
            if gateway["label"] != "candidate-canary":
                assert gateway["contract_returncode"] == 0 and gateway["contract"]["passed"]
                suites += 1
        rows.append(
            {
                "engine": attempt["engine"],
                "native_switches": switches["switches"],
                "native_strict_requests": switches["strict_success_requests"],
                "persistent_route_requests": count,
                "sdk_suites": suites,
                "upgrade_seconds": attempt["upgrade_transition"]["seconds"],
                "rollback_seconds": attempt["rollback_transition"]["seconds"],
                "native_identity_unchanged": True,
            }
        )
    result = {
        "integrity_and_accounting_passed": True,
        "release_gate_passed": False,
        "zero_downtime_certified": False,
        "rows": rows,
        "corpus_inputs": 1005,
        "default_auto_mismatches": 232,
        "owned_records_checked": 108,
        "numerical_quality_performance_requalified": False,
    }
    (BASE / "verified-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
