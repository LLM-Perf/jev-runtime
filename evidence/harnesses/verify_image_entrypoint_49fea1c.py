"""Verify image-input identity and failure-inclusive Linux host-entrypoint evidence."""

import hashlib
import json
import subprocess
from pathlib import Path

from jev_runtime.schema import DecisionResponse

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "evidence/dsw/image-entrypoint-49fea1c"
SCRIPTS = ROOT / "evidence/harnesses/image-entrypoint"
SOURCE = "49fea1ca833141d65f791caa9eeb2661d627cd67"
NATIVE = "b20d3f43d91c2e13ed2b044a77d35cc3037eb62b"
WHEEL = "6f0cda85f72619193be7a1d1cc89d75142899bf3"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_bytes(path):
    return subprocess.check_output(["git", "show", f"{SOURCE}:{path}"], cwd=ROOT)


def main():
    exported = read(BASE / "export-manifest.json")
    for name, item in exported["artifacts"].items():
        path = BASE / name
        assert sha(path) == item["sha256"] and path.stat().st_size == item["size_bytes"]
    audit = read(BASE / "audit.json")
    assert audit["passed"] and audit["source_commit"] == SOURCE
    assert (
        audit["script_sha256"] == exported["script_sha256"] == sha(SCRIPTS / "audit_export.py.txt")
    )
    assert audit["prior_audit_sha256"] == sha(
        ROOT / "evidence/dsw/native-rollout-cancel-b20d3f4/audit.json"
    )
    ids = {
        (r["identity"]["boot_id"], r["identity"]["pid"], r["identity"]["start_ticks"])
        for r in audit["records"]
    }
    assert len(ids) == len(audit["records"]) == audit["owned_records_checked"] == 153
    assert all(not r["matching_live_process"] and not r["live_group"] for r in audit["records"])
    assert int(audit["gpu7"]["free_mib"]) == 11990
    assert len(audit["model_files"]) == 6 and all(
        f["hub_digest_verified"] for f in audit["model_files"]
    )
    assert len(audit["profiles"]) == 2 and all(f["payload_verified"] for f in audit["profiles"])
    for name, digest in audit["source_files"].items():
        assert hashlib.sha256(git_bytes(name)).hexdigest() == digest
    for environment in audit["environments"].values():
        assert all("/releases/" + NATIVE + "/" in path for path in environment["imports"].values())
    context = read(BASE / "context/manifest.json")
    image = read(BASE / "context/image.json")
    locked = read(BASE / "gateway-lock-manifest.json")
    assert context["image"] == image
    for name in ["image.json", "Dockerfile"]:
        assert context["files"][name]["sha256"] == sha(BASE / "context" / name)
    assert context["files"]["locked/manifest.json"]["sha256"] == sha(
        BASE / "gateway-lock-manifest.json"
    )
    assert image["source_commit"] == locked["source_commit"] == WHEEL
    assert image["gateway_manifest_sha256"] == sha(BASE / "gateway-lock-manifest.json")
    assert image["platform"] == "linux/amd64" and image["target"] == locked["target"]
    assert image["target"]["python"] == "3.12"
    for name, digest in image["helpers"].items():
        assert digest == hashlib.sha256(git_bytes("deployment/" + name)).hexdigest()
        assert context["files"][name]["sha256"] == digest
    template = git_bytes("deployment/Dockerfile.gateway")
    assert hashlib.sha256(template).hexdigest() == image["template_sha256"]
    generated = (
        template.decode()
        .replace("__JEV_BASE_IMAGE__", image["base_image"])
        .replace("__JEV_LOCK_SHA256__", image["gateway_manifest_sha256"])
        .replace("__JEV_SOURCE_COMMIT__", WHEEL)
    )
    assert (BASE / "context/Dockerfile").read_text() == generated
    assert (
        image["preparer_sha256"]
        == hashlib.sha256(git_bytes("deployment/image_context.py")).hexdigest()
    )
    inspected = read(BASE / "base-inspection.json")
    assert inspected["tag_sha256"] == sha(BASE / "base-tag.json")
    assert inspected["base_image"] == image["base_image"]
    assert inspected["manifest_digest"] == "sha256:" + sha(BASE / "base-manifest-raw.json")
    assert inspected["config_digest"] == "sha256:" + sha(BASE / "base-config-raw.json")
    assert inspected["manifest"] == read(BASE / "base-manifest-raw.json")
    assert inspected["config"] == read(BASE / "base-config-raw.json")
    assert inspected["config"]["architecture"] == "amd64" and inspected["config"]["os"] == "linux"
    assert not inspected["layers_downloaded"] and not inspected["image_run"]
    installed = audit["gateway_installation"]
    assert installed["source_commit"] == WHEEL and installed["verified_core_files"] == 30
    assert installed["manifest_sha256"] == image["gateway_manifest_sha256"]
    assert len(locked["distributions"]) == 44
    first, campaign = read(BASE / "initial-attempt.json"), read(BASE / "campaign.json")
    assert (
        first["completed"] and not first["passed"] and campaign["completed"] and campaign["passed"]
    )
    assert campaign["first_report_sha256"] == sha(BASE / "initial-attempt.json")
    rows, failures = [], []
    for label, report, script, passed in [
        ("initial", first, "campaign.py.txt", False),
        ("retest", campaign, "campaign_r2.py.txt", True),
    ]:
        assert (
            report["source_commit"] == SOURCE
            and report["native_source"] == NATIVE
            and report["wheel_source"] == WHEEL
        )
        assert report["script_sha256"] == sha(SCRIPTS / script)
        assert report["image_built"] is False and report["descriptor"] == image
        assert report["context"]["manifest_sha256"] == sha(BASE / "context/manifest.json")
        assert len(report["attempts"]) == 2 and {a["engine"] for a in report["attempts"]} == {
            "sglang",
            "vllm",
        }
        for attempt in report["attempts"]:
            assert attempt["passed"] is passed
            assert attempt["healthcheck_passed"] and attempt["missing_key_probe_rejected"]
            assert (
                attempt["exec_replaced_same_process"] and "uvicorn" in attempt["observed_command"]
            )
            assert len(attempt["qualified"]["workers"]) == 2
            assert attempt["qualified"]["deployment"]["release"] == WHEEL
            assert not attempt["gateway_cleanup"]["remaining_group"]
            assert attempt["native_cleanup"]["passed"]
            assert (
                read(BASE / f"{label}-native-{attempt['engine']}/cleanup.json")
                == attempt["native_cleanup"]
            )
            if not passed:
                assert attempt["failure"] == {
                    "type": "AttributeError",
                    "message": "'DecisionResponse' object has no attribute 'partial'",
                }
                failures.append({"engine": attempt["engine"], "failure": attempt["failure"]})
                continue
            assert attempt["same_native_engine_identity"]
            assert (
                attempt["graceful_shutdown"]["returncode"] == 0
                and not attempt["graceful_shutdown"]["remaining_group"]
            )
            assert all(value == 0 for value in attempt["final_registry_counts"].values())
            assert len(attempt["responses"]) == 10
            for payload in attempt["responses"]:
                response = DecisionResponse.model_validate(payload)
                assert response.status == "completed" and response.usage.successful_questions == 1
                assert response.bundle == "default@1" and response.generation == 1
                assert response.engine["name"] == attempt["engine"]
            rows.append(
                {
                    "engine": attempt["engine"],
                    "strict_decisions": 10,
                    "workers": 2,
                    "healthcheck": True,
                    "missing_key_rejected": True,
                    "graceful_shutdown": True,
                    "zero_final_leases": True,
                }
            )
    summary = {
        "verified": True,
        "source_commit": SOURCE,
        "native_source": NATIVE,
        "wheel_source": WHEEL,
        "image_built": False,
        "container_executed": False,
        "initial_failures_preserved": failures,
        "results": rows,
        "owned_records_terminal": 153,
        "gpu7_free_mib": 11990,
        "qualification": (
            "Generated image inputs and Linux host-process startup/probe/serving/SIGTERM only; "
            "no container PID 1, UID 10001, mount, image ABI "
            "or cross-container rollout certification."
        ),
    }
    (BASE / "verified-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
