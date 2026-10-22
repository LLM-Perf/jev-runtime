"""Synthetic gate fixtures test rejection logic; they are never GPU evidence."""

import json
import zipfile

import pytest

from deployment.qualify_release import unpack, validate
from jev_runtime.production import (
    FAULTS,
    STAGES,
    ProductionProfile,
    qualify,
    sha256,
)


@pytest.fixture
def qualification(tmp_path):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    dist = tmp_path / "dist"
    dist.mkdir()
    wheels = {}
    for name in ("core", "vllm", "sglang", "tokenspeed"):
        path = dist / (name + ".whl")
        path.write_bytes(b"unit fixture, not a usable wheel")
        wheels[path.name] = sha256(path)
    identity = {
        "engine": "vllm",
        "engine_version": "fixture",
        "model_id": "fixture",
        "model_revision": "a" * 40,
        "tokenizer_sha256": "a" * 64,
        "chat_template_sha256": "b" * 64,
        "engine_config_sha256": "c" * 64,
        "gpu_model": "fixture",
        "compute_capability": "9.0",
        "dtype": "bfloat16",
        "readout_dtype": "float32",
        "quantization": None,
        "tensor_parallel_size": 1,
        "attention_backend": "fixture",
        "execution_mode": "eager",
        "max_context": 2048,
        "max_candidates": 8,
        "concurrency": 4,
    }
    quality = [
        {
            "dataset_id": name,
            "dataset_sha256": char * 64,
            "minimum_samples": 100,
            "minimum_accuracy": 0.9,
            "minimum_coverage": 0.9,
            "maximum_ece": 0.1,
        }
        for name, char in (("one", "1"), ("two", "2"))
    ]
    profile = ProductionProfile.model_validate(
        {
            "profile_id": "fixture",
            "engine": "vllm",
            "status": "candidate",
            "source_commit": "d" * 40,
            "identity": identity,
            "image_digest": "sha256:" + "e" * 64,
            "wheel_sha256": wheels,
            "acceptance": {
                "maximum_p99_ms": 100.0,
                "minimum_requests_per_second": 10.0,
                "maximum_logprob_error": 0.01,
                "maximum_probability_error": 0.01,
                "maximum_recovery_seconds": 30.0,
                "maximum_rss_growth_bytes": 100,
                "maximum_gpu_growth_bytes": 100,
                "maximum_storage_growth_bytes": 100,
                "quality": quality,
            },
        }
    )
    path = profile_dir / "fixture.json"
    path.write_text(profile.model_dump_json(indent=2))
    evidence = tmp_path / "evidence/fixture"
    evidence.mkdir(parents=True)
    raw = evidence / "raw.txt"
    raw.write_text("UNIT TEST ONLY - NOT GPU EVIDENCE")
    traffic = {
        "attempted": 10000,
        "strict_successes": 10000,
        "invalid_responses": 0,
        "mixed_bundle_responses": 0,
        "unaccounted_requests": 0,
        "all_four_types": True,
    }
    performance = {
        "p99_ms": 80.0,
        "requests_per_second": 12.0,
        "open_loop": True,
        "telemetry_complete": True,
        "load_generator_kept_up": True,
    }
    stages = {
        "installation": {
            "clean_wheel_install": True,
            "chat_template": True,
            "pip_check": True,
            "cli_smoke": True,
            "failed_tests": 0,
            "passed_tests": 618,
            "python_versions": ["3.11", "3.12"],
        },
        "functional": {
            **traffic,
            "auth_separation": True,
            "hot_switch": True,
            "confirmed_drain": True,
            "cancel_timeout_retains_work": True,
        },
        "numerical": {
            "samples": 100,
            "non_tie_samples": 100,
            "max_logprob_error": 0.001,
            "max_probability_error": 0.001,
            "non_tie_agreement": 1.0,
            "independent_reference": True,
            "native_plugin_parity": True,
            "cold_hot": True,
            "batch_order": True,
        },
        "quality": {
            "datasets": [
                {
                    "dataset_id": q["dataset_id"],
                    "dataset_sha256": q["dataset_sha256"],
                    "held_out": True,
                    "samples": 100,
                    "accuracy": 0.95,
                    "coverage": 0.95,
                    "ece": 0.05,
                }
                for q in quality
            ]
        },
        "performance": {
            **traffic,
            **performance,
            "repetitions": 3,
            "minimum_repetition_seconds": 180,
            "native_throughput_ratio": 0.95,
            "equivalent_outputs": True,
            "context_candidate_concurrency_coverage": True,
            "cold_hot": True,
        },
        "recovery": {
            **dict.fromkeys(FAULTS, True),
            "lost_acknowledged_publications": 0,
            "early_resource_releases": 0,
            "duplicate_owners": 0,
            "maximum_recovery_seconds": 10.0,
        },
        "image": {
            "image_digest": profile.image_digest,
            **dict.fromkeys(
                (
                    "built",
                    "runtime_smoke",
                    "persistent_volume",
                    "non_root",
                    "probes",
                    "sigterm_drain",
                ),
                True,
            ),
        },
        "soak": {
            **traffic,
            **performance,
            "duration_seconds": 86400,
            "rss_growth_bytes": 0,
            "gpu_growth_bytes": 0,
            "storage_growth_bytes": 0,
            "mixed_inputs": True,
            "multiple_tenants": True,
            "hot_updates": True,
            "faults_reported_separately": True,
        },
    }
    for stage, metrics in stages.items():
        (evidence / (stage + ".json")).write_text(
            json.dumps(
                {
                    "stage": stage,
                    "profile_sha256": sha256(path),
                    "source_commit": profile.source_commit,
                    "identity": identity,
                    "started_at": 100.0,
                    "finished_at": 86500.0,
                    "passed": True,
                    "metrics": metrics,
                    "artifacts": {"raw.txt": sha256(raw)},
                }
            )
        )
    return path, evidence, profile.source_commit, dist


def test_complete_synthetic_evidence_passes(qualification):
    assert qualify(*qualification)["passed"]


@pytest.mark.parametrize("stage", STAGES)
def test_missing_stage_fails(qualification, stage):
    (qualification[1] / (stage + ".json")).unlink()
    assert not qualify(*qualification)["passed"]


@pytest.mark.parametrize(
    "stage,key,value",
    [
        ("functional", "strict_successes", 9980),
        ("functional", "strict_successes", 10001),
        ("functional", "attempted", True),
        ("functional", "invalid_responses", 1),
        ("numerical", "non_tie_samples", 0),
        ("numerical", "max_logprob_error", 0.2),
        ("numerical", "max_probability_error", float("nan")),
        ("performance", "equivalent_outputs", False),
        ("performance", "repetitions", 1),
        ("performance", "native_throughput_ratio", 0.85),
        ("soak", "duration_seconds", 3600),
        ("recovery", "early_resource_releases", 1),
        ("recovery", "stale_owner_fenced", False),
        ("image", "runtime_smoke", False),
        ("installation", "failed_tests", 14),
        ("installation", "python_versions", ["3.12"]),
        ("quality", "datasets", []),
    ],
)
def test_insufficient_or_invalid_metrics_fail(qualification, stage, key, value):
    path = qualification[1] / (stage + ".json")
    report = json.loads(path.read_text())
    report["metrics"][key] = value
    path.write_text(json.dumps(report))
    assert not qualify(*qualification)["passed"]


@pytest.mark.parametrize(
    "change", ["source", "wheel", "raw", "identity", "profile", "symlink", "traversal", "duration"]
)
def test_evidence_identity_and_integrity_fail_closed(qualification, change, tmp_path):
    path, evidence, commit, dist = qualification
    report_path = evidence / "soak.json"
    report = json.loads(report_path.read_text())
    if change == "source":
        commit = "0" * 40
    elif change == "wheel":
        next(dist.glob("*.whl")).write_bytes(b"changed")
    elif change == "raw":
        (evidence / "raw.txt").write_text("changed")
    elif change == "identity":
        report["identity"]["dtype"] = "float16"
    elif change == "profile":
        path.write_text(path.read_text() + "\n")
    elif change == "symlink":
        original = evidence / "raw.txt"
        external = tmp_path / "external"
        external.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(external)
    elif change == "traversal":
        report["artifacts"] = {"../raw.txt": "0" * 64}
    else:
        report["finished_at"] = 1000.0
    report_path.write_text(json.dumps(report))
    assert not qualify(path, evidence, commit, dist)["passed"]


def test_draft_profile_never_certifies(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text(ProductionProfile(profile_id="draft", engine="tokenspeed").model_dump_json())
    assert not qualify(path, tmp_path, "a" * 40, tmp_path)["passed"]


def test_zip_rejects_traversal_and_wrong_hash(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="hash"):
        unpack(archive, "0" * 64, tmp_path / "out")
    with pytest.raises(ValueError, match="Unsafe"):
        unpack(archive, sha256(archive), tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def test_release_requires_all_three_engines(qualification, tmp_path):
    path, evidence, commit, dist = qualification
    archive = tmp_path / "one-engine.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(path, "profiles/fixture.json")
        for item in evidence.iterdir():
            bundle.write(item, "evidence/fixture/" + item.name)
    result = validate(archive, sha256(archive), dist, commit)
    assert not result["passed"]
    assert result["missing_engines"] == ["sglang", "tokenspeed"]
