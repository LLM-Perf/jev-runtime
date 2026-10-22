"""Version-bound production qualification. Missing evidence always fails closed.

This validates operator-produced evidence; hashes establish identity/integrity,
not independent attestation that a GPU campaign was honestly executed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
COMMIT = Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
STAGES = (
    "installation",
    "functional",
    "numerical",
    "quality",
    "performance",
    "recovery",
    "image",
    "soak",
)
FAULTS = (
    "worker_kill",
    "engine_kill",
    "container_replace",
    "activation_crash",
    "disk_write_failure",
    "restore_latest",
    "stale_owner_fenced",
    "rollback",
)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class RuntimeIdentity(Record):
    engine: Literal["vllm", "sglang", "tokenspeed"]
    engine_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^(?:[a-f0-9]{40,64}|local-sha256:[a-f0-9]{64})$")
    tokenizer_sha256: SHA256
    chat_template_sha256: SHA256
    engine_config_sha256: SHA256
    gpu_model: str = Field(min_length=1)
    compute_capability: str = Field(pattern=r"^\d+\.\d+$")
    dtype: Literal["float32", "bfloat16", "float16"]
    readout_dtype: Literal["float32", "bfloat16", "float16"]
    quantization: str | None
    tensor_parallel_size: int = Field(ge=1)
    attention_backend: str = Field(min_length=1)
    execution_mode: str = Field(min_length=1)
    max_context: int = Field(ge=1)
    max_candidates: int = Field(ge=2)
    concurrency: int = Field(ge=1)


class QualityTarget(Record):
    dataset_id: str = Field(min_length=1)
    dataset_sha256: SHA256
    minimum_samples: int = Field(ge=1)
    minimum_accuracy: float = Field(gt=0, le=1)
    minimum_coverage: float = Field(gt=0, le=1)
    maximum_ece: float = Field(ge=0, le=1)


class Acceptance(Record):
    minimum_requests: int = Field(default=10000, ge=10000)
    minimum_strict_success_rate: float = Field(default=0.999, ge=0.999, le=1)
    maximum_p99_ms: float = Field(gt=0)
    minimum_requests_per_second: float = Field(gt=0)
    maximum_logprob_error: float = Field(gt=0)
    maximum_probability_error: float = Field(gt=0, lt=1)
    minimum_non_tie_agreement: float = Field(default=1.0, ge=0.999, le=1)
    minimum_numerical_samples: int = Field(default=100, ge=100)
    minimum_native_throughput_ratio: float = Field(default=0.9, ge=0.9, le=1)
    maximum_recovery_seconds: float = Field(gt=0)
    maximum_rss_growth_bytes: int = Field(ge=0)
    maximum_gpu_growth_bytes: int = Field(ge=0)
    maximum_storage_growth_bytes: int = Field(ge=0)
    quality: list[QualityTarget] = Field(min_length=2)

    @model_validator(mode="after")
    def distinct_datasets(self):
        if len({q.dataset_sha256 for q in self.quality}) != len(self.quality):
            raise ValueError("Quality gates require distinct held-out datasets")
        return self


class ProductionProfile(Record):
    schema_version: Literal[1] = 1
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,100}$")
    engine: Literal["vllm", "sglang", "tokenspeed"]
    status: Literal["draft", "candidate"] = "draft"
    source_commit: COMMIT | None = None
    identity: RuntimeIdentity | None = None
    image_digest: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")
    wheel_sha256: dict[str, SHA256] = Field(default_factory=dict)
    acceptance: Acceptance | None = None

    @model_validator(mode="after")
    def candidate_is_complete(self):
        if self.identity and self.identity.engine != self.engine:
            raise ValueError("Profile engine and observed identity must agree")
        if self.status == "candidate" and (
            not all((self.source_commit, self.identity, self.image_digest, self.acceptance))
            or len(self.wheel_sha256) != 4
        ):
            raise ValueError(
                "Candidate requires identity, image, four wheels and acceptance targets"
            )
        if any(Path(name).name != name or not name.endswith(".whl") for name in self.wheel_sha256):
            raise ValueError("Wheel keys must be plain wheel filenames")
        return self


class StageReport(Record):
    schema_version: Literal[1] = 1
    stage: Literal[
        "installation",
        "functional",
        "numerical",
        "quality",
        "performance",
        "recovery",
        "image",
        "soak",
    ]
    profile_sha256: SHA256
    source_commit: COMMIT
    identity: RuntimeIdentity
    started_at: float = Field(gt=0)
    finished_at: float = Field(gt=0)
    passed: bool
    metrics: dict
    artifacts: dict[str, SHA256] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.finished_at < self.started_at:
            raise ValueError("Invalid evidence interval")
        return self


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(path.read_text(), parse_constant=invalid)


def member(root: Path, name: str) -> Path:
    if root.is_symlink():
        raise ValueError("Evidence root must not be a symlink")
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or not rel.parts:
        raise ValueError("Evidence path must stay inside its directory")
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Evidence must not contain symlinks")
    if not path.is_file():
        raise ValueError(f"Missing evidence file: {name}")
    return path


def check_metrics(report: StageReport, profile: ProductionProfile) -> list[str]:
    errors = []
    values, limits = report.metrics, profile.acceptance

    def bound(key, threshold, *, minimum=False):
        value = values.get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (not key.endswith("_growth_bytes") and value < 0)
            or (key in {"accuracy", "coverage", "ece", "non_tie_agreement"} and value > 1)
            or (
                key in {"samples", "non_tie_samples", "repetitions", "passed_tests"}
                and type(value) is not int
            )
            or (value < threshold if minimum else value > threshold)
        ):
            errors.append(f"{key}: required {'>=' if minimum else '<='} {threshold}")

    def exact(key, expected):
        value = values.get(key)
        if type(value) is not type(expected) or value != expected:
            errors.append(f"{key}: required {expected!r}")

    if report.stage == "installation":
        for key in ("clean_wheel_install", "chat_template", "pip_check", "cli_smoke"):
            exact(key, True)
        exact("failed_tests", 0)
        bound("passed_tests", 1, minimum=True)
        if set(values.get("python_versions", [])) != {"3.11", "3.12"}:
            errors.append("python_versions: require 3.11 and 3.12")
    if report.stage in {"functional", "performance", "soak"}:
        bound("attempted", limits.minimum_requests, minimum=True)
        attempted, succeeded = values.get("attempted"), values.get("strict_successes")
        if (
            type(attempted) is not int
            or type(succeeded) is not int
            or not 0 <= succeeded <= attempted
            or attempted <= 0
        ):
            errors.append("Invalid strict-success denominator")
        elif succeeded / attempted < limits.minimum_strict_success_rate:
            errors.append("Strict-success rate below target")
        for key in ("invalid_responses", "mixed_bundle_responses", "unaccounted_requests"):
            exact(key, 0)
        exact("all_four_types", True)
    if report.stage == "functional":
        for key in (
            "auth_separation",
            "hot_switch",
            "confirmed_drain",
            "cancel_timeout_retains_work",
        ):
            exact(key, True)
    if report.stage == "numerical":
        bound("samples", limits.minimum_numerical_samples, minimum=True)
        bound("non_tie_samples", limits.minimum_numerical_samples, minimum=True)
        bound("max_logprob_error", limits.maximum_logprob_error)
        bound("max_probability_error", limits.maximum_probability_error)
        bound("non_tie_agreement", limits.minimum_non_tie_agreement, minimum=True)
        if values.get("non_tie_samples", 0) > values.get("samples", 0):
            errors.append("non_tie_samples exceeds total samples")
        for key in ("independent_reference", "native_plugin_parity", "cold_hot", "batch_order"):
            exact(key, True)
    if report.stage == "quality":
        datasets = values.get("datasets", [])
        for target in limits.quality:
            matches = [
                d
                for d in datasets
                if isinstance(d, dict)
                and d.get("dataset_sha256") == target.dataset_sha256
                and d.get("dataset_id") == target.dataset_id
            ]
            if len(matches) != 1:
                errors.append(f"Missing or duplicate quality dataset: {target.dataset_id}")
                continue
            values = matches[0]
            exact("held_out", True)
            bound("samples", target.minimum_samples, minimum=True)
            bound("accuracy", target.minimum_accuracy, minimum=True)
            bound("coverage", target.minimum_coverage, minimum=True)
            bound("ece", target.maximum_ece)
    if report.stage in {"performance", "soak"}:
        bound("p99_ms", limits.maximum_p99_ms)
        bound("requests_per_second", limits.minimum_requests_per_second, minimum=True)
        exact("open_loop", True)
        exact("telemetry_complete", True)
        exact("load_generator_kept_up", True)
    if report.stage == "performance":
        bound("repetitions", 3, minimum=True)
        bound("minimum_repetition_seconds", 180, minimum=True)
        bound("native_throughput_ratio", limits.minimum_native_throughput_ratio, minimum=True)
        for key in ("equivalent_outputs", "context_candidate_concurrency_coverage", "cold_hot"):
            exact(key, True)
    if report.stage == "soak":
        bound("duration_seconds", 86400, minimum=True)
        if report.finished_at - report.started_at < 86400:
            errors.append("Evidence interval is shorter than 24 hours")
        bound("rss_growth_bytes", limits.maximum_rss_growth_bytes)
        bound("gpu_growth_bytes", limits.maximum_gpu_growth_bytes)
        bound("storage_growth_bytes", limits.maximum_storage_growth_bytes)
        for key in (
            "mixed_inputs",
            "multiple_tenants",
            "hot_updates",
            "faults_reported_separately",
        ):
            exact(key, True)
    if report.stage == "recovery":
        for fault in FAULTS:
            exact(fault, True)
        for key in (
            "lost_acknowledged_publications",
            "early_resource_releases",
            "duplicate_owners",
        ):
            exact(key, 0)
        bound("maximum_recovery_seconds", limits.maximum_recovery_seconds)
        if profile.engine == "tokenspeed":
            exact("coordinated_receipts_restore", True)
    if report.stage == "image":
        exact("image_digest", profile.image_digest)
        for key in (
            "built",
            "runtime_smoke",
            "persistent_volume",
            "non_root",
            "probes",
            "sigterm_drain",
        ):
            exact(key, True)
    return errors


def qualify(profile_path: Path, evidence_dir: Path, source_commit: str, dist: Path) -> dict:
    profile = ProductionProfile.model_validate(read_json(profile_path))
    errors = []
    if profile.status != "candidate":
        return {"passed": False, "profile_id": profile.profile_id, "errors": ["Profile is draft"]}
    if profile.source_commit != source_commit:
        errors.append("Source commit differs from release")
    wheels = {p.name: sha256(p) for p in dist.glob("*.whl") if not p.is_symlink()}
    if wheels != profile.wheel_sha256:
        errors.append("Built wheels differ from qualified wheels")
    profile_hash = sha256(profile_path)
    for stage in STAGES:
        try:
            report = StageReport.model_validate(read_json(member(evidence_dir, stage + ".json")))
            if (
                report.stage != stage
                or report.profile_sha256 != profile_hash
                or report.source_commit != source_commit
                or report.identity != profile.identity
            ):
                raise ValueError(
                    "Evidence belongs to a different stage, profile, source or runtime"
                )
            if not report.passed:
                raise ValueError("Campaign reported failure")
            for name, expected in report.artifacts.items():
                if sha256(member(evidence_dir, name)) != expected:
                    raise ValueError(f"Evidence artifact changed: {name}")
            errors.extend(f"{stage}: {error}" for error in check_metrics(report, profile))
        except (ValueError, OSError, TypeError, KeyError) as exc:
            errors.append(f"{stage}: {exc}")
    return {
        "passed": not errors,
        "profile_id": profile.profile_id,
        "source_commit": source_commit,
        "profile_sha256": profile_hash,
        "errors": errors,
    }
