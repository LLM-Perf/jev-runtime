from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field, TypeAdapter

from jev_runtime.errors import JevError
from jev_runtime.schema import Contract, Identifier, content_digest

# Expand only alongside native lifecycle/cache/cancellation GPU evidence.
# This limits the optional LoRA feature, not base-model typed scoring.
LORA_BASE_PROFILES = frozenset(
    {
        (
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "31b70e2e869a7173562077fd711b654946d38674",
        )
    }
)


def validate_lora_base(model_id: str, revision: str) -> None:
    if (model_id, revision) not in LORA_BASE_PROFILES:
        raise JevError(
            "adapter_profile",
            "Managed LoRA is restricted to lifecycle-tested checkpoint revisions; "
            "base-model scoring remains available",
            409,
        )


class AdapterArtifact(Contract):
    id: Identifier
    revision: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    base_model_id: str
    base_model_revision: str
    path: str
    rank: int = Field(ge=1, le=64)
    targets: tuple[str, ...]
    files: dict[str, str]
    bytes: int = Field(gt=0)

    @property
    def reference(self) -> str:
        return f"{self.id}@{self.revision}"


class AdapterBinding(Contract):
    artifact: AdapterArtifact
    engine_name: str
    engine_id: int = Field(ge=1, le=2**31 - 1)


class AdapterStore:
    """Copy an explicitly allowed local PEFT artifact into immutable owned storage.

    Registration never runs adapter Python code or downloads a repository. The
    engine only sees the verified copy, not a mutable caller-provided directory.
    This validates a narrow dense-linear LoRA format; engine/model certification
    is an additional requirement before an artifact can receive traffic.
    """

    files = ("adapter_config.json", "adapter_model.safetensors")
    targets = frozenset(
        {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    )
    tensor_name = re.compile(
        r"^base_model\.model\.model\.layers\.\d+\.(?:self_attn|mlp)\."
        r"(?P<target>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        r"\.lora_(?P<side>A|B)\.weight$"
    )

    def __init__(
        self, root: str | Path, allowed_roots: tuple[str, ...], max_bytes: int = 268435456
    ):
        self.root = Path(root).resolve()
        self.allowed_roots = tuple(Path(path).resolve(strict=True) for path in allowed_roots)
        self.max_bytes = max_bytes
        if max_bytes <= 0:
            raise ValueError("Adapter size limit must be positive")

    @staticmethod
    def _hash(path: Path) -> str:
        with path.open("rb") as file:
            return "sha256:" + hashlib.file_digest(file, "sha256").hexdigest()

    def _validate(self, directory: Path, base_model_id: str) -> tuple[int, tuple[str, ...]]:
        from safetensors import safe_open

        path = directory / "adapter_config.json"
        if path.stat().st_size > 1024 * 1024:
            raise JevError("adapter_config", "Adapter configuration is too large", 413)
        config = json.loads(path.read_text())
        if not isinstance(config, dict):
            raise JevError("adapter_config", "Adapter configuration must be an object", 409)
        # Untrusted JSON fields; the format gate below enforces their runtime types.
        rank: Any = config.get("r")
        alpha: Any = config.get("lora_alpha")
        targets = config.get("target_modules")
        if (
            config.get("peft_type") != "LORA"
            or config.get("task_type") != "CAUSAL_LM"
            or config.get("base_model_name_or_path") != base_model_id
            or type(rank) is not int
            or not 1 <= rank <= 64
            or type(alpha) not in (float, int)
            or not math.isfinite(alpha)
            or alpha <= 0
            or not isinstance(targets, list)
            or not targets
            or any(not isinstance(target, str) or target not in self.targets for target in targets)
            or config.get("bias", "none") != "none"
            or config.get("modules_to_save")
            or config.get("use_dora", False)
            or config.get("use_rslora", False)
            or config.get("fan_in_fan_out", False)
            or config.get("rank_pattern")
            or config.get("alpha_pattern")
        ):
            raise JevError("adapter_format", "Unsupported LoRA format or base-model binding", 409)
        pairs: dict[str, set[str]] = {}
        with safe_open(directory / "adapter_model.safetensors", framework="numpy") as tensors:
            for name in tensors.keys():
                match = self.tensor_name.fullmatch(name)
                if match is None or match["target"] not in targets:
                    raise JevError(
                        "adapter_tensor", "Only declared dense-linear LoRA tensors are allowed", 409
                    )
                tensor = tensors.get_slice(name)
                shape = tensor.get_shape()
                if (
                    len(shape) != 2
                    or min(shape) <= 0
                    or tensor.get_dtype() not in {"BF16", "F16", "F32"}
                    or shape[0 if match["side"] == "A" else 1] != rank
                ):
                    raise JevError(
                        "adapter_tensor", "LoRA tensor shape, rank or dtype is unsupported", 409
                    )
                stem = name.rsplit(".lora_", 1)[0]
                pairs.setdefault(stem, set()).add(match["side"])
        if not pairs or any(pair != {"A", "B"} for pair in pairs.values()):
            raise JevError("adapter_tensor", "Every adapted layer needs both LoRA matrices", 409)
        return rank, tuple(sorted(set(targets)))

    def register(
        self, adapter_id: str, source: str, base_model_id: str, base_model_revision: str
    ) -> AdapterArtifact:
        TypeAdapter(Identifier).validate_python(adapter_id)
        source_path = Path(source).resolve(strict=True)
        if not self.allowed_roots or not any(
            source_path.is_relative_to(root) for root in self.allowed_roots
        ):
            raise JevError("adapter_path", "Adapter source is outside configured local roots", 403)
        if not source_path.is_dir():
            raise JevError("adapter_path", "Adapter source must be a directory", 400)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        total = 0
        with tempfile.TemporaryDirectory(prefix=".staging-", dir=self.root) as temporary:
            staged = Path(temporary)
            for name in self.files:
                path = source_path / name
                if path.is_symlink():
                    raise JevError(
                        "adapter_path", "Adapter tensor/configuration symlinks are not allowed", 403
                    )
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as original:
                    if not stat.S_ISREG(os.fstat(original.fileno()).st_mode):
                        raise JevError("adapter_path", "Adapter files must be regular files", 400)
                    with (staged / name).open("xb") as target:
                        while block := original.read(1024 * 1024):
                            total += len(block)
                            if total > self.max_bytes:
                                raise JevError(
                                    "adapter_budget",
                                    "Adapter exceeds its configured byte limit",
                                    413,
                                )
                            target.write(block)
                        target.flush()
                        os.fsync(target.fileno())
            rank, targets = self._validate(staged, base_model_id)
            hashes = {name: self._hash(staged / name) for name in self.files}
            revision = content_digest(
                {
                    "base_model_id": base_model_id,
                    "base_model_revision": base_model_revision,
                    "files": hashes,
                }
            )
            destination = self.root / revision.removeprefix("sha256:")
            descriptor = os.open(staged, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                os.rename(staged, destination)
            except OSError:
                if (
                    destination.is_symlink()
                    or not destination.is_dir()
                    or any(
                        (destination / name).is_symlink()
                        or self._hash(destination / name) != digest
                        for name, digest in hashes.items()
                    )
                ):
                    raise JevError(
                        "adapter_corrupted", "Existing immutable adapter bytes differ", 409
                    ) from None
            for name in self.files:
                (destination / name).chmod(0o444)
            # Persist the rename before publishing a database reference.
            descriptor = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return AdapterArtifact(
            id=adapter_id,
            revision=revision,
            base_model_id=base_model_id,
            base_model_revision=base_model_revision,
            path=str(destination),
            rank=rank,
            targets=targets,
            files=hashes,
            bytes=total,
        )

    def verify(self, artifact: AdapterArtifact) -> None:
        directory = Path(artifact.path)
        expected = self.root / artifact.revision.removeprefix("sha256:")
        identity = content_digest(
            {
                "base_model_id": artifact.base_model_id,
                "base_model_revision": artifact.base_model_revision,
                "files": artifact.files,
            }
        )
        if set(artifact.files) != set(self.files) or identity != artifact.revision:
            raise JevError("adapter_corrupted", "Adapter content identity does not match", 409)
        if (
            directory != expected
            or directory.resolve() != expected
            or any(
                (directory / name).is_symlink() or self._hash(directory / name) != digest
                for name, digest in artifact.files.items()
            )
        ):
            raise JevError("adapter_corrupted", "Immutable adapter verification failed", 409)
