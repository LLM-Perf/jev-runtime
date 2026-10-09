"""Read-only startup checks; these do not load weights or certify GPU execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from jev_runtime.errors import JevError
from jev_tokenspeed.backend import validate_profile
from jev_tokenspeed.plugin import validate_binding, verify_source

SUPPORTED_NVIDIA_CAPABILITIES = {(9, 0), (10, 0), (10, 3), (10, 7)}


def check_hardware(device_index: int = 0) -> dict:
    import torch

    if type(device_index) is not int or device_index < 0:
        raise JevError("tokenspeed_hardware", "base_gpu_id must be a nonnegative integer", 409)
    if torch.version.hip or not torch.cuda.is_available():
        raise JevError(
            "tokenspeed_hardware", "This Jev TokenSpeed profile requires an NVIDIA CUDA GPU", 409
        )
    if device_index >= torch.cuda.device_count():
        raise JevError("tokenspeed_hardware", "base_gpu_id is outside the visible GPU set", 409)
    capability = torch.cuda.get_device_capability(device_index)
    if capability not in SUPPORTED_NVIDIA_CAPABILITIES:
        raise JevError(
            "tokenspeed_hardware",
            f"Visible GPU {device_index} is sm{capability[0]}{capability[1]}; "
            "the pinned profile requires sm90, sm100, sm103 or sm107. "
            "Select a supported GPU with CUDA_VISIBLE_DEVICES before launch.",
            409,
        )
    return {
        "visible_index": device_index,
        "device": torch.cuda.get_device_name(device_index),
        "capability": list(capability),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def installed_source() -> Path:
    spec = importlib.util.find_spec("tokenspeed")
    if spec is None or spec.origin is None:
        raise JevError(
            "tokenspeed_not_installed",
            "Install the pinned TokenSpeed source and its GPU dependencies first; "
            "jev-tokenspeed does not install an inference engine",
            409,
        )
    return Path(spec.origin).parent


def check(settings, engine_options: dict) -> dict:
    root = installed_source()
    revision = verify_source(root)
    hardware = check_hardware(engine_options.get("base_gpu_id", 0))
    from tokenspeed.runtime.utils.server_args import ServerArgs

    args = ServerArgs(**engine_options)
    validate_profile(args)
    validate_binding(settings, args)
    return {
        "passed": True,
        "scope": "source, hardware and configuration preflight; no model execution",
        "upstream_revision": revision,
        "source": str(root),
        "hardware": hardware,
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "gpu_validated": False,
    }
