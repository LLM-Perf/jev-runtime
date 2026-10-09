"""Prepare a pinned Qwen3-0.6B TokenSpeed demo in a fresh directory."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
from pathlib import Path

from huggingface_hub import snapshot_download

from jev_runtime.config import Settings
from jev_runtime.tokenizer_profiles import preserve_fast_tokenizer

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".jev/quickstart-tokenspeed"))
    parser.add_argument("--port", type=int, default=8796)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Offline override: Qwen3-0.6B weights at the documented pinned revision",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    root = args.output.resolve()
    if root.exists():
        parser.error(f"{root} already exists; reuse env.sh or choose a new --output")
    model = (
        args.model_path.resolve(strict=True)
        if args.model_path
        else Path(
            snapshot_download(
                MODEL,
                revision=REVISION,
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
            )
        ).resolve()
    )
    if not (model / "config.json").is_file() or not any(model.glob("*.safetensors")):
        parser.error("The model directory must contain config.json and safetensors weights")
    root.mkdir(parents=True, mode=0o700)
    (root / "model-source.json").write_text(
        json.dumps(
            {
                "model_id": str(model),
                "repository": MODEL,
                "revision": REVISION,
                "provenance": "operator-asserted local checkpoint"
                if args.model_path
                else "pinned Hub snapshot",
            },
            indent=2,
        )
        + "\n"
    )
    tokenizer = root / "tokenizer"
    preserve_fast_tokenizer(model, tokenizer)
    settings = Settings(
        backend="tokenspeed",
        engine_url=f"http://127.0.0.1:{args.port}",
        model_id=str(model),
        model_revision=REVISION,
        tokenizer=str(tokenizer),
        dtype="bfloat16",
        readout_dtype="bfloat16",
        registry_path=str(root / "registry.db"),
        host="127.0.0.1",
        port=args.port,
        bootstrap_alias="decision-model",
    )
    config = root / "config.json"
    config.write_text(settings.model_dump_json(indent=2) + "\n")
    engine_config = root / "engine.json"
    engine_config.write_text(
        json.dumps(
            {
                "model": str(model),
                "tokenizer": str(tokenizer),
                "revision": REVISION,
                "dtype": "bfloat16",
                "sampling_backend": "triton_full",
                "enable_output_logprobs": True,
                "enforce_eager": True,
                "world_size": 1,
                "max_model_len": 4096,
                "max_num_seqs": 16,
                "gpu_memory_utilization": 0.5,
                "enable_prefix_caching": True,
            },
            indent=2,
        )
        + "\n"
    )
    env = {
        "JEV_RUN": str(root),
        "JEV_CONFIG": str(config),
        "JEV_ENGINE_CONFIG": str(engine_config),
        "JEV_MODEL_PATH": str(model),
        "JEV_TOKENIZER_PATH": str(tokenizer),
        "JEV_MODEL_ID": str(model),
        "JEV_PORT": str(args.port),
        "JEV_URL": f"http://127.0.0.1:{args.port}/plugins/jev-runtime",
        "JEV_API_KEY": secrets.token_urlsafe(32),
        "JEV_ADMIN_KEY": secrets.token_urlsafe(32),
    }
    env_path = root / "env.sh"
    with os.fdopen(os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as file:
        file.write("# Local demo credentials. Do not commit or share this file.\n")
        for key, value in env.items():
            file.write(f"export {key}={shlex.quote(value)}\n")
    print(
        json.dumps(
            {
                "config": str(config),
                "engine_config": str(engine_config),
                "environment": str(env_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
