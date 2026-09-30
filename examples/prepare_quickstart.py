"""Prepare the pinned SmolLM2 README demo; run inside the chosen engine's environment."""

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

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
REVISION = "31b70e2e869a7173562077fd711b654946d38674"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--output", type=Path, help="New directory; existing paths are rejected")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Offline override: existing SmolLM2 weights at the documented pinned revision",
    )
    args = parser.parse_args()
    port = args.port if args.port is not None else (8000 if args.engine == "vllm" else 30000)
    if not 1 <= port <= 65535:
        parser.error("--port must be between 1 and 65535")
    root = (args.output or Path(f".jev/quickstart-{args.engine}")).resolve()
    if root.exists():
        parser.error(f"{root} already exists; reuse its env.sh or choose a new --output")
    model = (
        args.model_path.resolve(strict=True)
        if args.model_path
        else Path(
            snapshot_download(
                MODEL,
                revision=REVISION,
                allow_patterns=["*.json", "*.safetensors", "*.jinja"],
            )
        ).resolve()
    )
    if not (model / "config.json").is_file() or not any(model.glob("*.safetensors")):
        parser.error("The model directory must contain config.json and safetensors weights")
    root.mkdir(parents=True, mode=0o700)
    tokenizer = root / "tokenizer"
    preserve_fast_tokenizer(model, tokenizer)
    settings = Settings(
        backend=args.engine,
        engine_url=f"http://127.0.0.1:{port}",
        model_id=MODEL,
        model_revision=REVISION,
        tokenizer=str(tokenizer),
        dtype="bfloat16",
        registry_path=str(root / "registry.db"),
        host="127.0.0.1",
        port=port,
        bootstrap_alias="decision-model",
    )
    config = root / "config.json"
    config.write_text(settings.model_dump_json(indent=2) + "\n")
    env = {
        "JEV_RUN": str(root),
        "JEV_CONFIG": str(config),
        "JEV_MODEL_PATH": str(model),
        "JEV_TOKENIZER_PATH": str(tokenizer),
        "JEV_MODEL_ID": MODEL,
        "JEV_PORT": str(port),
        "JEV_URL": f"http://127.0.0.1:{port}/plugins/jev-runtime",
        "JEV_API_KEY": secrets.token_urlsafe(32),
        "JEV_ADMIN_KEY": secrets.token_urlsafe(32),
        "VLLM_PLUGINS" if args.engine == "vllm" else "SGLANG_PLUGINS": (
            "jev_runtime_api" if args.engine == "vllm" else "jev_runtime"
        ),
    }
    env_path = root / "env.sh"
    with os.fdopen(os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as file:
        file.write("# Local demo credentials. Do not commit or share this file.\n")
        for key, value in env.items():
            file.write(f"export {key}={shlex.quote(value)}\n")
    print(json.dumps({"config": str(config), "environment": str(env_path)}, indent=2))


if __name__ == "__main__":
    main()
