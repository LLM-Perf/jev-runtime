"""Create deterministic nonzero, untrained LoRAs for lifecycle/cache tests only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from jev_runtime.adapters import AdapterStore


def create(model_path: Path, destination: Path):
    config = json.loads((model_path / "config.json").read_text())
    source = json.loads((model_path / "jev-source.json").read_text())
    if config["architectures"] != ["LlamaForCausalLM"]:
        raise ValueError("This lifecycle fixture is limited to a dense Llama model")
    destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20260927)
    weights = {}
    rank, hidden = 8, config["hidden_size"]
    for layer in range(config["num_hidden_layers"]):
        prefix = f"base_model.model.model.layers.{layer}.self_attn.o_proj"
        weights[prefix + ".lora_A.weight"] = rng.normal(0, 0.1, (rank, hidden)).astype(np.float32)
        weights[prefix + ".lora_B.weight"] = rng.normal(0, 0.1, (hidden, rank)).astype(np.float32)
    manifest = {
        "qualification": "untrained nonzero lifecycle fixtures; not task-quality models",
        "base_model": source,
        "seed": 20260927,
        "artifacts": {},
    }
    for name, sign in (("positive", 1), ("negative", -1)):
        directory = destination / name
        directory.mkdir()
        (directory / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "base_model_name_or_path": source["model_id"],
                    "r": rank,
                    "lora_alpha": rank,
                    "lora_dropout": 0.0,
                    "target_modules": ["o_proj"],
                    "bias": "none",
                    "inference_mode": True,
                    "use_rslora": False,
                    "use_dora": False,
                },
                indent=2,
            )
            + "\n"
        )
        save_file(
            {key: value * sign if ".lora_B." in key else value for key, value in weights.items()},
            directory / "adapter_model.safetensors",
        )
        manifest["artifacts"][name] = {
            file: AdapterStore._hash(directory / file) for file in AdapterStore.files
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"destination": str(destination), "artifacts": list(manifest["artifacts"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    create(args.model_path, args.destination)
