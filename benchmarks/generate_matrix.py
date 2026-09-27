"""Freeze the 144 required performance scenarios; generates no measurement results."""

import argparse
import itertools
import json
from pathlib import Path


def main(args):
    inventory = {item["id"]: item for item in json.loads(args.inventory.read_text())["models"]}
    if len(set(args.models)) != 2 or any(model not in inventory for model in args.models):
        raise SystemExit("Choose exactly two distinct checkpoints from the frozen inventory")
    if args.output.exists():
        raise SystemExit("Output exists; do not overwrite a frozen matrix")
    cases = []
    for engine, model, context, candidates, concurrency, cache in itertools.product(
        ("sglang", "vllm"), args.models, (256, 2048, 8192), (2, 8, 32), (1, 16), ("cold", "hot")
    ):
        cases.append(
            {
                "id": (
                    f"{engine}-{model.rsplit('/', 1)[-1]}-L{context}"
                    f"-K{candidates}-C{concurrency}-{cache}"
                ),
                "engine": engine,
                "model_id": model,
                "model_revision": inventory[model]["revision"],
                "context_tokens": context,
                "candidates": candidates,
                "concurrency": concurrency,
                "cross_request_cache": cache,
                "duration_seconds": 180,
                "repeats": 3,
                "status": "not_run",
            }
        )
    report = {
        "schema_version": 1,
        "case_count": len(cases),
        "measurement_results": False,
        "release_gate_passed": False,
        "minimum_measurement_seconds_per_method": len(cases) * 180 * 3,
        "required_methods": [
            "native-label-scoring",
            "typed-plugin",
            "typed-gateway",
            "structured-generation",
            "independent-unoptimized",
            "existing-jev",
        ],
        "note": "144 scenarios per method; warmup, loading, baselines and soak add time.",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case_count": len(cases), "output": str(args.output), "measurements": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=Path("profiles/model-inventory.json"))
    parser.add_argument("--models", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
