# Jev Runtime

Private development repository for a typed decision runtime over SGLang and vLLM.

The target is a complete serving product: Choice, Boolean/Noul, Score and Rank;
versioned decision bundles; online activation and draining; model capability checks;
quality/calibration tools; and reproducible GPU evaluation.

**Status: implementation in progress.** A working unit test or simulated backend is
not GPU compatibility, model quality, performance certification, or a production SLA.
See [the implementation plan](docs/implementation-plan.md) and
[the completion ledger](docs/completion-ledger.md) for the full scope and evidence.

## Development

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,tokenizers]' -e packages/sglang -e packages/vllm
.venv/bin/pytest
.venv/bin/ruff check .
```

Engine adapters are separate distributions under `packages/` so the core does not
install two competing GPU dependency stacks. Run each engine in its own environment.
Model weights, credentials, private workload text, and raw service logs are excluded
from this repository.

See [engine setup](docs/engine-integration.md), [operations](docs/operations.md),
[calibration](docs/calibration.md), [public quality checks](docs/public-quality.md),
[managed LoRA](docs/adapters.md),
[performance experiments](docs/performance.md), and
[the TypeScript SDK](packages/typescript/README.md).

## Design

The model remains owned by its engine. The runtime compiles tasks into validated
label-scoring requests, obtains raw scores, applies a version-bound calibration and
policy, and returns typed results. An immutable bundle is pinned for the lifetime of
each request. Publishing a new version never mutates an in-flight request's bundle.

Configuration and calibration can change online. Installing new engine-side Python
or CUDA code requires a process rollout. A model or GPU adapter is only hot-swappable
after the relevant engine/model/parallel configuration has passed lifecycle tests.
