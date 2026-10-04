# Contributing to Jev Runtime

Thanks for helping improve Jev Runtime. Contributions should preserve its core
contract: typed decisions must remain versioned, observable, and safe under
cancellation and rollout failures.

## Before opening a change

- Use GitHub Issues for bugs, proposals, and support questions.
- Use a private security advisory for vulnerabilities; see [SECURITY.md](SECURITY.md).
- Keep engine-specific behavior isolated in its adapter or plugin package.
- Do not include model weights, credentials, generated registries, or private logs.

## Development setup

Python contract tests do not require a GPU engine:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,tokenizers,tokenizer-conversion]' \
  -e packages/sglang -e packages/vllm -e packages/tokenspeed
pytest -q
ruff check src tests packages deployment benchmarks
ruff format --check src tests packages deployment benchmarks
```

For the TypeScript client:

```bash
cd packages/typescript
npm ci --ignore-scripts
npm test
```

## Pull requests

1. Explain the user-visible problem and the smallest proposed solution.
2. Add a regression test for behavior changes.
3. Run the relevant tests, lint, formatting, and package builds.
4. State exactly which engines, models, and GPU paths were actually tested.
5. Preserve failed-attempt denominators in benchmark or certification evidence.

GPU evidence is welcome but not required for documentation, isolated contract,
or engine-independent fixes. Do not claim production certification from CPU
doubles or a single smoke run.

Unless stated otherwise, contributions intentionally submitted to this project
are licensed under the Apache License 2.0.
