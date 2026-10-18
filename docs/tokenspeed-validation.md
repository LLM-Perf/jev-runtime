# TokenSpeed support validation — 2026-10-09

This change updates the TokenSpeed source profile to
`f4ac1affe11ad404720bcd150970487f75fbf59a`, retaining the legacy
`7fa8acb1e885389825c077a6aec0326fbbbd7116` profile. It fixes local model/tokenizer
binding, adds source/hardware/configuration preflight, submits directly to the
Engine's owner loop, and persists native completion receipts across restarts.
The strict functional harness now understands TokenSpeed's per-label accounting.

**No TokenSpeed model engine, native kernel or GPU serving run was executed.**
The available DSW exposes L20Z GPUs, outside the pinned upstream NVIDIA profile
(`sm90`, `sm100`, `sm103`, `sm107`). No GPU stack was installed and no existing
inference service was replaced. GPU functional coverage remains **0/20**.

| Check actually executed | Result | What it establishes |
|---|---|---|
| Full local Python suite | **618 passed**, 0 failed, 0 skipped | Core/plugin/HTTP contracts, using CPU engine doubles |
| Owner-loop dispatch regression | 64 concurrent coroutines with a 2-thread HTTP executor | Pending model results do not need a blocking executor thread per request |
| TokenSpeed HTTP contract harness with an explicit CPU double | Passed all four types, both readout modes, bridge drain and live switching | The shared strict runner handles TokenSpeed sequence/token accounting; no model accuracy claim |
| Receipt lifecycle/error injection | Passed | Restart/cache-eviction recovery of completed IDs; unknown/pending IDs fail closed; duplicate IDs and failed writes handled |
| Hardware/model/tokenizer/source preflight | Passed CPU tests; current source-only CLI passed against the pinned checkout | Rejects unsupported/incorrectly selected GPUs and mismatched identities/files |
| Current upstream Python readout bodies, Torch CPU primitives | **6/6**, max absolute error 0 | Pre-bias readout ordering for FP16/BF16/FP32, batch sizes 1 and 3 |
| Legacy upstream Python readout bodies, Torch CPU primitives | **6/6**, max absolute error 0 | The retained legacy source profile still passes the same ordering checks |
| Maintained-source Ruff lint and format | Passed | `src tests packages deployment benchmarks examples`; historical captured evidence is excluded |
| TokenSpeed wheel + sdist | Built; wheel source bytes and both JSON profiles verified | Packaging includes the actual adapter and its source guards |
| TokenSpeed 10,000 strict GPU requests + 1,000 switches | **Not run** | Ready-to-run command only; not counted as success |
| Native numerical accuracy, performance and soak | **Not run** | No GPU speedup, production readiness or task quality claim |

The readout checks ran on the DSW's existing Python environment with
Torch `2.13.0+cu129`, `CUDA_VISIBLE_DEVICES=""` and CPU primitive doubles. They
execute the inspected upstream `sample` and `_write_logprob_outputs` Python bodies;
they do **not** execute its Triton/CUDA kernels, scheduler or transport. The measured
zero error is relative to the CPU `log_softmax` reference in that harness only.

## Retained evidence

- [Validation summary and source hashes](../evidence/tokenspeed/20261009/validation.json)
- [Full test-suite JUnit report](../evidence/tokenspeed/20261009/pytest.xml)
- [Current profile readout check](../evidence/tokenspeed/20261009/readout-f4ac1affe11a.json)
- [Legacy profile readout check](../evidence/tokenspeed/20261009/readout-7fa8acb1e885.json)

`validation.json` records the pre-change base commit plus tested source-file hashes.
The readout reports bind their harness and source-profile SHA256 values. These
artifacts do not contain API/admin credentials or model weights.

## Reproduce local validation

```bash
python -m pip install -e '.[dev,tokenizers,tokenizer-conversion]' -e packages/tokenspeed \
  -e packages/sglang -e packages/vllm
pytest -q
ruff check src tests packages deployment benchmarks examples
ruff format --check src tests packages deployment benchmarks examples
python -m build --wheel --sdist packages/tokenspeed
```

With Torch available, run the source-ordering check separately against a clean
pinned checkout; use a new output filename for every attempt:

```bash
python tests/integration/verify_tokenspeed_readout.py \
  --source-root /absolute/path/to/tokenspeed/python/tokenspeed \
  --output /absolute/path/to/new-readout-report.json
```

For GPU validation, follow the [TokenSpeed quick start](quickstart-tokenspeed.md).
Its live runner retains per-request responses and requires both 10,000 strict
successes and 1,000 bundle switches. It separately marks native chat as inapplicable
because this launcher exposes typed/scoring APIs only. This does not weaken the
vLLM/SGLang native-chat checks.

## Remaining work

1. Build and run the exact pinned engine on compatible GPUs; verify the generated
   Qwen3-0.6B configuration, raw-score parity and the full strict traffic gate.
2. Implement upstream selected-ID output plumbing for joint readout. K labels
   still cost K native one-token requests; prefix caching is not a guarantee of
   single-prefill execution.
3. Measure durable-receipt overhead and native/plugin throughput under matched
   workload and cache conditions, then optimize based on those results.
4. Add scheduler-confirmed abort/crash recovery and a safe receipt-retention
   protocol. Unknown/pending IDs currently remain unresolved; disk rows do not expire.
5. Expand model, parallelism, precision and long-run qualification to broaden
   deployment coverage.
