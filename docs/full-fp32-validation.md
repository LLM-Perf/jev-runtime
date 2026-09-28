# Full-FP32 diagnostic validation

**Three of four attempted profiles pass all 576 development reference comparisons;
SGLang ordinary execution fails during startup.** This narrows the earlier
Qwen3-0.6B numerical investigation. It does not qualify FP32 as the production
default, repair the retained BF16 results, or establish performance or task quality.

Source `f2ccc2a` adds an explicit FP32 backbone option to the isolated launcher and
the independent GPU reference. Core and plugin Python sources are unchanged from
`8bbdff6`. A subsequent launcher guard at `3233a80` rejects the observed unsupported
SGLang combination before allocating a run directory or GPU process. The GPU
campaign predates that guard; the guard has local tests, not a new GPU campaign.

## Execution contract

The checkpoint is `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`. The original checkpoint is BF16; loading
it as FP32 widens its stored values and changes execution precision. It does not
recover original training precision. All profiles configure FP32 backbone and
readout, TP1/API1, eager execution, context 2048, maximum four running sequences,
chunked prefill 512, and the preserved fast tokenizer.

vLLM `0.30.0+cu129` uses its explicitly FP32-capable `FLEX_ATTENTION` backend and
Transformers `5.17.0`. SGLang `0.5.19` uses Triton attention and Transformers
`5.12.1`. Both environments use Torch `2.13.0+cu129`. Runs share an L20Z GPU with
existing services preserved. These are colocated functional/numerical diagnostics.
vLLM changes attention backend relative to the previous BF16 experiments, so the
comparison does not isolate dtype as the only intervention.

The same 32 synthetic English/Chinese joint-label cases run in reset-serial,
repeat-serial and reversed-order client-concurrency-four states. Three completed
profiles produce **288 scoring responses**. Separate native/plugin comparisons
add **96 pairs / 192 responses**. Each completed profile runs 32 GPU reference
forwards with eager attention and 32 with SDPA after the serving engine exits:
**192 reference forwards and 576 comparisons**. The fourth profile contributes
one startup failure and zero collected suite responses; it stays in the attempt
denominator.

Independent references use FP32 parameters, batch size one, `use_cache=False`,
`float32_matmul_precision=highest`, and CUDA matmul TF32 disabled. Every floating
parameter is checked, and Qwen's shared embedding/output Parameter identities,
storage and FP32 dtypes remain intact. The native Linear projection is unchanged;
each forward verifies FP32 input, weight, output and returned logits. Both named
reference vectors match exactly between the two vLLM modes. The SGLang ordinary
reference is unavailable because scoring never starts.

All successful profiles reject legacy/wrong-readout/wrong-backbone bundles, wrong
execution-mode bundles, and mismatched attached runtime declarations. All 96
native/plugin pairs have zero error. As in prior campaigns, these sequential
pairs do not independently control two identical flushed cache states. Engine
capability values identify configured dtype/mode, not every kernel's internal
accumulation precision.

## Recomputed numerical results

The development check is unchanged: maximum selected-logprob error `<= 0.15`
and first-argmax agreement, with near ties included. Probability errors below are
absolute conditional-probability differences. Each completed row has 96 comparisons.

| Engine/mode | Reference | Passed | Max logprob error | Max probability error | Argmax mismatches |
|---|---|---:|---:|---:|---:|
| vLLM ordinary | eager | 96/96 | 0.000177383 | 0.0000134501 | 0 |
| vLLM ordinary | SDPA | 96/96 | 0.000123501 | 0.0000140993 | 0 |
| vLLM invariant | eager | 96/96 | 0.0326867 | 0.00416232 | 0 |
| vLLM invariant | SDPA | 96/96 | 0.0326276 | 0.00415739 | 0 |
| SGLang ordinary | — | Startup failed | — | — | — |
| SGLang invariant | eager | 96/96 | 0.0470114 | 0.00614203 | 0 |
| SGLang invariant | SDPA | 96/96 | 0.0470743 | 0.00614441 | 0 |

| Engine/mode | Exact repeat vectors | Exact concurrent vectors | Largest repeat/concurrent logprob change | Argmax changes |
|---|---:|---:|---:|---:|
| vLLM ordinary | 1/32 | 7/32 | 0.0000343323 / 0.0000972748 | 0 / 0 |
| vLLM invariant | 32/32 | 32/32 | 0 / 0 | 0 / 0 |
| SGLang invariant | 32/32 | 32/32 | 0 / 0 | 0 / 0 |

Both enabled modes preserve these vectors exactly under the tested state
interventions. vLLM ordinary execution has small variations but is closer to
these independent references than its invariant mode. Reproducibility and
reference agreement are separate properties. The data do not identify which
internal kernel causes the remaining differences.

All 33 cache resets per completed profile return 200; cold responses report zero
cached tokens. Input and label IDs match across completed profiles and the prior
BF16-backbone/FP32-head campaign. These counters and client submission states do
not establish realized GPU batch sizes or production throughput. The verifier
also records P50/P95 errors using linear interpolation at `(n-1)*q`; these are
descriptive statistics over correlated synthetic cases.

## Startup failure and retained registry

SGLang ordinary execution reaches its preparation canary but fails in
`layers/layernorm.py:forward_cuda` through `sgl_kernel.rmsnorm`:

```text
failed to dispatch data type Float
```

Its API/native process group exits. The campaign captures the failed readiness
attempt, engine traceback, process identity and successful process cleanup. It
does not turn the failure into a passing ordinary-mode result. The launcher now
requires explicit `--batch-invariant` for this pinned native SGLang FP32
diagnostic; it never silently enables the option and leaves gateway attachment
semantics unchanged.

The failed startup leaves one lease, one `lease_work` journal and one `PREPARING`
bundle in its isolated registry. The owner is verified dead. The first export
audit consequently fails its all-registries-drained assertion; its unchanged
script and a captured reproduction of that failure are retained. The final audit
records `passed=false`, `registries_drained=false` and
`owned_process_exit_passed=true` separately.

A consistent private SQLite snapshot preserves the failed registry, including
its integrity/content digest and blockers. Snapshotting leaves the source logical
content unchanged. No lease deletion, forced state rewrite or recovery operation
is performed. The runtime recovery path requires confirming cancellation against
the original engine target; this unsupported startup profile cannot currently
provide that confirmation. Recovery of this preparation-stage crash remains open.
The snapshot is evidence, not a restore-ready artifact: its blockers must not be
ignored or bypassed to activate it.

In the subsequent [recovery-mode exercise at `95de3d5`](recovery-mode-validation.md),
the retained source registry is recovered through the administrative cancellation
path, then serves a new invariant-mode bundle. This campaign's failed audit,
ordinary-mode startup failure and original snapshot remain unchanged. Recovery
of that registry does not make ordinary FP32 execution supported.

## Evidence and acceptance boundaries

The final audit verifies **269 retained owned process identities/groups are
terminal**, GPU7 returns to 11,990 MiB free, seven pinned checkpoint files match
Hub digests, and both tokenizer profiles verify. Completed-profile registries
drain; the single failed-profile registry remains as described above. Captured
reports, logs and snapshot are scanned for the campaign's credential values.
The offline verifier checks exported hashes, source identity, failure inclusion,
snapshot rows and independently recomputes all numerical metrics.

Local checks pass **442 tests at `f2ccc2a`** and **443 after the launcher guard at
`3233a80`**. Packaged core/plugin code and TypeScript are unchanged; no new wheel
installation or TypeScript run is claimed. DSW uses the recorded source checkout.

Distinct functional coverage remains **24/40**, and earlier BF16 numerical
failures remain in their original matrix entries. `full_release_gate_passed`
stays false. This development check supplies no held-out task labels, precision/
quantization/TP certification, controlled performance matrix, 24-hour soak,
deployment-image qualification or complete fault-recovery acceptance. Further
work must validate an explicit per-profile numerical budget on independent data,
measure the performance cost, and close preparation-crash recovery before making
a production recommendation.

- [Recomputed metrics](../evidence/dsw/full-fp32-f2ccc2a/verified-summary.json)
- [Campaign, including failed attempt](../evidence/dsw/full-fp32-f2ccc2a/campaign.json)
- [Audit with unresolved registry](../evidence/dsw/full-fp32-f2ccc2a/audit.json)
- [Failed registry snapshot manifest](../evidence/dsw/full-fp32-f2ccc2a/failed-registry-snapshot/manifest.json)
- [Offline verifier](../evidence/harnesses/verify_full_fp32_f2ccc2a.py)
- [Original local checks](../evidence/local-check-f2ccc2a.json)
- [Launcher-guard local checks](../evidence/local-check-3233a80.json)
