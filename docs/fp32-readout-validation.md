# Multi-input FP32 readout validation

**FP32 output projection alone does not resolve the retained Qwen3-0.6B numerical
failures.** With batch-invariant execution enabled, both engines still preserve
all 32 selected-logprob vectors exactly across the tested cache/concurrency
interventions. Independent-reference checks continue to fail. These are separate
findings, and neither establishes business accuracy or a performance gain.

Harness commit `55a1d26` extends the multi-input reference to the existing FP32
readout configuration. Product, plugin and deployment Python code is unchanged
from `0443dc5`; this is a validation extension, not a new engine kernel or default
precision change. The original [BF16-head A/B](batch-invariant-mode.md) remains
unchanged and its input/label IDs match this campaign for every case.

## Execution contract

The checkpoint is `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`. All four serving profiles use BF16
backbone, FP32 readout, TP1/API1, eager execution, context limit 2048, maximum four
running sequences and the preserved fast tokenizer. Each engine runs once with
ordinary execution and once with its explicit batch-invariance option. vLLM is
`0.30.0+cu129` with Transformers `5.17.0` and the logged `FLASH_ATTN` backend;
SGLang is `0.5.19` with Transformers
`5.12.1` and Triton attention. Torch is `2.13.0+cu129`. Runs use the same L20Z GPU
with existing services preserved; these are co-located development checks.

The 32 English/Chinese synthetic inputs run in reset-serial, immediate-repeat and
reversed-order client-concurrency-four states: **384 scoring responses**. A
separate 32 plugin/native pairs per profile add **256 responses**. After each
engine exits, two named GPU reference executions (eager and SDPA) perform 32
forwards each: **256 reference forwards and 768 comparisons** overall.

References keep resident BF16 model parameters, batch size one and `use_cache=False`.
At the Linear output module boundary, they recompute the projection from FP32
casts of activations and weights, then use FP32 log-softmax. They do not cast the
shared Parameter itself. Every forward records BF16 input and weight dtype and
FP32 output dtype; returned logits are also checked. Qwen's input/output weight
sharing, parameter objects/storage and BF16 dtypes remain intact. Both attention
reference vectors are exactly equal between modes within each engine environment.
Engine-specific determinism overrides are not injected into Transformers.

Native capability values establish configured precision/mode, not a GPU-kernel
attestation. In all four profiles, legacy and wrong-readout bundles fail with
`409 model_mismatch`; mismatched attached runtimes fail with
`engine_precision_mismatch`. Wrong-mode bundles/runtimes also fail. All 128
sequential plugin/native pairs have zero error. As before, these pairs do not
independently control two identical flushed cache states.

## Numerical results

Every reference profile fails part of the unchanged development check: maximum
selected-logprob error `<= 0.15` and first-argmax agreement. Near ties remain in
the denominator. Each row has 96 comparisons; probability errors are absolute
probability-point differences.

| Engine/mode | Reference | Passed | Max logprob error | Max probability error | Argmax mismatches |
|---|---|---:|---:|---:|---:|
| vLLM ordinary | eager | 18/96 | 0.898644 | 0.182363 | 7 |
| vLLM ordinary | SDPA | 32/96 | 0.686516 | 0.085798 | 7 |
| vLLM enabled | eager | 18/96 | 1.021853 | 0.140703 | 9 |
| vLLM enabled | SDPA | 27/96 | 0.554769 | 0.059031 | 6 |
| SGLang ordinary | eager | 20/96 | 0.814900 | 0.168379 | 8 |
| SGLang ordinary | SDPA | 28/96 | 0.574263 | 0.082695 | 9 |
| SGLang enabled | eager | 21/96 | 1.086738 | 0.122392 | 6 |
| SGLang enabled | SDPA | 27/96 | 0.500246 | 0.103704 | 3 |

The verifier additionally retains P50/P95 of per-comparison maximum logprob error,
P95 conditional-probability error and near-tie/non-near-tie argmax counts. Quantiles
use linear interpolation at position `(n-1)*q` in the sorted 96 observations.
This is a descriptive distribution over correlated synthetic cases/states, not
a confidence interval or held-out generalization estimate.

Both ordinary profiles have one eager-reference argmax mismatch outside the
declared near-tie range (reference top-two logprob gap `<= 0.3`): `text2-choice4`
in vLLM's concurrent state and SGLang's repeated state. The reference gap is
`0.6668853759765625`. All other argmax mismatches in this campaign are within
the declared near-tie range. The corpus has no business labels, so this does not
measure task error; it does prevent attributing every discrepancy to near ties.

Within-engine state comparisons use the reset-serial vector as baseline:

| Engine/mode | Exact repeat vectors | Exact concurrent vectors | Largest repeat/concurrent logprob change | Repeat/concurrent argmax changes |
|---|---:|---:|---:|---:|
| vLLM ordinary | 8/32 | 1/32 | 0.202833 / 0.423023 | 1 / 4 |
| vLLM enabled | 32/32 | 32/32 | 0 / 0 | 0 / 0 |
| SGLang ordinary | 0/32 | 4/32 | 0.240951 / 0.384167 | 2 / 2 |
| SGLang enabled | 32/32 | 32/32 | 0 / 0 | 0 / 0 |

All 33 cache resets per profile succeed; cold responses report zero cached tokens.
Collection takes 2.76–5.89 seconds per profile. These interventions and counters
do not establish realized GPU batch sizes, throughput, or latency SLOs.

Changing the readout to FP32 changes both the serving and reference contract.
Errors do not uniformly decrease relative to the BF16-head campaign, so the
result cannot be advertised as a numerical fix. Isolating remaining backbone and
attention-path differences needs further controlled evidence. A per-profile
budget still requires a declared reference, independent held-out inputs, explicit
near-tie treatment and task-quality validation; widening a tolerance on this
same corpus would not establish those requirements.

## Evidence and remaining gates

All four attempts complete without orchestration retries. Their registries drain
leases and admission counters. The audit verifies **241 retained owned process
identities/groups are terminal**, GPU7 returns to 11,990 MiB free, seven pinned
checkpoint files match Hub digests and both tokenizer profiles verify. Exported
reports/logs were checked for this campaign's credential values. The independent
offline verifier checks all artifact/source hashes and recomputes raw metrics.

Local **432 tests** pass, including 16 numerical-suite tests, and lint/format
checks pass. No product code changed, so wheel builds and TypeScript tests were
not repeated. DSW executes the recorded source checkout. Overall model functional
coverage remains **24/40**, numerical status remains failed and
`full_release_gate_passed` remains false. Quality, performance, broader model/TP/
precision coverage, deployment and soak gates remain open.

- [Recomputed metrics](../evidence/dsw/fp32-readout-55a1d26/verified-summary.json)
- [Campaign](../evidence/dsw/fp32-readout-55a1d26/campaign.json)
- [Audit](../evidence/dsw/fp32-readout-55a1d26/audit.json)
- [Offline verifier](../evidence/harnesses/verify_fp32_readout_55a1d26.py)
- [Local checks](../evidence/local-check-55a1d26.json)
