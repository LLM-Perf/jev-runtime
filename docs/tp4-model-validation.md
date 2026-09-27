# Qwen3-8B and OLMo-2 TP4 development checks

On 2026-09-28 (Asia/Shanghai), the two pinned public checkpoints passed native
functional validation on SGLang 0.5.19 and vLLM 0.30.0+cu129. The fixed matrix now
has **14/40 functional combinations, 7/20 per engine**. The requirement remains
18/20 per engine; numerical, quality, performance and other release gates remain
open. OLMo-2 adds the `Olmo2ForCausalLM` architecture to real GPU coverage.

Runtime/plugin source was `f6bda1a13e30b1d58cb4303455cce92c1601a683` and the
reusable validation harness was `b8742bd0d96dc98de7bf9a7860de407508043c39`.
No core or engine dependency upgrade was needed for either checkpoint.

| Model | Pinned revision | Download verification | Indexed weight bytes |
|---|---|---:|---:|
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | 14 files / 5 weight shards | 16,381,470,720 |
| allenai/OLMo-2-1124-7B-Instruct | `470b1fba1ae01581f270116362ee4aa1b97f4c84` | 12 files / 3 weight shards | 14,597,234,688 |

Both Hub revisions were public, ungated and declared Apache-2.0 at download time.
`hf cache verify` passed. The retained model manifests contain every downloaded
root file's size/SHA256 and verify that every indexed shard exists. Weights were
not added to Git and this is not in-memory weight attestation.

## Tested profile and results

All four runs used BF16 backbone and readout, TP4, one API worker, eager execution,
a 2,048-token context, four maximum running sequences and 512-token prefill chunks.
The GPU mapping was physical indices 3,4,5,6 on four L20Z devices, driver 550.127.08.
vLLM used memory fraction 0.065 of total device memory; SGLang used 0.65 of available
memory with 2,048 maximum total tokens. The launcher checked a 3,072 MiB reserve
on every selected device. Existing services stayed running, so none of these
checks is a controlled performance measurement.

| Model | Engine | Switches | Strict traffic successes during switches | Max CPU logprob error | Single-position result |
|---|---|---:|---:|---:|---|
| Qwen3-8B | sglang | 1,000 | 206 | 0.125000000 | Pass |
| Qwen3-8B | vllm | 1,000 | 362 | 0.250000000 | Fail |
| OLMo-2-7B-Instruct | sglang | 1,000 | 154 | 0.061897755 | Pass |
| OLMo-2-7B-Instruct | vllm | 1,000 | 215 | 0.062047958 | Pass |

Every run passed Choice/Boolean/Score/Rank contracts, default abstention and
explicit first-tie policy, independent candidate scoring, native chat coexistence,
auth separation, invalid-bundle rejection, generation conflicts and
activate/disable/drain/retire checks. Mixed-version responses and native/attached
logprob differences were zero. Legacy/unbound and mismatched-readout bundles were
rejected, and an independently attached runtime rejected a wrong precision
configuration before serving. Final leases/admission counters were zero and all
active bundles passed health checks.

The CPU reference uses each run's actual input IDs and label IDs, the pinned
checkpoint, BF16 eager Transformers and FP32 log-softmax. The unchanged development
tolerance is absolute logprob error 0.15. Qwen3-8B/vLLM failed at **0.25**; it remains
failed in the matrix and the runner exited nonzero after successful cleanup. The
other three positions pass, and all four top labels agree with the reference.
A single position with top-label agreement is not a full numerical or business
accuracy certificate. No tolerance was relaxed and no failed result was dropped.

For each model the two engines' saved input IDs and label IDs are identical.
Their maximum cross-engine differences at those positions are 0.25 for Qwen3-8B
and 0.123945713 for OLMo-2. These are separate-run diagnostics; they do not prove
cache-state invariance or establish the source of a numerical difference.

## Evidence and cleanup

The [validation index](../evidence/dsw/tp4-validation-b8742bd.json) binds 41 raw
JSON files by SHA256. The uploaded harness checkout had 101 files verified against
its immutable commit. Reports retain the installed runtime/plugin import paths,
engine/Torch/Transformers versions, model revisions and live topology. vLLM had
four distinct live rank-labelled workers in the recorded process group; SGLang's
live server configuration reported TP4/PP1/DP1. Both matched the parent CUDA mapping.
No direct mapping between container worker PIDs and host NVML PIDs is claimed.

All four owned process groups exited after validation. GPU 3/4/5/6 UUIDs matched
the preflight and each returned to the observed 10,792 MiB free baseline. A final scan of 70 saved process records
found no matching owned process still alive. The
runner requires a new attempt directory, fails early on process exit and performs
cleanup after numerical or harness failure. See the
[runner instructions](native-validation-runner.md). It does not automatically
kill unidentified orphan groups or restart after an observation timeout.

Local checks at harness source passed 261 Python tests, Ruff and formatting.
Core/plugin/SDK/launcher sources are unchanged from `f6bda1a`, so its verified
three-wheel evidence is explicitly reused; no new wheel-build claim is made.
TypeScript is unchanged; its earlier 62 passing tests were not rerun. Local, DSW
and hosted-CI evidence remain separate.

The remaining 26 model-engine combinations, full numerical budgets, business
quality/calibration, controlled 144-case performance matrix, 24-hour soak,
compatible-replica failover and broader LoRA/fault profiles are not certified by
these runs. The 20-model denominator and all original acceptance requirements
remain unchanged.
