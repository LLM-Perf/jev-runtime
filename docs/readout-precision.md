# Readout precision and bundle identity

The backbone dtype and output-layer dtype are separate parts of a serving
profile. BF16 weights do not imply a BF16 output projection: both tested engines
have official FP32 output-layer paths. Changing that computation can change
selected logprobs and invalidate a previously fitted calibrator.

At runtime source `f6bda1a13e30b1d58cb4303455cce92c1601a683`, newly built bundles
bind `model.readout_dtype`. The field participates in both the bundle digest and
the scoring-contract digest used by calibration artifacts. A calibrator fitted
for a BF16 readout cannot be attached to a different FP32 readout contract.

## Configuration and verification

The default remains the configured model dtype. For a BF16 backbone with an
FP32 output projection, set:

```yaml
dtype: bfloat16
readout_dtype: float32
```

Configure the host engine to match. The tested versions expose:

```sh
# vLLM 0.30.0
--hf-overrides '{"head_dtype": "float32"}'

# SGLang 0.5.19
--enable-fp32-lm-head
```

The task-owned DSW launcher accepts `--readout-dtype float32` and writes both
the Jev setting and native engine flag. Omitting it keeps the default readout.
For a gateway, the setting declares the required remote profile; it does not
change the existing remote engine.

Native adapters report the model and output-layer dtype from engine model
configuration. SGLang's HTTP adapter reads explicit startup configuration;
`auto` is not guessed. Unknown or mismatched reported precision prevents a
configured runtime from reaching serving with `engine_precision_mismatch`.
This checks engine-reported configuration, not an in-memory weights/kernel
attestation. A quantized readout is not inferred from a backbone dtype or an
FP32 flag; such profiles need separate implementation and validation.

Bundle preparation also checks readout precision before its scoring canary.
An uploaded legacy or wrong-precision bundle remains inspectable, but preparation
fails with `model_mismatch`. Existing periodic capability checks withdraw
readiness if the backend's reported configuration subsequently changes.

FP32 readout is opt-in. Its latency, throughput and quality are not certified.
The managed-LoRA gate remains the frozen BF16 backbone/readout, TP1/API1 profile;
enabling FP32 does not extend that gate. Crash-restart harnesses also reject
profiles outside their frozen readout and topology.

## Migration from earlier development snapshots

Stored manifests without `readout_dtype` keep their original serialization and
digest. They are not silently assigned a new precision or rewritten in place.
They must be rebuilt as new immutable versions before serving in a configured
runtime with precision binding. Previously active legacy versions can therefore
cause startup validation to fail after an in-place package upgrade.

For the initial migration, keep the old instance/registry available and launch
a new task-owned instance with a fresh registry, the intended precision and a
separate endpoint. Obtain `/admin/profile` from that serving worker and use
`jevctl bundle build-remote` to create new bundle versions. Copy the intended
task/policy, validate and prepare the new versions, and refit calibration for
the new scoring contract. Validate traffic before moving routing. Keep the old
instance available for rollback until its requests drain. Do not edit an old
registry manifest or copy its calibration digest to make the new version pass.

An offline build derives the default readout from `Settings.dtype` or the
explicit `Settings.readout_dtype`; its tokenizer implementation must still
match the real serving worker. Remote building avoids that separate mismatch.
Native plugin code upgrades still follow the host engine's deployment lifecycle;
this change concerns bundle identity and does not implement live Python code
replacement or compatible-replica failover.

## Numerical reference and evidence boundaries

`tests/integration/reference_logits.py` now checks the saved model's readout
identity. Use `--readout-dtype float32` for an FP32 profile. The CPU reference
keeps BF16 model parameters/hidden computation and recomputes only the Linear
output projection in FP32 at the module boundary. It does not cast a tied input
embedding to FP32. The report records the observed projection dtype, and existing
output files cannot be overwritten.

`tests/integration/live_precision_binding.py` checks legacy/wrong bundle
rejection against a real server and verifies that a new HTTP-attached runtime
with a mismatched declaration fails startup. It leaves the original engine's
precision unchanged and records both runtime and harness source revisions.

A single saved answer position is targeted numerical evidence. It does not
replace a multi-example budget frozen by dtype/quantization/TP, tie and argmax
analysis, cache/batch-state checks, task quality, or controlled performance.
The earlier BF16 failures in [the Phi TP2 report](phi-tp2-validation.md) remain
failures. A passing FP32 profile cannot overwrite that column or increase the
count of distinct model/engine combinations.

For memory-bounded CUDA references and the later Qwen/R1 attention/device
diagnosis, see [numerical reference execution profiles](numerical-reference-diagnostics.md).
The original CPU eager default and its failure records are retained.


## DSW validation, 2026-09-28 (Asia/Shanghai)

Runtime source was `f6bda1a`; the independent precision-binding harness was
`5aa59d6`. The tested engines were SGLang 0.5.19 and vLLM 0.30.0+cu129 with
Torch 2.13.0+cu129 on L20Z GPUs/driver 550.127.08. All runs used eager execution,
one API worker, a 2,048-token context and four maximum running sequences.
Existing services remained running; the tests were colocated development checks.

| Model | Engine | Backbone/readout | TP | Switches | Strict traffic successes | Single-position CPU max logprob error |
|---|---|---|---:|---:|---:|---:|
| Phi-4 mini | sglang | BF16/float32 | 2 | 1,000 | 182 | 0.049654961 |
| SmolLM2-1.7B | sglang | BF16/bfloat16 | 1 | 1,000 | 493 | 0.000161231 |
| Phi-4 mini | vllm | BF16/float32 | 2 | 1,000 | 318 | 0.089940071 |
| SmolLM2-1.7B | vllm | BF16/bfloat16 | 1 | 1,000 | 441 | 0.103232741 |

All four native functional suites passed, including four result types,
independent candidates, native chat, native/attach parity, authentication and
invalid-bundle handling, generation conflicts, disable/drain/retire and hot
switches. Native/attach maximum logprob differences and mixed-version counts
were zero. Each run additionally rejected both an unbound legacy bundle and a
wrong-readout bundle with HTTP 409, and a separately constructed attached
runtime rejected a mismatched precision declaration before serving.

The four CPU comparisons passed the targeted `atol=0.15` check with top-label
agreement. That development threshold is not a completed per-profile numerical
budget or a broad accuracy certification. Phi-4's FP32 comparisons use a different
engine **and reference** readout profile from its earlier BF16 comparisons;
these are not passing retests of the old BF16 profile. The earlier failures remain
in the matrix. Distinct functional coverage remains 10/40, or 5/20 per engine.

Final leases/admission counters were zero. All four owned process groups exited.
GPU 5/6 returned to 10,792 MiB free and GPU 7 to 11,990 MiB. The report checks
all selected device UUIDs; the TP2 runs also retain live topology evidence.
The original BF16 Phi-4 bundle digest was reconstructed from both retained
`737d814` reports and remained unchanged under the new serializer.

The [validation index](../evidence/dsw/precision-validation-f6bda1a.json) binds
40 evidence files by SHA256. The immutable runtime upload had 66 verified files;
the separate live harness file was also hash-verified. Local checks at runtime
source passed 257 Python tests, lint/format and all three wheel builds, whose
packaged Python files were compared with source. TypeScript was unchanged and
its prior 62 tests were not rerun. The first local test attempt exposed an old
unbound test fixture; the fixture was updated to use the configured identity.
The first remote editable-install attempt lacked a non-isolated Hatchling build
backend; using pip's isolated build backend succeeded without upgrading engine
dependencies. Both corrections are recorded separately from GPU outcomes.

Controlled performance, business quality, a full multi-example numerical suite,
TP/configuration-wide calibration validity, LoRA recertification at this source,
remaining model coverage and the 24-hour soak remain outstanding.
