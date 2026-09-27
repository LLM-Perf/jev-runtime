# Explicit batch-invariance profile

Set `batch_invariant: true` in the gateway/plugin configuration only when the
native engine has its matching execution mode enabled. The isolated DSW launcher
accepts `--batch-invariant`: it sets `VLLM_BATCH_INVARIANT=1` for vLLM or adds
`--enable-deterministic-inference` for SGLang. It records the mode in both config
and process manifest. Ordinary vLLM launches explicitly set the environment
variable to `0` instead of inheriting an accidental shell override.

The mode is bound to `ModelIdentity`, immutable bundle digest and calibration
contract. Runtime startup rejects an enabled engine paired with ordinary config,
or an enabled config whose engine does not report `true`. Bundle preparation
rejects a mode mismatch before scoring its canary. Existing ordinary-mode bundles
retain their historical serialization/digests; enabling the mode requires a newly
built bundle and new calibration. This is not an in-process runtime toggle.

The native vLLM plugin reports the host's `vllm.envs.VLLM_BATCH_INVARIANT` value;
native SGLang reports its resolved server argument. HTTP attachment transports
those reports (vLLM scoring capabilities or SGLang server info). Missing values
remain unknown. Historical ordinary configurations with unreported mode retain
their previous behavior; an explicitly enabled profile requires affirmative
reporting. These are configuration observations, not kernel or numerical
attestation. `Capabilities.verified` is not set to true by this flag.

Managed LoRA rejects the new mode because the frozen LoRA profile was tested only
with ordinary execution. Backend version, attention kernel, TP, cache behavior,
precision and hardware still require individual certification. This field binds
one significant execution choice; it does not replace a complete numerical
profile or solve all reference discrepancies.

Relevant upstream sources:

- [vLLM batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)
- [vLLM v0.30.0 batch-invariant implementation](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/determinism/batch_invariant.py)
- [SGLang v0.5.19 deterministic argument handling](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/arg_groups/attention_hook.py)

The installed versions and source are authoritative for a test. For example,
vLLM's installed environment-variable comment mentions SM90 while its current
implementation has an SM80-family path and Ada tuning; a documentation/comment
alone does not prove that a chosen model/hardware combination works.

Performance and numerical behavior must be measured separately. Upstream warns
that this mode can affect performance. The existing development tolerance remains
unchanged, and results must still be compared with the independently named
reference execution contract. Stable outputs can consistently disagree with a
reference. Neither stability nor this flag establishes task accuracy.

Validation tools:

- `tests.integration.numerical_suite` retains the same frozen 32 cases and three
  execution states used before this change.
- `tests.integration.live_execution_mode` checks a mismatched bundle and attached
  runtime, then compares 32 saved inputs through plugin and native scoring APIs.
- CPU tests cover mode binding, old calibration rejection, unknown/mismatched
  mode rejection, host-mode propagation, unchanged ordinary serialization and
  the managed-LoRA boundary.

## DSW A/B at `f9167c9`

Four profiles complete the same 32 synthetic inputs in three states, totaling
384 scoring responses. A separate 32-pair plugin/native check per profile adds
256 responses. Eight independent reference runs perform 256 forwards and yield
768 comparisons. Every profile uses identical input IDs and ordered label IDs.
The reference vectors are exactly equal between ordinary and enabled modes
within each engine environment and reference attention implementation.

The profile is pinned `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`, BF16 backbone/head, TP1/API1,
eager serving, context 2048, maximum four running sequences, preserved fast
tokenizer and one L20Z GPU. vLLM is `0.30.0+cu129` with Transformers `5.17.0`;
SGLang is `0.5.19` with Transformers `5.12.1` and Triton attention. Torch is
`2.13.0+cu129`. The GPU is co-located with existing services: this is a numerical
and lifecycle experiment, not an isolated performance measurement.

Each state comparison below is against that profile's serial result after a
confirmed cache reset. "Exact" means equality of every selected logprob, before
calibration or decision policy. Concurrency is four client submissions, not a
claim about realized GPU batch sizes.

| Engine/mode | Exact repeat vectors | Exact concurrent vectors | Largest repeat/concurrent logprob change | Repeat/concurrent argmax changes |
|---|---:|---:|---:|---:|
| vLLM ordinary | 8/32 | 1/32 | 0.249298 / 0.623706 | 1 / 4 |
| vLLM enabled | 32/32 | 32/32 | 0 / 0 | 0 / 0 |
| SGLang ordinary | 0/32 | 4/32 | 0.374370 / 0.485589 | 1 / 2 |
| SGLang enabled | 32/32 | 32/32 | 0 / 0 | 0 / 0 |

All 128 plugin/native pairs have zero error. These are sequential paired requests
using the same saved IDs, not separately flushed and independently controlled
cache states. They corroborate the plugin's selected values for this workload;
they do not certify every parser path or isolate a particular kernel.

Both enabled profiles become stable under these interventions, but neither
resolves independent-reference discrepancies. The development check remains
maximum selected-logprob error `<= 0.15` **and** first-argmax agreement. Each row
below has 96 comparisons, including near ties:

| Engine/mode | Reference | Passed | Largest logprob error | Largest conditional-probability error | Argmax mismatches |
|---|---|---:|---:|---:|---:|
| vLLM ordinary | eager | 22/96 | 0.998153 | 0.182498 | 7 |
| vLLM ordinary | SDPA | 33/96 | 0.744257 | 0.091741 | 7 |
| vLLM enabled | eager | 15/96 | 0.998384 | 0.121216 | 9 |
| vLLM enabled | SDPA | 33/96 | 0.744939 | 0.075711 | 6 |
| SGLang ordinary | eager | 25/96 | 0.756199 | 0.182076 | 7 |
| SGLang ordinary | SDPA | 31/96 | 0.707006 | 0.086882 | 8 |
| SGLang enabled | eager | 27/96 | 0.987908 | 0.113128 | 6 |
| SGLang enabled | SDPA | 21/96 | 0.502005 | 0.091280 | 3 |

References run resident on the same GPU after engine shutdown, batch one, no KV
cache, BF16 head and FP32 log-softmax. They use ordinary Transformers/PyTorch
execution; engine-specific invariance overrides are not injected into them.
Probability errors are absolute probability-point differences. These synthetic
inputs lack business labels, so argmax disagreement is not measured task error.

Every profile rejects a mismatched bundle with `409 model_mismatch` and a
mismatched attached runtime with `engine_execution_mismatch`. The enabled mode
is explicitly present in engine capabilities and model identity; ordinary model
identities retain their prior serialization. Final registries have no outstanding
admission or lease records.

The initial enabled SGLang attempt passed readiness but failed its first immediate
cache reset with HTTP 400 and collected zero cases. Its zero queue/running counts
did not establish `is_fully_idle()`: the scheduler also checks internal batch,
result and chunked-request state. The failed attempt is retained. Harness commit
`5e6b2c7` uses SGLang's native `GET /flush_cache?timeout=10` deferred idle barrier;
all 33 resets in the separate retest return 200. No forced reset, relaxed
comparison or overwritten run was used. Product/plugin/deployment Python files
are byte-identical between `f9167c9` and `5e6b2c7`.

The audit verifies all **205 retained owned process identities/groups are
terminal**, GPU7 is back to 11,990 MiB free, seven model files match pinned Hub
digests and both tokenizer profiles verify. Exported artifacts were checked
against this campaign's credential values. The offline verifier checks artifact
hashes and source commits, then independently recomputes all comparison and state
metrics. Local 430 tests, lint/format and three source-matched wheel builds pass;
DSW uses copied source through `PYTHONPATH`, not an installation of these wheels.
The collector-only follow-up also passes its 14 targeted tests.

Numerical status remains failed, functional coverage stays **24/40**, and
`full_release_gate_passed` remains false. The enabled profile still needs held-out
numerical/quality validation, declared precision/TP/attention coverage and measured
performance cost. Managed LoRA remains excluded. This result establishes one
useful execution option, not general batch invariance across supported models.

- [Recomputed results](../evidence/dsw/batch-invariant-f9167c9/verified-summary.json)
- [Initial campaign and retained reset failure](../evidence/dsw/batch-invariant-f9167c9/campaign.json)
- [SGLang bounded-idle retest](../evidence/dsw/batch-invariant-f9167c9/retest-campaign.json)
- [Audit](../evidence/dsw/batch-invariant-f9167c9/audit.json)
- [Offline verifier](../evidence/harnesses/verify_batch_invariant_f9167c9.py)
- [Local checks and wheel manifest](../evidence/package-check-f9167c9.json)
