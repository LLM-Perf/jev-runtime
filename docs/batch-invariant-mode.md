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
