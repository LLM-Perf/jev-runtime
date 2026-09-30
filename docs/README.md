# Documentation index

Entry points by goal. Every validation document records real evidence for its
checkpoint commit; a passing document is scoped to what it measured and does
not substitute for the remaining release gates in the
[completion ledger](completion-ledger.md).

## Start here

1. [Implementation plan](implementation-plan.md)（中文）— scope, architecture, acceptance criteria.
2. [API reference](api-reference.md) — endpoints, authentication and the error code table.
3. [Engine integration](engine-integration.md) — standing up SGLang/vLLM with the runtime.
4. [Operations](operations.md) — running the gateway day to day.
5. [Completion ledger](completion-ledger.md) — current checkpoint, evidence and open gates.
6. [Multi-engine status](multi-engine.md)（中文）— per-framework support and remaining gaps.

## Decision semantics and quality

- [Calibration](calibration.md) — fitting, metrics and immutable artifact binding.
- [Public quality checks](public-quality.md) — AG News/SST-2 runners and denominators.
- [Readout precision and bundle migration](readout-precision.md).
- [Batch-invariance mode and A/B results](batch-invariant-mode.md).
- [FP32 readout validation](fp32-readout-validation.md) and [full-FP32 diagnostics](full-fp32-validation.md).
- [Numerical reference diagnostics](numerical-reference-diagnostics.md) and [multi-input numerical suite](numerical-suite.md).

## Bundle lifecycle and rollout

- [Gateway traffic switching and drain](gateway-rollout.md) + [validation](gateway-rollout-validation.md).
- [Backend quiescence](quiescence.md) + [validation](quiescence-validation.md).
- [Combined stable-route reservation](combined-reservation.md) + [validation](combined-reservation-validation.md).
- [Typed response serialization](response-serialization.md) + [validation](response-serialization-validation.md).
- [Native request cancellation across rollback](native-rollout-cancellation.md).
- [Fault recovery](fault-recovery.md), [recovery-only startup](recovery-mode.md) + [validation](recovery-mode-validation.md).

## Registry

- [Registry schema and offline upgrade/rollback](registry-schema.md) + [migration validation](registry-migration-validation.md).
- [Registry snapshots and guarded restore](registry-snapshots.md).

## Engines and models

- [Reproducible native validation](native-validation-runner.md).
- [Managed LoRA](adapters.md) and [LoRA crash recovery](lora-crash-recovery.md).
- [Explicit chat templates](explicit-chat-templates.md), [tokenizer profiles](tokenizer-profiles.md),
  [fast tokenizer preservation](preserve-fast-tokenizers.md) + [validation](preserve-fast-validation.md).
- Model checkpoints: [Mistral](mistral-validation.md), [Qwen2.5/R1 Llama](public-model-pair-validation.md),
  [GLM4/R1 Qwen](glm-r1-model-validation.md), [TP4](tp4-model-validation.md), [Phi TP2](phi-tp2-validation.md).
- [Multi-engine validation round](multi-engine-validation.md)（中文）.

## Packaging and deployment

- [Immutable packages and offline installation](release-packaging.md) + [rollout validation](release-rollout-validation.md).
- [Offline gateway image preparation](container-images.md).
- [Serving health and readiness](serving-health.md).

## Performance

- [Performance experiments](performance.md).
- [Atomic admission comparison](atomic-admission-performance.md).

## Tracking files

- [backlog.json](backlog.json) — milestone/task ledger with acceptance criteria.
- [research-evidence-index.json](research-evidence-index.json) — pinned upstream sources and hashes.

Documents named `{topic}.md` describe design; `{topic}-validation.md` record the
corresponding executed evidence. A few older pairs predate this convention
(e.g. `registry-schema.md` ↔ `registry-migration-validation.md`).
