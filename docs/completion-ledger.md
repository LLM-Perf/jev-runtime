# Completion ledger

The objective remains the complete implementation plan, real DSW validation, and
delivery in the private `LLM-Perf/jev-runtime` repository. This ledger is not a
replacement for any requirement in `implementation-plan.md`.

## Current implementation checkpoint

- Core schemas, task compilation, joint-label and independent-candidate readout.
- Immutable bundles, SQLite CAS routing, persistent leases, prepare/activate/drain/retire.
- Explicit cancellation and retention of leases when abort cannot be confirmed.
- Gateway API and CLI; restricted System One compatibility layer.
- SGLang HTTP/native adapters and a native vLLM selected-logprob adapter.
- Separate SGLang/vLLM plugin distributions; vLLM real startup and serving checked.
- CPU contract tests with a controlled engine double. These are not model evaluations.
- Temperature/Platt fitting and collection CLI with grouped split checks, NLL/Brier/ECE/risk-coverage metrics,
  immutable artifact binding, and synchronous/asynchronous Python clients.
- Local checkpoint: 65 Python tests and Ruff passed; TypeScript SDK build and
  four client tests passed. All three Python distributions built successfully.
- Four real GPU combinations (two models per engine) each passed 1,000 bundle
  route switches with consistent snapshots; see the failure-inclusive matrix.
- Tenant credentials select per-tenant queues, request/token quotas and cancellation
  ownership. Quotas are per API process; multi-replica quotas remain unfinished.
- A real standalone SGLang gateway passed Python SDK serving, 16-question request
  cancellation (499, no retained gateway lease), disable/native-chat coexistence
  and rollback at `453ebf2`. Its initial repeated-cancellation failure at `790448e`
  is retained alongside the passing retest.
- Engine branch IDs are journaled before dispatch. Administrative recovery can
  operate after a worker restart, with PID/start-tick/boot-ID checks and matching
  engine identity. Real SGLang and vLLM gateway SIGKILL/restart tests passed: 128 branch
  IDs persisted, the dead owner was verified, explicit recovery released its lease,
  and serving resumed while the original engine remained alive. Offline CLI recovery
  also passed for SGLang. The initial launcher TIME_WAIT failure is retained with its fix.
- Per-worker startup revalidates persisted active versions. Concurrent local
  bootstrap runs a canary on every worker. A local activation barrier requires all
  serving workers to prepare the target digest. vLLM with two API workers passed
  its barrier, 1,000 switches and K=32/64 scoring at `a283bd5`. SGLang's first
  two-worker attempt exposed missing plugin routes; its worker-target fix is
  awaiting real retest. Multi-node rollout remains incomplete.
- Real-tokenizer checks of all 20 pinned entries at `53d8185`: 11 passed all 16
  combinations, Phi-3 passed 14/16 (joint K=64 unsupported; independent works), six
  gated repositories were inaccessible, and Mistral Small/GLM require specific
  template/tokenizer profiles. These are not GPU/model-quality certifications.

## Required evidence still outstanding

| Requirement | Status | Required next evidence |
|---|---|---|
| Private repository | Created, privacy verified | Verify pushed source and final visibility |
| SGLang real serving | Qwen3-0.6B and SmolLM2 functional checks passed | Remaining matrix; cross-implementation BF16 numerical differences remain |
| vLLM real serving | Qwen3-0.6B and SmolLM2 functional checks passed | Remaining matrix; cross-implementation BF16 numerical differences remain |
| Native plugin lifecycle | vLLM two-worker check passed; SGLang worker import failure fixed locally | SGLang two-worker retest and final-source checks |
| GPU hot-switch and drain | Four combinations each passed 1,000 route switches | Cancellation faults, adapter swaps, multi-replica rollout |
| 20-model/40-combination matrix | 2/20 functional checks per engine | Remaining 36 combinations and numerical/quality/performance gates |
| Calibration and quality tooling | CLI, collection, binary/multiclass fitting implemented | Real held-out tasks and accuracy evidence |
| Real business evaluation | Missing data | At least two approved tasks and grounded labels |
| Performance certification | Not run | Defined 144-case matrix and controlled native baselines |
| 24h soak and fault injection | Gateway crash recovery passed on both engines; soak not run | Remaining faults and complete 24h evidence |
| LoRA lifecycle | Incomplete | Load/unload adapters, engine identity, draining, cache isolation |
| SDKs and deployment productization | Python/TypeScript SDK checks and both standalone gateways pass | Deployment images, final-source recertification |
| Resource fairness/multiple tenants | Per-process implementation and contract tests pass | Real tenant load, replica-wide quota and mixed workloads |
| Multi-replica rollout | Incomplete | Per-replica READY state, shared activation, consistent request versions |
| Advanced readout/VLM roadmap | Not implemented | Follow the separate staged scope in the original plan |

## DSW preflight, 2026-09-27

Three reachable instances already host GPU services. Observed free memory is roughly
7–12 GiB per L20Z GPU, with driver 550.127.08 on the inspected candidate. Existing
processes have not been stopped or reconfigured. Independent capacity/performance
certification requires an available GPU allocation. Small-model co-located checks,
if run, will be labeled as such and will not become independent performance claims.

The pinned public engine versions require newer dependencies than the pre-existing
internal SGLang environment. New test environments must be isolated; do not upgrade
the running service's Python environment or GPU driver.

SGLang 0.5.20 retired its CUDA 12 lane. The isolated SGLang environment therefore
uses 0.5.19 and the dependency substitutions from its official CUDA 12.9 Dockerfile.
Torch CUDA GPU matmul, `pip check`, and two-model SGLang functional serving passed.
Optional Rust extensions were not built in this text-only functional environment.

GitHub Actions run 36301507239 never started its test job because organization
billing/spending eligibility rejected it. Hosted CI is not green; local checks
and DSW checks are reported separately.
