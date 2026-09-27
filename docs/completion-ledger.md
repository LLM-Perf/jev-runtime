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
- Local checkpoint: 38 CPU tests and Ruff passed. Real vLLM Qwen3-0.6B traffic also
  passed 1,000 bundle route switches with 432 completed requests and consistent snapshots.

## Required evidence still outstanding

| Requirement | Status | Required next evidence |
|---|---|---|
| Private repository | Created, privacy verified | Verify pushed source and final visibility |
| SGLang real serving | Not run | Model identity, complete scoring results, native/attach parity |
| vLLM real serving | Qwen3-0.6B functional checks passed | Remaining matrix; cross-implementation BF16 numerical differences remain |
| Native plugin lifecycle | Not run | Multi-process startup/shutdown and unchanged engine routes |
| GPU hot-switch and drain | vLLM Qwen3-0.6B passed 1,000 route switches | SGLang, cancellation faults, adapter swaps, multi-replica rollout |
| 20-model/40-combination matrix | Not run | Fixed revisions, failure-inclusive matrix, per-engine thresholds |
| Calibration and quality tooling | CLI, collection, binary/multiclass fitting implemented | Real held-out tasks and accuracy evidence |
| Real business evaluation | Missing data | At least two approved tasks and grounded labels |
| Performance certification | Not run | Defined 144-case matrix and controlled native baselines |
| 24h soak and fault injection | Not run | Live job logs, complete coverage and cleanup evidence |
| LoRA lifecycle | Incomplete | Load/unload adapters, engine identity, draining, cache isolation |
| SDKs and deployment productization | Incomplete | Installable SDKs, complete CLI, runbooks, image manifests |
| Resource fairness/multiple tenants | Incomplete | Tenant-aware admission, cancellation ownership, mixed workloads |
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
Torch CUDA GPU matmul and `pip check` passed; actual SGLang serving is still pending.
Optional Rust extensions were not built in this text-only functional environment.

GitHub Actions run 36301507239 never started its test job because organization
billing/spending eligibility rejected it. Hosted CI is not green; local checks
and DSW checks are reported separately.
