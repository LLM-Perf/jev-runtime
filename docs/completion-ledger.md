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
- Separate SGLang/vLLM plugin distributions, awaiting real engine lifecycle validation.
- CPU contract tests with a controlled engine double. These are not model evaluations.
- Temperature fitting with grouped split checks, NLL/Brier/ECE/risk-coverage metrics,
  immutable artifact binding, and synchronous/asynchronous Python clients.
- Local checkpoint: 31 CPU tests and Ruff passed. The 1,000-switch test at this stage
  exercises the registry, not GPU workers under traffic.

## Required evidence still outstanding

| Requirement | Status | Required next evidence |
|---|---|---|
| Private repository | Created, privacy verified | Verify pushed source and final visibility |
| SGLang real serving | Not run | Model identity, complete scoring results, native/attach parity |
| vLLM real serving | Not run | Model identity, complete scoring results, native/attach parity |
| Native plugin lifecycle | Not run | Multi-process startup/shutdown and unchanged engine routes |
| GPU hot-switch and drain | Not run | In-flight scoring across activation, cancellation and rollback |
| 20-model/40-combination matrix | Not run | Fixed revisions, failure-inclusive matrix, per-engine thresholds |
| Calibration and quality tooling | Partial | CLI/data collection, real held-out tasks, binary fitting and evidence |
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
