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
- Local response-contract checkpoint: 163 Python tests and 62 TypeScript tests
  passed, including shared valid/malformed answer cases and cancellation errors.
  Both SDKs enforce values, rankings, explicit score expectations, abstention,
  probability semantics and consistent success accounting. Saved DSW payload replay
  is distinct from new GPU execution: both clients accepted all 26 complete payloads
  in 17 retained DSW reports at `fe71bb3`. All three Python distributions built at
  that source; see `evidence/package-check-fe71bb3.json`.
- Six real GPU combinations (three models per engine) each passed 1,000 bundle
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
  two-worker attempt exposed missing plugin routes; its worker-target fix passed
  at `c5f8867`, as did an independent two-worker SGLang gateway. All owned process
  groups exited after the checks; GPU free memory returned to the observed baseline.
  Multi-node rollout remains incomplete.
- Real-tokenizer checks of all 20 pinned entries at `53d8185`: 11 passed all 16
  combinations, Phi-3 passed 14/16 (joint K=64 unsupported; independent works), six
  gated repositories were inaccessible, and Mistral Small/GLM require specific
  template/tokenizer profiles. These are not GPU/model-quality certifications.
- Server-side input export, a fixed 144-case performance manifest, paired native
  and typed HTTP runner, and strict cohort accounting are implemented. The runner
  preserves parity/setup failures, actual token lengths, drain and unknown cache
  observations. Five-second colocated SmolLM2 paired runs completed on both engines;
  the initial SGLang parity failure is preserved. They expose plugin overhead and
  do not certify performance. Bounded exact compiler caching and a persistent
  FULL-synchronous registry connection have local contract/crash-persistence tests;
  vLLM completed three ten-second paired repeats after optimization, with exact
  probability parity and all attempts successful. A real vLLM gateway SIGKILL and
  recovery retest passed with the reused WAL connection. These remain colocated
  development checks. SGLang also completed three ten-second paired repeats with
  all requests successful; its throughput still falls below the 90% target in some
  repeats. Neither result fills the controlled release matrix.
- New fast-tokenizer bundles bind the backend implementation fingerprint. Both
  native plugins use the host tokenizer. An admin profile and remote bundle-build
  CLI avoid a separately loaded tokenizer; old immutable digests are preserved,
  while serving requires rebuilding legacy bundles with the stronger identity.
  Contract tests and a real SGLang remote-build/prepare/reject-mismatched-fingerprint
  check pass. vLLM's host tokenizer served both fixed public tasks with two native
  API workers at `97271f8`.
- Explicit cancellation now addresses the original local API worker through a
  durable command, bound to tenant/backend/lease ID. Peer-worker success, failed
  abort retention, ownership and reused-ID races have local tests. Real two-worker
  GPU-serving validation passed through an independent two-worker SGLang gateway
  at `364b6e0`: worker B cancelled worker A's journaled 128-branch request, no lease
  remained, and serving resumed. vLLM's native two-worker path also passed at
  `97271f8`; SGLang's native two-worker path also passed at `febb5d8`, including
  terminal-response drain and serving after cancellation. This is not multi-node routing.
- Public AG News/SST-2 fixture preparation and HTTP quality/calibration runners
  are implemented with immutable source revisions and full attempted denominators.
  The existing raw-score collector now also journals work and preserves uncertain
  aborts. Both engines completed all 768 public requests and a live calibrated
  hot-switch check; per-example results include errors/abstentions and numeric
  quality evidence. This does not replace business data or certify cross-engine parity.
- Managed immutable LoRA artifacts, transactional drain guards and native completion
  fences have local tests. The initial real vLLM attempt at `67a6fbb` failed when a
  64-bit adapter ID reached vLLM's int32 GPU request array. Its report and verified
  process-group cleanup are retained. IDs now have an int32 bound and transactional
  collision checks; plugin cleanup now drains before host state/engine teardown.
  The corrected vLLM run at `500b197` passed 24 alternating switches, an old
  128-question request across publication, blocked premature unload, cancellation
  and six reload/rollback cycles. Output differences were zero against each warm
  baseline. The later `febb5d8` rerun below also requires per-adapter cache-hit observations.
  SGLang's first run hit its reserved base-model pool-slot constraint on the second
  pinned adapter; its report/cleanup are preserved. The launcher now reserves four
  slots and the adapter checks pinned capacity before dispatch. At `c36d65d`,
  SGLang passed nonzero output separation, cache-hit-observed alternating switches
  and the in-flight old-version drain check, then failed unload after cancellation.
  Its native generator had discarded request state before abort, leaving a LoRA
  usage count unreleased. The receiver now remains shielded through a bounded
  terminal-response wait. At `febb5d8`, both native engines passed the full suite:
  24 alternating switches with observed hits (SGLang 90/91 prompt tokens, vLLM
  80/91), a 128-question old request across publication, blocked premature unload,
  cancellation followed by GPU-fenced removal, and six reload/rollback cycles.
  Warm-baseline probability error was zero for switches and reloads. Every owned
  process group exited and GPU7 free memory returned to 11,990 MiB. The public
  managed-LoRA gate now permits only this checkpoint revision with BF16,
  TP/PP/DP=1 and one API worker. See `profiles/lora-lifecycle.json`; this is a
  synthetic lifecycle profile, not task-quality or overall release certification.

- DeepSeek-R1-Distill-Qwen-1.5B (Qwen2 architecture) passed both native engines
  at runtime `ad9b417`. Each passed 1,000 switches with no mixed-bundle responses
  and zero native/attach logprob difference. vLLM served 406 requests during the
  switch test and SGLang served 444. Both SDKs received
  live responses (TypeScript on vLLM, Python on SGLang). The initial vLLM harness
  wrongly iterated a legitimate rank abstention; the initial SGLang harness built
  a manifest using a different local tokenizer and was correctly rejected. Both
  failures are retained. The harness now checks default-policy abstention and an
  explicit `tie: first` bundle separately, and obtains serving identity/input IDs
  from the actual worker. Core runtime guards were not relaxed. Selected-label
  mass can be very small on this reasoning-distilled checkpoint; no business
  accuracy claim follows from these functional checks. Downloaded weight SHA256
  matches the immutable Hub LFS object; in-memory weight attestation remains false.
  All owned process groups exited and GPU7 returned to 11,990 MiB free. Local
  regression now has 165 passing Python tests; TypeScript retains 62 passing tests.

## Required evidence still outstanding

| Requirement | Status | Required next evidence |
|---|---|---|
| Private repository | Created, privacy verified | Verify pushed source and final visibility |
| SGLang real serving | Qwen3-0.6B, SmolLM2-1.7B and DeepSeek-R1-Distill-Qwen-1.5B functional checks passed | Remaining matrix; cross-implementation BF16 numerical differences remain |
| vLLM real serving | Qwen3-0.6B, SmolLM2-1.7B and DeepSeek-R1-Distill-Qwen-1.5B functional checks passed | Remaining matrix; cross-implementation BF16 numerical differences remain |
| Native plugin lifecycle | Both native two-worker paths and two-worker SGLang gateway passed | Additional configurations and final-source checks |
| GPU hot-switch and drain | Six combinations each passed 1,000 route switches | Cancellation faults, adapter swaps, multi-replica rollout |
| 20-model/40-combination matrix | 3/20 functional checks per engine | Remaining 34 combinations and numerical/quality/performance gates |
| Calibration and quality tooling | CLI, collection, binary/multiclass fitting implemented | Real held-out tasks and accuracy evidence |
| Real business evaluation | Missing data | At least two approved tasks and grounded labels |
| Performance certification | Not run | Defined 144-case matrix and controlled native baselines |
| 24h soak and fault injection | Gateway crash recovery passed on both engines; soak not run | Remaining faults and complete 24h evidence |
| LoRA lifecycle | Both native engines passed one frozen SmolLM2 BF16 TP1/API1 profile | GPU crash/restart checks; further profiles require separate evidence |
| SDKs and deployment productization | Python/TypeScript SDK checks and both standalone gateways pass | Deployment images, final-source recertification |
| Resource fairness/multiple tenants | Per-process implementation and contract tests pass | Real tenant load, replica-wide quota and mixed workloads |
| Local multi-worker coordination | Activation barrier and cross-worker cancellation passed on both native engines | Expanded local fault/load cases; multi-node coordination is later-stage scope |
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

The same eligibility rejection was rechecked at run `36313600555` for `febb5d8`.
Local checks at that source passed 103 Python tests, lint/format and all three
wheel builds. The subsequent conservative checkpoint gate has its own local tests;
GPU reports retain the exact runtime commit rather than claiming a later source ran.

Hosted CI run `36315081675` at `ad9b417` again had zero executed steps because
organization billing/spending eligibility rejected job startup. This is separate
from the passing local and DSW checks; see `evidence/hosted-ci-ad9b417.json`.
