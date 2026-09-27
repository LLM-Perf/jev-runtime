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
- Twenty-four real GPU combinations (twelve models per engine) each passed 1,000 bundle
  route switches with consistent snapshots; see the failure-inclusive matrix.
- Tenant credentials select per-tenant queues, quotas and cancellation ownership.
  Shared same-host quotas are now implemented and checked at `4944efd` below;
  multi-node quotas remain unfinished.
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


- Structured-generation choice baseline now preserves the exact same task/input,
  actual native prompt/output token traces, strict JSON/stop validation and output
  costs for successful and failed attempts. At harness `444fd1b` / runtime `60b3415`,
  both engines completed three five-second colocated SmolLM2 repeats: SGLang
  1,268/1,268 and vLLM 1,502/1,502 strict successes across all three methods. The
  JSON lane disagreed with the selected-label reference on every timed response;
  this fixture has no gold labels and establishes no equivalent-quality speedup.
  A real SGLang one-token-budget negative preflight correctly rejected HTTP 200
  truncation and retained its usage. Some plugin/native throughput ratios remain
  below 90%. Both process groups exited and GPU7 returned to 11,990 MiB free.
  See `docs/performance.md`. Local packaging at `444fd1b` built all three wheels;
  its 179 passing Python tests and checks are recorded separately from GPU runs.


- A real SGLang fault check reproduced collateral cancellation with the legacy IDs:
  aborting `parent.a.1` also aborted `parent.a.1.0`. At `0421d50`, fixed-length
  question/index suffixes remove this prefix relationship. Two CPU regressions
  cover dotted question names/candidate indices and actual runtime partial-result
  isolation; both failed before the fix. The live serving compiler's IDs passed
  the GPU retest: the selected branch aborted at three tokens while its sibling
  completed 128 tokens. Both engines then passed the full functional suite and
  1,000 hot switches again (SGLang 519 / vLLM 469 successful traffic
  requests, no mixed versions, zero native/attach logprob difference). The number
  of distinct supported combinations remains six. Both process groups exited and
  GPU7 returned to 11,990 MiB free. Local regression has 181 passing Python tests;
  lint/format and all three wheels passed at this source. These targeted checks
  do not recertify every earlier LoRA, multi-worker, quality or performance profile.

- Runtime phase timings, bounded branch/error metrics and known-token observation
  denominators are implemented. A failed durable lease release now also clears
  local request ownership while retaining the recoverable journal. Metrics remain
  process-local; multi-worker aggregation is not implemented. The benchmark can
  freeze and verify a round-robin input pool, retaining every attempted fixture
  index and timing observation. Multi-input structured generation remains unsupported.
- At `c11d2b3`, guarded IDs-only tokenization preserves full-context encoding and
  one-token label checks. vLLM uses an identity-checked private copy of its actual
  host pool prototype, without accessing the pool's shared mutable backend. The
  initial wrapper-batch experiment showed no consistent CPU gain and was removed;
  its negative evidence and the initial live fallback checks are retained.
  On 300 exact L=256/K=8 inputs, colocated vLLM compilation fell from 3.85–3.91 ms
  to 1.78–1.90 ms. The new source completed 2,608/2,608 vLLM and 2,078/2,078 SGLang
  timed requests. All 300 input/label-ID lists match across engines and each
  engine had zero native/typed probability error. The observed plugin/native
  throughput remains about 80% for vLLM and 76–83% for SGLang, below the 90% gate.
  These are short, shared-GPU development measurements; the controlled matrix
  remains unrun. Both engines passed another 1,000 switches (SGLang 548 / vLLM
  469 successful traffic requests, no mixed versions) and all owned process
  groups exited. The distinct functional denominator remains six combinations.
  Local regression at this source passed 207 Python tests, lint/format and all
  three wheel builds; TypeScript's previous 62 tests are retained separately.
  See `evidence/package-check-c11d2b3.json` and
  `evidence/dsw/input300-validation-c11d2b3.json`. No earlier LoRA, multi-worker,
  business-quality or full-release acceptance is implied by these targeted checks.

- At `4944efd`, default Runtime/gateway/native-plugin admission uses shared local
  SQLite transactions to reserve requests, expanded tokens and branches, with
  global/per-tenant queues and per-tenant FIFO/round-robin scheduling. Reservations
  live with durable request leases; uncertain aborts or failed lease cleanup do
  not silently return capacity. Policy disagreement prevents worker startup, and
  policy migration requires stopped workers and drained work. An explicit deadline
  check prevents synchronous compilation/admission from dispatching expired work.
  Local regression passed 220 Python tests, including an actual three-process
  branch-budget race; lint/format and three wheel builds passed. Both native
  engines passed a two-worker, two-tenant GPU check of shared quotas, queue-full
  rejection, eligible-tenant progress, queued timeout/cancellation, owner-bound
  cancellation, shared gauges and drain. All lease/ticket tables ended empty.
  Separate short two-worker timing cohorts completed 1,324/1,324 vLLM and
  1,168/1,168 SGLang attempts with zero within-engine native/typed parity error.
  Throughput ratios still fall below 90%; API-worker counts/workloads differ from
  prior checkpoints, so no causal regression or release pass is claimed. All
  owned process groups exited and GPU7 returned to 11,990 MiB free. See
  `evidence/dsw/shared-admission-validation-4944efd.json`. These targeted checks do
  not recertify LoRA, the full functional/quality matrix, 1,000 hot switches at
  this new source, database-failure recovery or GPU crash/restart quota retention.

- Durable quota recovery passed four real DSW fault cases: SGLang/vLLM standalone
  gateways at `aba3b52`, and both native API/GPU process groups at `0df49a1`.
  Every case retained one admitted request (128 branches / 134,290 expanded tokens)
  and one queued request after SIGKILL and restart with identical settings and
  credential bytes. The original dead owner was verified. New work could not use
  the retained capacity; recovering the queued request preserved the admitted
  reservation, then explicit recovery drained both journal and quota tables.
  Typed decisions and native chat resumed. Native checks also verified full old
  process-group exit and GPU memory returning to the baseline before restarting
  the same engine command. Final task-owned groups exited; GPU7 returned to
  11,990 MiB free. Core/package/deployment files are unchanged from `4944efd`;
  these commits extend the real fault harness. Its lint/format/compilation passed;
  the prior 220 CPU tests and wheel builds remain source-identical checkpoints,
  not newly rerun tests. Remote SHA256 checks cover 50 files at both revisions.
  See [the scoped fault report](fault-recovery.md) and
  `evidence/dsw/quota-crash-validation-0df49a1.json`. This does not cover hardware
  faults, database I/O failure, managed-LoRA restart, multi-node failover, a 24h soak
  or a performance gate. The distinct model/engine denominator remains 6/40.

- Managed-LoRA crash/restart passed at `5176a82` on vLLM and `533549a` on
  SGLang with the frozen SmolLM2 BF16 TP1/API1 eager profile. Each test killed the
  recorded full native process group during a 128-branch adapter request and
  verified 133,376 reserved expanded tokens remained after restart. Adapter
  residency became UNKNOWN, adapter routes paused, and explicit request recovery,
  reconciliation, fresh canaries and current generations were required. First and
  warm reload responses and 24 post-restart switches had zero probability error.
  A separate lifecycle recheck on each restarted engine passed 24 switches,
  old-request drain, cancellation/fenced unload and six reload/rollback cycles.
  Passing registries ended with four UNLOADED adapters and zero leases/tickets.
  All task-owned process groups exited; GPU7 returned to 11,990 MiB free.
  The first SGLang run failed before planned SIGKILL: upstream selected-logprob
  normalization called `.tolist()` on a Python list row. Its diagnostics and one
  retained preparation lease remain in the stopped failed attempt. The plugin now
  uses version-scoped official hooks for SGLang 0.5.19, preserving values and tensor
  objects without editing installed engine source. Actual upstream CPU reproduction
  passed for both affected consumers. A native 512-token SSE stream also completed
  with 12/12 typed requests wholly inside its lifetime; scheduler batch composition
  was not observed. Local validation at `533549a` passed 225 Python tests,
  lint/format and all three wheels with verified Python contents. TypeScript was
  unchanged and its prior 62 tests were not rerun. Core/vLLM/deployment sources are
  identical between these tested revisions, while the SGLang plugin changed.
  See [the LoRA crash report](lora-crash-recovery.md) and
  `evidence/dsw/lora-crash-validation-533549a.json`. Interrupted GPU load/unload,
  worker-only faults, other profiles, controlled performance and the 24h soak
  remain unverified. The functional denominator stays 6/40.

- Periodic per-worker engine canaries and typed-bundle circuits are implemented at
  `40b78c2`. Readiness requires fresh scoring evidence for every active bundle.
  New work checks health before compilation and after admission; failed or stale
  canaries reject dispatch. Unconfirmed probes retain their durable lease and do
  not repeat until explicit recovery. Real DSW gateways on both engines withdrew
  readiness after their recorded engine groups were paused, rejected 10/10 new
  requests, and recovered on fresh canaries after resume. vLLM passed the original
  comparison. SGLang's first comparison failed against a partially cached baseline;
  a separate native replay without process interruption reproduced a 0.02238
  cold/warm probability difference across three pairs. The revised harness at
  `02addc7` retains failure responses and compares matching warm states with the
  same `1e-4` threshold; SGLang passed with zero error. Runtime source stayed
  unchanged. Both native engines subsequently passed four-type/native-parity
  checks and 1,000 switches (vLLM 453 / SGLang 510 successful traffic requests,
  zero mixed versions). All four groups exited and GPU memory returned to baseline.
  Local validation passed 237 Python tests, lint/format and all three wheels.
  See [serving health](serving-health.md) and
  `evidence/dsw/health-validation-40b78c2.json`. Replica selection/failover, cache-state
  numerical/quality impact, metrics aggregation and the full release gates remain.
  The functional denominator remains 6/40; LoRA/multi-worker profiles were not
  recertified by these targeted health checks.

- At `737d814`, Phi-3 mini and Phi-4 mini passed native BF16 TP2/API1 functional
  checks on both engines, including 1,000 switches per combination and zero
  native/attach logprob difference. Traffic successes were 244/448 for Phi-3
  (SGLang/vLLM) and 184/317 for Phi-4, with zero mixed versions. This raises the
  fixed matrix to 10/40 (5/20 per engine). Pinned model files and 56 release files
  were verified; live topology and zero final leases/admission were recorded.
  The first vLLM pre-ready harness failure and SGLang TP memory-balance startup
  failure are retained. All five owned groups exited and selected GPUs returned
  to their observed baselines. Three CPU BF16 single-position comparisons exceed
  the unchanged 0.15 tolerance; only Phi-4/SGLang passes that targeted check,
  which is not full numerical certification. Local checks passed 243 Python tests
  and Ruff (102 files). Package sources are unchanged from `40b78c2`; its wheel
  evidence is explicitly reused. See [Phi TP2 evidence](phi-tp2-validation.md).


- At `f6bda1a`, new bundles/calibration contracts bind readout dtype separately
  from backbone dtype. Startup checks the engine report; legacy/wrong bundles
  fail preparation without rewriting stored digests. Both engines passed Phi-4
  FP32-readout TP2 and SmolLM2 default-BF16 TP1 native suites, 1,000 switches each,
  and live precision rejection checks with harness `5aa59d6`. The four targeted
  CPU reference positions pass the development 0.15 tolerance; this does not
  replace a full per-profile numerical budget or repair earlier BF16 failures.
  All groups exited and selected GPUs returned to baseline. Local tests passed
  257 checks and all three wheel builds. The base denominator stays 10/40;
  calibration migration, deployment scope and raw evidence are documented in
  [readout precision](readout-precision.md). Remaining release gates are unchanged.


- With harness `b8742bd` and unchanged runtime `f6bda1a`, Qwen3-8B and OLMo-2-7B
  passed BF16 native TP4/API1 suites on both engines: 1,000 switches per run,
  zero mixed versions, zero native/attached logprob difference and live precision
  rejection checks. Coverage is now 14/40, or 7/20 per engine. Qwen3-8B/vLLM
  fails the CPU single-position development tolerance (0.25 > 0.15); the other
  three positions pass. All four owned groups exited, all selected GPU UUIDs
  matched and free memory returned to the recorded baseline. A reusable runner
  now preserves attempts, verifies live topology and attempts cleanup on failures.
  Local tests passed 261 checks; 101 harness files and 41 evidence hashes were
  verified. See [TP4 model validation](tp4-model-validation.md). Original release
  gates and failed numerical evidence remain unchanged.


- At `e17918e`, hash-bound local Jinja/JSON templates are supported by both
  native plugins and the gateway compiler, with unchanged default behavior and
  rejection of renderers that can ignore the override. Mistral7B passes both
  native BF16 TP4 suites and 1,000 switches each using its official template,
  raising coverage to 16/40 (8/20 per engine). At `75649bf`, explicit tokenizer
  regex profiles are checked against native host vocabulary/backend fingerprints.
  Mistral Small's explicit HF profile passes 16/16 tokenizer compilation cases in
  each environment; this adds no GPU/model-quality pass. Initial missing-template
  and renderer-rejection attempts remain, as does GLM's custom-loader rejection.
  The real-tokenizer mismatch guard passes in both environments. All owned GPU
  groups exited; local checks passed 277 tests and three verified wheel builds.
  GPU evidence remains at `e17918e`, separate from later CPU/tokenizer checks.
  See [Mistral validation](mistral-validation.md) and
  [explicit template configuration](explicit-chat-templates.md).

- At `3377c95`, shared admission journals recovery IDs and reserves quota in one
  FULL-synchronous transaction, reducing immediately admitted requests from four
  commits to three. Atomic visibility, rollback, before-dispatch persistence and
  uncertain-abort retention pass local tests (284 total). Both engines completed
  matched before/after two-worker BF16 TP1 cohorts: 45,970/45,970 timed requests,
  zero warm-fixture native/plugin probability difference, and independently
  recomputed summaries. Journal-plus-queue time decreases, but single-concurrency
  throughput remains below 90%; high-concurrency results were already above 90%
  before the change and do not show a stable overall gain. Both new plugins pass
  real two-worker/two-tenant quotas and peer cancellation. All groups exit, GPU7
  returns to baseline, and 79 owned records have no matching live process.
  The failed readiness setup is retained. No model count or release gate changes.
  See [atomic admission comparison](atomic-admission-performance.md).

- Runtime `3377c95` and launcher/runner `b28bb46` add Qwen2.5-7B and R1-Distill-
  Llama-8B on both engines: 20/40 functional combinations, 10/20 per engine.
  A tokenizer artifact audit invalidates the original vLLM R1 default-loader
  profile despite its passing raw-token checks. The separately tested explicit
  ByteLevel tokenizer preserves checkpoint IDs on both engines. `--tokenizer-path`
  now reaches the Jev configuration and native engine consistently. Qwen BF16 and
  FP32-head CPU-reference checks fail the unchanged 0.15 tolerance; the corrected
  R1 vLLM position also fails (0.160690), while SGLang passes one position (0.129480).
  All eight attempts, original tokenizer failure and numerical failures remain.
  Every group exits and GPUs 3–6 return to baseline; 87 owned records have no live
  match. Local checks pass 292 tests, with unchanged core/plugin source packages.
  See [public-model pair validation](public-model-pair-validation.md).

- Reference tooling at `5730449` adds bounded module-state offload for independent
  CUDA computation. Both installed Python environments passed 24 full-logit
  resident/offload equalities across tiny Qwen2, tied Llama and real SmolLM2,
  including state and exception cleanup. Eighteen new matched reference profiles
  cover only two saved Qwen2.5/R1 compiled inputs: nine pass and nine fail the
  unchanged 0.15 development threshold. GPU SDPA checks pass for both BF16 model
  rows on both engines, but SGLang Qwen's FP32-head profile still fails (0.289116).
  Reference attention/device affects the result; this is not a resolved full
  numerical gate. Core/plugins and 20/40 functional coverage remain unchanged.
  Local checks passed 298 tests. Model-file rehash and final owned-process/group
  checks passed, and GPU7 returned to 11,990 MiB free. See
  [reference diagnosis](numerical-reference-diagnostics.md).

- At `c48f18d`, the GLM4 vocabulary converter produces a bound fast tokenizer
  without executing checkpoint Python in serving. Its 588 built-in cases pass;
  both engine environments match the pinned original wrapper on 630 encoding
  comparisons, 40 rendered templates and 16 compiler configurations each. Actual
  saved GPU input/label IDs also match. GLM-4-9B and R1-Distill-Qwen-7B add four
  native TP4 combinations, each with 1,000 switches and zero mixed responses:
  24/40 functional profiles, 12/20 per engine. R1's vLLM CPU reference passes one
  position (0.124721); SGLang fails (0.187206). Default GLM reference loading fails;
  the isolated official Transformers 4.44.2 reference runs but exceeds tolerance
  on both engines (0.875 / 0.625). Failures remain visible. All four groups exit,
  30 checkpoint files rehash unchanged, 91 historical owned server records have no
  live matches, and GPUs 3–6 return to 10,792 MiB free each. Local 307 tests and
  three source-matched wheels pass. No quality/performance/soak release gate is
  closed. See [GLM/R1 validation](glm-r1-model-validation.md).

- At installer `4e49524`, immutable three-wheel Git snapshots, exact platform locks,
  offline fresh gateway installation and installed-code verification pass. DSW
  candidate `861ed3a` (44 packages) and previous `3377c95` (40 packages) serve both
  native engines at `c48f18d`. Previous/canary/active/rollback stages preserve the
  same registry route; 64 persistent decisions and six complete Python SDK suites
  pass. Native process identities remain unchanged. Measured maintenance upgrade
  intervals are 3.346/3.487 s and rollback 3.214/2.976 s (vLLM/SGLang); automatic
  uninterrupted rollout is not implemented. The initial metadata-resolution,
  duplicate-cleanup controller, tokenizer-identity and alternate-constraint
  failures remain. All 108 historical owned records have no live match/group.
  A Smol default AutoTokenizer fidelity defect is exposed: 232/1,005 inputs differ
  from its checkpoint Digits pipeline. An explicit standard fast profile matches
  checkpoint/native encodings on all 1,005 inputs and both engines pass another
  1,000 switches (458/546 strict traffic requests). Functional coverage stays
  24/40. Historical numerical and development results retain their original
  profiles; the new profile has no independent numerical/quality/performance or
  LoRA requalification. Local 317 tests, Ruff and pip check pass; rebuilt `4e49524`
  wheels are byte-identical to the tested candidate. P18/P24 remain partial. See
  [installed-wheel rollout validation](release-rollout-validation.md).

- At `cdd0caf`, the product `preserve-fast` CLI creates bound, checkpoint-preserving
  JSON tokenizer profiles. Both isolated engine environments verify all 11 eligible
  local checkpoints: 22,110 independent encodings and 348/352 compiler cases pass;
  Phi-3 joint K=64 still fails. GLM4 format rejection and eight absent checkpoints
  remain in each 20-model inventory. Initial identity-postprocessor rejections are
  preserved; only the exact single-text identity representation is now accepted.
  A separate OLMo SGLang native-getter check has zero corpus differences and matches
  its historical native fingerprint, despite default AutoTokenizer differences.
  Generated Smol profiles pass native four-type/precision checks and 1,000 switches
  per engine (479 vLLM / 537 SGLang strict requests, zero mixed versions). One CPU
  BF16 reference position passes each unchanged 0.15 tolerance (0.000211239 /
  0.086620569). This is partial numerical evidence, not the full gate. An initial
  missing-Hatchling installation failure occurs before any server starts and is
  retained. All 110 historical owned records are terminal; GPU7 returns to 11,990
  MiB free and six model files rehash against fixed Hub metadata. Local 332 tests
  and three source-matched wheels pass. Functional coverage stays 24/40; the other
  ten new tokenizer profiles have CPU checks only. See
  [serialized tokenizer validation](preserve-fast-validation.md).

## Required evidence still outstanding

At `7739145`, a journaled HAProxy gateway rollout controller qualifies all workers,
binds registry/model/bundle/auth identities, runs a real candidate canary, switches
one live map entry, reconciles interrupted operations without replay and observes
HTTP stream/queue plus durable-lease drain. Both native engines pass actual
offline-installed `accef86` ↔ `7739145` wheel upgrades/rollback: two workers per
slot, 13 switches each, 333 vLLM/338 SGLang continuous requests, zero request errors,
unchanged proxy/native process identities. Initial same-code exercises separately
pass 446/326 native requests and 1,116 CPU fixture requests; the fixture also tests
cross-slot cancellation. All 129 historical owned records are terminal, GPU7 is
back to 11,990 MiB free, and six public model files rehash. Local 357 tests and both
sets of three source-matched wheels pass. This qualifies one compatible-schema
single-host gateway profile, not engine failover, GPU upgrade cancellation, SLOs,
images, migration, soak or complete release acceptance. See
[gateway rollout validation](gateway-rollout-validation.md).

At harness `b20d3f4`, both engines also pass three real native scoring cancellations
across rollback from installed green `6f0cda8` to blue `7739145`. Every cancelled
request has 128 planned sequences, at least two completed scoring RPCs and one
active RPC at both observations. The same worker/request/bundle/lease remains
pinned; early drain blocks, cancellation through blue returns 499 in green,
leases disappear and serving resumes. Continuous traffic is 580/580 vLLM and
627/627 SGLang, separate from the six expected cancellations; each observes 19
map generations. Both initial admin-client setup failures are retained. The
independent audit finds 145 distinct owned records terminal and GPU7 at 11,990
MiB free. Local 368 tests pass. This covers gateway rollback while native requests
execute, not an engine-process upgrade or the full fault/performance/soak gates.
See [native cancellation validation](native-rollout-cancellation.md).

At `49fea1c`, offline gateway image-context preparation and guarded startup/probe
helpers pass 392 local tests and a real two-engine DSW host-process check. Each
engine passes 10/10 decisions with two gateway workers, authenticated readiness,
missing-key rejection, exec identity preservation and SIGTERM cleanup with zero
leases/tickets. The initial test-script failures are retained; all 153 historical
owned records are terminal. The actual image is neither built nor run: selected
base ABI, PID 1, non-root/mount/network behavior, engine images and container
restart/rollout remain open. See [image input and entrypoint evidence](container-images.md).

At `c4e9cef`, consistent private registry snapshots and guarded inactive restore
staging pass 406 local tests, three wheel builds and a real two-engine DSW
exercise. Each engine serves 10 requests before restore, one generation-control
request and 10 after restore, retaining generation 3 and rejecting generation 1.
Live/stale staging is rejected, the original registry remains unchanged and final
work/quotas drain. All 159 historical owned records are terminal; six snapshots
verify remotely. This preserves identical stopped same-host state; schema
migration/downgrade, destroyed-source recovery and container replacement remain
unimplemented. See [registry snapshots](registry-snapshots.md).

At `877f319`, a 32-case multi-input Qwen3-0.6B numerical suite completes 192
scoring responses and 384 GPU reference comparisons across both engines. Every
reference profile fails part of the unchanged development check; same-engine
cache/concurrency interventions also change some near-tie argmax labels. The
initial SGLang environment-recording error and separate reference completion
remain visible. All 171 retained owned identities are terminal; 420 local tests
pass. Numerical certification remains failed, and no model/quality/performance
release gate is newly passed. See [multi-input numerical diagnostics](numerical-suite.md).

At runtime `f9167c9`, execution mode is bound to model/bundle/calibration identity
and checked against the native engine report. A four-profile Qwen3-0.6B DSW A/B
collects 384 scoring responses, 128 plugin/native pairs and 768 independent GPU
reference comparisons. Both enabled profiles preserve all 32 vectors exactly
under repetition and reversed concurrent submissions; ordinary profiles change.
All native pairs match, but every independent-reference profile still fails part
of the unchanged numerical check. The first SGLang enabled run fails its immediate
cache reset; retained retest `5e6b2c7` completes with the native bounded idle barrier
and identical product code. All 205 retained owned records are terminal. Local
430 tests and three source-matched wheels pass; DSW executes source, not these
new wheels. Coverage remains 24/40 and all broader release gates remain open. See
[batch-invariance validation](batch-invariant-mode.md).

| Requirement | Status | Required next evidence |
|---|---|---|
| Private repository | Created, privacy verified | Verify pushed source and final visibility |
| SGLang real serving | Twelve checkpoints pass functional profiles, including explicit R1 ByteLevel and GLM4 fast tokenizers | Remaining matrix; independent numerical failures remain |
| vLLM real serving | Twelve checkpoints pass functional profiles, including explicit R1 ByteLevel and GLM4 fast tokenizers | Default R1 Llama tokenizer is unqualified; remaining matrix and numerical failures |
| Native plugin lifecycle | Both native two-worker paths and two-worker SGLang gateway passed | Additional configurations and final-source checks |
| GPU hot-switch and drain | Twenty-four combinations each passed 1,000 route switches | Cancellation faults, adapter swaps, multi-replica rollout |
| 20-model/40-combination matrix | 12/20 functional profiles per engine | Remaining 16 combinations and numerical/quality/performance gates |
| Calibration and quality tooling | CLI, collection, binary/multiclass fitting implemented | Real held-out tasks and accuracy evidence |
| Real business evaluation | Missing data | At least two approved tasks and grounded labels |
| Performance certification | Not run | Defined 144-case matrix and controlled native baselines |
| 24h soak and fault injection | Gateway/native quota recovery and READY-LoRA native group crash/restart passed on both engines; soak not run | Interrupted load/unload, worker-only/database/other fault profiles and complete 24h evidence |
| LoRA lifecycle | Both native engines passed one frozen SmolLM2 BF16 TP1/API1 eager profile, its crash/restart and lifecycle recheck | Other fault phases and execution/model profiles require separate evidence |
| SDKs and deployment productization | SDKs, immutable wheels/offline locks, and stable-proxy two-worker installed-wheel gateway upgrade/rollback pass on both engines | Expanded rollout faults/load, deployment images, migration/downgrade, handoff and final-source recertification |
| Resource fairness/multiple tenants | Durable same-host budgets, targeted two-worker/two-tenant load and single-worker crash/restart quota retention pass on both engines | Representative mixed workloads, fault/load performance and multi-node quotas |
| Engine health and fault routing | Periodic real canaries, stale/failure circuits and gateway pause/resume checks pass on both engines | Certified compatible-replica selection/failover and expanded load/fault profiles |
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

Hosted CI run `36318099621` at `0c1e38d` again executed zero steps: account
billing/spending eligibility rejected job startup. See
`evidence/hosted-ci-0c1e38d.json`; local and DSW checks are separate evidence.
