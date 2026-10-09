# Operating the current development release

Use separate Python environments for SGLang, vLLM and TokenSpeed. A gateway may attach to an
already-running engine without owning its process. Native plugins share the host
engine's lifetime. Do not use a plugin update to restart an unrelated service.

For versioned wheels, offline dependency locks and independent gateway environments,
see [release packaging](release-packaging.md). Retain complete prior environments
for rollback; an install receipt does not replace real serving/readiness checks.

For same-host compatible gateway code changes, use the
[journaled proxy rollout procedure](gateway-rollout.md). It qualifies every worker,
switches requests through a stable listener and observes HTTP/lease drain before
graceful retirement. See the [two-engine installed-wheel exercise](gateway-rollout-validation.md)
for its tested scope; engine/CUDA rollout and schema migration remain separate.

Periodic engine canaries now withdraw readiness and block new typed dispatch when
scoring fails or its evidence expires. See [health configuration and recovery](serving-health.md)
for per-worker scope, probe leases, explicit recovery and outstanding replica failover.

Newly built bundles bind the output-layer dtype separately from the backbone.
The runtime checks it against the engine report before serving. Older manifests
remain readable but need a new version and matching calibration before use;
see [readout configuration and migration](readout-precision.md).

## Current tested scope

DSW functional checks cover Qwen3-0.6B, SmolLM2-1.7B and
DeepSeek-R1-Distill-Qwen-1.5B on SGLang 0.5.19 and vLLM 0.30.0 with CUDA 12.9,
BF16, TP=1 and eager execution. Phi-3 mini and Phi-4 mini additionally passed
scoped BF16 TP2/API1 native checks on both engines at `737d814`; see
[the TP2 report and numerical limits](phi-tp2-validation.md). Check the evidence files for
the exact source commit: these are development snapshots, not a certification of
every later commit. Config switches have been tested. Managed LoRA swaps also
passed a scoped, frozen SmolLM2 BF16 TP1/API1 profile on both native engines at
`febb5d8`; scoped crash/restart and lifecycle rechecks also passed at `5176a82`
(vLLM) and `533549a` (SGLang), as detailed in the
[LoRA recovery report](lora-crash-recovery.md). See
[adapter operations and limits](adapters.md). Other checkpoints and
parallel configurations are not certified for managed LoRA.

For the R550 test host, `deployment/install_sglang_cu129.sh` uses the official
0.5.19 CUDA 12 dependency substitutions and records the source revision and patch.
The plugin does not modify installed engine source files. It uses the official
hook registry, including a version-scoped SGLang 0.5.19 host-logprob-row
compatibility hook described in [engine integration](engine-integration.md).
This environment omits optional Rust
extensions. SGLang 0.5.20/CUDA 13 requires a separate compatible-host certification.

## TokenSpeed operations (experimental)

Use the [TokenSpeed quick start](quickstart-tokenspeed.md) for its pinned source,
compatible GPU, generated model/tokenizer configs and `--check` preflight. The
launcher owns one native Engine and one HTTP worker; it exposes typed/admin/raw
routes, without native chat. The DSW launch/cleanup utility `deployment/dsw_service.py`
supports vLLM/SGLang only, so do not pass it `--engine tokenspeed`. Start TokenSpeed
with `jev-tokenspeed` and track the process you own for ordered shutdown.

Persist two separate stores on local durable storage:

| Store | Purpose | Recovery boundary |
|---|---|---|
| Configured `registry_path`, e.g. `registry.db` | Bundles, routes, leases, quota and work journals | Shared core registry lifecycle and explicit recovery rules |
| Sibling receipt DB, e.g. `registry.tokenspeed-receipts.db` | Native request reservations and observed terminal completions | Completed IDs can be confirmed after restart; unknown/pending IDs remain unresolved |

The receipt filename replaces the registry suffix with `.tokenspeed-receipts.db`.
Its namespace binds model/tokenizer/template identity, upstream profile, engine
options and engine URL. Preserve those identities when recovering the same service.
It stores request IDs and states, not prompts. Same-namespace native IDs cannot be
reused. The 4,096-entry in-memory cache is bounded; durable rows have no automatic
expiry. Monitor disk growth and include the full-sync writes in performance tests.

`jevctl registry snapshot`, staging and schema migration cover the **core registry
only**, not the receipt store. Quiesce, stop all affected writers and preserve both
databases using SQLite-aware backups before a deployment change. Do not copy a
live database without its WAL, delete receipt rows to clear an error, or assume a
core-registry backup alone preserves TokenSpeed completion evidence. Coordinated
receipt snapshot/restore/retention tooling is not implemented.

Cancellation waits up to 4.5 seconds for the native one-token request to complete.
A terminal receipt permits confirmation after HTTP response loss, cache eviction
or a same-identity restart. No receipt or a `pending` row returns
`cancellation_unconfirmed`; it is not proof that GPU work stopped. Retain journals
and leases, and inspect recovery state. There is no scheduler-confirmed abort or
automatic recovery of unknown requests. The launcher closes its owned Engine even
when Runtime drain fails, while preserving the failure and unresolved records.

Follow the shared [quiescence protocol](quiescence.md) before stopping the owned
parent; follow its offline resume step before reusing a quiesced registry. Hot
updates apply to decision bundles; plugin/engine code or base weights require a
process restart. The [618-test CPU suite and readout checks](tokenspeed-validation.md)
do not certify a real TokenSpeed GPU deployment, crash recovery or performance.

## Keys, tenants and limits

Read data-plane and administrative credentials from separate environment variables.
For authenticated tenant scheduling, map stable tenant IDs to environment names:

```yaml
tenant_key_envs:
  support: JEV_SUPPORT_API_KEY
  routing: JEV_ROUTING_API_KEY
admission:
  max_requests: 64
  max_tokens: 1048576
  max_queue: 256
  max_tenant_requests: 16
  max_tenant_tokens: 262144
  max_tenant_queue: 64
  max_branches: 1024
  max_tenant_branches: 256
```

The server derives the tenant from the credential; clients cannot choose an
arbitrary tenant name in request bodies. Queues use FIFO within a tenant and
round-robin admission between eligible tenants. Fairness is by admitted request,
not GPU execution time. Tokens and branches count every expanded scoring sequence,
including branches not yet dispatched within an admitted request. This conservative
reservation lasts until the request lease is safely released. Unknown cache hits
do not discount token demand.

Runtime, gateway and native plugins default to one admission pool per exact engine
identity in a shared local registry. API workers atomically reserve the same global
and tenant budgets; worker count does not multiply limits. Every worker must use
identical limits. `/admin/profile` reports scope, limits and current occupancy.
This covers typed decisions, System One requests and leased score collection.
Privileged preparation canaries, the service-key-only raw scoring bridge and
native chat are outside this pool. Use a separate decision engine when isolation
from native chat is required. Separate registry files or differently named engine
identities are separate pools. Multi-node/replica-wide quotas are not implemented;
never place the SQLite database on a network filesystem.

An admitted ticket is deleted in the same transaction as its request lease.
Unconfirmed aborts, failed durable cleanup and dead owners retain their entire
reservation until explicit recovery confirms cancellation. Queued cancellations
and deadlines remove never-dispatched tickets. A verified dead queued owner is
skipped when considering other tenants, but its queue slot and lease still need
recovery. No elapsed-time TTL reclaims potentially live GPU work. Under contention,
workers poll the shared queue every 20 ms; this is not a GPU scheduler or a claim
of latency certification.

If a database failure prevented recording `abort_pending`, the retained lease can
still be `inflight`. Recovery then requires a verified dead owner: stop/restart
the affected API **process** before using the existing recovery procedure. Merely
closing a Runtime object does not prove process death. This is conservative
capacity retention, not automatic recovery from database failure.

Migration from a process-local release requires stopping all old API workers and
draining/recovering their leases before starting the shared-admission release.
New workers reject incompatible live/unknown workers or retained work with
`admission_policy_conflict`. Limit changes likewise require all workers to stop
and all leases to drain. Bundle hot publication is unaffected. The explicit
in-memory `Admission` class remains available for isolated embedding/tests; it is
not the deployment default and cannot join a shared-policy worker group.

## Publication, rollback and removal

1. Build a new bundle version with the loaded checkpoint revision and tokenizer.
2. Upload the immutable manifest. Reusing its ID/version with different contents fails.
3. Prepare it. A real scoring canary must succeed before it receives traffic.
4. Read `/admin/bundles` and activate using the exact current alias generation.
5. To roll back, activate the retained prior version with the **new** generation.
6. Disable an alias to stop new requests. Retire only after its leases have drained.

A generation conflict requires refreshing state and deciding which version should
win. Never blindly retry a stale activation. Existing requests retain their bundle
digest, calibration and policy through a route change. Installing new Python/CUDA
code requires rolling a process; changing a manifest does not reload Python modules.

New bundles include `tokenizer_implementation_digest` when the tokenizer exposes
its backend. This binds normalization, pretokenization and BPE rules, in addition
to the vocabulary and template. The native integrations derive compilation from the
host's actual tokenizer. The verified standard vLLM pool may supply an
identity-checked private copy so optimized encoding never bypasses the shared
pool. A local AutoTokenizer can differ after engine adjustments, so
build from the administrative serving profile when targeting a native plugin:

```sh
jevctl bundle build-remote new-bundle.json \
  --url http://127.0.0.1:30000/plugins/jev-runtime --name decisions --version 2
```

This creates a local manifest without uploading or activating it. It refuses to
overwrite an existing file. `/admin/profile` uses the separate administrative key
and distinguishes tokenizer fingerprint availability from engine-weight identity
verification; the latter is still incomplete. The profile's checkpoint revision
is configured, not proof that every loaded weight matches that revision.

Legacy manifests without the backend fingerprint remain readable with unchanged
digests. A fast-tokenizer runtime refuses to prepare/serve them. During an upgrade,
retain the old process until its requests drain; bring up a separate registry and
new bundle IDs/versions, prepare and switch routing, then retire the old deployment.
Do not rewrite a persisted manifest or reuse an old calibration artifact: the
stronger tokenizer identity changes the scoring contract. A tokenizer without an
exportable backend still has only vocabulary/template checking and must not be
treated as having complete implementation identity coverage.

The registry is local SQLite in WAL mode. Multiple processes can share a local file;
network filesystems and multi-node SQLite are unsupported. Back up the database with
SQLite's online backup API, not by copying a live `.db` without its WAL.
Use the [registry snapshot and guarded staging commands](registry-snapshots.md);
they preserve the complete stopped state and never activate or overwrite a registry.

Each API worker also prepares its own engine path. On restart it revalidates active
versions; a shared database's READY state does not bypass that worker's canary.
Prepare a version on every traffic-serving worker before activation. Inspect
`/admin/workers` to see each local worker's state and prepared versions. Preparation
returns the worker ID; `/ready` and the `X-Jev-Worker` decision response header
identify the handling process. The activation transaction rejects a version until
all serving workers for that backend have prepared its digest. Verified exited or
explicitly stopped workers do not block it; unknown owners fail closed. Startup
joins serving only after its active-route snapshot is rechecked atomically.
Concurrent initial bootstrap waits for global and per-worker preparation. This
coordinates a shared local registry; multi-node rollout remains under development.

Set `workers: 2` for a multi-process standalone gateway. Native engine API worker
counts are controlled by their own engine flags. The vLLM HTTP scoring bridge
currently requires exactly one verified **engine API worker** because upstream
`AsyncLLM.abort` resolves external IDs in that worker's OutputProcessor. A gateway
refuses an unknown or multi-worker vLLM bridge instead of acknowledging a potentially
misrouted cancellation. Native typed endpoints may use multiple engine API workers.

## Cancellation and failure accounting

Clients may set a unique `request_id` before submitting. Cancellation is scoped to
the authenticated tenant. `/v1/requests/{request_id}/cancel` confirms cancellation
only after cleanup; a failed engine abort returns `cancellation_unconfirmed` and
retains the bundle lease. Duplicate in-flight IDs are rejected across workers
sharing the registry. Workers sharing the same local registry route explicit
cancellations through a durable owner-addressed command. Tenant and engine identity
are checked against the persisted lease; its random lease ID prevents a delayed
command from cancelling a later request that reuses the public request ID. Only
the owning worker calls the engine abort and confirms cleanup. A dead/unreachable
owner, failed abort or ten-second control timeout returns `cancellation_unconfirmed`
instead of acknowledging success. Administrative recovery remains explicit.
This covers local API workers, not multiple nodes or a multi-worker vLLM HTTP bridge.
Readiness fails while the local cancellation control loop reports database errors.

Engine branch IDs are opaque internal identifiers, separate from client question
IDs. Each request has a random namespace; each question uses a fixed-length SHA256
suffix and a fixed-width branch index. SGLang matches abort IDs by prefix, so
variable-length concatenation such as `parent.a.1` and `parent.a.1.0` could abort an
unrelated question. The compiler now prevents that relationship among its branches.
Recovery continues using the exact IDs already persisted in a lease; changing code
does not rewrite existing dispatch journals. The SGLang regression uses IDs exported
by the live compiler and confirms that a sibling completes after one branch is
aborted. See [the retained reproduction](../evidence/dsw/sglang-abort-prefix-60b3415-reproduction.json)
and [the fixed GPU check](../evidence/dsw/sglang-prefix-0421d50/abort-prefix.json).

Client disconnects and request deadlines propagate to scoring branches. `partial`
retains per-question errors; unknown or incomplete engine usage is `null`, not zero.
Do not count a partial response as a successful whole request in a load test.

Failed-abort leases require confirming the recorded engine requests have stopped.
Inspect `/admin/requests/recovery`, then retry aborts through
`/admin/requests/{request_id}/recover`. Every engine ID is journaled transactionally
before dispatch. Recovery survives a new API process using the same local registry.
An explicitly `abort_pending` request is eligible; an interrupted in-flight request
requires proof that its original Linux PID/start-tick/boot-ID owner has exited
within the same hostname and PID namespace.
Unverifiable owners and legacy leases without a dispatch journal fail closed.
A changed engine target or failed abort retry preserves the lease.
The engine abort APIs acknowledge request cancellation; this is not a certified GPU
LoRA unload barrier. Adapter lifecycle certification still requires separate checks.
Recovery is an explicit administrative operation, not a time-based lease expiry.
Preparation state, lease creation, and completion each use atomic transactions so
a crash cannot leave a PREPARING version without its recovery journal. If a crashed
bootstrap prevents HTTP startup, use the same config and engine credentials:

```sh
jevctl recovery list config.yaml
jevctl recovery recover config.yaml prepare-REQUEST_ID
```

When native startup cannot expose its recovery endpoint because bundle preparation
is incomplete, use the explicit [recovery-only startup mode](recovery-mode.md).
It keeps Jev scoring/publication closed and requires a normal restart after recovery.

The recovery CLI does not activate routes or clear unknown leases. It contacts the
configured engine and applies the same owner, backend and abort checks as the API.
Unit tests cover cross-instance recovery and fail-closed identity checks. A real
SGLang and vLLM gateway SIGKILL/restart tests and SGLang offline CLI recovery passed;
see `evidence/dsw/gateway-sglang-crash-7eee70e.json` and
`evidence/dsw/gateway-vllm-crash-7eee70e.json`. Each engine survived throughout,
but this does not establish multi-node recovery or a GPU unload barrier.
The shared-admission release also passed explicit quota-retention fault checks on
both gateways (`aba3b52`) and both native API/GPU process groups (`0df49a1`).
Restart preserved an admitted 128-branch request and a queued request; no automatic
capacity reset occurred. Native tests waited for the old group to exit and GPU
memory to return before loading the same checkpoint and engine configuration.
See [fault scenarios, commands and boundaries](fault-recovery.md). Killing and
restarting a process group does not certify GPU hardware faults, a managed-LoRA
restart or database I/O failure.

Unattended recovery is not certified.

## Health, metrics and evidence

`/ready` requires initialized capabilities, prepared active bundles and fresh per-worker
model canaries. Startup/prepare and periodic probes update this evidence; stale or
failed evidence withdraws readiness and blocks new typed dispatch. See
[health configuration](serving-health.md) for scope and recovery. The current
metrics expose decision outcome counts and HTTP latency. Engine usage, canceled
branches and registry state must also be retained in evaluation artifacts.
Enable the `jev_runtime.runtime` logger at INFO to emit `jev_scoring` records
correlating the public request ID, authenticated tenant,
bundle digest/generation and all internal engine IDs, without input text or keys.

Use `tests/integration/live_contract.py` only against task-owned aliases/services.
Pass `--source-commit` for the harness revision and `--runtime-source-commit` for
the deployed server revision, and use a new output path for every attempt.
It creates temporary versions and runs real traffic. The harness uses the actual
serving profile and compiled IDs, accepts legitimate default-policy abstentions,
and separately exercises selected values with an explicit `tie: first` test bundle.
Its reports include strict
successes, native/attach parity, switches and mixed-version failures. It does not
measure a dedicated throughput baseline. `reference_logits.py` separately reports
cross-implementation numerical differences and preserves failures.

The DSW launcher captures PID, Linux start ticks and boot ID before signaling any
test process. A PID mismatch refuses a stop. Confirm both process exit and released
GPU memory before launching a replacement. Existing services are not stopped.
Use `--gpus 5,6` only after a fresh inventory to request ordered TP2 devices;
each device is budget-checked and recorded by UUID. The engine may impose
additional memory-balance/workspace constraints. Wait for native plugin readiness
before running the contract harness. See [TP2 reproduction](phi-tp2-validation.md).


## Runtime metrics

Administrators can inspect `GET /admin/requests/{request_id}/progress` on the
owning worker. It returns `scope: local_worker`, `worker_id`, the pinned bundle
digest/generation/lease and the current phase, plus started/active/succeeded/
failed-or-cancelled counts for scoring calls, branch queueing, assembly and abort.
No prompt, candidate text or per-branch database write is added. An active engine
call is an adapter RPC, not evidence that a particular CUDA kernel is executing.
A successful call means the adapter returned a validated score result; it does
not certify prediction quality.

The endpoint requires the separate admin credential. A null `request` means only
that this worker has no in-memory observation. It does not mean the request is
globally complete or its durable lease is released: another worker can own it,
or an unconfirmed abort can outlive the request task. Use the shared registry
inventory/recovery endpoint for that distinction, and pin an HTTP/1 connection to
the worker ID when polling progress. Observations disappear when that local task
finishes, including failure/cancellation; no history is retained by this endpoint.
Successful cancellation replies additionally identify the handling worker through
`X-Jev-Worker`, consistent with decision responses.

Authenticated `/metrics` exposes observations for both decision and System One
requests after they enter the runtime route. Request counters and histograms are
process-local; admission gauges use the scope described below. Request observations
exclude rejected authentication, body-schema validation, management endpoints and
native engine APIs.

- `jev_runtime_stage_seconds{stage,outcome}`: seven serial phases and inclusive
  `total`; includes durable lease release and failure/cancellation cleanup.
- `jev_branch_work_seconds{kind}`: individual branch queue, engine call, assembly
  and abort durations. Concurrent samples overlap; these are not GPU kernel times.
- `jev_questions_total{outcome}`: returned per-question states, including failed
  questions in a partial response.
- `jev_request_errors_total{category}`: five bounded runtime failure categories.
- `jev_observed_tokens_total{quantity}` and `jev_token_observations_total{quantity}`:
  summed known usage and the matching response observation counts. A known zero
  contributes an observation; unavailable usage contributes neither. These are
  response-level counters, not inference of work discarded by a failed request.
- `jev_shared_admission{quantity}`: admitted requests, reserved expanded tokens and
  branches, and queued requests across this engine's shared registry. Every API
  worker reads the same pool; **do not sum these gauges across workers**. Retained
  uncertain work remains included. The explicit process-local implementation uses
  `jev_admission{quantity}` instead.

No prompt, request ID, arbitrary task ID or tenant name is used as a metric label.
Multiple API workers do not share the request counters or histograms:
load-balanced scrapes are not a complete service aggregate. Use explicit
per-worker collection before reporting global
rates; automatic multi-worker Prometheus aggregation remains outstanding. Use
`X-Jev-Timing: 1` for a request-bound phase header when profiling through a shared
HTTP endpoint. It changes no decision JSON or scoring behavior. See performance
for the exact timing boundaries and missing-observation rules.

## Installed environments and tokenizer upgrades

Use [backend quiescence and ordered shutdown](quiescence.md) before signalling a
native process group. The product handshake preserves aliases, stops canaries on
all local API workers and reports durable blockers; restart requires an explicit
offline resume. Native chat traffic needs its own drain.

Both engines have a scoped [installed-wheel gateway upgrade/rollback report](release-rollout-validation.md).
The successful exercise preserves a registry route while replacing gateway processes;
it includes a maintenance interruption and does not certify uninterrupted routing.
Keep the previous environment and manifest for rollback. Match the full native/gateway
model and tokenizer identity before allowing the candidate to score.

For pinned SmolLM2, use the report's explicit checkpoint-preserving tokenizer recipe
in both native engine and gateway. Default AutoTokenizer changes the number splitting
pipeline in the tested environments. Changing tokenizer identity requires new bundles
and calibration plus process restart; prior numerical, quality, performance and LoRA
evidence is not automatically valid for that new profile.
