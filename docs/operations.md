# Operating the current development release

Use separate Python environments for SGLang and vLLM. A gateway may attach to an
already-running engine without owning its process. Native plugins share the host
engine's lifetime. Do not use a plugin update to restart an unrelated service.

## Current tested scope

DSW functional checks cover Qwen3-0.6B and SmolLM2-1.7B on SGLang 0.5.19 and vLLM
0.30.0 with CUDA 12.9, BF16, TP=1 and eager execution. Check the evidence files for
the exact source commit: these are development snapshots, not a certification of
every later commit. Config switches have been tested; LoRA weight swaps have not.

For the R550 test host, `deployment/install_sglang_cu129.sh` uses the official
0.5.19 CUDA 12 dependency substitutions and records the source revision and patch.
The plugin itself does not patch the engine. This environment omits optional Rust
extensions. SGLang 0.5.20/CUDA 13 requires a separate compatible-host certification.

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
```

The server derives the tenant from the credential; clients cannot choose an
arbitrary tenant name in request bodies. Queues use FIFO within a tenant and
round-robin admission between eligible tenants. Fairness is by admitted request,
not GPU execution time. Tokens count every expanded scoring sequence. These
limits are per API process; deployment-wide quota enforcement is not yet provided.
The raw scoring bridge is restricted to the separate `JEV_API_KEY` service key.

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

The registry is local SQLite in WAL mode. Multiple processes can share a local file;
network filesystems and multi-node SQLite are unsupported. Back up the database with
SQLite's online backup API, not by copying a live `.db` without its WAL.

## Cancellation and failure accounting

Clients may set a unique `request_id` before submitting. Cancellation is scoped to
the authenticated tenant. `/v1/requests/{request_id}/cancel` confirms cancellation
only after cleanup; a failed engine abort returns `cancellation_unconfirmed` and
retains the bundle lease. Duplicate in-flight IDs are rejected across workers
sharing the registry. Cancellation lookup currently belongs to the handling API
process; use sticky routing for an explicit cancellation call in a multi-worker setup.

Client disconnects and request deadlines propagate to scoring branches. `partial`
retains per-question errors; unknown or incomplete engine usage is `null`, not zero.
Do not count a partial response as a successful whole request in a load test.

Failed-abort leases require confirming the recorded engine requests have stopped.
Automatic orphan recovery across a worker crash remains unfinished. Never remove a
lease merely because it is old. This limitation blocks an unattended production SLA.

## Health, metrics and evidence

`/ready` requires initialized capabilities and an active alias. Startup/prepare
executes a canary; readiness is not yet a periodic model liveness canary. The current
metrics expose decision outcome counts and HTTP latency. Engine usage, canceled
branches and registry state must also be retained in evaluation artifacts.

Use `tests/integration/live_contract.py` only against task-owned aliases/services:
it creates temporary versions and runs real traffic. Its reports include strict
successes, native/attach parity, switches and mixed-version failures. It does not
measure a dedicated throughput baseline. `reference_logits.py` separately reports
cross-implementation numerical differences and preserves failures.

The DSW launcher captures PID, Linux start ticks and boot ID before signaling any
test process. A PID mismatch refuses a stop. Confirm both process exit and released
GPU memory before launching a replacement. Existing services are not stopped.
