# Managed LoRA lifecycle

Install `jev-runtime-core[lora]` before enabling this optional feature. The base
text-serving install intentionally does not include the safetensors reader.

Managed LoRA is opt-in and initially limited to native plugins, BF16, no
quantization, TP/PP/DP=1 and one API/tokenizer worker. The exposed checkpoint is
`HuggingFaceTB/SmolLM2-1.7B-Instruct` at revision
`31b70e2e869a7173562077fd711b654946d38674`, using `LlamaForCausalLM` on SGLang
0.5.19 or vLLM 0.30.0. Startup rejects other checkpoint revisions for this optional
feature; ordinary base-model scoring has its separate, broader compatibility path.
The GPU lifecycle evidence uses an L20Z and untrained synthetic adapters. It is
not business-quality or overall release certification. HTTP gateway LoRA is
intentionally unavailable. See [the tested profiles](../profiles/lora-lifecycle.json).

Add to the Jev configuration (server-local absolute paths):

```yaml
workers: 1
adapters:
  enabled: true
  allowed_roots: [/srv/approved-loras]
  store_path: /srv/jev/adapters
  max_bytes: 268435456
  operation_timeout_seconds: 120
```

Start vLLM with `--enable-lora`, enough `--max-cpu-loras` for all pinned adapters,
and `--worker-extension-cls jev_vllm.worker.LoRAWorkerExtension`. Set
`--max-lora-rank` to accommodate the artifacts. SGLang also needs `--enable-lora`,
appropriate rank/capacity/target-module options and the Jev plugin. Its official
scheduler hooks add a GPU completion fence and return a marker that the coordinator
requires before publishing a successful update. Keep native engine administration
restricted to the service operator; do not mutate Jev's opaque adapter names/IDs
through a second LoRA control plane.

SGLang reserves one memory-pool slot for base/unpinned traffic: two pinned adapters
need `--max-loras-per-batch` of at least 3, regardless of `--max-loaded-loras`.
The DSW functional launcher uses 4. Capacity is checked before native dispatch.

Registration accepts `adapter_config.json` and `adapter_model.safetensors` from an
allowed local directory. It copies, validates, hashes and fsyncs them into owned
content-addressed storage. No Python code, remote download, pickle, embedding or
LM-head replacement is accepted. The initial PEFT format covers paired dense
q/k/v/o/gate/up/down projection matrices with uniform rank 1–64, no bias, DoRA,
RSLoRA or modules-to-save. Artifact identity includes the configured immutable base
model revision. This binding does not independently prove the engine's loaded
weight-file hash.

## Publish and remove

Set `JEV_ADMIN_KEY` in the environment. For native plugins, `--url` includes
`/plugins/jev-runtime`. `SOURCE` is a directory on the server, not the CLI host.

```sh
jevctl adapter register refunds /srv/approved-loras/refunds --url "$JEV_URL"
jevctl adapter load 'refunds@sha256:ARTIFACT_DIGEST' --url "$JEV_URL"
jevctl bundle build-remote refunds.json --name refunds --version 1 \
  --adapter 'refunds@sha256:ARTIFACT_DIGEST' --url "$JEV_URL"
jevctl bundle upload refunds.json --url "$JEV_URL"
jevctl bundle prepare refunds@1 --url "$JEV_URL"
jevctl bundle activate refunds@1 refunds 0 --url "$JEV_URL"
```

Load and prepare a different immutable adapter/bundle before switching the alias
with its current generation. Existing requests retain their old bundle and adapter.
Disable every alias pointing to the old adapter, or move those aliases to another
prepared bundle. Then call `jevctl adapter unload REFERENCE --url "$JEV_URL"`.
The server returns `adapter_in_use` while any referencing route or lease exists;
it never forcibly clears them. Poll `/admin/bundles` for drain or explicitly cancel
the request. A failed abort retains its lease and must be recovered first.

Lifecycle operations persist `LOADING`/`UNLOADING` before GPU calls. Successful
load becomes `READY`; successful unload becomes `UNLOADED` and invalidates every
referencing bundle's canary. Failure, cancellation or timeout becomes `UNKNOWN`.
Reload requires a confirmed unload/reconciliation first. Raw `/v1/scores` rejects
adapter IDs, so it cannot bypass the typed endpoint's leases. Unload/reload does
not delete immutable files; garbage collection is not yet implemented.

## Restart and uncertainty

On a new managed engine session, an alive or unverifiable old coordinator blocks
startup. Once the old worker is verified dead or gracefully stopped, previous
`READY`/in-progress adapters are quarantined and their aliases paused, advancing
route generations. Durable request leases remain. Base-model routes can still
serve. Use the admin recovery API to settle eligible old requests, unload the
uncertain adapters, load them again, prepare new canaries, then explicitly
reactivate with the refreshed generations. SQLite `READY` alone is never treated
as proof of GPU residency after restart.

vLLM unload checks residency around fixed worker RPC GPU fences. SGLang unregisters
the name, waits its engine usage counter, unloads, then returns the fenced reply.
If an interrupted SGLang operation leaves a missing frontend name with uncertain
backend residency, it returns `adapter_restart_required`: restart the complete
isolated engine, then reconcile. Do not restart only its frontend. The `--recover`
CLI flag can take over an in-progress operation only from a verified dead owner;
it cannot override active routes, leases or a live coordinator.

Native SGLang scoring keeps its response receiver alive when the client cancels.
It sends abort and waits up to 4.5 seconds for terminal output before confirming
drain. Closing the generator first would discard upstream request state and lose
the LoRA usage-counter release. A timeout retains the runtime's recovery lease.

## Validation

`tests/integration/lora_fixture.py` creates two deterministic nonzero, untrained
LoRA artifacts for a dense Llama checkpoint. `live_lora.py` checks distinct base/A/B
outputs, 24 alternating cache-sensitive switches, a 128-question old request
surviving publication, rejected premature unload, cancellation/drain and six
reload/rollback cycles. It retains a report on failures. These tests establish
lifecycle behavior for the tested configuration; they do not establish business
accuracy, LoRA training quality or controlled performance.

`live_lora_crash.py` also passed on both native engines with this same frozen
profile: it kills the recorded complete engine process group during a 128-branch
adapter request, verifies retained quotas and quarantined residency after restart,
then requires explicit recovery, new canaries and current route generations.
Both first and warm reload outputs matched their pre-crash baselines, followed by
24 switches and a separate full lifecycle recheck. SGLang required a version-scoped
host-logprob-row compatibility hook; the initial failed attempt remains preserved.
See [the crash/recovery report](lora-crash-recovery.md) for source identities,
cleanup, evidence and remaining fault/configuration coverage.
