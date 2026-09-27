# Process-crash recovery of durable quotas

Four targeted DSW checks passed on SmolLM2-1.7B-Instruct at revision
`31b70e2e869a7173562077fd711b654946d38674`, BF16, TP=1 and one API worker.
They establish that process restart does not erase outstanding admission budgets.
The overall release gate remains unpassed.

| Engine | Fault target | Harness/deployed revision | Result |
|---|---|---|---|
| SGLang 0.5.19 | Standalone gateway; engine stays alive | `aba3b52` | Passed |
| vLLM 0.30.0+cu129 | Standalone gateway; engine stays alive | `aba3b52` | Passed |
| SGLang 0.5.19 | Native API and GPU process group | `0df49a1` | Passed |
| vLLM 0.30.0+cu129 | Native API and GPU process group | `0df49a1` | Passed |

These are sequential, colocated functional checks on GPU7 of the shared L20Z DSW
host. Each run has its own local registry and credentials. The engine environments
remain separate. The source tree for core, packages and deployment is unchanged
from `4944efd`; the two revisions extend only the fault harness. Remote hashes
match 50 source/deployment/harness files at each revision. The previously recorded
220 CPU tests and three wheel builds were not rerun for this harness-only change.

## Assertions exercised

The policy allows one admitted request, 262,144 expanded tokens and 128 expanded
branches globally and for the tenant. The queue allows four requests globally and
two per tenant. A synthetic 128-question request reserves 134,290 tokens and all
128 branches. A second one-question request reserves its queue slot with a demand
of 1,048 tokens and one branch. Both HTTP calls must still be pending immediately
before the fault.

1. Recovery refuses the still-live owner with HTTP 409.
2. SIGKILL targets only the launcher's recorded process group, after verifying its
   Linux PID, start ticks and boot ID. Both pending clients observe transport errors.
3. The exact admitted/queued rows and journaled branch IDs survive. Both old lease
   owners are provably dead. Native mode additionally waits for the complete old
   non-zombie process group to disappear and free GPU memory to reach the observed
   11,990 MiB baseline before restarting.
4. Restart preserves normalized settings, their serialized bytes and all credential
   bytes. Native mode also verifies identical engine command and GPU UUID. Both
   active bundles pass startup canaries while the old capacity remains reserved.
5. The restarted profile still reports one admitted request, one queued request,
   134,290 expanded tokens and 128 branches. Another short request enters the queue
   and reaches its 150 ms deadline with HTTP 504; the original reservations persist.
6. A tenant credential cannot perform administrative recovery (HTTP 401). Recovering
   the old queued request removes only its ticket/lease. The admitted reservation
   remains unchanged until its own explicit recovery confirms cancellation.
7. All leases/tickets drain; a fresh typed Boolean request completes and native chat
   returns generated tokens. The temporary alias is disabled and retired. Independent
   postchecks verify zero tables/gauges and the dead old process group. Final cleanup
   verifies the restarted task-owned groups exit and GPU7 returns to 11,990 MiB free.

The retained decision responses also pass `DecisionResponse` validation, including
exactly one successful Boolean answer. Readiness alone is not the success criterion.
Credentials, model weights and raw service logs are excluded from these artifacts.

## Reproduction

Use a disposable task-owned service created by `deployment/dsw_service.py` with
this admission policy, `--api-workers 1`, and `--tenant alpha`. Do not target a
shared production run directory. The harness refuses to overwrite its output and
preserves settings/tenant mappings on restart. Native mode rejects managed LoRA.

```json
{
  "max_requests": 1,
  "max_tokens": 262144,
  "max_queue": 4,
  "max_tenant_requests": 1,
  "max_tenant_tokens": 262144,
  "max_tenant_queue": 2,
  "max_branches": 128,
  "max_tenant_branches": 128
}
```

Run with the intended engine environment's Python. Here `SOURCE_COMMIT` is the
exact deployed source revision; use a separate `--runtime-source-commit` if the
harness and server differ.

```sh
python tests/integration/live_crash_recovery.py \
  --gateway-run-dir "$GATEWAY_RUN" --engine-run-dir "$ENGINE_RUN" \
  --verify-admission --tenant alpha \
  --source-commit "$SOURCE_COMMIT" --output "$GATEWAY_RUN/quota-crash.json"

python tests/integration/live_crash_recovery.py \
  --native-run-dir "$NATIVE_RUN" --verify-admission --tenant alpha \
  --source-commit "$SOURCE_COMMIT" --output "$NATIVE_RUN/quota-crash.json"
```

The native harness uses the recorded engine memory fraction, resolves the GPU by
its recorded UUID and applies the launcher's reserve check (3,072 MiB by default).
A failure to verify complete exit or restored memory stops the sequence before
replacement launch. After either test, use the pinned launcher to stop the current
recorded process and separately confirm full group exit and GPU memory. The harness
leaves the successful restarted service available for this inspection and cleanup.

## Evidence and remaining scope

The [validation index](../evidence/dsw/quota-crash-validation-0df49a1.json) retains
all four cases, reservation values, process identities and artifact SHA256 hashes.
It points to the raw reports, postchecks, profiles, metrics and cleanup records.
The [gateway source identity](../evidence/dsw/source-identity-aba3b52.json) and
[native source identity](../evidence/dsw/source-identity-0df49a1.json) distinguish
deployed code from later documentation commits. Native environment files record
the imported package paths and actual installed engine/Torch/Transformers versions.

This verifies a full native process-group restart, not a GPU worker-only failure
while the API stays live. It does not cover hardware resets/OOM, failed database
I/O, active LoRA session recovery, multiple nodes, mixed-load fairness, long-run
reliability or performance. Gateway cases keep their original engines running;
native cases restart the engine. Recovery stays an explicit administrator action,
with no time-based release of uncertain work. These checks add no new checkpoint
to the model coverage denominator, which remains 6/40 functional combinations.
