# Managed LoRA crash recovery and SGLang coexistence

Both native engines passed a real process-group crash/restart while a typed
request held a READY LoRA adapter. Restart preserved its lease and quota,
quarantined adapter residency, paused adapter routes, and required explicit
reconciliation, new canaries and fresh route generations before serving again.
These are development checks of one frozen profile, not release certification.

## Tested identity and results

The model was `HuggingFaceTB/SmolLM2-1.7B-Instruct`, revision
`31b70e2e869a7173562077fd711b654946d38674`, BF16, no quantization,
TP/PP/DP=1, one API/tokenizer worker and eager execution. Both runs used the same
two deterministic, nonzero, untrained rank-8 `o_proj` LoRA fixtures on a colocated
L20Z. Fixture hashes are in each report and [the profiles](../profiles/lora-lifecycle.json).

| Check | vLLM | SGLang |
|---|---|---|
| Engine | 0.30.0+cu129 | 0.5.19 |
| Runtime source | `5176a82` | `533549a` |
| Crash/recovery checks | 28 passed | 28 passed |
| Retained request | 128 branches / 133,376 reserved expanded tokens | Same |
| First and warm output error after reload | 0 against the corresponding pre-crash baseline | 0 |
| Post-restart base/A/B switches | 24, zero probability error | 24, zero probability error |
| Observed cached prompt tokens per switch | 80 / 91 logical tokens | 90 / 91 logical tokens |
| Separate lifecycle recheck | 24 switches, 6 reload/rollback cycles | Same |
| Passing registry after cleanup | 4 adapters UNLOADED, zero leases/tickets | Same |
| GPU free memory after full group exit | 11,990 MiB | 11,990 MiB |

Core, vLLM and deployment sources are identical between these two revisions.
The SGLang plugin changed to fix the incompatibility below. Remote SHA256 checks
cover [53 files at 5176a82](../evidence/dsw/source-identity-5176a82.json) and
[55 files at 533549a](../evidence/dsw/source-identity-533549a.json).

The [validation summary](../evidence/dsw/lora-crash-validation-533549a.json)
includes hashes of the underlying reports. Full attempts are retained for
[vLLM](../evidence/dsw/vllm-lora-crash-5176a82/lora-crash.json) and
[SGLang](../evidence/dsw/sglang-lora-crash-533549a/lora-crash.json), with
environment, process, postcheck, metrics and cleanup records alongside them.

## Recovery assertions

The harness starts with distinct base/A/B output distributions and immutable
adapter bindings. It journals all 128 branch IDs before killing the exact recorded
native API/GPU process group. A live-owner recovery attempt is rejected first.
After SIGKILL, it verifies whole-group exit and GPU memory return before restarting
the same command, GPU, configuration, credential bytes and admission limits.

On restart, both old READY adapters become UNKNOWN with `engine_session_changed`.
Their routes become paused and their generations advance; their bundles require
preparation again. The base route remains unchanged and serves while the old
adapter request retains its full reservation. Paused serving, premature loading,
preparation and activation are rejected with their respective state errors.

`unload --recover` cannot bypass the active adapter's old request lease. The idle
adapter can be reconciled without releasing that other adapter's reservation.
Explicit request recovery then drains the old lease and ticket. Reload keeps the
immutable binding identity, but activation still requires a successful new canary
and the current route generation. Both the first measured response after reload
and a subsequent warm response match their pre-crash counterparts. The first
response is not a cold-cache claim: preparation canaries may have populated cache.

After 24 cache-hit-observed switches, the harness disables its aliases, unloads
both adapters and retires the test bundles. A separate `live_lora.py` recheck on
that restarted service exercises an old 128-question request across publication,
rejected premature unload, cancellation followed by fenced removal, six
reload/rollback cycles and refusal of raw adapter-ID bypass. Final passing-run
registries contain no leases or quota tickets, and native chat still serves.
All task-owned engine process groups were then stopped and their exit verified.

## Preserved SGLang failure and compatibility fix

The [first SGLang attempt at 5176a82](../evidence/dsw/sglang-lora-crash-5176a82/lora-crash.json)
failed during adapter preparation, before the harness sent SIGKILL. The engine
raised `AttributeError: 'list' object has no attribute 'tolist'` while normalizing
selected-token logprobs. Startup native generation was still active. The
[diagnostics](../evidence/dsw/sglang-lora-crash-5176a82/failure-diagnostics.json)
preserve the stack, engine source identity and failed registry state; one old
preparation lease remains in that stopped attempt's registry. It is not counted
as a successful recovery or silently removed to make the evidence appear clean.

In SGLang 0.5.19, the upstream logprob producer returns a tensor for a request
with selected token IDs and a Python `[]` for a request without them. Its output
type permits both tensors and lists. Two consumers nevertheless call `.tolist()`
on every selected-logprob row. This fails on host-list rows in mixed traffic.

The plugin registers official BEFORE hooks on `move_logprobs_to_cpu` and
`_normalize_decode_outputs` for version 0.5.19 only (including build suffixes).
A plain list row gets a list subclass exposing `.tolist()`; the original consumer
then proceeds. Values, token-ID rows and tensor objects are preserved. The hook
does not allocate a GPU tensor, cast precision or synthesize scores, and does not
modify installed SGLang source files. Other engine versions do not receive this
compatibility hook and require their own validation.

The [installed-upstream CPU reproduction](../evidence/dsw/sglang-logprob-rows-533549a.json)
uses the actual producer and both consumers from source revision
`0bcd822377da7b5718e674eaf9c870d349424dd1`: both fail without the hook and preserve
exact float64 values and empty rows with it. The full LoRA crash/restart suite
then passed on GPU at `533549a`.

A separate [concurrent serving check](../evidence/dsw/sglang-lora-crash-533549a/mixed-native-typed.json)
completed a 512-token native SSE generation with a terminal DONE and 12/12 typed
requests. All 12 typed requests started and completed within the native stream's
lifetime. This establishes overlapping request lifetimes, not an observed
scheduler batch composition or a mixed-load performance certification.

## Reproduction

Use an isolated engine environment, a fresh run directory and the exact model
revision/fixtures. `deployment/dsw_service.py launch` must enable managed adapters
and point `--adapters-root` at the immutable fixture root. The recorded admission
limits allow two requests, 262,144 tokens and 129 branches globally and per tenant,
with global queue size four and tenant queue size two. Keep at least 3,072 MiB
free-memory reserve at launch. Set these shell variables to the chosen immutable
release, its Python environment, fresh run directory and approved fixture root:

```sh
"$JEV_PYTHON" "$JEV_RELEASE/tests/integration/live_lora_crash.py" \
  --run-dir "$JEV_RUN" --fixtures "$JEV_FIXTURES" \
  --output "$JEV_RUN/lora-crash.json" --source-commit "$JEV_COMMIT" \
  --reserve-mib 3072
"$JEV_PYTHON" "$JEV_RELEASE/tests/integration/live_lora.py" \
  --run-dir "$JEV_RUN" --fixtures "$JEV_FIXTURES" \
  --output "$JEV_RUN/lifecycle.json" --source-commit "$JEV_COMMIT"
```

For SGLang, run `sglang_logprob_rows.py` in its installed engine environment and
`live_sglang_mixed.py --run-dir ... --source-commit ... --output ...` against the
restarted service. Reports refuse to overwrite earlier attempts. The crash harness
leaves the restarted service running for postchecks; stop the recorded process
group through the launcher afterward and independently verify that every member
exited. A successful stop command alone is not full GPU-worker cleanup evidence.

## Remaining scope

These checks cover a full native process-group crash during scoring with READY
adapters. They do not cover interruption during GPU load/unload, worker-only or
hardware faults, database I/O failure, other checkpoints/dtypes/parallelism,
CUDA-graph execution, multi-node recovery, task quality, a controlled load matrix
or a 24-hour soak. Synthetic adapter output separation is a lifecycle test, not
evidence of learned business accuracy. The functional model denominator remains
6/40 and the performance/quality release gates remain unpassed.

At `533549a`, local validation passed 225 Python tests, Ruff lint/format and all
three wheel builds with byte-for-byte Python source verification. TypeScript was
unchanged; its earlier 62 passing tests were not rerun. See
[package evidence](../evidence/package-check-533549a.json). Hosted CI is tracked
separately from these local and DSW checks.
