# Performance experiments

`profiles/performance-matrix.json` freezes 144 scenarios using Qwen3-8B and
Phi-4-mini, two engines, L=256/2048/8192, K=2/8/32, concurrency=1/16 and cold/hot
cache states. All entries are currently **not run**. The 21.6 hours is a lower
bound **per applicable measurement method**; baselines, loading, warmup and the
24-hour soak add time. Small-model development runs do not fill this matrix.

The paired runner currently covers native selected-label HTTP scoring and the
configured typed plugin or gateway. For independent-candidate mode it dispatches
the native branches serially and records that distinction. Optional structured
JSON generation is also implemented for the joint-label choice fixture. Other Jev
implementations, upstream decisions, open-loop saturation sweeps and full GPU/CPU
telemetry collection remain separate outstanding work.

The first colocated SmolLM2 development runs at runtime `5c4fcbe` and runner
`ebec8b3` completed on both engines. SGLang recorded 202/202 native and 132/132
typed successes; vLLM recorded 228/228 and 144/144. Each method ran for only five
seconds. These expose overhead and validate the runner, not the release gates.
SGLang's initial pre-warmup parity failure is retained; after both paths were
warmed, their selected-label probabilities matched exactly at the unchanged
1e-4 tolerance. Cold/cache numerical differences remain a separate investigation.

The compiler now caches exact rendered-prompt/label encodings within each worker,
bounded by `compiler_cache_tokens` (262144 token IDs) and `compiler_cache_entries`
(256 entries). Set either to zero to disable it. Entries retain token IDs and a
prompt digest, not raw text. Changed input, labels or template cause misses;
request IDs, bundle limits and routing are checked afresh. Repeated-prompt results
must therefore be reported separately from varying-input/cache-disabled tests.
The registry reuses one serialized connection per owner while keeping WAL and
`synchronous=FULL`. This avoids per-transaction connection and final-WAL-close
overhead without removing durable dispatch journals or reducing crash guarantees.

## Exact inputs and denominators

The administrator-only `/admin/compile` endpoint pins the active bundle and uses
the actual serving worker's compiler. It exports input IDs, selected label IDs,
question order, bundle digest, generation, input digest and the full tokenizer
implementation digest when a fast tokenizer exposes it. It does not submit GPU
work. This avoids substituting a similarly named local tokenizer in a baseline.
New fast-tokenizer bundles bind the implementation fingerprint as well as the
vocabulary fingerprint. An old manifest retains its original digest but must be
rebuilt as a new version before serving with the stricter runtime. See operations.

`benchmarks/run_case.py` uses this endpoint to build an exact context, rejects a
context that cannot fit, and records every branch length. Before measurement it
compares native and typed probabilities against a fixed tolerance. It does not
silently relax the tolerance or change scoring semantics after failure. Synthetic
padding is a throughput fixture, not a quality dataset.

Cache mode must match the reported engine capability. Cold means engine prefix
caching is disabled, including candidate reuse. Hot means an identical prompt is
warmed before each method; actual cached-token coverage is reported separately.
An unknown cache setting fails setup. No cache flush is issued to a shared engine.

Each repeat alternates which method runs first. Warmup is excluded. Dispatch stops
at the prescribed window and outstanding calls drain before the next method. The
report distinguishes strict successes over the full cohort makespan from successes
completed inside the measurement window. A partial result never counts as a fully
successful request. HTTP and response-contract failures remain in the denominator.
Cache ratios use summed observed tokens; missing observations remain unknown.
P99 is omitted below 10,000 successful samples and is descriptive even above that
threshold. The runner never marks the release gate passed.

## Running a development measurement

Use only a task-owned service launched with `deployment/dsw_service.py`. The
run directory contains the config and local credential files; credentials are
not copied into the evidence. The matching engine must already be running with
a sufficient context, logprob limit and the requested cache configuration.

```sh
python benchmarks/run_case.py \
  --run-dir /absolute/task-owned/run \
  --output /absolute/new/evidence-directory \
  --source-commit FULL_COMMIT_SHA \
  --context-tokens 256 --candidates 8 --concurrency 1 --cache hot \
  --duration 180 --repeats 3 --warmup 32
```

For a gateway, also supply `--engine-run-dir` for the original task-owned engine's
credential. `--scoring-mode independent-candidate` measures a different scoring
contract and must be reported separately. A short `--duration` is useful for
harness validation; it does not satisfy the release matrix.

The output directory retains the fixture, per-attempt JSONL, summaries and setup,
parity or cleanup failures. The temporary benchmark alias is disabled and retired
when its work drains. A failed retirement is recorded and never forces deletion of
a retained lease. Do not mix measurements from different source commits, engine
flags, checkpoints or background-load conditions into a single performance claim.

## Structured generation comparison

Add `--structured-generation --generation-max-tokens 64` to the runner command.
This opt-in lane uses the same task, ordered candidate IDs/descriptions and input
text, with a JSON Schema constraining the output to `{"choice": "candidate_id"}`.
The benchmark bundle explicitly sets `tie: first` for selected-choice comparisons;
the serving default remains `tie: abstain`. Independent-candidate mode cannot be
combined with this generation lane.

The label fixture retains its exact requested L. The generation prompt has a
different instruction and JSON format, so its actual length is reported separately,
not forced to L by changing the shared input text. Before timing, the runner asks
the same native engine to return actual prompt/output token IDs and checks their
lengths against its usage report. The fixture preserves these IDs, the exact HTTP
request body, task/input digest, and serving tokenizer implementation digest.
Timed generation requests omit large token-ID arrays and check the fixed prompt
count. Grammar compilation/preflight and per-method warmups are outside timing.
Three methods rotate first position across three repeats.

Every timed generation request must complete with `finish_reason: stop`, one
assistant choice, exactly one JSON key, and a candidate from the frozen set.
Duplicate keys, malformed JSON, tool calls and truncation fail even with HTTP 200;
a parseable JSON object at `finish_reason: length` still fails. Missing usage is
not fabricated from content length. Trustworthy generated-token/byte observations
remain attached to failed attempts, and summaries report all-attempt and
strict-success cost denominators separately. Native/typed scoring also record
observed completion tokens. Cached-token observations remain optional and unknown
when the engine does not supply them.

The report counts selections matching the label-score reference separately from
HTTP/contract success. That agreement is not accuracy against labels. This fixture
is synthetic padding: a faster JSON or typed path does not establish equivalent
business quality, calibrated confidence or an overall product speedup. The held-out
quality gate and controlled 144-scenario performance gate remain outstanding.

## Recorded three-method development check

At harness `444fd1b` and runtime `60b3415`, both native engines ran the same frozen
SmolLM2-1.7B checkpoint on the shared DSW L20Z GPU7 (BF16, TP1, eager,
context limit 2048). The scenario was joint L=256/K=8/C=1 with a hot repeated
prompt, two warmups and five seconds per method per repeat, for three repeats.
Generation used the same task/input, an actual 299-token prompt and a 64-token
output budget. Both preflights retained actual input/output token IDs. All JSON
responses used eight completion tokens; the native selected-label lane reported
zero completion tokens on SGLang and one on vLLM. These are observed engine
counters, not interchangeable estimates of computation.

| Engine | Method | Strict successes / attempts | Full-cohort RPS, three repeats | P95 ms, three repeats | Observed completion tokens, all attempts |
|---|---|---:|---|---|---:|
| sglang | native-label | 628/628 | 42.61, 40.73, 41.89 | 27.76, 27.91, 27.92 | 0 |
| sglang | native-plugin | 554/554 | 35.29, 38.64, 36.57 | 34.28, 29.88, 30.77 | 0 |
| sglang | structured-generation | 86/86 | 5.41, 5.62, 5.65 | 217.44, 204.63, 206.24 | 688 |
| vllm | native-label | 739/739 | 49.57, 49.07, 48.90 | 23.18, 23.26, 23.23 | 739 |
| vllm | native-plugin | 654/654 | 44.13, 42.44, 43.89 | 25.48, 27.83, 26.05 | 654 |
| vllm | structured-generation | 109/109 | 7.08, 7.40, 7.15 | 170.63, 155.28, 159.34 | 872 |

Native/typed probability parity error was zero. The label reference selected `c1`;
all 86 SGLang and 109 vLLM timed JSON responses selected `c0`. This is zero
agreement with that reference, not zero accuracy: the padding fixture has no gold
label. No equivalent-quality speedup can be concluded. Plugin/native throughput
also remains below 90% in multiple repeats. No P99, independent capacity claim,
confidence interval or controlled release-gate pass follows from this short run.

The SGLang negative preflight with `max_tokens=1` returned HTTP 200, one completion
token and `finish_reason: length`; the harness correctly failed it as
`generation_truncated` and retained its observed output cost. It dispatched no timed
cohort and retired its temporary alias. The original failure is preserved alongside
an explicit expected-negative check. Both task-owned engine process groups exited;
GPU7 returned to 11,990 MiB free. Other services were not changed.

Evidence: [SGLang](../evidence/dsw/sglang-generation-444fd1b/report.json),
[vLLM](../evidence/dsw/vllm-generation-444fd1b/report.json), and
[truncation negative check](../evidence/dsw/sglang-generation-truncated-444fd1b/report.json).
The native candidate/other Jev baselines, controlled matrix, representative
varying-input workloads and business-quality equivalence remain outstanding.

## Runtime phase profiling

Use `--runtime-timing` to request per-response `Server-Timing` observations. The
seven serial phases partition the runtime's `total`, including durable cleanup:

| Phase | Measured boundary |
|---|---|
| pin | Resolve route and acquire the persisted bundle lease |
| compile | Validate bundle/request and render/tokenize/validate scoring sequences |
| journal | Emit the correlation log; the in-memory admission fallback also persists branch IDs here |
| queue | Atomically persist branch IDs and shared admission capacity, then wait when needed |
| execute | Schedule branches, await engine results, assemble answers and drain child tasks |
| finalize | Aggregate question outcomes/usage and construct the response contract |
| release | Persist lease completion or uncertain-abort state and clear local ownership |

`total` excludes HTTP body parsing, authorization, response serialization, network
transport and metrics publication. Client end-to-end latency remains the throughput
and overhead comparison boundary. `execute` is engine queue/compute/adapter wall
time plus local scheduling/assembly, not a GPU kernel timer. Separate Prometheus
branch-operation samples cover semaphore waits, engine calls, answer assembly and
abort; samples can overlap and cannot be summed into wall time.

The shared admission path combines its recovery journal and reservation in one
`synchronous=FULL` transaction. A normal, immediately admitted request now commits
three transactions: pin the bundle lease, journal/reserve, and release the lease.
Previously journal and reservation committed separately, for four transactions.
Waiting requests still use admission polling transactions. Canary preparation and
the in-memory admission fallback retain their independent journaling path.

The combined commit preserves the dispatch boundary: another connection sees
neither write before commit and both writes afterward. A failure in either write
rolls back both. An admitted ticket still survives context exit and uncertain
aborts, until confirmed completion/recovery releases its lease. WAL and FULL
durability are unchanged. Compilation stays outside the write transaction.

Historical reports before this change assign journal persistence to `journal`;
new shared-admission reports include it in `queue`. Compare `journal + queue`
or total/end-to-end latency across these versions, not either phase alone.

The header is opt-in (`X-Jev-Timing: 1`) and adds no response JSON fields. Handled
runtime errors include timings for phases actually reached. Authentication/schema
rejections before runtime have no runtime observation. The benchmark rejects a
missing/invalid requested header rather than inventing zero. Each attempt retains
its observed stages; summary statistics expose stage-specific observation counts
for all attempts and strict successes separately. Stage percentiles are descriptive
and must not be added. Independent-candidate work may have overlapping branches;
its serial runtime phases still partition the request's observed total.

`--input-variants N` (1–1024, default 1) extends the native/typed scoring comparison
to a frozen round-robin pool. Every input is independently padded to the exact
requested length using the serving compiler; every branch length must match the
first fixture. The pool preserves all actual token IDs, distinct request texts and
digests. All variants must pass numeric parity after an explicit per-variant
rewarm. Each timed method/repeat starts at variant 0; at least one full traversal
warms the pool beforehand. Per-attempt fixture indices and observed counts expose
unequal distributions when methods dispatch at different rates. Missing indices
on failed attempts remain unknown.

With multiple inputs, `hot` means prefix caching is enabled and the pool is warmed;
it does not guarantee that the entire pool fits in KV or the compiler encoding
cache. Report actual token hits and configured capacity. A pool larger than the
compiler's 256-entry bound exercises encoding misses during sequential traversal.
This pool consists of synthetic variants, not representative business prompts.
Multi-input structured generation is explicitly rejected until its per-input
native token traces and output contracts are implemented.

## Full-context encoding optimization

The compiler can use the Rust tokenizer's `encode_batch_fast` for continuation
IDs when the tokenizer uses the inspected standard `PreTrainedTokenizerBase.encode`
and `TokenizersBackend._encode_plus` methods. The base prompt still goes through
its normal wrapper first. Padding, truncation and special-token splitting settings
must match before using the backend path. Every full `prompt + label` is encoded;
every result must retain the complete prompt prefix and add exactly one token.
Only unused offset mapping is omitted. At most 32 candidate texts are copied per
batch. Unknown/custom encode methods retain the original wrapper path.

vLLM 0.30's host tokenizer pool is never bypassed on its shared backend. For a
standard pool prototype, the endpoint plugin creates an independent deep copy
from the host's pickle-reduction protocol, verifies backend serialization,
vocabulary, template and special-token identity, then uses that private object
for synchronous compiler calls. Unsupported reductions or changed identity keep
the original pooled API. This adds a CPU tokenizer copy per API worker; its memory
cost is not yet a certified capacity result. It never reloads a tokenizer from a
repository name. The admin profile records concrete method implementations,
encoding eligibility, cache bounds and whether a host-pool copy was created.

The initial HF wrapper-batch experiment (`9fb3e8f`) matched all 300 frozen inputs
but showed no consistent CPU speedup; that generic batch path was removed. The
IDs-only implementation at `8a126a8` matched all 300 serving fixtures and 12
additional synthetic Unicode/whitespace/special-token/control-character cases.
Its three CPU mean compile times were 2.34/2.36/2.30 ms versus the original
3.13/3.32/3.13 ms with encoding caching disabled and tokenizer parallelism disabled.
These CPU checks alone establish no serving speedup. The initial live vLLM
eligibility check correctly selected the pooled fallback; the private-copy
integration at `c11d2b3` enables the path only after identity verification.


## Recorded multi-input profiling check

The colocated vLLM SmolLM2 profile used the same 300 exact L=256/K=8 inputs at
concurrency 1, BF16/TP1/eager, prefix caching enabled, and three ten-second timed
repeats per method. The input pool exceeds the 256-entry compiler cache. Actual
typed token-weighted KV hits were 18.75%, so this is not an all-cache-hit workload.
All 300 input/label-ID sequences match across versions and passed per-variant
native/typed probability parity. The comparison artifact records a canonical
workload hash separately from run-specific aliases and bundle digests.

| Runtime / method | Strict successes / attempts | RPS, repeats 1/2/3 | P95 ms, repeats 1/2/3 | Mean compile ms, repeats 1/2/3 |
|---|---:|---|---|---|
| da2293b before / native-label | 1418/1418 | 47.68, 47.33, 46.64 | 24.19, 24.00, 24.40 | not observed |
| da2293b before / native-plugin | 1052/1052 | 34.95, 35.00, 35.05 | 32.43, 32.71, 32.22 | 3.86, 3.91, 3.85 |
| c11d2b3 after / native-label | 1449/1449 | 47.73, 48.49, 48.53 | 24.02, 23.77, 23.49 | not observed |
| c11d2b3 after / native-plugin | 1159/1159 | 38.04, 38.94, 38.75 | 30.58, 29.43, 30.06 | 1.90, 1.81, 1.78 |

The observed compilation reduction did not satisfy the native-overhead gate:
plugin/native throughput remained about 80% after the change. Native throughput
also varied across the sequential runs; no isolated causal speedup or confidence
interval is claimed. Fixed-duration methods have different request counts and
per-variant frequencies, all retained in JSONL. The earlier one-input warm run
had roughly 0.22–0.23 ms compilation and 1.4 ms combined pin/journal/release work;
its cache behavior and results must not be substituted for this varying-input run.

Evidence: [before](../evidence/dsw/vllm-input300-11791e0/report.json),
[after](../evidence/dsw/vllm-input300-c11d2b3/report.json),
[exact-input comparison](../evidence/dsw/vllm-encoding-comparison-c11d2b3.json),
[one-input phase profile](../evidence/dsw/vllm-timing-da2293b/report.json), and
[CPU encoding checks](../evidence/dsw/compiler-ids-8a126a8.json).

SGLang at the same `c11d2b3` source completed the same 300 exact prompt/label-ID
fixtures and three ten-second repeats. It uses the standard host tokenizer
directly; the private host-pool copy applies only to vLLM. Every input passed
within-engine native/typed parity with zero error. This does not establish
cross-engine numeric parity. All 2,078 timed attempts succeeded:

| Method | Strict successes / attempts | RPS, repeats 1/2/3 | P95 ms, repeats 1/2/3 | Mean compile ms, repeats 1/2/3 |
|---|---:|---|---|---|
| native-label | 1155/1155 | 39.54, 38.55, 37.22 | 31.14, 31.13, 32.47 | not observed |
| native-plugin | 923/923 | 30.06, 31.10, 30.95 | 38.81, 37.74, 37.44 | 2.41, 2.37, 2.28 |

SGLang plugin/native throughput ratios were 76.0%, 80.7% and 83.2%, below the
90% target. Its actual token-weighted cache ratio was about 20.1% for typed
requests and 20.1% for native requests; both denominators are observed. This
differs from vLLM's cache behavior. No SGLang before/after encoding speedup is
claimed because the matching pre-change varying-input GPU cohort was not run.

Both engines also passed the full functional suite and 1,000 route switches at
`c11d2b3`: SGLang served 548 requests and vLLM served 469 during switching, with
no mixed versions. All owned process groups exited and GPU7 returned to the
11,990 MiB free-memory baseline. The raw JSONL summaries were independently
recomputed, including per-variant frequencies and timing observation counts.
These checks do not recertify previous LoRA/multi-worker profiles or the release
matrix. See the [SGLang report](../evidence/dsw/sglang-input300-c11d2b3/report.json)
and [two-engine evidence validation](../evidence/dsw/input300-validation-c11d2b3.json).

## Shared-admission two-worker checkpoint

At `4944efd`, both native engines used two API workers and a single shared local
registry with durable request/token/branch reservations. The functional check
holds a 128-branch alpha-tenant request on one worker, observes another alpha
request queued on its peer, rejects a further alpha request with 429, serves an
eligible beta request, enforces tenant-bound cancellation, and verifies queued
cancellation/deadline cleanup. A beta request with two branches remains queued
while the shared 129-branch budget has 128 reserved. Both workers report the same
shared gauges. Peer cancellation drains the long request, subsequent serving
succeeds and retirement leaves zero leases and admission tickets. Both engines
passed these checks. This is a targeted functional load, not representative
fairness/performance certification or a GPU crash-restart experiment.

A separate short timing check then used one warm L=256/K=8 input, concurrency 1,
BF16/TP1/eager, with three five-second repeats per method. Neither method's timed
cohort includes the preceding quota/cancellation cases. Native/typed probability
parity error was zero, and every timed attempt succeeded:

| Engine / method | Successes / attempts | RPS, repeats 1/2/3 | P95 ms, repeats 1/2/3 |
|---|---:|---|---|
| vLLM native-label | 712/712 | 47.12, 46.96, 48.18 | 24.54, 24.35, 23.67 |
| vLLM native-plugin | 612/612 | 40.54, 41.74, 39.99 | 28.51, 27.48, 29.89 |
| SGLang native-label | 641/641 | 42.21, 41.71, 43.91 | 27.69, 27.37, 27.06 |
| SGLang native-plugin | 527/527 | 34.96, 35.00, 35.07 | 32.37, 31.95, 33.73 |

Plugin/native throughput was 83.0–88.9% for vLLM and 79.9–83.9% for SGLang, still
below the 90% target. Observed mean `queue` phase time, which includes the durable
admission transaction even without waiting, was 0.63–0.76 ms for vLLM and
0.62–0.80 ms for SGLang. These timings are not an isolated ablation of admission:
the API-worker count and workload differ from prior checkpoints. They do not
establish a controlled regression magnitude, stable P99 or confidence interval.
The controlled release matrix remains unrun.

All raw cohort summaries were recomputed from JSONL. Both task-owned process
groups exited and GPU7 returned to 11,990 MiB free. Evidence:
[shared-quota and timing validation](../evidence/dsw/shared-admission-validation-4944efd.json),
[vLLM raw timing report](../evidence/dsw/vllm-shared-admission-perf-4944efd/report.json),
[SGLang raw timing report](../evidence/dsw/sglang-shared-admission-perf-4944efd/report.json).


## Atomic journal and admission comparison

The `3377c95` change combines recovery IDs and shared quota in one durable commit.
Matched two-worker BF16 TP1 checks on both engines completed 45,970/45,970 timed
requests across 48 cohorts. Combined journal/queue time decreases, while the
single-concurrency profile remains below 90% of native throughput. High-concurrency
results were already above 90% before the change. These are short colocated samples,
not a consistent throughput-gain or release certification claim. Both plugins also
pass real two-tenant quota and peer-cancel checks at the new source. See the
[full comparison, failure-inclusive artifacts and reproduction](atomic-admission-performance.md).

The subsequent [combined lease/admission comparison](combined-reservation-validation.md)
uses two commits on stable routes. It reduces commit-related work but does not
establish a consistent throughput gain or pass the low-concurrency release gate.
