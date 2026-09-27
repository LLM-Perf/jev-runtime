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
The native candidate/other Jev baselines, controlled matrix, varying-input workloads
and business-quality equivalence remain outstanding.
