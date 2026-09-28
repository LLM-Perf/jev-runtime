# Decision response serialization validation

`dadb516` declares the existing `DecisionResponse` as the decision endpoint return
model. FastAPI uses its model serializer and exposes the output schema in OpenAPI.
Both native engines eliminate the recursive response-encoding calls in the
controlled diagnostic window. Response validation still runs. The separate
colocated throughput samples do **not** establish a consistent relative speedup
or pass the 90% native-throughput gate.

## Evidence and scope

- Runtime before: `7391ed362117ab39a32bfa1dd1554356ee7a6178` (same runtime code as delivery `a475ad4`).
- Runtime after: `dadb516b38816b32254c710b8f33cba6dd4a6af3`.
- Frozen launcher/unprofiled runner: `7391ed3`; the instrumented copy only adds
  diagnostic headers and its qualification. Source and wrapper hashes are retained.
- SmolLM2-1.7B-Instruct revision `31b70e2e869a7173562077fd711b654946d38674`;
  BF16 backbone/readout, TP1, API2, eager, GPU7 L20Z. Engine versions remain
  vLLM 0.30.0+cu129 and SGLang 0.5.19.
- One hot L=256/K=8 input, C=1. Exact input/label IDs, launch flags, tokenizer,
  model and admission settings match before/after in each engine.
- Health checks use a matched 300-second interval, 10-second timeout and 900-second
  evidence age. This differs from earlier development experiments; comparisons here
  use only this campaign. No engine package is changed.
- Each version first runs a separate diagnostic window, then three unprofiled
  10-second native/plugin repeats with 32 warmups per method and alternating order.
  The diagnostic header hook remains installed but inactive in the latter repeats.
  This extra header gate is present in both versions.

## Serialization path

The corrected collector runs one contiguous cProfile window per lane, using the
default monotonic wall clock. After skipping 64 matching requests, it collects 128
successful requests. Idle/background callbacks can appear and function times can
overlap; this is not a CPU-only, GPU-time or ordinary-latency measurement. All eight
corrected profiles pass nonnegative-time and total-window sanity checks. Raw `.prof`
statistics independently match their JSON exports.

| Engine | Plugin requests per version | Recursive encoder calls before → after | DecisionResponse validator calls before → after |
|---|---:|---:|---:|
| vllm | 128 | 10,240 → 0 | 128 → 256 |
| sglang | 128 | 10,240 → 0 | 128 → 256 |

The additional validation is the framework response-model validation; the runtime
already validates each constructed response. Shared HTTP fixtures preserve all ten
valid response forms, including nulls, numeric/Boolean/rank values, Unicode/escaping,
partial/failed outcomes, timing headers and worker identity.

The copied profiling runner retains an old `thread-CPU` qualification string in its
raw report. That label is stale: the corrected collector metadata and this report
define the actual wall-clock window. Raw evidence is preserved rather than rewritten.

## Separate throughput measurements

All **10,048/10,048** requests in 24 unprofiled cohorts succeed, and warm native/plugin
probability differences are zero. Ranges below span three repeats and are not
confidence intervals. P99 is omitted below 10,000 successes per cohort.

| Engine | Before plugin/native RPS | After plugin/native RPS | Before plugin RPS | After plugin RPS | Before native RPS | After native RPS |
|---|---:|---:|---:|---:|---:|---:|
| vllm | 0.816–0.847 | 0.814–0.863 | 37.2–39.1 | 41.8–42.8 | 45.0–46.3 | 48.7–51.3 |
| sglang | 0.851–0.865 | 0.842–0.864 | 36.4–37.5 | 35.6–36.3 | 42.3–43.7 | 42.0–42.8 |

vLLM absolute throughput rises in both lanes, so the plugin rise cannot be attributed
solely to this change. SGLang plugin throughput is slightly lower in these samples.
Neither engine demonstrates a stable relative gain. The client-latency minus
runtime-total observation also includes HTTP transport, metrics, serialization and
client validation; it must not be labeled pure serialization time.

## Failed diagnostic and recovery

The first per-request `time.thread_time` profiler produced negative SGLang cumulative
values and totals larger than wall time. Those timings are invalid and excluded from
comparisons. Its raw data, scripts and nonzero campaign result are retained.
In the same trial, process-group SIGTERM terminated the scheduler while two background
health probes were active. Cancellation could not be confirmed, two abort-pending
journals remained, and the API workers did not finish shutdown within the deadline.
This does not prove that profiling caused the shutdown race.

The exact owned parent PID/start-tick/boot identity and launch command were checked
before SIGKILL. A new profiler-free recovery-only native service confirmed cancellation
of both journaled branches via the supported recovery API. Before/after SQLite
snapshots pass integrity checks and show two leases/journals becoming zero.
The original failed cleanup record remains unchanged.

The corrected test campaign disables its isolated test aliases and waits for leases
to drain before terminating each engine. All four corrected lifetimes then exit.
This is explicit test-service quiescence, not a product fix guaranteeing clean shutdown
under arbitrary native-engine SIGTERM. That operational boundary remains open.

## Validation and delivery boundaries

- 473 local Python tests; Ruff checks and 121-file format check pass. All three
  wheels build with source-matched 31/4/4 Python payloads; DSW executes source.
- Each actual DSW engine environment passes 40 response/API/telemetry/recovery tests
  using CPU engine doubles. These are separate from the real GPU cohorts above.
- Framework versions/source hashes are recorded at final audit; unchanged TypeScript
  tests were not rerun.
- Independent verification checks 171 artifact hashes and 239 Python source files
  against the declared commits, raw call profiles, exact scoring IDs, all cohort
  summaries and recovery snapshots.
- All 295 retained owned process identities/groups are terminal; all seven test-run
  registry views have zero retained work. GPU7 returns to 11,990 MiB free. The eight
  model files still match the prior immutable-Hub-verified audit.

- [Verified summary](../evidence/dsw/response-model-dadb516/verified-summary.json)
- [Export manifest](../evidence/dsw/response-model-dadb516/export-manifest.json)
- [Independent verifier](../evidence/harnesses/verify_response_model_dadb516.py)
- [Initial diagnostic and failure](../evidence/dsw/response-model-dadb516/initial/campaign.json)
- [Confirmed recovery](../evidence/dsw/response-model-dadb516/initial/recovery.json)

Coverage remains 12/20 models per engine (24/40 combinations). The controlled
144-case performance matrix, broader numerical/model certification, business quality,
24-hour soak, deployment images and complete release/rollback acceptance remain
outstanding. No original release gate changes status.
