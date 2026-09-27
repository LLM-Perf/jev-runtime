# Atomic journal and admission: DSW development comparison

At runtime `3377c95`, recovery branch IDs and the shared admission ticket commit in
one SQLite transaction. Normal immediately admitted requests use three durable
commits instead of four. Local tests verify independent-connection visibility,
rollback after both writes, dispatch only after persistence, and retained quota
until uncertain cancellation is explicitly recovered. WAL and `synchronous=FULL`
remain enabled. Waiting requests still need polling transactions.

The optimization reduces observed journal/reservation time. These short samples
do **not** establish a consistent throughput improvement or pass the full release
performance gate. The single-concurrency profile remains below 90% of native
scoring throughput on both engines. The higher-concurrency profile was already
above 90% before this change.

## Matched profile and results

Before: core/plugins `75649bf`. After: core/plugins `3377c95`. Both use the exact
benchmark and launcher harness at `322782a`, with unchanged orchestration bytes.
The core/plugin files at `75649bf` and `322782a` are identical; later changes there
were documentation, evidence and hosted-CI dependencies.

Both sides use SmolLM2-1.7B-Instruct revision
`31b70e2e869a7173562077fd711b654946d38674`, BF16 model/readout, TP1, two API workers,
eager execution, selected GPU7 UUID `GPU-b57fb933-0e5a-dd28-7041-a177da03405e`,
256-token input, eight labels, one fixed input variant and enabled engine caches.
The warm fixture is intentionally narrow and does not represent varying prompts.
vLLM is `0.30.0+cu129`; SGLang is `0.5.19`. Each engine keeps its own unchanged
launch command and dependency versions. Native execution limits remain four
running sequences; client concurrency is 1 or 16.

Each method has 32 warmup requests and three 10-second measured cohorts; method
order alternates. Throughput counts strict successes over the complete cohort,
including drain. Table cells give the three-repeat range, not confidence intervals.
The phase columns are the sum of journal and queue means in milliseconds: the
journal write moved into the queue transaction, so either phase alone would give
a misleading before/after comparison.

| Engine | Client concurrency | Before plugin/native RPS | Before journal+queue ms | After plugin/native RPS | After journal+queue ms |
|---|---:|---:|---:|---:|---:|
| vllm | 1 | 81.5–82.7% | 1.008–1.174 | 80.9–85.2% | 0.839–0.862 |
| vllm | 16 | 94.4–98.1% | 0.991–1.042 | 97.2–100.7% | 0.864–0.962 |
| sglang | 1 | 80.5–84.9% | 1.041–1.171 | 82.4–87.7% | 0.841–0.905 |
| sglang | 16 | 96.0–99.2% | 0.906–1.055 | 95.9–99.9% | 0.769–0.865 |

All **45,970 / 45,970 timed attempts** across 48 cohorts succeeded. Exact input and
label IDs, compiler profiles, engine commands/versions, model fingerprints,
admission limits and health cadence match within each before/after comparison.
Every native/plugin warm-fixture probability check had zero absolute difference.
All 48 summary objects were independently recomputed from retained JSONL rows.
Warmups, setup/parity checks and quota-suite requests are outside that timed
attempt denominator. Native cached-token reporting is unavailable on vLLM; no
invented native hit rate is used. No P99 is reported because each individual
cohort has fewer than 10,000 samples.

The vLLM C=16 repeat above 100% is a descriptive ratio of separate short cohorts,
not evidence that the plugin accelerates the GPU engine. SGLang C=16 plugin RPS
was 169.1–175.7 before and 168.2–174.3 after; this does not demonstrate a stable
throughput gain. The host is shared, without controlled background load, clock or
thermal conditions. No large-model, cold-cache, varying-input, business-quality,
24-hour stability or 144-case release certification follows from this experiment.

## Safety and evidence checks

Both new native plugins also pass the existing separate two-worker/two-tenant
suite: durable global branch quotas, same-tenant queue limit (429), another
eligible tenant progressing, tenant isolation, queued cancellation (499), queued
deadline (504), matching shared gauges, peer cancellation of the active owner,
zero retained tickets and serving after drain. This is a real GPU-serving check;
it does not claim a new full-engine SIGKILL test at this source. Local uncertain
abort/recovery and multiprocessing quota tests are included in the 284-test pass.

SGLang reports `api_workers: null` in capabilities; the evidence does not coerce
that to a measured value. Its launch flag, two distinct registry process identities
with prepared default bundles, and separate live quota-worker connections support
the two-worker profile. The registry snapshots were collected after cleanup and
do not claim live liveness.

Every owned process group exited and GPU7 returned to 11,990 MiB free after each
attempt. Final identity checks found no live match across 79 historical owned
process records. Existing unrelated services were preserved. The initial vLLM
readiness attempt used the wrong credential class; it ran no timed cohort and its
failure and successful cleanup remain in the evidence.

Local verification: 284 Python tests, Ruff checks/format checks, dependency check,
and three wheels with Python source bytes verified (28 core, four per plugin).
This adds no new model to the 16/40 functional matrix and no performance release
pass. Hosted CI eligibility remains separate from these local results.

## Reproduction and retained artifacts

- [Recomputed comparison and quota results](../evidence/dsw/atomic-admission-comparison-3377c95.json)
- [117 artifact SHA-256 checks](../evidence/dsw/atomic-export-manifest-3377c95.json)
- [Final owned-process and GPU snapshot](../evidence/dsw/preflight-atomic-3377c95.json)
- [Local package checks](../evidence/package-check-3377c95.json)
- [Exact orchestration scripts and verifier](../evidence/harnesses/atomic-admission/README.md)
- Before: [vLLM](../evidence/dsw/vllm-atomic-before-75649bf-r2/performance-attempt.json),
  [SGLang](../evidence/dsw/sglang-atomic-before-75649bf-r2/performance-attempt.json)
- After: [vLLM](../evidence/dsw/vllm-atomic-after-3377c95/performance-attempt.json),
  [SGLang](../evidence/dsw/sglang-atomic-after-3377c95/performance-attempt.json)
- New quota checks: [vLLM](../evidence/dsw/vllm-atomic-quota-3377c95/quota.json),
  [SGLang](../evidence/dsw/sglang-atomic-quota-3377c95/quota.json)

Each performance run contains `c1/` and `c16/` directories with all reports,
fixtures and per-attempt JSONL rows. From the repository root:

```sh
.venv/bin/python evidence/harnesses/verify_atomic_comparison_3377c95.py
```
