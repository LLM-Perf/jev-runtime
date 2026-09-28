# Combined reservation performance and validation

Runtime `7391ed3` combines stable-route lease creation, branch journaling and shared
admission into one FULL-synchronous transaction. With release, an immediately
admitted request now commits twice instead of three times. See
[the concurrency and recovery design](combined-reservation.md).

All 46,835 timed requests in 48 cohorts succeeded. The measured commit-related
phase generally fell, but there is no consistent end-to-end throughput gain.
At concurrency 1 the new plugin/native ratios remain below 90% for both engines.
The absolute plugin throughput is also slightly lower in these C=1 samples;
this colocated experiment cannot establish either a causal speedup or absence
of a throughput regression. No performance release gate is passed.

## Matched experiment

- Before: `d8a2649339da391e506dd9cab17570c659267716`.
- After and frozen launcher/benchmark harness: `7391ed362117ab39a32bfa1dd1554356ee7a6178`.
- SmolLM2-1.7B-Instruct, revision `31b70e2e869a7173562077fd711b654946d38674`.
- Native vLLM 0.30.0+cu129 / SGLang 0.5.19; BF16 backbone/readout, TP=1,
  two API workers, eager execution, GPU7 L20Z. Engine concurrent-running limit
  remains four. This is a shared GPU development environment.
- Exact L=256 input IDs, K=8 label IDs and one repeated hot input; engine caches enabled.
- C=1 and C=16; 32 warmups per method; three 10-second repeats per method,
  alternating native/plugin order. Source revision is the intended changed variable.
- Serving profile, launch command, tokenizer implementation, limits and input/label
  IDs match before/after within each engine. Warm native/plugin probability difference
  is zero. This fixture does not certify task quality or broad numerical accuracy.

Each cell is the range across the three repeats, not a confidence interval.
`pin + journal + queue` accounts for the durable pin phase moving into the combined
reservation transaction; comparing `pin` alone would be misleading.

| Engine | C | Before plugin/native RPS | After plugin/native RPS | Before pin+journal+queue ms | After pin+journal+queue ms |
|---|---:|---:|---:|---:|---:|
| vllm | 1 | 0.846–0.933 | 0.832–0.843 | 1.540–1.685 | 1.364–1.408 |
| vllm | 16 | 0.977–1.002 | 0.971–1.040 | 1.622–1.727 | 1.272–1.420 |
| sglang | 1 | 0.848–0.865 | 0.822–0.880 | 1.425–1.646 | 1.347–1.524 |
| sglang | 16 | 0.957–1.008 | 0.969–1.006 | 1.416–1.754 | 1.299–1.331 |

| Engine | C | Before native RPS | After native RPS | Before plugin RPS | After plugin RPS |
|---|---:|---:|---:|---:|---:|
| vllm | 1 | 44.9–49.2 | 48.5–49.1 | 41.0–41.9 | 40.4–41.3 |
| vllm | 16 | 122.8–127.8 | 117.3–130.1 | 122.0–127.0 | 119.7–129.1 |
| sglang | 1 | 42.0–43.5 | 41.3–43.5 | 35.6–37.6 | 35.6–36.3 |
| sglang | 16 | 176.8–185.4 | 177.1–183.4 | 174.3–178.6 | 177.7–179.1 |

P99 is omitted because each cohort has fewer than 10,000 successes. vLLM native
cached-token observations remain unknown where the API does not report them.
Ratios above one in individual C=16 repeats are not evidence of a causal speedup.

## Correctness, failures and evidence

- 463 local Python tests pass; Ruff checks/format pass for 120 files. All three
  wheels build and their 31/4/4 packaged Python files match source. DSW uses source
  checkouts, not these wheel artifacts; unchanged TypeScript tests were not rerun.
- The first DSW CPU-test launch failed because the engine environment lacks pytest.
  Its nonzero report/log remain in the export. A separate target directory containing
  pytest 9.1.1 and pytest-asyncio 1.4.0 then ran 49 tests successfully in 4.67 seconds.
  Existing engine packages were not changed. The Linux process-exit case confirms
  a dead owner and retained complete journal/quota; the local macOS case checks
  conservative rejection when Linux process identity cannot be established.
- Both real native engines pass all 13 shared-admission checks using two API workers:
  global/per-tenant queue and expanded-branch budgets, isolation of cancellation,
  eligible-tenant progress, queued cancellation/deadline cleanup, peer-worker active
  cancellation, drain, subsequent serving and retirement. These are separate from
  the timed cohorts and from a GPU-process crash or new 1,000-switch campaign.
- Six native service lifetimes and their harnesses finish. The final audit covers
  288 retained owned process identities/groups, all terminal; GPU7 returns to
  11,990 MiB free. All six registries have zero leases, journals, tenant bindings
  and admission tickets. Eight model files match the immutable Hub revision.
- The verifier checks 153 exported artifact hashes, 237 source Python files against
  their declared Git commits, matched workload/profile fields, and recomputes every
  summary from the retained request rows. Credential values are excluded from export.

Raw evidence and independently computed results:

- [Verified summary](../evidence/dsw/combined-reservation-7391ed3/verified-summary.json)
- [Export manifest](../evidence/dsw/combined-reservation-7391ed3/export-manifest.json)
- [Original campaign, including missing-pytest failure](../evidence/dsw/combined-reservation-7391ed3/campaign.json)
- [Isolated Linux tests and real quota campaign](../evidence/dsw/combined-reservation-7391ed3/quota-campaign.json)
- [Independent verifier](../evidence/harnesses/verify_combined_reservation_7391ed3.py)

The 144-case controlled matrix, 24-hour soak, business-quality acceptance, broader
model/numerical certification and deployment/image gates remain outstanding.
Model coverage stays 12/20 per engine (24/40 combinations); this checkpoint does
not promote any matrix status or close the original project goal.
