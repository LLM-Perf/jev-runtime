# Performance experiments

`profiles/performance-matrix.json` freezes 144 scenarios using Qwen3-8B and
Phi-4-mini, two engines, L=256/2048/8192, K=2/8/32, concurrency=1/16 and cold/hot
cache states. All entries are currently **not run**. The 21.6 hours is a lower
bound **per applicable measurement method**; baselines, loading, warmup and the
24-hour soak add time. Small-model development runs do not fill this matrix.

The paired runner currently covers native selected-label HTTP scoring and the
configured typed plugin or gateway. For independent-candidate mode it dispatches
the native branches serially and records that distinction. Structured generation,
other Jev implementations, upstream decisions, open-loop saturation sweeps and
full GPU/CPU telemetry collection remain separate outstanding work.

## Exact inputs and denominators

The administrator-only `/admin/compile` endpoint pins the active bundle and uses
the actual serving worker's compiler. It exports input IDs, selected label IDs,
question order, bundle digest, generation, input digest and the full tokenizer
implementation digest when a fast tokenizer exposes it. It does not submit GPU
work. This avoids substituting a similarly named local tokenizer in a baseline.
The implementation fingerprint supplements the current vocabulary fingerprint;
a migration that binds it into every bundle is still required.

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
