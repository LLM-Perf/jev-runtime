# Retained DSW orchestration for atomic admission

The `.py.txt` files preserve the exact executed script bytes, including the first
failed readiness attempt. Python can execute these text files directly. They are
run-specific evidence scripts, not general deployment commands. Do not launch them
against an unrelated service or a host without the documented GPU allocation.

- `run_perf_attempt.py.txt`: runs the unchanged `322782a` benchmark harness against
  installed core `75649bf` or `3377c95`. It launches only a fresh task-owned service
  on GPU7, with two API workers, runs C=1 and C=16 paired samples, checks health and
  drain, and stops the recorded process group in `finally`.
- `run_perf_attempt_failed_auth.py.txt`: original wrapper mistakenly used the admin
  credential for `/ready`; no timed cohort ran. The service was stopped, the wrapper
  detected its exit, and the cleanup record is retained.
- `run_quota_attempt.py.txt`: launches a separate two-worker/two-tenant service with
  explicit shared quotas and runs the existing real quota/cancellation suite at
  `3377c95`. Failure status is read from the JSON report, not only the exit code.
- `export_evidence.py.txt`: exports only allowlisted JSON/JSONL evidence and creates
  SHA-256 hashes. It excludes keys, engine logs, model files and the registry itself.
  The worker registry observation is read-only and occurs after service cleanup;
  it proves recorded identities/preparation, not current liveness.

Exact service commands, model revision, engine versions, memory snapshots,
source paths and wrapper hashes are in each retained run. The performance driver
uses subprocess success to record stage completion; cohort success must additionally
be checked in the benchmark report and raw rows. The verifier does this explicitly.

Recompute all 48 cohort summaries and matched-profile comparisons from the repository:

```sh
.venv/bin/python evidence/harnesses/verify_atomic_comparison_3377c95.py
```

The verifier checks all attempts, input/label ID identity, engine settings and
versions, queue limits, health cadence, before/after source identity, final leases,
worker registration, shared-quota results, process-group exit and GPU memory return.
It does not turn these short colocated measurements into release certification.
