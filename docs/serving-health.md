# Engine canaries and readiness

Each API worker periodically checks its active bundles and locally prepared
standbys in READY/ACTIVE/DRAINING state against the actual scoring backend.
Active routes run first; otherwise a frequently switched version could be absent
at every polling instant and expire after 90 seconds despite receiving traffic.
Retired or unprepared versions are excluded. Monitoring always acquires a
revalidation lease and cannot revive a concurrently retired/unloaded version.
Startup/manual preparation still validates all fixed questions. A
periodic check of an already prepared immutable version verifies the current
capabilities and scores its first question with synthetic `ready` input; dynamic
bundles use a small Boolean canary. The full score/assembly contract must pass.
This establishes current scoring availability, not task accuracy or weight hashes.

```yaml
health:
  interval_seconds: 30
  timeout_seconds: 10
  max_age_seconds: 90
```

These defaults apply to both native plugins and gateways. The maximum age must
allow the interval, probe timeout and five-second cancellation deadline. Probes
run serially per worker, with one interval between sweeps. For many active/standby bundles
or congested engines, size the maximum age and timeout for that workload. Probes
hold durable bundle/adapter leases and journal branches before dispatch. Like
manual preparation, they are privileged control-plane work outside tenant quota
accounting; they can consume GPU time and warm prefix cache.

`GET /ready` remains a cheap authenticated read. It requires healthy cancellation
control, a running monitor, at least one active bundle for this backend, and
current successful canaries for **every** such bundle on this worker. Failure or
expired evidence returns HTTP 503 `engine_unavailable`. The endpoint does not
start extra GPU work on each poll. `/admin/profile` exposes per-bundle readiness,
last successful canary age, a bounded error code, monitor state and settings.
Health state is local to the API worker; it is not a replica-wide health verdict.

A serving transport/backend error or invalid score contract opens the affected
bundle's circuit immediately. New typed requests check that circuit before
compilation and again after admission, before GPU dispatch. Other healthy bundles
can continue serving, while overall `/ready` remains false. Ordinary client
cancellation, input rejection and quota/deadline handling do not automatically
trip this circuit. In-flight requests keep their original snapshot and existing
cancellation/quota rules. No request is automatically replayed onto another engine.

Only a successful new canary closes the circuit. A canary that began before a new
serving failure cannot erase that failure when it completes. Failed checks remove
the worker's publication-barrier preparation record; a fresh successful check
restores it. Activating or rolling back to a version with expired local canary
evidence requires preparing it again. Bundle aliases and route generations do not
change during temporary health failures or recovery.

If probe cancellation cannot be confirmed, its durable lease remains visible in
`/admin/requests/recovery`. That worker does not issue another canary for the same
bundle until explicit recovery confirms cancellation and releases the old lease.
This bounds automatic probe accumulation without treating a timeout as GPU drain.
The existing authenticated recovery operation is available even while readiness
is false. A changed engine capability profile stays unavailable until the service
is restarted and the configured profile validated. Unchanged reported metadata
does not prove unchanged weights or tokenizer files on a remote engine.

The privileged raw scoring bridge and the engine's native generation API are
outside the typed bundle circuit. A standalone gateway can report an unresponsive
engine while keeping its administrative endpoints alive. A native plugin shares
the host API's process lifetime; if that process is unavailable, its health
endpoint is unavailable too.

The implementation does not yet select a compatible replica or provide cross-engine
failover. That remains a separate requirement in the original implementation plan.
It requires certified equivalent task/backend profiles and must preserve request
identity, leases and cancellation semantics. Current checks return an explicit
failure when the configured engine cannot serve.

## Validation

CPU tests cover stale evidence, all-active readiness, independent healthy bundles,
queue-time circuit changes, error/canary races, score-contract failures, capability
changes, minimal versus full preparation, periodic recovery, shutdown drain and
unconfirmed-probe retention. `tests/integration/live_health.py` pauses an exact
recorded task-owned engine process group while its independent gateway remains
alive, then verifies withdrawal, rejected traffic, resume and fresh scoring.
The harness always attempts to resume the same verified group on an error and
preserves unsuccessful reports. It is not a replica failover or performance test.

At runtime source `40b78c2`, both DSW gateways passed the pause/resume check with
SmolLM2-1.7B BF16, TP1/API1, eager execution and the pinned CUDA 12.9 engines.
The test policy used a one-second interval, one-second probe timeout and ten-second
maximum age. This is an accelerated fault test, not the production default.

| Observation | vLLM 0.30.0+cu129 | SGLang 0.5.19 |
|---|---:|---:|
| Readiness withdrawal after group pause | 1.71 seconds | 2.01 seconds |
| New requests rejected with 503 | 10/10 | 10/10 |
| Maximum observed rejection response time | 3.67 ms | 17.51 ms |
| Probability error on two recovered responses | 0 | 0, against pre-pause warm baseline |
| Native route switches afterward | 1,000 | 1,000 |
| Successful traffic requests during switches | 453 | 510 |
| Mixed-version responses | 0 | 0 |

Both native rechecks passed four types, explicit first-token policy, independent
candidate scoring, native/typed probability parity, native chat, disable/drain/retire
and CAS checks. Gateway and native postchecks showed healthy active bundles and
zero leases/admission. All four task-owned process groups exited; GPU7 returned
to 11,990 MiB free after each engine stopped. The distinct functional denominator
remains 6/40. These checks do not recertify LoRA, multi-worker or other model profiles.
See [the hashed evidence summary](../evidence/dsw/health-validation-40b78c2.json).

The initial SGLang attempt is retained as a failure. It passed withdrawal,
rejection and same-process recovery, then failed the `1e-4` probability comparison
against its first, partially cached baseline (34/90 tokens cached). That harness
did not retain the original failing post-resume response. Subsequent repeated
responses were stable with 89/90 cached tokens. A separate direct native replay,
without pausing/restarting the engine, flushed cache before each of three pairs:
all three showed 0 then 89 cached tokens and a 0.0223848795 maximum probability
difference. This establishes a native cache-state numerical difference for that
fixture; it does not establish its impact across tasks or replace a dtype-specific
numerical budget and business-quality evaluation.

The revised harness at `02addc7` preserves responses before asserting and records
both first and warm baselines. Its recovered responses are compared with the warm
baseline using the unchanged `1e-4` threshold. SGLang passed that comparison with
zero error. Runtime/package/deployment sources remained exactly `40b78c2`; the
installed runtime was not modified to obtain the retest. The initial failure,
post-failure observations and native cache replay are linked in the summary.

Local validation at `40b78c2` passed 237 Python tests, Ruff lint/format and three
wheel builds with verified Python source contents. TypeScript was unchanged; its
earlier 62 tests were not rerun. Hosted CI, full numerical/quality gates,
controlled performance, replica failover and the 24-hour soak remain separate.
