# Native scoring cancellation across gateway rollback

Both engines passed three real request cancellations after rolling traffic from
the candidate gateway back to the previous installed wheel. This closes the
native-request gap in the earlier CPU-fixture cancellation test for the profile
below. It does not certify engine-process upgrades, multiple hosts, all model
profiles, performance, task quality or release acceptance.

## Tested artifacts and profile

- Harness/native editable source: `b20d3f43d91c2e13ed2b044a77d35cc3037eb62b`.
- Candidate green gateway wheel: `6f0cda85f72619193be7a1d1cc89d75142899bf3`.
- Previous blue gateway wheel: `773914587b4fab287ca1e4b17f60ff66eb20d261`.
- The candidate core/package code is byte-identical at `6f0cda8` and `b20d3f4`;
  the latter changes only integration harness code and its regression tests.
- Both gateways use offline-installed wheels, two workers per slot, one shared
  local SQLite registry and one stable HAProxy 3.2.24 process with `nbthread=1`.
  All 30 core Python files imported from each wheel match its exact source commit.
- Native engines: vLLM `0.30.0+cu129` and SGLang `0.5.19`, isolated environments,
  SmolLM2-1.7B-Instruct revision `31b70e2e869a7173562077fd711b654946d38674`,
  BF16, TP1, eager execution, context 2048 and preserved engine tokenizer profiles.
  Each native engine uses one API worker; the two-worker counts refer to gateways.
- Tests used spare memory on shared DSW GPU7. No latency, throughput or production
  availability target is certified by these short functional runs.

## Results and denominators

| Observation | vLLM | SGLang |
|---|---:|---:|
| Cross-version native cancellation cases passed | 3/3 | 3/3 |
| Planned scoring sequences in each cancelled request | 128 | 128 |
| Completed scoring RPCs before each switch | 2, 2, 2 | 2, 2, 2 |
| Completed scoring RPCs after each switch | 10, 9, 9 | 11, 6, 10 |
| Active scoring RPCs at both observations | 1 | 1 |
| Strict successful continuous requests | 580/580 | 627/627 |
| Continuous requests through blue / green | 338 / 242 | 343 / 284 |
| Observed proxy-map generations | 19 | 19 |

The continuous traffic is a separate cohort from the six intentionally cancelled
requests, readiness probes, controller canaries and recovery requests. HTTP 499 is
the expected result of those six explicit cancellations; it is not counted as a
successful decision. The 19 map generations include ten alternating traffic
switches, six transitions for cancellation cases, two initial transitions and one
interrupted operation reconciled from observation.

Each cancellation case binds the same request ID, worker, durable lease, immutable
bundle digest and route generation before and after switching. At least two native
scoring RPCs have returned validated results before switching, and one is active
at each sample. The old slot initially fails drain with an outstanding stream and
lease. Cancellation sent through the stable frontend reaches blue while the
original request runs in green; the original returns `request_cancelled`/499,
its journal lease and local progress disappear, drain succeeds and a new typed
request succeeds. The helper bundle is disabled and retired after all cases.

The old blue release does not emit `X-Jev-Worker` on cancellation. Its
`X-Jev-Deployment: blue` header identifies the selected slot; no exact blue worker
identity is claimed. RPC counters show adapter execution progress, not instantaneous
CUDA kernel occupancy, engine scheduler composition or business accuracy.

Partial HTTP request bodies surviving a switch, HTTP keep-alive selection on the
next request, incorrect-release rejection, interrupted-map reconciliation and
unchanged native/proxy process identities also pass in both runs.

## Failure retained and local checks

The initial `6f0cda8` harness incorrectly entered an httpx client context after its
first request had already opened it. Both initial runs failed during admin
connection setup with `Cannot open a client instance more than once.`, before any
native cancellation case. Their 73 vLLM / 60 SGLang continuous requests and complete
cleanup records remain in `initial-attempt.json`; they are not cancellation passes.

The fix registers client cleanup before I/O and avoids reopening the client.
Regression tests cover normal/failed readiness connection cleanup and rejected
progress snapshots. Full local validation at `b20d3f4`: 368 Python tests passed;
Ruff checks passed. Three candidate wheels were built at `6f0cda8`, with source
contents verified. The TypeScript code is unchanged and was not rerun.

The final audit verifies 145 distinct historical task-owned process identities and
their groups are terminal, GPU7 free memory is back to 11,990 MiB, both tokenizer
profile payloads verify and six model files rehash against fixed public Hub
metadata. This audits on-disk model bytes, not in-memory weight attestation.

## Reproduce the evidence check

```sh
.venv/bin/python evidence/harnesses/verify_native_rollout_cancel_b20d3f4.py
```

The verifier checks the exported artifact hashes, retained initial failure,
installation/source identity, every cancellation observation, request denominator
and terminal ownership. It writes
[`verified-summary.json`](../evidence/dsw/native-rollout-cancel-b20d3f4/verified-summary.json).
The frozen campaign scripts are in
[`evidence/harnesses/native-rollout-cancel`](../evidence/harnesses/native-rollout-cancel).
They document these exact DSW attempts; running them requires the isolated paths
and resources they name and is not a generic deployment command.

Functional model coverage stays 24/40 combinations. The controlled 144-case
performance matrix, independent numerical failures, business labels/acceptance,
24-hour soak, deployment image validation, database migration/downgrade and
operational handoff remain open in the [completion ledger](completion-ledger.md).
