# Versioned registry migration: real DSW upgrade and rollback

Date: 2026-09-28. Candidate: `0ac5935477cd7265087c3a2548794f0a1f26c15c`.
Baseline and rollback runtime: `e3f82d705b5db37504ec88f7c9ab89cf5a9c041c`.

Both native engines passed an offline database upgrade and code rollback with
two API workers. The baseline published route generation 3; the upgraded runtime
preserved it, then published generation 5; the old runtime served generation 5
after explicit schema downgrade. No route history was rewound. Each phase ran
fresh readiness and real Choice/Boolean/Score/Rank requests. The original baseline
database remained unchanged after staging. See [the operating procedure](registry-schema.md).

## Scope and profile

The candidate introduces an exact database schema/version check, transactional
fresh initialization and a client-registration handshake. Legacy startup against
version 1 fails before joining admission; new startup against an unversioned
database requires explicit migration. The migration copies a verified latest
snapshot under a writer reservation on the stopped source, validates shared-table
content preservation, and publishes a separate inactive database with a receipt.

The exact supported old schemas were compared with historical Git DDL. Upgrade
from the pre-quiescence format and legacy format has CPU contract coverage.
**The real GPU maintenance cycle here uses the legacy-quiescence format only.**
It does not qualify arbitrary earlier/future/custom schemas.

| Property | Actual test profile |
|---|---|
| Model | SmolLM2-1.7B-Instruct, revision `31b70e2e869a7173562077fd711b654946d38674` |
| Engines | vLLM 0.30.0+cu129 and SGLang 0.5.19 |
| Execution | BF16 backbone/readout, TP1, two native API workers, eager, context 2048 |
| Device | Shared L20Z GPU 7; UUID `GPU-b57fb933-0e5a-dd28-7041-a177da03405e` |
| Workload | Three sequential four-question requests on each of two distinct API workers per phase |
| Health | Interval 1 s, timeout 10 s, maximum evidence age 30 s |
| Deployment | Exact source checkouts through PYTHONPATH; native environment packages unchanged |
| Maintenance | Stop old group, stage into a new directory, explicit backend resume, `--no-bootstrap`, start new group |

This is a maintenance operation with downtime. The test does not measure
uninterrupted rolling traffic, throughput, prediction quality or model-wide
compatibility. CPU test doubles and the separate Linux migration-process fault
are not counted as additional GPU requests.

## Observed results

| Check | vLLM | SGLang |
|---|---:|---:|
| Baseline generation 3: complete responses / questions | 6 / 24 | 6 / 24 |
| Upgraded generation 5: complete responses / questions | 6 / 24 | 6 / 24 |
| Old code after downgrade, generation 5 | 6 / 24 | 6 / 24 |
| Stale route writes rejected with 409 | 3 | 3 |
| New code rejects original legacy database without mutation | Passed | Passed |
| Old constructor rejects version 1 without registering an owner | Passed | Passed |
| Migration rejects QUIESCED but still-live API owners | Passed | Passed |
| New code rejects downgraded database without mutation | Passed | Passed |
| Original baseline state unchanged after upgrade/rollback | Passed | Passed |
| Owned native process groups terminal | 3/3 | 3/3 |

The six native runs retain 36 complete responses and 144 successfully scored
questions, excluding health/preparation probes. Normal policy abstention is valid
contract behavior; these counts are not accuracy measurements. Response bundle
digests are stable across each engine's three phases, and every response's route
generation matches the expected phase. New workers have distinct identities.

Before the live-migration rejection, the candidate explicitly quiesces so periodic
health writes stop. The snapshot still records two living API owners. Staging
rejects those owners even though work is drained and worker states are QUIESCED.
This avoids confusing a snapshot-state race with the owner-liveness check.

## Actual migration-process interruption

A separate vLLM-environment Python subprocess used the stopped baseline registry
and was killed by SIGKILL immediately before linking the final database name.
The child returned -9. Its intent, completed receipt and pending database remain
as evidence, while `registry.sqlite3` was never published. New startup rejected
the incomplete directory without creating an empty database. Verification also
rejected its inventory. A fresh-directory retry completed and verified using the
same unchanged source/snapshot.

This is one Linux process-crash timing in the migration tool. It is not an engine
crash, host power-loss test, filesystem corruption test or exhaustive crash matrix.
The source writer lock was released on process death, and no source rows were deleted.

## Evidence and checks

- 523 local Python tests passed; 40 migration/backup tests cover both old formats,
  data/work retention, concurrent startup, incompatible clients, stale snapshots,
  writer exclusion, interrupted publication and tampered artifacts.
- 140 CPU contracts passed independently in each actual engine environment using
  isolated test tools. No engine environment was upgraded or patched.
- Ruff and formatting passed. Three local wheels built and matched 34 core and
  4+4 plugin source files. GPU execution used source, not these wheels. TypeScript
  was unchanged and not rerun.
- The verifier checks 102 artifact hashes and 337 Python source hashes across the
  two declared commits. It validates six stopped registry snapshots, all response
  contracts/generations, snapshot manifests and migration/crash receipts.
- Initial staged payloads were verified on DSW before first activation. The exported
  live-registry copies are explicitly named `registry-final.sqlite`; they include
  subsequent serving state and are not claimed to match the original staging hash.
  The local verifier independently replays all four transformations from immutable
  snapshot inputs and compares every logical result against the initial receipts.
- All 313 retained native/gateway process records are terminal. The separate fault
  child has its -9 exit receipt. GPU 7 free memory returns to 11,990 MiB. Eight model
  files rehash against the previously verified immutable Hub audit; no new Hub query.

The vLLM logs contain no forced-child-cleanup, leaked-semaphore or traceback
messages in these runs; startup and sibling-port warnings remain. Each SGLang run
retains four SystemExit/CancelledError traceback occurrences, and some logs include
a native NCCL process-group teardown warning. The groups and GPU allocations were
reclaimed, but warning-free native shutdown and absence of all resource leaks are
not claimed. Native parent exit codes were not captured by the detached launcher.

Inspect the [campaign](../evidence/dsw/registry-migration-0ac5935/campaign.json),
[fault receipt](../evidence/dsw/registry-migration-0ac5935/faults.json),
[export manifest](../evidence/dsw/registry-migration-0ac5935/export-manifest.json),
[verified summary](../evidence/dsw/registry-migration-0ac5935/verified-summary.json),
[audit](../evidence/dsw/registry-migration-0ac5935/audit.json) and
[local checks](../evidence/local-check-0ac5935.json).

Run `.venv/bin/python evidence/harnesses/verify_registry_migration_0ac5935.py` from
the repository. The evidence archive SHA256 is
`6e6e4b357a71aaa452374564a217aa3639d5241de4513cb3099aab9415dd7d3b`.

## Remaining boundaries

This qualifies the tested same-host legacy-quiescence ↔ version-1 maintenance
cycle and one interrupted-publication case. Supervisors must remain fenced, and
only the chosen registry copy may serve after cutover. The tool cannot prevent an
external supervisor from later starting an independent obsolete copy. The current
owner check requires the original Linux host, boot ID and PID namespace.

Lost-source restoration, container/namespace replacement, additional crash phases,
engine-version upgrades, other models and full deployment handoff remain open.
Functional model coverage stays 24/40 (12/20 per engine), and numerical, approved
business quality, controlled performance, 24-hour soak, image execution and overall
release gates are not newly accepted.
