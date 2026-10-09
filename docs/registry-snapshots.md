# Consistent registry snapshots and restore staging

`jevctl registry` adds an executable backup boundary for the local SQLite registry.
Snapshots preserve every supported table, including immutable manifests, route
CAS generations, owners, work journals, cancellation commands, admission quotas and
adapter state. They do not reset generations, delete leases or mark workers dead.

Use this before a compatible deployment change and after a drained shutdown. A
staged restore is an inactive copy of the exact latest stopped state in a **new**
directory. It never replaces the source file, starts a worker or changes traffic.
The `stage-restore` command is not a schema migration, recovery of a lost source database, cross-host
restoration or authorization to revert publications made after the snapshot.
For explicit supported upgrades and code rollback, use the separate
[versioned migration procedure](registry-schema.md).

## Create and verify

Keep backups in a private local directory with enough space. The parent must exist;
the destination itself must be new:

```sh
jevctl registry snapshot /data/jev/registry.db /backups/jev/checkpoint-001 \
  --timeout-seconds 30
jevctl registry verify-snapshot /backups/jev/checkpoint-001 SNAPSHOT_MANIFEST_SHA256
```

Retain the returned manifest SHA256 in the release record. The snapshot uses
[SQLite's online backup API](https://www.sqlite.org/backup.html), including committed
WAL state. It does not copy a live `.db` while ignoring its WAL. The output is a
standalone database with DELETE journal mode; no WAL sidecar is needed. The timeout
bounds backup copying/retries, not the subsequent integrity and table-hash scans.

The manifest binds the database bytes, exact supported schema, row counts and
content hashes of every table. `integrity_check` and `foreign_key_check` must pass.
The tool recognizes the exact version-1 schema and two frozen unversioned legacy
schemas. Unknown versions or altered structures fail closed. Snapshot/restore
never converts schemas; ordinary startup requires the current format. Use the
explicit migration procedure to change a supported format.

Snapshots may be taken while serving. Their `restore_blockers_at_snapshot` counters
identify work or process ownership observed then; those observations are not a
lasting claim that processes are stopped. An in-flight backup is useful evidence
and retains its leases; it cannot become a clean restart image by deleting them.
Snapshotting neither registers a new runtime owner nor changes serving state.

Directories use mode 0700 and database/manifest files use 0600. They can contain
private task instructions, identifiers and adapter paths: apply normal restricted
backup storage and retention controls. They do not contain model/LoRA weights,
credentials supplied only through environment variables or external tokenizer
files. Preserve those artifacts separately with their immutable identities.

TokenSpeed's separate `*.tokenspeed-receipts.db` is also outside this snapshot:
it is not a table in the core registry. The snapshot, verification and staging
commands do not back up, validate or restore native completion receipts. Before
a TokenSpeed deployment change, quiesce and stop affected writers, then preserve
both stores with SQLite-aware backups. Keep the corresponding receipt database
at the sibling path expected by the selected registry and retain the same engine
identity. See [TokenSpeed operations](operations.md#tokenspeed-operations).
Automatic coordinated snapshot/restore and receipt expiry are not implemented.

## Stage the latest stopped state

1. Remove traffic and observe HTTP and durable-work drain.
2. Gracefully stop **all** gateway/API/embedded processes that may open this registry.
   `STOPPED` in a worker row alone is insufficient; its Linux process must be
   verifiably dead. Keep supervisors disabled through cutover.
3. Resolve uncertain work through the existing recovery APIs before the final
   shutdown. Unload managed adapters through their normal fenced lifecycle.
4. Take a fresh final snapshot and verify its returned manifest digest.
5. Stage it into a new, inactive local directory:

```sh
jevctl registry stage-restore /backups/jev/checkpoint-001 SNAPSHOT_MANIFEST_SHA256 \
  /data/jev/registry.db /data/jev/staged-001
```

The command takes a SQLite writer reservation on the original source and requires
its complete logical state to still match the snapshot. It rejects live/unknown
owners, leases, work/tenant journals, admission tickets, in-progress preparations,
unconfirmed cancellation commands and resident/uncertain adapters. It checks
source file identity and copies the verified snapshot without resetting any data.
Other SQLite writers cannot change the source during validation/publication; WAL
readers may continue. See [SQLite transaction semantics](https://www.sqlite.org/lang_transaction.html).

The database is fsynced before the exclusive, atomic `restore.json` completion
receipt is published. A failed operation preserves its incomplete directory for
diagnosis; use a new destination on retry and never serve an incomplete artifact.
The source remains unchanged. The returned receipt digest and database digest
identify the staged copy.

Keep old processes fenced after the command returns: its database lock is released
then, and it cannot prevent an external supervisor from restarting an old instance.
Select the new `registry_path` explicitly, keep the original file intact, and start
only one intended deployment. Require fresh readiness for every worker and real
scoring canaries before restoring traffic. Existing READY rows do not bypass this
runtime validation. Retire the old registry from all future startup configurations.
Do not serve original and restored copies concurrently as independent quota pools.

## Limits and remaining release work

The current restore gate requires the original registry, Linux host, boot ID and
PID namespace to remain available for exact-state and owner-death checks. A host
reboot, container namespace change, destroyed database or unknown owner fails
closed. Those cases require separately implemented fencing/recovery procedures.
Restoring an old database while continuing from its generation numbers can reuse
stale CAS tokens or discard acknowledged state; this tool rejects that operation.
Compatible code rollback should preserve current state, not rewind its history.

Explicit supported DDL transitions are provided by [stage-migration](registry-schema.md).
Backup encryption/retention automation, lost-host recovery, container replacement
and final release acceptance remain open work.
The target engine, model, tokenizer, calibration and adapter artifacts must still
match their existing compatibility contracts after a restore.

## Real DSW validation at c4e9cef

Both native engines passed the same-host exercise with SmolLM2-1.7B-Instruct
revision `31b70e2e869a7173562077fd711b654946d38674`, BF16, TP1, eager execution,
context 2048 and preserved tokenizer profiles. Native serving remained at
`b20d3f4` (SGLang 0.5.19 and vLLM 0.30.0+cu129); the two-worker gateway and backup
tool ran exact source `c4e9cef`. Every imported gateway Python file was hashed
against that source. This was source execution with existing isolated gateway
dependencies, not an installation test of the newly built wheel.

| Check | vLLM | SGLang |
|---|---:|---:|
| Original-registry decisions at generation 1 | 10/10 | 10/10 |
| Control decision after disable/reactivate, generation 3 | 1/1 | 1/1 |
| Restored-registry decisions at generation 3 | 10/10 | 10/10 |
| Live restore rejected with two live owners | Passed | Passed |
| Stale snapshot rejected after publication/shutdown | Passed | Passed |
| Old generation 1 write rejected after restore | HTTP 409 | HTTP 409 |
| Original registry unchanged while restored registry served | Passed | Passed |
| Final leases, work/tenant journals and admission tickets | 0 | 0 |

The original seed was a private snapshot/staging copy of the previous stopped
image-entrypoint test registry. The historical registry was never started or
modified by this campaign. During the new original gateway's lifetime, an online
snapshot succeeded but staging it was rejected because its two owners were alive.
The route was then disabled and reactivated (generation 1 → 2 → 3). After shutdown,
that old snapshot was rejected for changed source state. A fresh stopped snapshot
and staged copy had identical table content and database hashes before activation.
The restarted workers ran fresh canaries; generation 3 and the bundle digest were
preserved, and the old generation remained invalid.

There are 21 successful decisions per engine: the two ten-request cohorts plus
one generation-control request. Readiness and preparation canaries are outside
that denominator. These are functional/state-preservation checks, not a latency,
throughput, cross-engine numerical or business-quality comparison.

All four gateway process groups exited with status zero. The native engines kept
their process identities throughout each restore exercise and were then stopped.
The final independent audit verifies all 159 distinct historical task-owned
process identities/groups are terminal, GPU7 free memory is 11,990 MiB, both
tokenizer profiles verify and six model files rehash against fixed Hub metadata.
Six snapshot payloads were verified on DSW. Only their manifests and restore
receipts were exported; private SQLite database contents remain outside Git.

Local validation at this source passed 406 Python tests, Ruff and three
source-matched wheel builds (31 core Python files and four per engine plugin).
The TypeScript source was unchanged and was not rerun. Use:

```sh
.venv/bin/python evidence/harnesses/verify_registry_snapshot_c4e9cef.py
```

[The verified summary](../evidence/dsw/registry-snapshot-c4e9cef/verified-summary.json)
retains the exact sources and checks. This closes the tested backup/staging case;
DDL migration/downgrade, destroyed-source recovery, container restart, full model
coverage, controlled performance and the 24-hour soak remain separate gates.
