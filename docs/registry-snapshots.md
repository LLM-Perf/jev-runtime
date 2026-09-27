# Consistent registry snapshots and restore staging

`jevctl registry` adds an executable backup boundary for the local SQLite registry.
Snapshots preserve every supported table, including immutable manifests, route
CAS generations, owners, work journals, cancellation commands, admission quotas and
adapter state. They do not reset generations, delete leases or mark workers dead.

Use this before a compatible deployment change and after a drained shutdown. A
staged restore is an inactive copy of the exact latest stopped state in a **new**
directory. It never replaces the source file, starts a worker or changes traffic.
This is not a schema migration, recovery of a lost source database, cross-host
restoration or authorization to revert publications made after the snapshot.

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
The schema matches this tool's current registry DDL with `user_version=0`;
unrecognized or altered schemas fail closed rather than receiving an automatic
migration. Extracting the shared DDL constant did not change its SQL bytes.

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

Actual DDL migrations/downgrades, backup encryption/retention automation, lost-host
recovery, container replacement and final release acceptance remain open work.
The target engine, model, tokenizer, calibration and adapter artifacts must still
match their existing compatibility contracts after a restore.
