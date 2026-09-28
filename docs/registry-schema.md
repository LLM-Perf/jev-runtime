# Registry versions, offline migration and code rollback

New registries use `user_version=1` and an exact schema contract. Startup rejects
known legacy formats with migration instructions, and rejects unknown versions or
altered schemas before registering an owner. It never repairs a partial existing
database or silently adds new tables to one. Fresh database creation and owner
registration are transactional and support concurrent API-worker startup.

## Supported transitions

| Source | Target | Behavior |
|---|---|---|
| `legacy-pre-quiescence-v0` | `versioned-v1` | Add closed backend gates and raw/recovery tables, then client protocol registration |
| `legacy-quiescence-v0` | `versioned-v1` | Preserve existing gates and all business state; add client protocol registration |
| `versioned-v1` | `legacy-quiescence-v0` | Remove the version-1 registration handshake; preserve latest business state and gate generations |

The legacy schemas match the actual DDL at `c4e9cef`/`e70a5e2` and `7ffc8e3`,
respectively. Matching is by the complete SQLite structure and version, not just
the presence of selected tables. Older, future or customized structures are not
implicitly supported. There is no downgrade to the pre-quiescence format: that
would discard the admission gate and raw/recovery safety semantics.

Version 1 adds `owner_protocols` and a trigger on owner registration. A new runtime
registers protocol 1 and its owner in one transaction with a deferred foreign key.
Historical owner rows retain protocol 0 provenance. An old constructor that inserts
an owner directly fails before joining admission or calling the engine. This is
a compatibility guard, not an authorization boundary against direct database edits.
It cannot stop old code already executing; all owners must be stopped before migration.

## Upgrade procedure

1. Remove traffic, disable supervisors and complete the Jev drain handshake.
   Resolve retained/uncertain work through the normal recovery APIs. Unload managed
   adapters and stop every process that can open this registry.
2. Use the new tool to inspect and snapshot the original database. These commands
   do not instantiate a Registry or add an owner:

```sh
jevctl registry inspect /data/jev/registry.db
jevctl registry snapshot /data/jev/registry.db /backups/jev/stopped-001
```

3. Retain the returned manifest SHA256. Stage an upgrade in a new private directory:

```sh
jevctl registry stage-migration /backups/jev/stopped-001 MANIFEST_SHA256 \
  /data/jev/registry.db /data/jev/upgraded-001 \
  --target-format versioned-v1
jevctl registry verify-migration /data/jev/upgraded-001 RECEIPT_SHA256
```

4. Inspect the staged backend gates. For a QUIESCING backend, run the guarded
   `jevctl registry resume-backend` command using its exact generation. Configure
   only the new deployment to use `/data/jev/upgraded-001/registry.sqlite3`.
   Suppress bootstrap publication when preserving existing aliases. Require fresh
   worker preparation/readiness and real scoring checks before returning traffic.
5. Retire the original path from all startup configurations. Keep its unchanged
   data and the verified snapshot for diagnosis, not as another active quota pool.

The source must exactly match the snapshot, including owners, events and route
generations. A writer reservation blocks source changes during validation and
publication. Live or unverifiable owners, leases, branch/tenant journals, raw work,
recovery claims, admission tickets, PREPARING bundles, pending cancellation and
resident/uncertain adapters block migration. A STOPPED flag alone is insufficient.
Do not delete these rows to make the command pass.

The migration operates on a private staged copy, preserving every shared table's
row identities and content hashes. It checks integrity and foreign keys. The
source remains unchanged. New gates for the pre-quiescence format start QUIESCING
at generation 1; existing routes and their CAS generations are unchanged.

## Roll back code without rewinding state

Drain and stop the current version, then take a **new** snapshot of its latest
state. Stage the explicit reverse transition with the new tool:

```sh
jevctl registry stage-migration /backups/jev/latest-stopped MANIFEST_SHA256 \
  /data/jev/upgraded-001/registry.sqlite3 /data/jev/rollback-001 \
  --target-format legacy-quiescence-v0
jevctl registry verify-migration /data/jev/rollback-001 RECEIPT_SHA256
```

Use the selected compatible legacy runtime's resume command and start it against
the rollback copy. Do not point old code at a version-1 registry, and do not use
the pre-upgrade snapshot to erase acknowledged publications. New code rejects the
downgraded database until an explicit upgrade. This procedure preserves schema
state only; model, bundle, tokenizer, adapter, calibration and engine compatibility
still need their normal checks. It does not certify arbitrary code revisions.

## Interrupted staging and operating boundary

The destination begins with `migration-intent.json`. Schema changes occur in one
transaction on `registry.pending.sqlite3`; no final database name is published
while transformation runs. The tool fsyncs the database and publishes
`migration.json` before exclusively linking the final `registry.sqlite3` name.
New startup rejects an incomplete declared migration instead of creating an empty
database in that directory. The verifier checks the full inventory, intent/receipt
hashes, database hash/size, target schema and logical content before first startup.
Normal serving changes database contents, so re-verifying the original staging
hash after activation is expected to fail.

A failed command leaves its directory for diagnosis. Do not start either runtime
from an incomplete result; use a fresh destination when retrying. If interruption
occurred after publication, inspect and verify the complete artifact explicitly
before deciding to use it. The original database is never replaced automatically.

Keep supervisors fenced throughout cutover. The source writer reservation is
released when staging finishes; it cannot stop an external supervisor from later
starting a separate old copy. Staging is an offline maintenance operation with
interruption, not a zero-downtime cross-schema rolling upgrade. Cross-host restore,
changed boot/PID namespaces, lost-source recovery, replica coordination and future
schema transformations require separate support. Current migration owner checks
require the same Linux host, boot ID and PID namespace.

See [real DSW upgrade/rollback and interruption evidence](registry-migration-validation.md)
for the exact qualified source versions, workload and remaining boundaries.
