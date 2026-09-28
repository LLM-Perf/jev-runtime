# Start a runtime for explicit recovery

Set `recovery_only: true` to expose inspection and administrative recovery when
normal startup would revalidate an active bundle or wait on a crashed PREPARING
bundle. This mode probes the configured engine and retains precision/execution
checks, but does not join admission/publication, prepare bundles, start canaries,
load adapters or bootstrap a route. It never declares Jev readiness.

This is an explicit maintenance startup mode. Restart the process to leave it;
there is no HTTP switch that enables serving inside a recovery process.

## Procedure

1. Preserve the failed registry with a consistent private snapshot and retain the
   old engine/API process identity and dispatch journal. Do not delete leases or
   edit immutable bundle manifests.
2. Confirm which engine target owns the recorded request IDs. For a native engine
   restart, verify the entire original owned process group has exited before
   starting its replacement. Preserve model/revision/endpoint identity and use an
   engine configuration that actually starts. A changed execution profile must
   have newly built bundles/calibrators before serving; recovery does not migrate
   their identities.
3. Start with the original registry and `recovery_only: true`, using separate
   data/admin keys. In this repository's isolated launcher, use an explicit
   `--registry-path /absolute/existing/registry.db --recovery-only` and a fresh run
   directory. The explicit path must already be a regular file, not a symlink.
   The launcher does not copy or restore a database. Keep normal settings such as
   the engine URL, model revision, tokenizer, dtype and admission limits explicit.
4. Inspect authenticated `GET /admin/profile`, `/admin/bundles`, `/admin/workers`
   and `/admin/requests/recovery`. The profile reports `recovery_only: true`;
   `/ready` returns 503 `recovery_only`. Native plugins use the usual
   `/plugins/jev-runtime` prefix.
5. Call `POST /admin/requests/{request_id}/recover` with the admin key. The existing
   recovery implementation checks the original backend identity, recorded phase
   and owner status, then awaits cancellation of every journaled native branch.
   Only confirmed cancellation permits the lease CAS cleanup. An inflight request
   with a live or unverifiable owner remains retained; a failed cancellation also
   leaves the lease. An absent request returns `recovered: false`.
6. Confirm leases/work/admission reservations are drained and the failed
   PREPARING bundle has become FAILED. Stop the recovery process. Restart a normal
   runtime with the intended verified profile and prepare/activate appropriate
   immutable bundles. `--no-bootstrap` permits explicit administrative publication
   when an old default bundle must remain inspectable. Confirm Jev readiness and
   real decisions before restoring traffic.

The recovery API does not force reclaim by elapsed time, bypass a dead-owner
check, guess process ownership from a port, or mark unfinished engine work as
cancelled. It does not modify adapter residency records. A recovery process is
not a prepared rollout candidate and must not receive decision traffic.

## API boundaries

Authenticated reads remain available. In recovery mode, Jev decisions, System One
requests, compile previews, bundle publication/activation/disable/retire, adapter
mutations and the native plugin's raw `/v1/scores` endpoint return 503
`recovery_only`. Recovery still requires the separate admin credential. The
internal raw cancellation endpoint remains available for attached recovery
clients; its existing data/service credential requirement is unchanged.

The host engine's own chat/completion/generate APIs are outside Jev's router and
are not disabled by this setting. Drain or isolate native traffic using the
engine deployment's normal controls. This mode also does not solve an engine
that cannot initialize, host/PID-namespace ambiguity, destroyed-source recovery,
container replacement, schema migration or replica failover.

## Validation scope

Local contract tests exercise active and PREPARING bundles, no startup scoring or
admission enrollment, all guarded writes, native raw scoring rejection, separate
admin authentication, live-owner rejection, unconfirmed-abort retention,
confirmed recovery, and retained precision/mode/cancellation capability guards.
Real DSW evidence is recorded separately; local doubles do not establish GPU
cancellation or restart behavior.
