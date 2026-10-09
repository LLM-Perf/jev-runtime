# Backend quiescence and ordered shutdown

Quiescence closes Jev admission for one exact backend in one local SQLite registry,
stops every API worker's periodic canaries, and drains accepted work before the
operator signals the native engine. It preserves route references, generations,
bundle digests and preparation state. It is distinct from disabling an alias.

This protocol addresses the SGLang shutdown race retained in
[response serialization validation](response-serialization-validation.md): sending
SIGTERM to the whole native process group can stop the scheduler before API-worker
health probes finish. ASGI lifespan cleanup alone cannot order that parent shutdown.

## Operator sequence

Use the normal API base URL for a gateway, or append `/plugins/jev-runtime` for a
native plugin. The following require the separate `JEV_ADMIN_KEY` environment variable.

1. `jevctl backend status URL` reads `generation`, `state`, worker states and outstanding
   lease/raw/adapter/recovery counts.
2. `jevctl backend quiesce URL GENERATION --timeout-seconds 30` atomically closes the
   backend gate. The result uses `drained: false` and CLI exit 1 on timeout. **A timeout
   leaves the gate closed**; neither a timeout nor a client disconnect resumes traffic.
3. Repeat status or quiesce using the returned generation. Do not signal the native
   engine until `drained: true`. Existing requests, including already queued requests,
   finish under their original snapshot; tenant-scoped cancellation remains available.
4. Stop the owned API/native processes and verify process identities and complete exit.
   `deployment/dsw_service.py stop --run-dir ...` now performs this handshake before
   SIGTERM to the native parent and rechecks the exact PID/start-tick/boot identity
   after the handshake. Do not signal the whole group: that kills the scheduler
   concurrently with API workers that still need their host's shutdown machinery.
   Its output still distinguishes signal delivery from verified process exit.
   Native vLLM receives an explicit 30-second teardown budget by default; override
   it with `--vllm-shutdown-timeout` at launch. Its native zero-timeout default means
   immediate abort, even after Jev has drained. The native budget starts after
   the Jev handshake; a container/service-manager stop deadline must cover both.
5. If upgrading an unversioned registry to the current runtime, first complete the
   [offline schema migration](registry-schema.md) with all old processes stopped.
   Before restarting a compatible registry, run
   `jevctl registry resume-backend REGISTRY_PATH BACKEND_ID GENERATION` locally.
   This CAS operation requires no retained work and every old worker STOPPED or
   verifiably dead. Unknown process identities fail closed. New workers revalidate
   the preserved active bundles before becoming ready; startup never silently reopens
   a quiescing backend.

The HTTP equivalents are `GET /admin/quiescence` and `POST /admin/quiescence` with
`{"expected_generation": 0, "timeout_seconds": 30}`. The timeout range is 0–300
seconds. A stale generation returns 409. Repeating with the current quiescing
generation is idempotent. Readiness returns 503 as soon as the durable gate closes.

## Ownership and failure behavior

- Decisions, canaries, preparation, adapter operations and publication check the
  gate in their existing authoritative writer transactions. A stale local control
  poll cannot admit new engine work. Starting/serving workers cannot join after
  the gate closes. Normal typed decisions retain the two-commit admission/release
  path; the gate adds a SQL read, not another fsync.
- Health loops wake when quiescence is observed. An already running probe is allowed
  to finish. A worker acknowledges QUIESCED only after its tracked preparation,
  scoring, management and cancellation tasks and health loop have finished, and its
  durable work is clear. New preparation waiting on a management lock rechecks the
  gate before dispatch. Cancellation polling continues during drain.
- Configured native-plugin raw scoring now records a durable `raw_work` row before
  dispatch. Normal completion or confirmed cancellation removes it; uncertain abort
  retains it. Inspect `GET /admin/raw-requests/recovery`, quiesce, then use
  `POST /admin/raw-requests/WORK_ID/recover`. A live/unverifiable in-flight owner
  cannot be recovered. Existing typed recovery remains
  `POST /admin/requests/REQUEST_ID/recover`.
- Recovery claims serialize concurrent administrative aborts and keep resume blocked
  throughout the operation. Inspect `GET /admin/recovery-operations` after a recovery
  worker crashes; retry that resource's recovery endpoint only after its owner is
  verifiably dead. A retained claim is not removed merely because time passed, even
  if the original work completed before the recovery process died.
- UNKNOWN adapters block completion. Explicit `recover: true` unload can reconcile an
  uncertain adapter while quiescing; normal new adapter work stays blocked. Existing
  route/lease checks still apply, so an operator may need to disable that adapter's
  routes deliberately before reconciliation.

## Deployment boundaries

The TokenSpeed launcher exposes these Jev administrative gates, but has no real
GPU shutdown certification yet. Use `jev-tokenspeed`, not the vLLM/SGLang-only
`deployment/dsw_service.py`, to start its owned process. After a successful Jev
drain, stop that parent through its owning terminal or supervisor and verify exit.
A failed drain remains an error even though lifespan cleanup shuts down the owned
Engine. Keep both its core registry and separate completion-receipt database;
[unknown/pending requests are not confirmed by elapsed time](operations.md#tokenspeed-operations).

All participating API workers must run this protocol version against the same local
registry. Known live legacy workers cause quiescence to fail. Do not start an older
binary against the registry after closing the gate: old software does not implement
its semantics. This is not an online mixed-version migration protocol.

The schema adds backend controls, worker protocol markers, raw work and recovery
claims. Existing registries acquire these additive tables when opened by this version.
The exact-schema backup verifier deliberately rejects snapshots from the prior schema;
keep old binaries and snapshots together. Downgrade/schema migration certification
remains open. Backups report retained raw/recovery work as restore blockers.

Scope is **Jev work for one registry/backend on one host**. Native chat/completion
traffic, other registries, another host, and raw-only plugins without a configured
Runtime are outside this handshake. Stop or drain those clients separately before
terminating their engine. Arbitrary SIGTERM, Kubernetes preStop/image behavior,
multi-node failover and 24-hour stability are not certified by CPU contract tests.
