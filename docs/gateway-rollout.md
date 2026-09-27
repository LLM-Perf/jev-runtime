# Gateway code rollout

`jevctl rollout` manages two gateway slots behind a stable, local HAProxy listener.
Each slot may contain multiple uvicorn workers. Slots use the same local SQLite
registry and the same native engine. The proxy selects a slot once per HTTP
request; existing streams retain their backend. A later request on an existing
client connection evaluates the current selection again.

This implements a gateway traffic-switching primitive. It does not migrate a
registry schema, roll an engine/CUDA plugin, replace base weights, or recover
from host/proxy failure. Those changes require separate redundant infrastructure.
The qualification record must distinguish same-code replica replacement from
an actual old-wheel/new-wheel upgrade. Two differently named slots do not prove
two different code artifacts were tested.

## Prerequisites

- Linux, Python 3.11 or newer (3.12 for the bundled source build helper), local
  filesystems, matching native-engine and model identity, identical tenant keys
  and admission policies, and an authenticated gateway in each slot.
- Each gateway config has a unique `deployment_id` for its slot incarnation and
  an immutable `release_id` identifying its verified code artifact. Both set
  `workers` to the actual number of uvicorn workers. Do not reuse an incarnation
  for a different release. The release ID is a declaration, not code attestation;
  the installer must verify source/wheel hashes separately.
- Only the owned HAProxy listener receives application traffic. Gateway ports
  and the mode-0600 Runtime API socket remain private to the service account.
  All administration of that proxy uses one controller directory. External
  Runtime API writers, HAProxy reloads, and gateway changes must be serialized
  by the service manager with rollout operations.
- Keep the old slot running and healthy until rollback is no longer needed.
  Do not restart a proxy while an operation is pending: reconcile first whenever
  possible. Proxy failure can interrupt streams and is outside this guarantee.

The qualified proxy is HAProxy **3.2.24**, pinned to upstream source SHA256
`f765638cc4819f25e20d974a4a1bc24ed54342467c88363d7fc34a1fa95b725a`.
Build in a new directory using `python3.12 deployment/build_haproxy.py /srv/jev/proxy-build`.
This requires a C compiler, make, OpenSSL and zlib development headers, and makes
no host package changes. [Upstream source](https://www.haproxy.org/download/3.2/src/)
and [checksum](https://www.haproxy.org/download/3.2/src/haproxy-3.2.24.tar.gz.sha256).

## Prepare and start

Example gateway configuration additions (blue; green uses a distinct deployment):

```yaml
deployment_id: blue-release-20260928
release_id: SOURCE_COMMIT_OR_WHEEL_DIGEST
workers: 2
registry_path: /srv/jev/registry.db
host: 127.0.0.1
port: 8796
```

Start both gateways with their verified environments. Keep their engine URL,
model revision, tokenizer/template fingerprints, keys and shared registry equal.
`JEV_API_KEY` and `JEV_ADMIN_KEY` for the controller come from the environment;
credentials are never written into the proxy configuration.

```sh
jevctl rollout init /srv/jev/router 8796 8797 8795
/srv/jev/proxy-build/haproxy-3.2.24/haproxy -c -f /srv/jev/router/haproxy.cfg
/srv/jev/proxy-build/haproxy-3.2.24/haproxy -db -f /srv/jev/router/haproxy.cfg
```

Run HAProxy under the service manager in the last command; do not start a second
copy during rollout. The listener is loopback HTTP, intended behind the existing
TLS/authenticated ingress. The Unix-socket directory must use a short path without
spaces. `init` requires a ready blue group and refuses any existing destination.
Initialization failure leaves an explicit partial directory to inspect.

## Switch and rollback

Prepare a normal `DecisionRequest` JSON canary for an active alias. The selected
canary must represent the application's scoring contract; a successful generic
boolean request is not a business-quality regression suite.

```sh
jevctl rollout status /srv/jev/router
jevctl rollout switch /srv/jev/router green 0 deploy-20260928 green-release-20260928 SOURCE_COMMIT_OR_WHEEL_DIGEST canary.json
jevctl rollout drain /srv/jev/router deploy-20260928 --timeout 60
```

The switch requires the current rollout generation, a unique operation ID and
the exact candidate deployment/release. It samples **every worker** on fresh
HTTP connections; starvation, unknown process identity, missing workers, mixed
releases and starting/stopped workers fail qualification. Every worker must
have current health evidence for all active aliases. The registry inode, engine,
model, capabilities, routes, bundle digests and keyed authentication-policy
fingerprint must agree. A real candidate decision must complete all questions
on the expected bundle digest/generation. Both groups are rechecked afterwards.
Concurrent bundle publication still uses the registry's all-serving-worker
preparation barrier. There is no atomic transaction spanning HTTP health probes,
the registry and the proxy; a crash or new health failure after qualification
remains possible and must be detected by normal health monitoring.

Rollback is the same command targeting blue with the **new** generation, a new
operation ID and blue's expected incarnation/release. An unavailable old version
is not silently reactivated. Requests are never retried onto the other slot.
`X-Jev-Deployment` identifies the proxy backend; `X-Jev-Worker` identifies the
gateway worker. Bundle digest and generation remain in each decision response.

## Interrupted controller

A single filesystem lock serializes controller mutations. State, map and journal
updates use fsync plus atomic replace. Before changing the live map, the controller
persists an intent. Live map update, disk map update and final receipt are three
separate operations, so a crash can leave a pending intent.

```sh
jevctl rollout status /srv/jev/router
jevctl rollout reconcile /srv/jev/router
```

Reconcile observes the live selection, revalidates that exact source/target
incarnation, persists the observed disk map and consumes one generation. It never
replays the switch. If the process stopped before map update, reconciliation may
commit the original slot. Inspect `observed` in the receipt. Retrying an operation
ID with identical arguments returns its historical receipt; different arguments
are rejected. A receipt does not claim that its slot is still active today.

If a registry/worker change prevents reconciliation, preserve the pending state
and investigate. Do not delete the intent, edit map files, or bypass it with an
unrecorded Runtime API command. The proxy continues with its last live selection
while the controller is unavailable.

## Drain and retirement

The bundled proxy deliberately uses one event loop (`nbthread 1`) so Runtime API
selection and backend stream accounting have one serialization point. It retains
HTTP keep-alive and upstream connection reuse. No yielding frontend rule precedes
backend assignment. Do not alter this topology without requalifying drain semantics.
The single-thread limit must be included in performance testing; current functional
tests do not certify the eventual throughput/SLO target.

Drain checks both backend/server active streams and queues, plus durable leases
for every historical worker in the departing incarnation (including dead owners).
It rechecks stream counters after the registry read. This includes a request whose
body has not yet reached the gateway parser and therefore has no registry lease.
Unconfirmed abort leases prevent a successful drain. A timeout returns
`drained: false` and a nonzero CLI status, preserving the old process.

The controller **never signals a gateway**. After successful drain, the service
manager must keep its deployment lock, prevent direct/admin traffic and gracefully
terminate that recorded old process group. The observation is not a permanent
permission to kill: a subsequent rollback, direct request or new worker incarnation
invalidates it. Normal graceful server shutdown must finish before retiring its
environment. The existing runtime's `close()` actively cancels remaining work;
do not invoke it as a substitute for HTTP and durable-work drain.

## Validation

`tests/test_rollout.py` covers crash windows, lost acknowledgements, duplicate IDs,
stale generations, candidate mismatches, all-worker qualification and retained work.
`tests/integration/live_rollout.py` starts real HAProxy and two two-worker gateways,
checks partial-body drain, persistent client connections, continuous switches,
bad-release rejection, controller interruption and unchanged proxy PID. Its CPU
fixture additionally checks peer cancellation across a switch. Native mode accepts
an existing isolated engine run directory and uses real inference; it preserves
the native engine for its owning campaign to stop. Reports state fixture/native
mode, exact source, process identities, traffic errors and cleanup.

Primary HAProxy references: [Runtime API and statistics](https://docs.haproxy.org/3.2/management.html),
[HTTP routing and map configuration](https://docs.haproxy.org/3.2/configuration.html).
