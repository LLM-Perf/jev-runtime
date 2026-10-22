# Production qualification and operation

Production acceptance belongs to an exact framework/model/hardware/artifact
combination. The typed APIs and three integrations are implemented; the initial
production profiles are drafts until real acceptance evidence is supplied.
No existing historical report is automatically promoted to a new release.

## 1. Install and validate the distributable

Install the core tokenizer extra and the matching engine adapter. The tokenizer,
conversion and reference extras explicitly include Jinja2 for chat templates.
The CI matrix runs Python 3.11 and 3.12 without fail-fast, then builds all four
wheels and installs them into a second, clean environment. It runs `pip check`,
offline chat-template rendering, backend discovery and CLI smoke tests using
`python -I`, so editable source and the developer's installed dependencies cannot
hide missing wheel contents or runtime dependencies.

For a local wheel-only check after building into `dist/`:

```sh
python -m venv /tmp/jev-wheel-check
/tmp/jev-wheel-check/bin/python -m pip install \
  'dist/jev_runtime_core-0.1.0a1-py3-none-any.whl[tokenizers,tokenizer-conversion]' \
  dist/jev_vllm-*.whl dist/jev_sglang-*.whl dist/jev_tokenspeed-*.whl
/tmp/jev-wheel-check/bin/python -m pip check
/tmp/jev-wheel-check/bin/python -I deployment/wheel_smoke.py
```

Use the actual built version instead of `0.1.0a1` for a later release. Never reuse
an existing check directory. These checks do not install inference engines.

## 2. Define a candidate and its evidence

Start from `profiles/production/{vllm,sglang,tokenspeed}-initial.json`. Drafts are
valid planning records but can never pass qualification. Before setting
`status: candidate`, supply all fields defined by `ProductionProfile` in
`src/jev_runtime/production.py`:

- Full source commit, four wheel SHA256 values, and deployment image digest.
- Engine version, immutable model revision, tokenizer/template/config hashes.
- GPU and compute capability, model/readout dtype, quantization, TP, attention
  backend, execution mode, maximum context/candidates/concurrency.
- Business-approved latency/throughput, numerical error and recovery targets;
  resource-growth budgets; two distinct held-out quality datasets and thresholds.

Generate a machine-readable schema from the installed package:

```sh
python -c 'import json; from jev_runtime.production import ProductionProfile; print(json.dumps(ProductionProfile.model_json_schema(), indent=2))' > production-profile.schema.json
jevctl production validate-profile /releases/profile.json
```

Schema validation alone never certifies a deployment. The gate has these minimum
requirements, which a profile cannot weaken:

| Stage | Required evidence |
|---|---|
| installation | Clean wheels, template and CLI smoke, `pip check`, both Python versions, zero failed tests |
| functional | At least 10,000 requests, >=99.9% strict success, four types, auth isolation, hot updates, confirmed drain, retained work on uncertain cancellation |
| numerical | Independent reference, native/plugin parity, cold/hot and batch-order coverage, >=100 non-tie samples, declared error bounds and agreement |
| quality | Two held-out dataset identities, sample counts, accuracy, coverage and ECE targets; abstentions remain in coverage accounting |
| performance | Open-loop traffic, complete telemetry, three >=180s repeats, context/K/concurrency coverage, equivalent outputs, >=90% native throughput |
| recovery | Worker/engine kill, container replacement, activation crash, write failure, latest-state restore, stale-owner fencing and rollback; no lost acknowledged publication or early resource release |
| image | Exact deployment digest, built/running image, non-root and persistence checks, probes and SIGTERM drain |
| soak | >=24h actual interval, mixed input/tenants, hot updates, resource budgets and separately reported fault periods |

These are acceptance targets, not achieved benchmark results. Supply real latency,
throughput, quality and recovery values for the intended workload; arbitrary sample
values must never be used to qualify a release.

Every `StageReport` binds its source, complete observed identity and the exact
profile file SHA256. It contains start/end times, `passed`, metrics and a nonempty
map of raw artifact paths to SHA256 values. Store reports as `installation.json`,
`functional.json`, etc. alongside their artifacts. Paths must remain inside the
evidence directory; missing/changed files and symlinks are rejected. Failed and
incomplete campaigns must remain retained. Do not hand-convert a smoke run into a
soak report or treat synthetic test fixtures as model evidence.

```sh
jevctl production qualify /releases/profile.json /releases/evidence \
  FULL_SOURCE_COMMIT /releases/wheels
```

All raw campaign producers currently retain their own report formats. A release
maintainer must review those outputs and assemble the corresponding `StageReport`
with its raw artifacts. This gate checks evidence and targets, not independent
attestation or unattended GPU provisioning.

For publishing, archive the following relative paths in `qualification.zip`:

```text
profiles/vllm-example.json
profiles/sglang-example.json
profiles/tokenspeed-example.json
evidence/<profile_id>/installation.json
evidence/<profile_id>/functional.json
evidence/<profile_id>/numerical.json
evidence/<profile_id>/quality.json
evidence/<profile_id>/performance.json
evidence/<profile_id>/recovery.json
evidence/<profile_id>/image.json
evidence/<profile_id>/soak.json
evidence/<profile_id>/<raw artifacts referenced by reports>
```

The release archive is limited to 256 MiB uncompressed. Keep large private traces
in controlled storage; include an audited export with trace identity, coverage,
metrics and durable retrieval information in the archive. Do not publish prompts,
credentials or private SQLite contents. See [PyPI release](pypi-release.md) for the
reviewed archive digest and workflow dependency.

## 3. Run common traffic against all three engines

The same `benchmarks.production_traffic` entrypoint supports vLLM, SGLang and
TokenSpeed. It does not launch, kill, reconfigure or switch any service. Start an
isolated, configured service first. Use the existing `tests/integration/live_contract.py`
for the four types, bundle lifecycle and strict hot-switch functional campaign;
its TokenSpeed branch uses the native raw-scoring bridge.

Prepare a JSONL file of explicit `DecisionRequest` objects with varied inputs and
all four question types. Prepare a JSON list of allowed route tuples read from
the qualified service, with `bundle`, `bundle_digest`, and `generation`. The three
fields are checked together; independently allowed generations/digests cannot be
combined into a mixed version. Export the data-plane credential as `JEV_API_KEY`.

```sh
python -m benchmarks.production_traffic \
  --url http://127.0.0.1:8795/plugins/jev-runtime \
  --engine tokenspeed --engine-version EXACT_REPORTED_VERSION \
  --requests /validation/requests.jsonl --routes /validation/routes.json \
  --rate 10 --duration 180 --concurrency 4 --queue-size 16 \
  --output /validation/run-001
```

Replace `--engine` and the exact observed version for vLLM/SGLang. A gateway uses
its base URL without the plugin suffix. Select rate/concurrency from the intended
profile, then sweep them; the values above only illustrate syntax. For long-run
collection use `--duration 86400`, with separately coordinated telemetry and fault
campaigns. Do not use a shared busy GPU to claim controlled performance.

Every planned arrival remains in the denominator, including client queue overflow,
429, transport errors and invalid HTTP 200s. Validation checks request identity,
engine/version, bundle tuple, completed status, question coverage, candidate sets,
answer types and numeric score levels. Queue delay is included in latency to avoid
hiding overload. Raw rows stream to a private JSONL; latency samples use disk-backed
SQLite to keep memory bounded. Output includes exact planned/accounted/sent counts,
strict success, errors, P99, input/route/log hashes and measured throughput.
The standalone command exits nonzero for any non-success; release SLO thresholds
are evaluated separately by the production gate. A cancelled/failed load generator
records an incomplete report and cannot pass.

This collector alone does not prove numerical quality, concurrent hot-switch
atomicity, telemetry completeness or fault recovery. Do not fill those gate fields
from its success count.

## 4. Back up and restore TokenSpeed state together

Remove traffic, quiesce the backend, resolve uncertain work and stop every process
that can access the registry/receipts. Keep supervisors disabled. With the latest
stopped registry and its sibling `*.tokenspeed-receipts.db`:

```sh
jevctl registry snapshot-pair /data/jev/registry.db /backups/jev/pair-001
jevctl registry verify-pair /backups/jev/pair-001 MANIFEST_SHA256
jevctl registry stage-restore-pair /backups/jev/pair-001 MANIFEST_SHA256 \
  /data/jev/registry.db /data/jev/restored-001
```

Parents must exist; destinations must be new. Both database writer reservations
are held during copying/validation. Any live or unknown owner, retained work or
pending native receipt prevents the operation. The snapshot uses SQLite backup
APIs and captures committed WAL state. Restore checks that **both** original stores
still match the snapshot and preserves every route generation and duplicate-request
receipt. It never activates traffic or changes the original stores.

A successful restore produces `registry.sqlite3`, its correctly named receipt
sibling and an atomic completion record. A staged pair missing its completion
record cannot open as a runtime registry. Keep the old deployment stopped after
the command returns; its short-lived SQL locks cannot fence a future supervisor
restart. Start only the intended restored path with the same engine/model binding,
then require fresh readiness before switching traffic.

## 4a. Fence a lost host or replaced container

Same-host process death is already classified as `dead`. A reboot, a new
container PID namespace, or a registry opened on another host leaves those
owners `unknown`. Unknown owners continue to block recovery, restore and
admission. Fence them only after the previous deployment is stopped and its
supervisor cannot restart it:

```sh
jevctl registry fence-owners /data/jev/registry.db lost_host \
  previous-deployment-stopped
```

Use `container_namespace` when the host and boot ID still match but the PID
namespace changed, and `unverifiable_identity` when the recorded identity has
no boot ID, PID namespace or start ticks. The command refuses a reason that
does not match every unknown owner, and it refuses while any owner is
verifiably alive in this PID namespace. It writes a `deployment_fence` event
and does not delete leases, journals or receipts.

A process of this runtime version that registered before that generation fails
its next registry write with `owner_fenced` and fails `/ready`. `/live` stays
a process-liveness signal. Repeat the command only after another boot or
namespace change; while the same observation still covers those owners, the
second call does not advance the generation.

This is not a distributed lock. A copy of the database cannot stop a process
that is still serving another copy, and a binary that does not perform this
generation check will ignore the fence. Do not start traffic until explicit
recovery has released every fenced lease and a new process reports ready.

## 4b. Retire completed TokenSpeed receipts

Completed receipts are idempotency records. Deleting them would let the same
native request ID be dispatched again. Retirement moves completed,
unreferenced IDs into `retired_receipts` in the receipt database and appends
a `receipts_retired` audit event, in one transaction across both files.
`reserve` of a retired ID still fails closed. `completed` stays false for a
retired ID, so retirement is never treated as fresh completion proof.
Pending rows are never removed. There is no time-based expiry.

Stop every writer, fence or confirm every owner dead, and drain leases first.
The registry must have no restore blockers:

```sh
jevctl registry retire-receipts /data/jev/registry.db NAMESPACE 1024 \
  retire-completed-unreferenced-receipts
```

`NAMESPACE` is the TokenSpeed receipt namespace (a `sha256:` digest of the
engine binding). Each call retires at most 1,024 IDs. Repeat until it reports
that nothing completed remains. Then take a new paired snapshot; the tombstone
table is part of that snapshot. Do not hand-delete either SQLite file.

## 5. Build images and operate probes/metrics

On a separate Linux Docker builder with Python 3.12, `build`, `hatchling` and core
dependencies installed:

```sh
python -m deployment.container_smoke --ref FULL_SOURCE_COMMIT \
  --output /validation/container-001
```

The tool builds committed wheels, resolves a Linux lock, resolves the official
Python image to a digest, prepares the offline context and performs an actual Docker
build. It runs with UID 10001, no network, read-only root and restricted capabilities;
checks tokenizer rendering/plugin discovery; and verifies persistence across two
containers. It removes only its unique test volume and image tag. GitHub's
`Gateway container packaging` workflow runs the same procedure and retains logs.
Its local image **config** digest is not a registry manifest digest. GPU serving,
readiness, engine images and deployment rollbacks require separate evidence.

Use `/live` (or container `livecheck`) for event-loop liveness. Use `/ready` (or
container `healthcheck`) for accepting traffic after current GPU canaries pass.
Both retain authentication. Engine unavailability makes readiness fail without
making liveness fail. Configure startup grace and termination grace for actual
model startup and drain times. Do not restart a healthy event loop merely because
GPU work is saturated.

`/metrics` now exposes registry lease/journal counts, oldest lease age, database/WAL
size and per-worker canary state; `X-Jev-Worker` identifies the sampled worker.
For the first production configuration use one API worker per scrape endpoint,
or provide a separate endpoint for every worker. Randomly scraping a multiworker
listener does not collect global counters. Use `max`, not `sum`, for copies of
registry-wide gauges and shared admission. Add `deployment` and `registry` target
labels, and load `deployment/prometheus-rules.yaml`. The shipped 0.1% failure,
300-second retained-work and WAL-growth alerts are initial operational thresholds.
Tune them to the approved profile. Collect ingress 401/422/429/5xx, GPU telemetry,
disk capacity and TokenSpeed receipt-store growth separately.

## Remaining acceptance work

Fence and receipt retirement are executable and covered by CPU contracts. They
do not close the real-GPU, business-data or infrastructure gates. The
2026-10-14 read-only check of the existing project DSW found SM89 GPUs with
other workloads occupying most memory; it is unsuitable for a controlled
capacity claim and does not meet the pinned TokenSpeed profile's GPU
requirements. A free supported GPU and business quality datasets/targets are
still required before a candidate profile can qualify. The drafts and
historical performance matrix remain unpassed until new results are collected.
No 24-hour production soak is claimed by adding its runner. No cross-host
quorum is claimed for the fence.
