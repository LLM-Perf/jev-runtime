# Gateway image preparation

`deployment.image_context` replaces the old source-install Dockerfile with a
snapshot of an already verified Linux gateway wheelhouse. It requires an explicit
base-image digest and emits a separate context manifest. The gateway contains no
engine wheels or CUDA dependencies. Engine images and their dependency profiles
remain separate deliverables.

**Current evidence:** context integrity and startup/probe tests pass locally.
Actual BuildKit image construction and container runtime validation have not run;
no usable image tag or image digest is claimed. A build context is not an image.
The existing DSW workload is already a container and must not host nested Docker.
Use a separate Linux BuildKit/Docker builder for the remaining image checks.

## Prepare and build

First create the gateway lock using [release packaging](release-packaging.md) on
the deployment's Linux Python ABI/architecture. Select a trusted Python base image
containing the same Python ABI, glibc-compatible OS libraries, `venv`, `ensurepip`
and `python` on PATH. Resolve and review its immutable digest. Preparation validates
the reference syntax; it does not contact a registry or attest the publisher.

```sh
python -m deployment.image_context prepare \
  --bundle /releases/gateway-COMMIT \
  --manifest-sha256 GATEWAY_MANIFEST_SHA256 \
  --base-image 'docker.io/library/python@sha256:BASE_MANIFEST_DIGEST' \
  --output /releases/image-context-COMMIT
python -m deployment.image_context verify \
  --context /releases/image-context-COMMIT \
  --manifest-sha256 CONTEXT_MANIFEST_SHA256

docker buildx build --platform linux/amd64 --network=none --load \
  --metadata-file /releases/image-build-COMMIT.json \
  --tag jev-gateway:COMMIT /releases/image-context-COMMIT
```

Replace every uppercase placeholder with the emitted/verified value. Use the
context's `platform` value (`linux/amd64` or `linux/arm64`) in the build command.
The base-image pull may need registry access; every Dockerfile `RUN` executes with
network disabled. Installation uses the copied hash lock, not PyPI. The same base
digest is used in the builder and final stages so the copied virtual environment
retains matching interpreter/library paths. Base-image digest pinning and
multi-stage copies follow the [Dockerfile reference](https://docs.docker.com/reference/dockerfile/)
and [Docker build guidance](https://docs.docker.com/build/building/best-practices/).

The installer verifies the target ABI, complete distribution inventory, core Python
bytes and imports before writing `/opt/jev/venv/jev-install-complete.json`. The
final image retains that receipt and `/opt/jev/image.json` (source, dependency,
base and helper hashes); build wheels/logs stay out of the final stage. These
records identify accepted inputs, not bit-for-bit reproducibility or an image
security certification. Record the actual output image digest and build metadata
separately after a successful build.

## Runtime contract

Mount the configuration and tokenizer read-only, and a host-local persistent
registry directory writable by UID/GID `10001:10001`. The configuration must set:

- `release_id` to `image.json`'s exact core `source_commit`.
- An absolute `registry_path` on the persistent local volume.
- An absolute `tokenizer` path with its matching, verified tokenizer profile.
- An engine URL reachable from the container and the exact loaded model revision.
- `host: 0.0.0.0` (or `::`) for externally exposed service; `port: 8795` by default.
- The desired worker count and normal credential environment-variable names.

Use the normal separate data-plane, admin and engine keys as runtime secrets;
never bake them into the context or image. `JEV_CONFIG` defaults to
`/config/gateway.yaml`. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` make startup
use the supplied tokenizer artifacts. Weights remain owned by the separate engine.

The entrypoint replaces itself with the Uvicorn supervisor, allowing SIGTERM to
reach it directly. `JEV_GRACEFUL_TIMEOUT_SECONDS` defaults to 180 (range 1–3600);
configure the orchestrator's termination grace longer than that. Removing traffic,
observing HTTP/lease drain and confirming cleanup remain required before retirement.
A hard shutdown must retain uncertain leases for the existing recovery procedure.

The image's health probe uses the configured data-plane credential and checks
`/ready` for an active prepared bundle and current engine canary evidence. It
samples one worker. It does not replace all-worker qualification by the rollout
controller. Probe diagnostics omit response bodies, headers and credential values.

Run the registry-sharing gateway workers within one container's PID/UTS namespaces.
The existing same-host rollout evidence does not certify two containers sharing
SQLite: default container PID namespaces and hostnames differ, and owner-death
verification may deliberately return unknown. Do not override those guards or
copy a live SQLite file to force a cross-container rollout. Compose/Kubernetes
examples, ownership/recovery across container replacement, persisted-state upgrades,
image rollback, engine images and actual two-engine container serving still need
validation before deployment acceptance.
