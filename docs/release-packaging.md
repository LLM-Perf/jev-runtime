# Immutable packages and offline gateway installation

`deployment/release.py` builds all three project wheels from an explicit Git
commit, verifies their Python files against that commit, and records SHA256 hashes
and build-tool versions. It can resolve a platform-specific gateway wheelhouse,
then create a fresh isolated environment using only that wheelhouse. It does not
change an installed SGLang/vLLM environment or activate traffic automatically.

These are development release artifacts. Release acceptance still requires the
model, numerical, business quality, performance and lifecycle gates in the plan.
The source commit and manifest SHA256 identify an artifact even when the package
version is still the same prerelease number. Never use version number alone to
identify two development builds.

## Build and resolve

From a trusted checkout with the development extra installed:

```sh
python deployment/release.py build --repo . --ref FULL_COMMIT_SHA --output /releases/wheels-COMMIT
```

The output must be new. Building uses an isolated Git snapshot and the builder's
installed `build`/`hatchling`, with no automatic build-dependency resolution.
Uncommitted files are excluded. The final `manifest.json` records the actual
build-tool versions, builder script hash, source commit and wheel hashes. Keep
the emitted manifest SHA256 in the release record through a trusted channel.
This proves artifact integrity against that supplied digest, not publisher
signatures or bit-for-bit reproducibility across different build environments.

Resolve on the same Python ABI, OS and architecture as the intended deployment:

```sh
python deployment/release.py resolve \
  --bundle /releases/wheels-COMMIT --manifest-sha256 WHEEL_MANIFEST_SHA256 \
  --output /releases/gateway-COMMIT \
  --constraints /releases/approved-versions.txt
```

The optional constraints file accepts only `name==version` lines. Resolution
uses the public PyPI index, excludes source distributions and ignores inherited
pip/Python configuration. It does not install dependencies into the resolver's
environment. The result contains the core wheel, tokenizer dependencies, all
resolved transitive dependencies, exact versions and SHA256 hashes in
`requirements.txt`. The builder's pip version and, when present, setuptools
version are pinned as well. Resolution success alone is not compatibility proof;
validate the resulting environment with the actual service.

SGLang and vLLM plugin wheels are delivered in the first bundle; they do not pull
engine dependencies into this gateway environment. Engine images/environments
still need their separately certified version and CUDA dependency profiles.

## Verify and install offline

Transfer the complete gateway directory and its manifest digest. Use the matching
Python interpreter and a trusted copy of the installer:

```sh
python deployment/release.py verify \
  --bundle /releases/gateway-COMMIT --manifest-sha256 GATEWAY_MANIFEST_SHA256
python deployment/release.py install \
  --bundle /releases/gateway-COMMIT --manifest-sha256 GATEWAY_MANIFEST_SHA256 \
  --output /envs/gateway-COMMIT
```

Installation uses `--no-index`, `--require-hashes`, `--only-binary=:all:` and the
complete pinned package list, followed by `pip check`. It checks the exact
installed distribution inventory and every core Python file against the wheel,
then imports the API/tokenizer dependencies. It writes `jev-install-complete.json`
only after these checks pass. It rejects an existing destination, payload changes,
extra payload files, symlinks, incorrect manifest digests and incompatible targets.
A failed operation keeps its incomplete output/log for diagnosis; it cannot be
retried onto that same directory. Do not serve an incomplete environment.

The interpreter and operating-system libraries are external prerequisites; this
is not a container image or OS-level dependency lock. See
[gateway image preparation](container-images.md) for digest-bound build contexts
and the remaining container validation gates. The first environment
creation uses that interpreter's bundled `venv`/`ensurepip` before installing the
locked pip wheel. Hash-checked installation follows the
[pip installation guidance](https://pip.pypa.io/en/stable/topics/secure-installs/).

To audit the installed environment later:

```sh
python deployment/release.py inspect-install \
  --bundle /releases/gateway-COMMIT --manifest-sha256 GATEWAY_MANIFEST_SHA256 \
  --environment /envs/gateway-COMMIT
```

## Upgrade and rollback contract

Keep application configuration, credentials and the SQLite registry outside the
versioned environment. Start a candidate on a different local port, using a
compatible registry/migration policy, then require real readiness and a typed
scoring canary. Existing routes and versions must be revalidated by each worker;
a successful Python import is insufficient.

Select the validated process through the deployment's traffic router, drain old
requests, and retain the previous environment and its original manifest digest.
Rollback selects that retained environment and revalidates it before traffic is
sent. Do not `pip install --upgrade` over a running process or swap its tokenizer.
See [registry migration and worker coordination](operations.md).

This tool supplies the immutable installation boundary. Automatic traffic
switching, compatible-replica failover, database migration/downgrade and a complete
zero-downtime rollout remain separate work. A stop/start rollback exercise must
be reported with its interruption window and must not be called uninterrupted
service. Never roll back across an incompatible database/schema or scoring
identity change merely because an older wheel is available.

The first real installation and maintenance rollout on both engines is recorded in
[the DSW validation report](release-rollout-validation.md), including exact artifact
commits, dependency counts, interruption intervals and preserved failed attempts.
