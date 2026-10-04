# Publishing Python packages to PyPI

The release workflow builds, validates, and publishes the four Python
distributions together:

- `jev-runtime-core`
- `jev-sglang`
- `jev-vllm`
- `jev-tokenspeed`

It uses PyPI Trusted Publishing. No long-lived API token is stored in GitHub.

## One-time maintainer setup

1. Make `LLM-Perf/jev-runtime` public.
2. Create a protected GitHub environment named `pypi`.
3. For each distribution above, create a pending publisher on PyPI with:
   - Owner: `LLM-Perf`
   - Repository: `jev-runtime`
   - Workflow: `publish-pypi.yml`
   - Environment: `pypi`
4. Require a maintainer approval on the `pypi` environment.

## Release

All four `project.version` values must be identical. Publish a GitHub release
whose tag is `v` followed by that exact PEP 440 version, for example
`v0.1.0a1`. The workflow rejects mismatched tags, runs strict metadata checks,
and uploads eight files: one wheel and one source distribution per project.

PyPI releases are immutable. If a workflow partially publishes, bump all four
versions before retrying; never reuse an already uploaded filename.
