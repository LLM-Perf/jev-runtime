# Publishing Python packages to PyPI

The release workflow builds, validates, and publishes the four Python
distributions together:

- `jev-runtime-core`
- `jev-sglang`
- `jev-vllm`
- `jev-tokenspeed`

It uses PyPI Trusted Publishing. No long-lived API token is stored in GitHub.

## One-time maintainer setup

1. Verify the repository's package publishing settings and permissions.
2. Create a protected GitHub environment named `pypi`.
3. For each distribution above, create a pending publisher on PyPI with:
   - Owner: `LLM-Perf`
   - Repository: `jev-runtime`
   - Workflow: `publish-pypi.yml`
   - Environment: `pypi`
4. Require a maintainer approval on the `pypi` environment.

## Release

All four `project.version` values must be identical. Prepare a draft GitHub release
whose tag is `v` followed by that exact PEP 440 version, for example
`v0.1.0a1`. The workflow rejects mismatched tags, runs strict metadata checks,
and builds eight files: one wheel and one source distribution per project.

Before publishing the release, attach `qualification.zip` in the layout documented
in [production qualification](production.md). Have a maintainer verify the evidence
and set repository variable `JEV_QUALIFICATION_SHA256` to the archive's SHA256.
The archive must contain candidate profiles for vLLM, SGLang and TokenSpeed at
the release commit, with hashes matching the exact wheels built by this workflow.
Use the same pinned build tools for qualification and release; wheel hash drift
requires rebuilding/requalifying the artifact, never editing an old test result.

Publishing the release triggers the reusable Python/TypeScript contract workflow,
clean wheel-install checks, and the eight-stage production gate. The upload job
depends on both the build and successful qualification. Missing evidence, a draft
profile, mismatched artifacts or failed tests prevents upload. The protected
`pypi` environment remains the final publishing approval boundary.

Evidence hashes prove integrity and version binding; they do not attest that an
operator truthfully ran a GPU test. Restrict release and repository-variable write
permissions and review the underlying logs before approving publication.

PyPI releases are immutable. If a workflow partially publishes, bump all four
versions before retrying; never reuse an already uploaded filename.
