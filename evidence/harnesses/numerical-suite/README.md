# Numerical suite campaign at 877f319

The `.py.txt` files preserve the exact executed orchestration source. They are
evidence, not generic launch commands: paths, source archive digest and run names
are frozen, and previous output directories must never be overwritten.

- `campaign.py.txt` launches one engine at a time and captures 32 inputs across
  three states. Its SGLang postcheck calls package metadata from the vLLM parent
  environment and fails after scoring/live profile capture; cleanup succeeds.
- `finish_references.py.txt` preserves that failure, checks terminal engine
  identities and saved live postcheck profiles, records each environment with
  its own interpreter, and completes SGLang references using the original data.
- `audit_export.py.txt` verifies model files against the pinned public Hub
  metadata, tokenizer profiles, final registries, source hashes and 171 historical
  owned process identities/groups. It exports selected JSON reports only.
- `model-metadata.json` is the public Hub metadata for the fixed model revision.

The product runtime/plugins are unchanged from the previous source. The generic
collector/reference/comparison tool and its CPU tests are committed at
`877f319cf03932193db362a9475bd6a5aebc723d`. Native engines import the source files
from that exact checkout through an explicit `PYTHONPATH`; this campaign does
not claim a new installed-wheel test.

The transferred evidence archive had SHA256
`8d4fe2a110930c7c6d0e11f8471d6d5c55365ad80e90efce8db4281e367cd96c`
and size 81,398 bytes. `export-manifest.json` binds every exported payload;
`verified-summary.json` is generated locally afterward. No registry database or
credential file is included.

The independent verifier checks those hashes, exact committed Python source,
input identity and denominators, then recomputes all comparison/state-change
statistics from the raw vectors without Torch or GPU:

```sh
.venv/bin/python evidence/harnesses/verify_numerical_suite_877f319.py
```

See [findings and limits](../../../docs/numerical-suite.md). All four numerical
reference profiles fail the existing development check on part of the corpus.
This is useful failure evidence, not release certification.
