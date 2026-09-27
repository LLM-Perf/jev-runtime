# Installed-wheel gateway upgrade and rollback

Both SGLang and vLLM passed the corrected DSW gateway campaign using fresh,
offline-installed wheel environments. The candidate was tested beside the previous
version, then activated on the stable port and rolled back. Each native engine
remained the same live process throughout its gateway transitions. This is a
scoped development exercise with measured maintenance interruptions, not a
zero-downtime rollout or final release acceptance.

## Tested artifacts and environment

| Component | Exact source / scope |
|---|---|
| Native core and engine plugins | `c48f18dfc9715dae09588e0c592c38e25c9e4b80`, existing isolated editable engine environments |
| Previous installed gateway | `3377c95ca9955ae96ce7c7deb2c4d97b41b1d9c1`, 40 locked distributions |
| Candidate installed gateway | `861ed3ad99df9028ef5d3df2d1dfc4127ab0a946`, 44 locked distributions |
| Dependency resolver and installer | `4e49524b54aade12988a8249faad6a90c736a173` |
| Native integration harness | `861ed3ad99df9028ef5d3df2d1dfc4127ab0a946` |
| SGLang | 0.5.19; Transformers 5.12.1; PyTorch 2.13.0+cu129 |
| vLLM | 0.30.0+cu129; Transformers 5.17.0; PyTorch 2.13.0+cu129 |
| Model | `HuggingFaceTB/SmolLM2-1.7B-Instruct`, revision `31b70e2e869a7173562077fd711b654946d38674` |
| Execution | L20Z GPU7 shared with pre-existing services, BF16 backbone/readout, TP1, one API worker, eager, context 2048 |
| Tokenizer | Explicit checkpoint-preserving profile described below, configured in both native engine and gateway |

All three wheels rebuilt at `4e49524` are byte-identical to the tested `861ed3a`
payloads; only installer/development tooling changed. This does not mean the
native engine was wheel-installed or that every earlier feature was requalified.
The package version remains `0.1.0a1`; source commit and manifest SHA256 distinguish
these development artifacts.

The candidate wheel manifest SHA256 is
`14b2ece0b86063ac9c87972015ce50d34ed0592ae5be9b0cf88aef707a3ed36f`.
The gateway manifests, covering complete offline wheelhouses, are:

- Candidate: `78e13bef7006dc4f481d0f15706cf822fb0e910c1c5fab6b3a497dc062100e9f`.
- Previous: `a79ad00cef9f52c01b1050e3c06fc3728291f2e5c83fd104332911ec336379bd`.

The installer checked exact installed package versions, isolated environment
prefixes, `pip check`, imports and every core Python file against its wheel.
Gateways ran the installed `jevctl` with Python isolated mode. The final audit
re-inspected both environments. Engine dependencies were not upgraded.
See [installation commands and boundaries](release-packaging.md).

## What passed

| Check | vLLM | SGLang |
|---|---:|---:|
| Native bundle switches under traffic | 1,000 | 1,000 |
| Strict successful native traffic requests during switches | 458 | 546 |
| Mixed-version responses | 0 | 0 |
| Native/attached logprob maximum absolute difference | 0 | 0 |
| Persistent-route responses over four gateway stages | 32/32 | 32/32 |
| Complete live Python SDK suites | 3/3 | 3/3 |
| Measured upgrade transition | 3.346 s | 3.487 s |
| Measured rollback transition | 3.214 s | 2.976 s |

The four gateway stages were previous, candidate canary, candidate active and
rollback. Every stage checked actual readiness, exact native/gateway model
identity, compiled input and label token IDs, and eight completed decisions through
`release-persisted`. Its bundle `release-round-trip@1` and generation 1 survived
all stages in the same SQLite registry. Previous, candidate-active and rollback
also ran the full Python SDK suite: decision serving, explicit 16-question
cancellation returning 499 with no retained lease, recovery, disable/native-chat
coexistence and alias rollback. Final admission and lease checks drained.

The candidate canary served on port 18797 while the previous version continued
on 18796. Activation stopped the previous process and canary, then started the
candidate on 18796. Rollback stopped the candidate and started the previous
version there. An unavailable endpoint was observed during upgrade. The durations
above are controller transition intervals, not a continuous availability SLO or
precise request-loss window. Automatic traffic routing and compatible-replica
failover remain unimplemented.

The recorded native PID/start ticks/boot identity stayed unchanged across each
gateway transition. Every owned gateway and native group exited afterward. A
final audit of 108 historical owned server records found no matching live
processes or group members. GPU7 returned to 11,990 MiB free; unrelated services
were preserved. Six model files were rehashed against fixed-revision Hub metadata.

## SmolLM2 tokenizer fidelity and explicit profile

The first SGLang gateway attempt correctly failed the exact tokenizer fingerprint
guard. Inspection showed a semantic difference, not just unequal metadata:
checkpoint `tokenizer.json` has a `Sequence(Digits(individual_digits=True),
ByteLevel(...))` pre-tokenizer; default `AutoTokenizer` reconstructed a GPT2
pipeline with plain `ByteLevel`. The default SGLang native getter retained the
checkpoint pipeline. Direct AutoTokenizer probes in both tested Transformers
versions reproduced the reconstruction, so changing Transformers version alone
was not a supported fix.

Among 1,005 fixed-seed text/number/Unicode inputs, default AutoTokenizer differed
from the explicit profile on 232. For `9 ²,abc1①`:

- Default AutoTokenizer: `[41, 3351, 127, 28, 25276, 33, 173, 235, 250]`.
- Checkpoint-preserving profile: `[41, 216, 19133, 28, 25276, 33, 173, 235, 250]`.

All 1,005 inputs matched across the raw checkpoint Tokenizer, the explicit generic
fast tokenizer and the SGLang native getter using that profile. Both final native
engines and every gateway bound the same implementation digest:
`sha256:08e96741a4f1c7a56136f9400ddf6d8f4f0e50fd09743f83579328e76e6cb6ef`.
The live parity fixture includes the numeric counterexample.

For this pinned checkpoint, create a **new, data-only** tokenizer directory:

1. Verify the checkpoint revision and source `tokenizer.json` SHA256:
   `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.
2. Copy that `tokenizer.json` byte-for-byte. Copy `tokenizer_config.json`, set
   `tokenizer_class` to `PreTrainedTokenizerFast`, and remove `auto_map` if present.
   The tested profile contains only these two files. Do not execute checkpoint
   Python or mutate the original checkpoint.
3. Pass the directory to both the native engine tokenizer option and Jev's
   `model.tokenizer_path` (the native validation launcher exposes
   `--tokenizer-path`). Verify actual serving fingerprints and input/label IDs
   before preparing new bundles. A successful load alone is insufficient.
4. Rebuild immutable bundles and any calibration bound to the changed tokenizer
   identity. Stop/drain and restart model processes to select another tokenizer;
   do not replace tokenizer files beneath live workers.

This is a tested manual profile recipe, not an automatic generic converter.
Changing JSON whitespace can change the config file hash; the exact tested config
hash is recorded in `smol-explicit-profile.json`. Encoding checks on this corpus
are not a proof of equivalence for every possible input or every chat template.
New-profile independent numerical, quality, performance and LoRA qualification
were not run. Historical results retain their original profiles and are not
transferred to this one. Functional coverage remains 24/40 model-engine rows,
12/20 per engine, with the same required 18/20 per engine.

## Retained failures

| Attempt | Outcome and follow-up |
|---|---|
| First candidate dependency resolution | Rejected the pip wheel because vendored nested `.dist-info/METADATA` was counted as a second root metadata record. Fixed detection and regression tests in `4e49524`; incomplete output retained. |
| First vLLM gateway campaign | Four stages executed, then controller exit 1 from a duplicate exclusive `cleanup.json` write. Original campaign has no completed marker. Corrected campaign reran all stages. |
| First SGLang gateway campaign | Tokenizer identity rejection before the SDK suite. The guard remained strict; the explicit profile above passed the subsequent complete campaign. |
| Two alternate SGLang-constrained resolutions | Explicit resolver `setuptools==80.10.2` conflicted with constraints `84.0.0`. Neither installed an environment. They do not establish a Transformers-version cause for the tokenizer issue. |

These failures are included in the export and verifier. No failed report was
rewritten as passing. The successful campaign uses the same original 40/44-package
locked gateway environments with the corrected explicit tokenizer configuration.

## Evidence and remaining acceptance

The allowlisted export is under
[`evidence/dsw/release-packaging-4e49524`](../evidence/dsw/release-packaging-4e49524).
It contains hashed manifests, locks, install inspections, failures, structured
campaign reports, token-ID comparisons and final source/model/process audits.
Credentials, SQLite databases, model weights and raw service logs are excluded.
Exact executed controller/export scripts and their limitations are described in
[the harness README](../evidence/harnesses/release-packaging/README.md).

Run the read-only consistency verifier from the repository (it writes only its
small derived summary):

```sh
python evidence/harnesses/verify_release_packaging_4e49524.py
```

Local source checks at `4e49524` passed 317 Python tests, Ruff lint/format and
`pip check`; TypeScript was unchanged and was not rerun. The verifier checks
artifact integrity/accounting, including expected failures, not GPU reproduction.

P18/P24 remain partial. Engine container images, database migration/downgrade,
automatic uninterrupted rollout, replica failover, operator handoff, final-source
recertification, remaining model/numerical/business-quality gates, the full
144-case controlled performance matrix and a 24-hour soak remain outstanding.
No production release or 90% throughput claim follows from this campaign.
