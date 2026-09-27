# Serialized tokenizer preservation: real-model and native validation

At source `cdd0caf330a645f15279cd346f2ffa8ba62190f9`, the shipped
`jevctl tokenizer preserve-fast` command successfully prepared all 11 locally
available standard `tokenizer.json` checkpoints in both isolated DSW engine
Python environments. The generated SmolLM2 profiles also passed new native GPU
serving, hot-switch and targeted independent-reference checks on both engines.
This does not increase the fixed functional denominator: coverage remains 24/40,
12/20 per engine. It makes profile preparation and validation reproducible.

## Inventory and CPU results

Each environment retained all 20 entries from the original frozen inventory.
Eleven had eligible local JSON tokenizers, GLM4 had its different tiktoken format,
and eight checkpoints were absent locally (six gated and two larger public models).
The missing models and GLM4 rejection were not removed from the report. GLM4
continues to use the separately tested `convert-glm4` command.

| Checkpoint | vLLM-environment compiler cases | SGLang-environment compiler cases |
|---|---:|---:|
| Qwen3-0.6B | 16/16 | 16/16 |
| Qwen3-8B | 16/16 | 16/16 |
| Qwen2.5-7B-Instruct | 16/16 | 16/16 |
| Mistral-7B-Instruct-v0.3 | 16/16 | 16/16 |
| Phi-3-mini-4k-instruct | 14/16 | 14/16 |
| Phi-4-mini-instruct | 16/16 | 16/16 |
| R1-Distill-Qwen-1.5B | 16/16 | 16/16 |
| R1-Distill-Qwen-7B | 16/16 | 16/16 |
| R1-Distill-Llama-8B | 16/16 | 16/16 |
| OLMo-2-1124-7B-Instruct | 16/16 | 16/16 |
| SmolLM2-1.7B-Instruct | 16/16 | 16/16 |

The compiler matrix is system role on/off × joint-label/independent-candidate ×
K=2/8/32/64. Phi-3's two joint-label K=64 failures remain; independent-candidate
K=64 passes. Totals are 174/176 per environment, 348/352 overall. These are
real-tokenizer/compiler checks, not 22 newly certified GPU models.

Each generated profile passed the converter's full backend and encode/decode
checks (612–2,338 encoding cases depending on checkpoint metadata). An independent
fixed 1,005-input corpus then compared the saved checkpoint Tokenizer with the
loaded profile: 22,110/22,110 exact token-ID comparisons passed. Both environments
used the same corpus, with actual sampled input digests and ledger hashes retained.
Finite corpus equality does not prove arbitrary custom Python wrapper equivalence.

Default AutoTokenizer diagnostics on that corpus found:

- R1 Llama: 999/1,005 inputs differed from serialized checkpoint IDs in each environment.
- SmolLM2: 232/1,005 differed in each environment.
- OLMo: 557/1,005 differed in the SGLang environment's default AutoTokenizer; zero
  differed in the vLLM environment.

The last result does **not** show a SGLang native serving defect. Its own native
`get_tokenizer` was tested separately on all 1,005 inputs, with zero differences;
its implementation fingerprint matches the previously retained native OLMo
profile. The generic AutoTokenizer and actual engine getter are distinct paths.

## Representational difference found and handled

The first `fbd5b36` campaign rejected Phi-4, OLMo and Smol on both environments
because their raw backend has no post-processor, while the standard fast loader
inserts a `TemplateProcessing` identity. The exact difference was retained.

The final implementation recognizes only the exact single-sequence identity:
sequence A, type ID 0, no inserted tokens. Every other backend field must remain
identical. Manifest fields distinguish full structural equality from this explicit
single-text equivalence. Prefix/special-token insertion is still rejected by a
regression test, and both add-special-token modes and both decode modes are
checked. Text-pair type IDs are outside this profile's contract.

Generated artifacts retain a pending marker until publication completes. Both
this command and the existing GLM4 converter now reject partially published
profiles at startup; injected write failures verify that incomplete output cannot
fall through as an ordinary checkpoint directory. Unexpected files, named-template
changes, symlinks, unsafe manifest paths and host-pipeline mismatches are also
rejected. No source weights or tokenizer files were edited.

## Native SmolLM2 results

The pinned checkpoint revision was `31b70e2e869a7173562077fd711b654946d38674`.
Runtime, native plugin and validation harness all used `cdd0caf`. Profiles were
created by the real CLI in the corresponding environment. Serving used GPU7 on
the shared L20Z host, BF16 backbone/readout, TP1, one API worker, eager execution
and context 2048. SGLang remained 0.5.19 / Transformers 5.12.1; vLLM remained
0.30.0+cu129 / Transformers 5.17.0. Both used PyTorch 2.13.0+cu129.

| Check | vLLM | SGLang |
|---|---:|---:|
| Native functional and precision-binding suites | Pass | Pass |
| Bundle switches under traffic | 1,000 | 1,000 |
| Strict successful traffic requests during switches | 479 | 537 |
| Mixed-version responses | 0 | 0 |
| Native/attached maximum logprob difference | 0 | 0 |
| Single-position CPU-reference maximum absolute logprob error | 0.000211239 | 0.086620569 |
| Unchanged development tolerance | 0.15 | 0.15 |
| Selected top-label equality | Yes | Yes |

The reference is a separate Transformers CPU BF16 eager forward over the actual
saved input IDs, with FP32 log-softmax. Passing one saved answer position is
partial numerical evidence; it does not close the full numerical gate or resolve
other checkpoints' failed comparisons. Different inputs/profiles/cache states
must not be treated as a causal numerical improvement over historical reports.

The first native campaign stopped before starting any server because a
`--no-build-isolation` editable install could not import `hatchling.build`.
Its original failed report and log remain. A fresh controller used isolated build
dependencies and `--no-deps` editable installation in the task-owned environments,
then passed both complete native runs. No existing engine service was restarted.
All 110 historical owned process records had no matching live processes or groups
afterward, and GPU7 returned to 11,990 MiB free. Six checkpoint files, including
weights, were rehashed against fixed-revision Hub metadata. This is disk-artifact
verification, not in-memory weight attestation.

## Evidence and remaining scope

Local validation passed 332 Python tests, project-scoped Ruff and `pip check`.
All three wheels were built from the immutable source and their Python contents
checked against it. GPU engines used the isolated editable source installations;
this campaign is not a new installed-wheel upgrade/rollback qualification.
TypeScript was unchanged and not rerun.

See [the command guide](preserve-fast-tokenizers.md),
[retained reports](../evidence/dsw/preserve-fast-cdd0caf), and
[exact executed harnesses](../evidence/harnesses/preserve-fast/README.md).
Run `python evidence/harnesses/verify_preserve_fast_cdd0caf.py` to verify local
hashes, source identities, full inventory accounting, retained failures and native
results. The verifier does not reproduce GPU execution.

The other ten generated model profiles have CPU validation only at this new
configuration. Existing native certifications keep their original profile scopes.
The remaining model matrix, wider numerical checks, approved business datasets,
controlled performance matrix, 24-hour soak, broader fault cases, deployment
images and uninterrupted rollout/handoff requirements remain incomplete.
