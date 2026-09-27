# GLM4 tokenizer conversion and R1 Qwen native validation

Four new model/engine combinations pass the colocated functional profile. The
frozen matrix now contains **24/40 functional passes, 12/20 per engine**. The
release target remains 18/20 per engine plus the separate numerical, quality,
performance and lifecycle gates. This checkpoint does not meet that release target.

## Product change

At runtime `c48f18dfc9715dae09588e0c592c38e25c9e4b80`,
`jevctl tokenizer convert-glm4 MODEL_DIR DESTINATION` converts standard GLM4
`tiktoken` vocabulary and special-token metadata to a bound fast-tokenizer profile.
It preserves the original weights and chat template. The converter and production
serving tokenizer loader do not execute checkpoint Python. The two engines use
their native ChatGLM model implementation and the same converted tokenizer directory.
See [conversion and deployment](tokenizer-profiles.md).

The conversion is specific to GLM4 single-text/rendered-chat encoding. It is not
an arbitrary Python tokenizer transpiler, a weight conversion, or multimodal
support. A tokenizer change requires an engine startup configuration change and
new bundles. Online bundle activation continues to use the existing immutable
version/lease protocol; it does not hot-replace Python or CUDA code.

A completed profile binds source/output file hashes, library versions, vocabulary,
implementation and template fingerprints. File tampering, an incompatible native
compiler, or a supplied compiler with a different pipeline is rejected. Local
regressions cover these boundaries and verify that checkpoint Python is not run.

## Fixed inputs and execution identity

| Checkpoint | Immutable revision | Verified selected files | Weight bytes |
|---|---|---:|---:|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` | 9 | 15,231,271,850 |
| `zai-org/glm-4-9b-chat` | `bd8234fe5e0c09c48637a92abb0c797cb5fa0e73` | 21 | 18,799,941,256 |

All 30 selected files matched Hub LFS SHA256 or Git blob SHA1 and size at download.
They were rehashed with SHA256 after all serving/reference runs and remained
unchanged. Checkpoint Python files were downloaded for audited independent
reference checks, not installed into either production engine. Model files and
credentials are not included in this repository. This is artifact verification,
not attestation of every tensor resident in GPU memory.

- R1 serving runtime: `3377c95ca9955ae96ce7c7deb2c4d97b41b1d9c1`;
  harness: `d1ef50e8420cc0e0af82000c462499e8180929b1`.
- GLM serving runtime and harness: `c48f18dfc9715dae09588e0c592c38e25c9e4b80`.
- vLLM `0.30.0+cu129`, Transformers `5.17.0`; SGLang `0.5.19`,
  Transformers `5.12.1`; both use Torch `2.13.0+cu129`.
- DSW L20Z GPUs 3–6, TP=4, PP=DP=1, one API worker, BF16 backbone/readout,
  eager execution, context limit 2,048. Selected cards each had 10,792 MiB free
  before the test. Existing services remained running.
- vLLM uses a 0.075 fraction of total device memory. SGLang uses a 0.65 static
  fraction of available memory at startup. These flags have different denominators.
  Each launch additionally requires at least 3,072 MiB reserve per selected GPU.
- The production GLM tokenizer profile is
  `/root/jev-runtime/tokenizers/GLM4-fast-bd8234f-c48f18d`, with implementation
  fingerprint `sha256:781719944c8597db5954229c2942077f81de1c342b68aed0252b58a8dd25282b`.

## Functional and numerical results

Each row passed Choice/Boolean/Score/Rank, independent-candidate scoring,
native-chat coexistence, invalid bundle and precision rejection, native/attached
selected-logprob parity, route switching under traffic, disable/drain/retire/CAS,
and the final zero-lease/zero-admission/healthy-active-bundle checks.

| Model | Engine | Route switches | Strict successful traffic requests | Mixed-version responses | Maximum CPU-reference logprob error | Single-position numerical check |
|---|---|---:|---:|---:|---:|---|
| R1 Qwen 7B | vLLM | 1,000 | 455 | 0 | 0.124721 | Pass |
| R1 Qwen 7B | SGLang | 1,000 | 262 | 0 | 0.187206 | Fail |
| GLM4 9B | vLLM | 1,000 | 322 | 0 | 0.875000 | Fail |
| GLM4 9B | SGLang | 1,000 | 202 | 0 | 0.625000 | Fail |

Native/attached logprob difference was zero in each row. Independent reference
uses BF16 CPU eager forward, observed BF16 output projection and FP32 log-softmax,
with the unchanged absolute tolerance 0.15. All four top labels agree. These are
four comparisons over **two distinct saved inputs**, one per model, not a
multi-input numerical certification. Traffic request counts are functional
observations during switching, not controlled throughput comparisons.

R1 uses the existing reference runner in each engine environment. GLM's default
reference loader rejects checkpoint custom code. A direct attempt using audited
local official code with Transformers 5.17.0 also fails because `ChatGLMConfig`
lacks `max_length`. These failures are preserved; the GLM campaign's overall
`passed` remains false even though `functional_passed` is true.

The additional GLM reference uses the unchanged official local model code in a
separate Transformers 4.44.2 / tokenizers 0.19.1 / huggingface_hub 0.36.0 environment.
It reuses Torch from the vLLM environment through a read-only dependency path;
engine dependency versions are unchanged. Its HF cache is isolated and loading
is local-only. It completes both forwards but both numerical comparisons fail.
This isolates a reference-loader compatibility issue; it does not establish the
root cause of the remaining numerical differences. The original model head is
`transformer.output_layer`; a hook verifies BF16 output in the reference.

The GLM default runner reports a missing reference JSON after the loader exits;
its retained stderr supplies the underlying custom-code rejection. Do not mistake
that secondary `FileNotFoundError` for the reason model serving failed: serving
completed successfully. Likewise, a completed reference process is not a passed
numerical test.

## Tokenizer fidelity

The product converter passes 588 deterministic tiktoken encoding/decoding cases.
Independent checks load the hash-verified original GLM tokenizer wrapper and
compare the actual production profile in **each** engine environment:

- 630 original/profile encoding comparisons, including prefix-on/prefix-off and
  default decoding with special tokens skipped.
- 40 rendered chat templates with exact text equality.
- 16 compiler configurations: system-role on/off × joint/independent scoring ×
  K=2/8/32/64. Every emitted sequence's full-context IDs and label continuation
  IDs match the original wrapper.
- The actual saved native GPU fixture has identical input and label IDs on both
  engines and matches the pinned original wrapper. This is separate from the
  generated compiler test cases above.

The R1 Qwen saved input and label IDs match the checkpoint tokenizer JSON on both
engines. Its loaded tokenizer has metadata differences from the raw serialized
backend, but these checks found no encoded-input mismatch. This result must not
be conflated with the previously discovered R1 **Llama** Metaspace/ByteLevel issue.

The initial conversion prototype failed when raw added-token metadata was passed
straight to `PreTrainedTokenizerFast`. A second prototype serialized the standard
configuration and loaded it correctly. Both the failure and succeeding prototype
are retained. GPU tests use the product-generated `c48f18d` profile, not either
prototype. The documented scope remains finite equivalence evidence, not proof
for every possible Unicode string or arbitrary custom tokenizer behavior.

## Cleanup, local checks and reproduction

All four owned serving groups exited. A final audit checked 91 historical owned
server records using PID/start-tick/boot identity and checked their process groups:
no matching live process or non-zombie group member remained. GPU UUIDs matched
and GPUs 3–6 each returned to 10,792 MiB free. The R1 vLLM shutdown emitted a
resource-tracker shared-memory cleanup warning; process-group cleanup still
passed. No unrelated service was stopped or reconfigured.

The final model-file rehash covers 30 files. The export contains 54 allowlisted
artifacts plus its hash manifest. Original engine logs, runtime credentials,
SQLite state and weights are excluded. Only the reference/prototype failure
stderr is retained as text evidence. Executed orchestration scripts are stored
byte-for-byte under [harnesses/glm-r1](../evidence/harnesses/glm-r1/README.md).

At `c48f18d`, local checks passed 307 Python tests, the CI-scoped Ruff check and
format check, `pip check`, and all three wheel builds. The wheels contain 29/4/4
Python files for core/SGLang/vLLM respectively, all equal to committed source.
TypeScript checks were not rerun because its implementation did not change.
An extra repository-wide formatter check found one previously existing Markdown
code block outside the CI scope; it was not rewritten as part of this change.
See [package evidence](../evidence/package-check-c48f18d.json).

To verify the retained evidence without GPU execution:

```sh
python evidence/harnesses/verify_glm_r1_c48f18d.py
```

The verifier checks export hashes, uploaded source against Git blobs, artifact
identity, tokenizer equality, native contracts, cleanup, exact reference inputs,
and recomputes numerical errors including the failures. Its result is
[glm-r1-validation-c48f18d.json](../evidence/dsw/glm-r1-validation-c48f18d.json).
Remote re-execution needs fresh run/output names: evidence scripts intentionally
reject overwriting completed outputs.

Six gated model entries remain inaccessible; the remaining public large-model
profiles need suitable capacity. The two approved business tasks, controlled
144-case performance matrix, full numerical coverage, 24-hour soak and expanded
lifecycle/deployment checks remain outstanding. Functional matrix progress does
not close those requirements or qualify arbitrary model/engine configurations.
