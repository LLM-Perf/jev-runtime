# Preserve serialized checkpoint tokenizers

`jevctl tokenizer preserve-fast MODEL_DIR DESTINATION` creates an immutable,
data-only tokenizer profile from a checkpoint's existing `tokenizer.json`. It
prevents a model-specific AutoTokenizer constructor from rebuilding a different
normalizer/pre-tokenizer/model/decoder pipeline. This is the reusable form of the
explicit SmolLM2 and R1 Llama configurations used in previous DSW checks.

The reference is the serialized tokenizer data. This does not convert weights,
train a judgment head or prove equivalence to arbitrary custom checkpoint Python.
For GLM4's non-JSON vocabulary use [convert-glm4](tokenizer-profiles.md). Engine
model support, numerical accuracy, task quality and performance remain separate
requirements after tokenizer preparation.

## Prepare a profile

From the private repository, install `.[tokenizers]` in the intended environment:

```sh
jevctl tokenizer preserve-fast /models/SmolLM2-1.7B-Instruct /tokenizers/smol-fast-v1
```

The destination must be new and outside the checkpoint directory. Source files
and weights remain unchanged. The command copies `tokenizer.json` byte-for-byte,
selects the standard `PreTrainedTokenizerFast` loader, and removes custom-code,
class reconstruction and alternate-file selection fields. It copies applicable
`special_tokens_map.json`, `added_tokens.json`, `chat_template.jinja` and named
`additional_chat_templates/*.jinja` data. No checkpoint Python is loaded or copied.

A missing default chat template can be supplied explicitly:

```sh
jevctl tokenizer preserve-fast /models/model /tokenizers/model-fast-v1 \
  --chat-template /profiles/model.jinja \
  --chat-template-sha256 VERIFIED_SHA256
```

The digest is required. This override replaces the complete default/named template
set and is recorded separately from the original inputs. Template files otherwise
have the standard loader's precedence over `tokenizer_config.json`. Unsupported
or silently ignored templates cause validation failure.

## Validation and manifest

Before publishing, the command checks the entire loaded backend representation
against the serialized reference: normalization, splitting, vocabulary/merges,
added-token flags and IDs, prefixes and decoding must agree. One explicit exception
is a standard loader inserting an exact single-sequence identity post-processor
where the serialized backend has none: it forwards sequence A with type ID 0 and
adds no tokens. This representational change is recorded in the manifest; all
other backend fields must match exactly. Its text-pair type IDs are not covered.
It also checks fixed Unicode/number/whitespace/special-token/boundary samples and
rendered default/named chat prompts, with special-token insertion both enabled
and disabled. Decoding is compared with and without skipping special tokens.
Source data and an optional override are reread before output creation to detect
changes during the operation.

`jev-tokenizer-profile.json` records input/output hashes and sizes, implementation
and dependency versions, vocabulary/pipeline/template fingerprints, validation
counts and ledger hashes. Only a completed manifest qualifies the output. Existing
paths are never overwritten; an I/O failure can leave an incomplete destination,
which retains a pending marker and is rejected by the loader. Choose a fresh output
for a retry. Startup rejects missing, changed, additional,
unsafe-path or symlinked profile payload files and verifies the actual host
compiler against the manifest. Ordinary model directories without a generated
profile manifest retain their existing behavior.

The deterministic single-text serving profile rejects serialized active padding,
truncation, BPE dropout, split-special-token mode, absent default templates, or
config metadata that changes the raw backend. A rejection is not permission to
strip those semantics silently. Text pairs, custom wrapper methods, multimodal
processors and global all-input equivalence are not certified by this operation.

## Use in serving

Set Jev's `tokenizer` configuration and the native engine tokenizer option to the
same completed directory. vLLM uses `--tokenizer`; SGLang uses `--tokenizer-path`.
The task-owned DSW launcher binds both through `--tokenizer-path`. Keep the
original weights/model path. Validate actual native profile fingerprints and
compiled input/label IDs before preparing bundles.

New tokenizer identity requires fresh immutable bundles and bound calibration,
and a separately started engine process. Bundle activation does not mutate a
running tokenizer. Historical numerical, quality, performance and LoRA results
cannot automatically qualify a newly prepared profile.

The loader and special-token/template interfaces follow the
[Transformers tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer).
Actual profile validation, rather than a library version alone, determines
whether the installed loader preserved this checkpoint's serialized behavior.

The [real-model/DSW validation report](preserve-fast-validation.md) records the
11-checkpoint CPU inventory, native Smol checks, preserved failures and exact
remaining qualification limits.
