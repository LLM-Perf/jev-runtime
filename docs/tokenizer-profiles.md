# Preparing GLM4 tokenizer profiles

`jevctl tokenizer convert-glm4 MODEL_DIR DESTINATION` converts the standard GLM4
tiktoken vocabulary, special tokens and prefix behavior into a serialized fast
tokenizer. It reads local data and uses the installed Transformers converter;
checkpoint Python is never imported or copied. Source model weights/configuration
remain unchanged. This extends the typed text-serving path to this tokenizer
format without enabling arbitrary repository code in the serving tokenizer loader.

```sh
python -m pip install 'jev-runtime-core[tokenizer-conversion]'
jevctl tokenizer convert-glm4 \
  /models/glm-4-9b-chat \
  /tokenizers/glm4-fast-v1
```

The destination must be new and outside the source checkpoint tree. The command
checks contiguous vocabulary/special IDs, byte coverage, special-token flags and
prefix tokens. It compares the derived tokenizer against a tiktoken reference on
fixed Unicode, whitespace, numeric, special-token and rendered-chat samples,
with and without prefix insertion. Encoded IDs and byte decoding must match.
The same literal chat template is retained. The original GLM4 Python wrapper is
not a runtime dependency of the generated profile.

The completed directory contains `tokenizer.json`, `tokenizer_config.json` and
`jev-tokenizer-profile.json`. The manifest binds input/output file hashes,
conversion library versions, vocabulary and tokenizer-implementation fingerprints,
template identity and the validation ledger hash. Existing outputs are never
overwritten. A publication I/O failure can leave an incomplete directory; only a
successful command with its final manifest qualifies the artifact.

This is the **GLM4 single text / rendered chat prompt profile**. It does not claim
equivalence for arbitrary custom Python tokenizer behavior, text pairs, generated
position-ID fields or multimodal processing. Another checkpoint or converter
version needs validation before claiming compatibility. The target checkpoint
for native GPU validation is `zai-org/glm-4-9b-chat` at
`bd8234fe5e0c09c48637a92abb0c797cb5fa0e73`.
The encoding pattern and prefix contract come from its
[original tokenizer implementation](https://huggingface.co/zai-org/glm-4-9b-chat/blob/bd8234fe5e0c09c48637a92abb0c797cb5fa0e73/tokenization_chatglm.py).

## Use with the two native plugins

Configure the **same directory** for the host engine and Jev:

```yaml
tokenizer: /tokenizers/glm4-fast-v1
```

For vLLM, add `--tokenizer /tokenizers/glm4-fast-v1`; for SGLang use
`--tokenizer-path /tokenizers/glm4-fast-v1`. The task-owned DSW launcher accepts
`--tokenizer-path` and binds both settings together. Keep the original model
directory as the engine's model path. No converted weight checkpoint is created.

Jev validates generated-profile file hashes before local loading and checks the
actual native host compiler against the manifest. A changed file or different
normalization/BPE pipeline fails with `tokenizer_profile_mismatch`, including when
the runtime receives an already constructed compiler. Explicit chat-template
overrides remain separately bound and validated by the existing template contract.

The tokenizer is loaded at engine startup. To change it, deploy a separately
validated engine profile and build new bundles against that host identity.
Bundle hot-switching does not mutate an active engine's tokenizer. Task quality,
quantization, numerical budgets and performance remain separate model-profile gates.
