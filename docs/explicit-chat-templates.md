# Explicit chat templates and tokenizer profiles

Some checkpoints put their Jinja template in a processor file instead of the
tokenizer configuration. Jev supports an explicit, hash-bound local template for
the standard Transformers chat renderer. It does not guess a template from the
model's name, enable remote Python execution or silently switch tokenizer backends.

## Configuration

Add the following to a normal Jev configuration. This example uses Mistral Small
3.1's official template at revision `68faf511d618ef198fef186659617cfd2eb8e33a`:

```yaml
chat_template:
  path: /models/Mistral-Small-3.1-24B-Instruct-2503/chat_template.json
  sha256: d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f
  format: json
tokenizer_options:
  fix_mistral_regex: true
```

`format: json` reads the `chat_template` string from a JSON object. `format: jinja`
reads the file as UTF-8 Jinja text. The SHA256 covers the original file bytes, not
its decoded JSON value. The file is limited to 1 MiB and must contain a nonempty
template. Missing files, changed hashes and malformed documents fail startup.

The template is passed explicitly to the standard HF renderer for every compile;
the plugin does not modify its host tokenizer. Other renderers are rejected for
this override because they can ignore `chat_template` keyword arguments. Default
behavior remains the host tokenizer's own template when no override is configured.

The existing model/bundle `template_digest` binds the effective template string.
Changing that string requires new bundle versions and matching calibration;
stored manifests are not rewritten. This setting configures process startup, not
online replacement of the host's tokenizer or rendering code. Task instructions,
policies and calibrated bundles retain their existing hot-activation lifecycle.

`fix_mistral_regex` is an explicit tokenizer profile option, defaulting to unset.
It is forwarded to `AutoTokenizer` with the pinned revision and
`trust_remote_code=False`. It affects the tokenizer backend selected by the
tested Transformers versions and can change pretokenization. The complete fast
backend fingerprint, already bound into bundles/calibration, captures this change.
Setting this option is not itself evidence of compatibility or accuracy.

When a native host supplies the tokenizer and an explicit tokenizer option is
configured, startup independently loads the requested profile and compares both
its vocabulary and full backend fingerprints with the host compiler. A mismatch
fails with `tokenizer_implementation_mismatch` before registry creation/serving.
The option does not modify or reload an already running engine's tokenizer.
Configure the host engine for the same profile before enabling the plugin.

## Isolated native tests

Both `deployment/dsw_service.py` and `tests/integration/run_native_validation.py`
accept this pair, with an optional file format:

```text
--chat-template-path /models/checkpoint/tokenizer_config.json
--chat-template-sha256 <SHA256-of-that-file>
--chat-template-format json
```

The DSW launcher verifies and decodes the source file, writes a canonical
`chat_template.jinja` snapshot in the fresh run directory, and binds the snapshot
in `JEV_CONFIG`. It passes that same snapshot to the engine's `--chat-template`
flag so native chat and typed decisions use the same template body. The process
record retains both source and snapshot hashes. Direct deployments using only
`JEV_CONFIG` must configure their native chat endpoint separately if they also
want its template changed.

The tokenizer options above are configured through `JEV_CONFIG`; the isolated
launcher does not invent engine-specific flags to force a different tokenizer.
`tests/integration/tokenizer_contract.py` accepts the template flags and explicit
`--fix-mistral-regex` / `--no-fix-mistral-regex` options. Such an override probe must
select exactly one pinned model. The complete 20-model denominator is unchanged.

## Observed scope

At runtime/harness `e17918e`, Mistral-7B-Instruct-v0.3 passed both native engines
with an explicit snapshot of its own official template, BF16, TP4, one API worker
and eager execution. Each passed the full functional contract and 1,000 switches.
The follow-up tokenizer-profile change at `75649bf` has separate CPU/tokenizer
evidence; the earlier native reports retain their original source identity.

Mistral Small 3.1 initially still failed the explicit-template probe because
`AutoTokenizer` chose `MistralCommonBackend`. That renderer uses its own protocol
and is not interchangeable with the HF Jinja renderer. The guarded rejection is
retained. Explicit `fix_mistral_regex: true` plus the pinned official template
selects the standard HF backend in both tested environments and passes the 16
compilation cases. These cases cover two role policies, two scoring modes and
2/8/32/64 candidates for a choice task. They do not prove native GPU serving,
multimodal support, quality, calibration, performance or broad numerical parity.

GLM-4-9B-Chat still requires its custom tokenizer loader. The default loader's
rejection remains recorded. This work does not grant it a functional pass, execute
its remote Python code, replace its frozen revision or remove it from the matrix.

See [Mistral evidence](mistral-validation.md) for results and retained failures.
