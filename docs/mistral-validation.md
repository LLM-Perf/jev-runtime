# Mistral native validation and explicit tokenizer profiles

On 2026-09-28 (Asia/Shanghai), Mistral-7B-Instruct-v0.3 passed native functional
validation on both engines. The fixed matrix is now **16/40 functional combinations,
8/20 per engine**. The original target remains 18/20 per engine, with all numerical,
quality, performance and lifecycle acceptance gates intact.

## Mistral 7B native GPU evidence

The pinned checkpoint is `mistralai/Mistral-7B-Instruct-v0.3` at revision
`c170c708c41dac9275d15a8fff4eca08d52bab71`. Its 13 downloaded files passed
`hf cache verify`; all downloaded root files have recorded sizes/SHA256 and every
indexed weight shard is present. Indexed weight bytes total 14,496,047,104. The
Hub revision was public, ungated and declared Apache-2.0. Weights are not in Git.

Runtime and native harness source was `e17918e79d587f5fabe6ec65740492cf8f37fcef`.
Both runs used BF16 backbone/readout, TP4, one API worker, eager execution,
2,048-token context and four maximum running requests on L20Z GPUs 3,4,5,6 with
driver 550.127.08. vLLM 0.30.0+cu129 used memory fraction 0.065; SGLang 0.5.19 used
0.65 and 2,048 maximum total tokens. Each GPU passed the 3,072 MiB reserve check.
Existing services remained running: these are colocated development checks.

The launcher verified the pinned `tokenizer_config.json` SHA256, extracted its
actual `chat_template`, and gave the engine and plugin the same hash-bound Jinja
snapshot. A retained check confirms source content equality, snapshot SHA256,
engine command path and the effective template digest in the serving profile.
The template was the checkpoint's official template, not a replacement prompt
chosen to make the test pass.

| Engine | Functional suite | Switches | Strict traffic successes | Native/attach logprob error | CPU single-position max error |
|---|---|---:|---:|---:|---:|
| sglang | Pass | 1,000 | 229 | 0 | 0.148433685 |
| vllm | Pass | 1,000 | 423 | 0 | 0.093746185 |

Both passed all four output types, default abstention and explicit first-tie policy,
independent candidate scoring, native chat coexistence, authentication and invalid
bundle rejection, CAS conflicts and activate/disable/drain/retire. Mixed-version
responses were zero. Both also rejected legacy/wrong readout bundles and an
attached runtime with a mismatched precision declaration. Final leases/admission
were zero and active bundles were healthy.

The independent CPU BF16 eager reference uses the actual saved input/label IDs.
Both positions have top-label agreement and pass the unchanged development
absolute logprob tolerance 0.15. SGLang's 0.148433685 is close to that threshold;
neither result is a full numerical budget, accuracy guarantee or business-quality
evaluation. No throughput claim follows from the switch-traffic counts.

## Mistral Small and GLM checks

Mistral Small 3.1 remains pinned at `68faf511d618ef198fef186659617cfd2eb8e33a`.
Its official `chat_template.json` SHA256 is
`d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f`.

Three first attempts at `e17918e` are retained: default loading failed for a missing
chat template; explicit-template attempts in both environments failed because
`AutoTokenizer` selected `MistralCommonBackend`. That renderer does not have the
standard HF Jinja contract. Its rejection was not disabled.

At follow-up source `75649bfb633ab5d6219e4b1c6bc4d7e9b9780bd5`, an explicit
`tokenizer_options.fix_mistral_regex: true` plus the pinned template selects the
HF tokenizer in both installed environments. Each passes **16/16** compilation
cases: system-role on/off, joint-label/independent-candidate, and 2/8/32/64 choice
candidates. These are real tokenizer checks, not Mistral Small GPU certification.
No model weights or native engine for that 24B checkpoint were loaded in this work.

A separate check with those real tokenizer assets demonstrates why the implementation
fingerprint matters. With the regex profile disabled/enabled, vocabularies match,
but backend fingerprints differ. In both environments, `build_runtime` rejects the
mismatched host compiler with `tokenizer_implementation_mismatch` before creating
the registry. The check uses a placeholder backend and performs no GPU scoring;
its exact harness SHA256 and source are retained.

GLM-4-9B-Chat at `bd8234fe5e0c09c48637a92abb0c797cb5fa0e73` still fails the default
loader because it requires custom tokenizer Python. That current-source rejection
is recorded. No remote custom Python was executed, no native GLM test is claimed,
and both GLM combinations remain in the original denominator.

## Reproduction and limits

See [explicit template configuration](explicit-chat-templates.md) and the
[native validation runner](native-validation-runner.md). The
[validation index](../evidence/dsw/template-validation-75649bf.json) binds 36
retained artifacts by SHA256, including failures, model provenance, source upload
manifests, local packaging checks and the exact real-tokenizer rejection harness.
Each immutable source upload had 107 verified files.

Both native process groups exited and GPU 3/4/5/6 each returned to the recorded
10,792 MiB free baseline with matching UUIDs. A final scan of 72 saved process
records found no matching owned process still alive. No unrelated service was
stopped or reconfigured.

Source `e17918e` passed 275 local Python tests; `75649bf` passed 277. Both built
all three wheels and compared packaged Python files with source. Ruff and formatting
passed. TypeScript was unchanged and its earlier 62 passing tests were not rerun.
The GPU runs remain attributed to `e17918e`; the later tokenizer-profile work has
its own CPU and real-tokenizer evidence and does not imply broad final-source GPU
recertification. Engine dependency versions were not upgraded. The CI install step now includes
the `tokenizers` extra required by the new real-HF unit fixtures, matching the
README development command; this does not establish hosted CI eligibility or success.

Remaining model combinations, full per-profile numerical and quality gates,
controlled performance, the 24-hour soak, compatible-replica failover and broader
LoRA/fault configurations remain unfinished. Hosted CI is separate from all local
and DSW evidence.
