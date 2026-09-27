# Phi native TP2 development validation

At `737d8146562cb0dd906e7f826a3eea8b4d38875f`, Phi-3 mini and Phi-4 mini
passed the native functional suite on both engines. This adds four distinct
combinations: the fixed matrix now has **10/40 functional passes, 5/20 per
engine**, against the unchanged requirement of at least 18/20 per engine.
Functional coverage does not certify numerical accuracy, business quality or
performance. The complete release remains unfinished.

## Exact scope and provenance

- `microsoft/Phi-3-mini-4k-instruct` at
  `f39ac1d28e925b323eae81227eaba4464caced4e`.
- `microsoft/Phi-4-mini-instruct` at
  `cfbefacb99257ffa30c83adab238a50856ac3083`.
- SGLang 0.5.19, vLLM 0.30.0+cu129, Torch 2.13.0+cu129, driver 550.127.08,
  NVIDIA L20Z; isolated Python environments on a host with existing services.
- BF16, TP2, one API worker, eager execution, 2,048-token context, 512-token
  prefill batches and four maximum running sequences. No managed LoRA.
- SGLang used Triton attention on GPU 5,6; vLLM used GPU 6,7. The two services
  ran sequentially. Native serving used ports 18794/18795.

The Hub CLI downloaded only selected weights/config/tokenizer/license files at
the pinned revisions. `hf cache verify` passed for 12 Phi-3 files and 13 Phi-4
files. The [source manifests](../evidence/dsw/phi-model-sources-737d814/)
record their sizes and SHA256 values, including every indexed weight shard.
Custom remote Python was not downloaded or executed. These are disk provenance
checks; in-memory weight attestation is not implemented.

The uploaded release's 56 deployment/runtime/plugin/harness files were verified
against their local SHA256 manifest. Environment reports point to the exact
immutable release. SGLang's live server configuration reported TP2. vLLM's two
rank-labelled workers were alive in the recorded API process group, whose CUDA
mapping was verified. Its development server-info route was unavailable; it was
not enabled for the test. Container and NVML PID namespaces differ, so the report
does not claim a direct worker-to-NVML PID mapping.

## Functional results

| Model | Engine | Route switches | Successful traffic requests during switching | Mixed versions | Native/attach max logprob error |
|---|---|---:|---:|---:|---:|
| Phi-3 mini | SGLang | 1,000 | 244 | 0 | 0 |
| Phi-3 mini | vLLM | 1,000 | 448 | 0 | 0 |
| Phi-4 mini | SGLang | 1,000 | 184 | 0 | 0 |
| Phi-4 mini | vLLM | 1,000 | 317 | 0 | 0 |

Every run also checked choice/boolean/score/rank responses, default-policy
abstention, a separate explicit `tie: first` bundle, independent-candidate
scoring, authentication separation, native chat, invalid bundles, generation
conflicts, disable, drain and retire. Final leases and admission counters were
zero, and active bundles had fresh healthy/prepared state.

These request counts are functional-test denominators, not throughput results.
The traffic, scheduling and GPU occupancy were not a controlled performance
comparison. Other TP degrees, API-worker combinations and LoRA remain outside
this profile. The existing Phi-3 tokenizer limitation for joint K=64 remains;
independent scoring is a separate supported route, not a reason to erase that
limitation.

## Independent numerical results

Each run separately evaluated its saved input IDs and label IDs with a CPU
Transformers BF16 eager forward and FP32 log-softmax. The tolerance remained
`atol=0.15`; top-label agreement was also required.

| Model | Engine | Maximum absolute logprob error | Top label agrees | Single-position result |
|---|---|---:|---|---|
| Phi-3 mini | SGLang | 0.250208855 | Yes | Failed |
| Phi-3 mini | vLLM | 0.247813225 | Yes | Failed |
| Phi-4 mini | SGLang | 0.000003338 | Yes | Passed |
| Phi-4 mini | vLLM | 0.250003815 | Yes | Failed |

The passing Phi-4/SGLang example is marked **partial** in the numerical column;
one answer position cannot certify a model's numerical behavior. The other
three rows retain **failed** status. The native/attach zero-error checks show
that the plugin reproduced each engine's selected scores on these fixtures;
they do not resolve the independent numerical differences.

Phi-3's two engines compiled different input/label IDs, so their reference
results are separate per-engine comparisons. Phi-4 used identical input/label
IDs, and both CPU references produced identical selected logprobs, despite
Transformers 5.12.1 versus 5.17.0. The engine scores still differed by about
0.25 on one selected label. This narrows that observation but does not establish
its kernel/precision root cause or its effect on business quality.

## Failed attempts and cleanup

The first vLLM/Phi-3 harness connected before readiness and returned
`ConnectError`. Its report is retained as `contract.json`; the same engine
process subsequently passed as `contract-ready.json` after readiness was checked.

The first SGLang/Phi-3 launch on GPU 6,7 failed the engine's native TP memory
balance check: after distributed initialization the minimum available memory
was about 8.950 GiB versus 10.120 GiB locally. That run never entered the
functional suite. Its traceback and full-group cleanup are preserved. The
successful retry used GPU 5,6 with equal observed spare memory; the engine
guard was not disabled. An initial readiness-helper invocation for that retry
used the wrong key-field name and exited before sending a readiness request;
it was corrected to `api`
before running the suite, without restarting the engine.

All five owned process groups, including the failed launch, exited. The cleanup
reports verify every selected UUID and memory snapshot. GPU 5 and 6 returned to
10,792 MiB free; GPU 7 returned to 11,990 MiB free. Unrelated services were not
signaled. The launcher's memory check is a preflight budget, not a continuous
memory reservation or a guarantee against engine-specific workspace use.

## Reproduction and remaining gates

`deployment/dsw_service.py launch --gpus 5,6 ...` preserves device order and sets
the appropriate engine TP size. Every device is checked for its UUID, memory
budget and reserve. Existing `--gpu` single-device use is unchanged. Managed-LoRA
and crash-restart harnesses explicitly retain their TP1 restriction.

Select GPUs using a fresh inventory; do not copy this host's device IDs blindly.
Wait for `/plugins/jev-runtime/ready` before invoking `live_contract.py`, use a
new output path, and pass both source-commit arguments. Run
`reference_logits.py --device cpu --atol 0.15` separately. Capture topology and
postconditions before stopping, then confirm full process-group exit and every
selected GPU before starting another model.

The [validation index](../evidence/dsw/phi-tp2-validation-737d814.json) binds
42 raw/local-check artifacts by SHA256, including failures. Local regression
at this source passed 243 Python tests and Ruff checks (102 formatted files).
Core and plugin package sources are unchanged from `40b78c2`, whose wheel
verification is reused explicitly; wheels and the unchanged 62 TypeScript tests
were not rerun for this deployment/harness-only change.

Thirty base-model combinations, full numerical certification, approved business
evaluations, the controlled performance matrix, 24-hour soak and other release
gates remain. This TP2 result does not authorize a broader managed-LoRA profile.
