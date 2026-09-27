# DSW evidence, 2026-09-27

These files are observations, including failures, not a release certificate.

- Engine: vLLM `0.30.0+cu129`; Torch `2.13.0+cu129`.
- GPU: one NVIDIA L20Z, SM90, shared with an existing unrelated service.
- Driver: 550.127.08; localhost-only test API; independent Python environment.
- Source tested: `4359927bf4705413a3a7434b3a4f3959cf89e182`.
- Model: Qwen3-0.6B at the revision in each JSON report, BF16.
- Engine settings: eager, max model length 2048, max sequences 4, max batched
  tokens 512, GPU memory fraction 0.07, selected logprob limit 128.

`vllm-qwen06b-contract.json` records real GPU requests. The four output types,
independent candidates, native chat coexistence, authentication separation,
explicit invalid-version failure, CAS, disable/drain/retire, and native/attach
selected-logprob parity passed. During 1,000 route switches, 432 decision requests
completed with consistent bundle version/digest/generation. It does not prove
1,000 adapter weight swaps or a 24-hour soak.

The CPU and GPU Transformers reference files **failed** the fixed 0.15 absolute
logprob tolerance. Largest errors were approximately 0.50 and 0.25 respectively,
on low-probability labels. Top labels agreed. The same-engine OpenAI completion
baseline and Jev selected-logprob API agreed exactly (maximum absolute error 0).
This distinguishes adapter extraction correctness from cross-implementation
BF16 numerical equivalence; the latter is still under investigation.

No throughput, overhead, task-accuracy or majority-model support claim follows
from these co-located checks. The frozen inventory remains 20 checkpoints and
40 engine combinations; untested and inaccessible checkpoints remain in it.

Additional runs at source `a90e002` passed the same functional contract:

| Engine | Model | Config switches | Strict completed requests during switches |
|---|---|---:|---:|
| SGLang 0.5.19 | Qwen3-0.6B | 1,000 | 501 |
| vLLM 0.30.0+cu129 | SmolLM2-1.7B-Instruct | 1,000 | 534 |
| SGLang 0.5.19 | SmolLM2-1.7B-Instruct | 1,000 | 582 |

Every run observed both bundle versions and zero mixed version/digest/generation
responses. SGLang native/HTTP selected-logprob parity was exact in both runs.
The Qwen SGLang CPU BF16 reference comparison failed the 0.15 tolerance with about
0.25 maximum absolute logprob error; the top label agreed. SmolLM2 independent
reference/quality/performance comparisons have not yet run.

SGLang used Triton attention, max-total-tokens 2048, max-running-requests 4,
chunked-prefill-size 512 and no CUDA graphs. Its memory fractions were 0.35 for
Qwen and 0.55 for SmolLM2, measured against pre-load available memory. An initial
Qwen launch at 0.07 failed its memory preflight and was not counted as a successful
test. Both test processes were stopped before replacements; other services remained.

`runs.json` maps the artifacts to source versions. The frozen 40-combination
status is in `profiles/certification-matrix.json`; functional coverage is 2/20 per
engine, numerical reference failures are retained, and the release gate is false.

The test runner reads credentials locally; they are never written into evidence.
