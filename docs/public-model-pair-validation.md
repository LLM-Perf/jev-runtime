# Qwen2.5 and DeepSeek R1 Llama: failure-inclusive native validation

Four additional engine/checkpoint combinations now have qualified functional
profiles: Qwen2.5-7B-Instruct and DeepSeek-R1-Distill-Llama-8B on SGLang and vLLM.
The fixed matrix is **20/40 functional combinations, 10/20 per engine**. The target
remains 18/20 per engine, with independent numerical, quality, performance and
lifecycle gates. No entry was removed from the denominator.

This campaign also found a tokenizer defect that the existing raw-token functional
suite alone did not catch. The original vLLM R1 run passed its API/native-logprob
and single-position CPU checks, but its input IDs differed from the checkpoint's
serialized tokenizer. That run does **not** qualify the default tokenizer profile.
The matrix selects the separately tested explicit ByteLevel profile instead.

## Exact scope and observations

Core and plugins remain at `3377c95`. Initial and FP32-head runs use that harness;
the explicit-tokenizer reruns use launcher/runner `b28bb46`. The core/plugin source
bytes did not change between those commits. Engine versions remain vLLM
`0.30.0+cu129` and SGLang `0.5.19`, with Transformers `5.17.0` and `5.12.1` respectively.
All eight attempts use BF16 backbone, native TP4, one API worker, eager execution,
context 2048 and GPUs 3,4,5,6. Each selected card had 10,792 MiB free before launch.
vLLM uses 0.075 of total device memory; SGLang uses 0.65 of pre-load free memory.
The launcher retains at least 3,072 MiB outside the requested allocation per GPU.

The public artifacts are pinned and verified against Hub metadata:

- Qwen/Qwen2.5-7B-Instruct: `a09a35458c702b33eeacc393d103063234e8bc28`,
  13 files, 15,231,271,888 weight bytes.
- deepseek-ai/DeepSeek-R1-Distill-Llama-8B: `6a6f4aa4197940add57724a7707d069478df56b1`,
  nine files, 16,060,556,354 weight bytes.

All 22 selected files match their size and LFS SHA-256 or Git blob SHA-1; a local
SHA-256 is also retained for each. No gated access was used.

Every attempt completed four typed outputs, native chat coexistence, independent
candidate scoring, precision-binding rejection checks, CAS/lifecycle checks and
1,000 bundle route switches without mixed-version responses. The traffic column
counts strict successes during those switches; it is not a throughput measurement.
Native/attached-plugin logprob difference was zero for every attempt. CPU reference
uses the saved input/label IDs, BF16 eager backbone and FP32 log_softmax; the head
precision is explicit. The original absolute tolerance remains **0.15**.

| Profile | Engine | Checkpoint tokenizer fidelity | Successful switch-traffic requests | Maximum CPU logprob error | Numerical result |
|---|---|---|---:|---:|---|
| Qwen2.5-7B / BF16 head | vllm | yes | 463 | 1.000000 | **fail** |
| Qwen2.5-7B / BF16 head | sglang | yes | 266 | 0.875000 | **fail** |
| R1-Llama-8B / original loader | vllm | **no** | 432 | 0.093750 | pass (one position) |
| R1-Llama-8B / original loader | sglang | yes | 249 | 0.129480 | pass (one position) |
| Qwen2.5-7B / FP32 head | vllm | yes | 433 | 1.121746 | **fail** |
| Qwen2.5-7B / FP32 head | sglang | yes | 244 | 0.866352 | **fail** |
| R1-Llama-8B / explicit ByteLevel | vllm | yes | 427 | 0.160690 | **fail** |
| R1-Llama-8B / explicit ByteLevel | sglang | yes | 244 | 0.129480 | pass (one position) |

Qwen2.5's explicit FP32 head does not resolve its CPU/GPU discrepancy. These
results do not localize the difference to a specific backbone operation, and no
threshold was relaxed. The corrected R1 vLLM profile also remains above tolerance.
The SGLang R1 result covers one answer position only and therefore remains
`partial` in the full numerical column. All quality/performance columns remain
`not_run` for these model combinations.

The later [reference-execution diagnosis](numerical-reference-diagnostics.md)
compares CPU/GPU and eager/SDPA on these saved inputs. It preserves all results
above and does not promote the numerical gate.

## Tokenizer mismatch and the compatible profile

The R1 checkpoint ships a ByteLevel tokenizer in `tokenizer.json` while declaring
`LlamaTokenizerFast` in `tokenizer_config.json`. Direct AutoTokenizer probes in the
installed Transformers environments produce a `LlamaTokenizer` with a Metaspace
pre-tokenizer and a different decoder. The original vLLM host path produces 85
input tokens for the saved fixture; the checkpoint tokenizer produces 83. Its
normalization/pre-tokenization/BPE fingerprint differs even though vocabulary and
chat-template fingerprints match. The actual SGLang host input matches the
checkpoint; this is distinct from the standalone AutoTokenizer probe.

The compatible artifact copies `tokenizer.json` without changing its bytes, copies
the tokenizer configuration and changes only `tokenizer_class` to
`PreTrainedTokenizerFast`. AutoTokenizer then selects `TokenizersBackend` and
preserves the serialized ByteLevel pipeline. This profile is specific to the pinned
checkpoint. Six CPU samples cover whitespace, newlines, Unicode and the complete
saved task prompt. Both native reruns use the exact same 83 input IDs and label IDs
as the checkpoint. Their saved-position cross-engine logprob difference is
0.0312099457; this is not broad cross-engine numerical certification.

`deployment/dsw_service.py` and the native validation runner now accept
`--tokenizer-path`. The launcher passes the same directory to the Jev config and
the engine (`--tokenizer` for vLLM, `--tokenizer-path` for SGLang), and records it in
the process manifest. The original model directory and weights stay intact. Bundle
fingerprints continue to bind the loaded tokenizer implementation. A tokenizer
change requires an engine rollout; bundle publication does not mutate the host
model/tokenizer while requests are active.

For this checkpoint, create a separate profile from the verified model files:

```python
import json
import shutil
from pathlib import Path

model = Path('/root/jev-runtime/models/DeepSeek-R1-Distill-Llama-8B')
profile = Path('/root/jev-runtime/tokenizers/R1-Llama-8B-bytelevel-6a6f4aa')
profile.mkdir(parents=True, exist_ok=False)
shutil.copyfile(model / 'tokenizer.json', profile / 'tokenizer.json')
config = json.loads((model / 'tokenizer_config.json').read_text())
config['tokenizer_class'] = 'PreTrainedTokenizerFast'
(profile / 'tokenizer_config.json').write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + '\n'
)
```

Supply that directory with `--tokenizer-path` when launching or validating. The
retained fidelity report contains original/derived file hashes and exact expected
IDs; verify those before treating another artifact as the tested profile.

## Cleanup, reproduction and remaining gates

All eight owned process groups exited; selected GPU UUIDs match and each returned
to 10,792 MiB free. Health checks passed with zero leases and admission tickets.
A final identity check found no live match among 87 historical owned records.
Existing unrelated services were preserved. Local checks passed 292 tests and
Ruff; no engine dependency upgrade or core/plugin rebuild was needed for the
launcher-only change.

- [Verified profiles and cross-engine diagnostics](../evidence/dsw/public-pair-validation-3377c95.json)
- [Failure-inclusive certification matrix](../profiles/certification-matrix.json)
- [Pinned model file verification](../evidence/dsw/model-verification-public-pair-3377c95.json)
- [R1 exact input and profile fidelity](../evidence/dsw/r1-llama-tokenizer-fidelity-3377c95.json)
- [Direct tokenizer-component probes](../evidence/dsw/tokenizer-component-manifest-b28bb46.json)
- [Qwen exact input fidelity](../evidence/dsw/qwen25-tokenizer-fidelity-3377c95.json)
- [89 exported artifact hashes](../evidence/dsw/public-pair-export-manifest-3377c95.json)
- [Local launcher checks](../evidence/local-check-b28bb46.json)
- [Exact orchestration and verification scripts](../evidence/harnesses/public-pair/README.md)

Run the verifier from the repository root:

```sh
.venv/bin/python evidence/harnesses/verify_public_pair_3377c95.py
```

It verifies evidence/source hashes, source attribution, original-model file
verification, saved-input fidelity, exact reference errors, lifecycle counters and
owned-process/GPU cleanup. Passing this verifier means the evidence is internally
consistent, including its recorded failures. It does not mark the full numerical,
quality, performance, multi-worker, LoRA or production-readiness gates complete.
