# Numerical reference execution profiles

Numerical evidence depends on the checkpoint, exact input IDs, backbone/readout
precision, attention implementation and device. `reference_logits.py` continues
to default to the original CPU BF16 eager reference and `atol=0.15` development
check. An additional reference profile cannot overwrite or turn an earlier
failure into a pass. The complete multi-example, dtype/quantization/TP-specific
numerical budget in the implementation plan remains required.

## Bounded CUDA reference

The optional `--device cuda --cpu-offload` mode runs Transformers computation on
one GPU while transferring module-owned leaf parameters/buffers around their
forward calls. This is test tooling, separate from the serving plugins. It uses
local checkpoint files and exact saved input/label IDs; it does not retokenize or
alter checkpoint files. KV caching is disabled. A Linear output-head hook runs
before the offload hook, so an explicitly requested FP32 readout sees the original
BF16 parameters on the device and leaves tied input embeddings unchanged.

```sh
CUDA_VISIBLE_DEVICES=7 /path/to/engine-env/bin/python \
  tests/integration/reference_logits.py \
  --model-path /path/to/verified/checkpoint \
  --contract-report /path/to/retained/contract.json \
  --output /path/to/new/reference-cuda-sdpa.json \
  --device cuda --cpu-offload --attention sdpa --atol 0.15
```

Use `--readout-dtype float32` only for a saved serving contract with that exact
readout identity. The script rejects a mismatch. An output path must be fresh.
`--attention eager` and `--attention sdpa` are distinct reference profiles; SDPA
uses PyTorch's device-specific dispatch. They are not interchangeable evidence.

Expose exactly one GPU. The default PyTorch allocator cap is 6,144 MiB, reduced
when necessary to retain at least 3,072 MiB of observed free memory. The reserve
cannot be lowered below that amount. `--gpu-budget-mib` may set a smaller cap;
allocation failure remains a failed attempt. The allocator cap does not account
for every CUDA driver allocation, so also inspect physical free memory and owned
process cleanup after the process exits. Do not launch on an occupied device
without independently checking that the remaining memory is sufficient.

Models whose stateful parent modules contain stateful children are rejected.
Ancestor code that directly reads another module's weights is unsupported and
can fail during execution. This is not a generic offload guarantee for arbitrary
Transformers architectures. New families require a resident/offload comparison.
The report records registered/visited modules, GPU UUID, allocator and peak
memory, attention/device, relevant math flags and script/input hashes.

## Validate the offload helper

`tests/integration/check_reference_offload.py` compares full logits bit for bit
between resident CUDA and offloaded CUDA. It covers tiny Qwen2 and tied-embedding
Llama models plus an explicitly supplied real checkpoint, eager and SDPA
attention, and BF16 and FP32 heads. It verifies unchanged parameter values and
weight aliasing, and injects a forward failure to check CPU-state/hook cleanup.
The supplied checkpoint must fit the resident allocator budget.

```sh
CUDA_VISIBLE_DEVICES=7 /path/to/engine-env/bin/python \
  tests/integration/check_reference_offload.py \
  --model-path /path/to/verified/small/checkpoint \
  --contract-report /path/to/retained/contract.json \
  --output /path/to/new/offload-equality.json
```

Passing this helper check validates reference execution for the tested models.
It does not establish engine parity, task quality or production performance.

## DSW observations at reference source `5730449`

The helpers ran in both existing engine environments: Torch `2.13.0+cu129`,
Transformers `5.17.0` for vLLM and `5.12.1` for SGLang. The served scores are from
the retained `3377c95` runtime runs, with the corrected R1 ByteLevel tokenizer
profile selected by launcher `b28bb46`. No engine dependency or serving plugin was
changed, and no new serving process was started in this campaign. This compares
new independent references with the saved native/plugin scores.

All **24 resident/offload full-logit comparisons** passed bit for bit: three
models × two attention implementations × two readout precisions × two Python
environments. The models were tiny Qwen2, tiny Llama with tied embeddings and the
real SmolLM2-1.7B checkpoint. Every case preserved parameters and tied-weight
identity; all six model/environment failure-injection checks restored CPU state
and removed hooks. This supports the helper on these tested implementations,
not arbitrary model families.

The matched diagnosis then ran **18 reference profile comparisons over two
distinct compiled inputs**: one Qwen2.5 answer position and one corrected R1
answer position. Nine comparisons pass and nine fail the unchanged `atol=0.15`
development check. All retain top-label agreement. The table shows maximum
absolute selected-logprob error; the first column is the earlier retained CPU
eager result, while the other three columns are the new computations.

| Saved model/readout | Engine | Prior CPU eager | CPU SDPA | GPU eager | GPU SDPA |
|---|---|---:|---:|---:|---:|
| Qwen2.5 / BF16 | vLLM | **1.000000** | 0.125000 | **0.500000** | 0.125000 |
| Qwen2.5 / BF16 | SGLang | **0.875000** | **0.250000** | **0.375000** | 0.125000 |
| Qwen2.5 / FP32 head | vLLM | **1.121746** | 0.101162 | **0.549337** | 0.126654 |
| Qwen2.5 / FP32 head | SGLang | **0.866352** | **0.184076** | **0.293943** | **0.289116** |
| R1-Llama / BF16 | vLLM | **0.160690** | **0.156250** | 0.093451 | 0.031294 |
| R1-Llama / BF16 | SGLang | 0.129480 | **0.156210** | 0.093491 | 0.031254 |

Bold values exceed 0.15. Both backbone and saved engine precision stay fixed in
each row. Qwen's attention implementation strongly affects the discrepancy on
this input; switching only the output head to FP32 did not resolve it. CPU and GPU
references are also not numerically interchangeable. This evidence does not
identify a defective engine operation or prove that the lowest-error reference is
the correct release oracle. The SGLang FP32-head profile still fails against GPU
SDPA, so it cannot be described as a blanket resolution of the earlier failures.

The generated summary also recomputes conditional-label probability differences.
For the R1 input, CPU SDPA differs from the engine by up to about 0.0364, while GPU
SDPA differs by up to about 0.00730. These are probability-point differences, not
task accuracy. R1's selected labels have very small full-vocabulary probability
mass; normalization and top-label agreement alone do not establish useful task
quality. The Qwen fixture is saturated on one label and is insufficient for
testing close decisions.

The reference allocator cap was 6 GiB with at least a 3 GiB free-memory reserve.
Maximum recorded CUDA reserved memory across the 12 large-model GPU references
was 4,370,464,768 bytes. Every campaign child exited and GPU7 returned to
11,990 MiB free after each attempt. The final audit found no live match or group
member among 107 retained owned process records and rehashed all 22 Qwen/R1
checkpoint files against their previously verified fixed revisions. The first
helper's tracked SSH command also completed with exit 0 before the campaign.

Core/plugin functionality and model coverage are unchanged: **20/40 functional
combinations, 10/20 per engine**. Original CPU eager failures and the incomplete
numerical/quality/performance gates remain in the matrix. The next numerical
gate needs a named reference execution contract, calibration and independent
held-out inputs, dtype/TP-specific budgets, near-tie cases and cache/batch-state
comparisons. These two inputs do not supply those denominators.

- [Recomputed comparison summary](../evidence/dsw/reference-diagnostics-5730449.json)
- [All raw attempts and process accounting](../evidence/dsw/reference-diagnostics-5730449/campaign.json)
- [Model rehash and final cleanup audit](../evidence/dsw/reference-diagnostics-5730449/final-audit.json)
- [Exact executed orchestration scripts](../evidence/harnesses/reference-diagnostics/README.md)
- [Local checks at the committed reference source](../evidence/local-check-5730449.json)

Recompute the summary with:

```sh
.venv/bin/python evidence/harnesses/verify_reference_diagnostics_5730449.py
```
