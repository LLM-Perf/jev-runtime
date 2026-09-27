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
