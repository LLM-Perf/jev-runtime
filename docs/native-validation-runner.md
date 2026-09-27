# Reproducible native validation

`tests/integration/run_native_validation.py` runs one isolated SGLang or vLLM
attempt using the selected engine environment's Python. It requires a new run
directory and full runtime/harness commit IDs. Upload and verify that immutable
harness checkout before running; the declared commit alone is not verification.
The existing `deployment/dsw_service.py` launcher requires a pinned local model
with `jev-source.json` and checks every selected GPU's memory budget and reserve.

Example for a colocated BF16 TP4 development check:

```bash
ENGINE_PYTHON=/root/jev-runtime/envs/vllm/bin/python
HARNESS_SHA=b8742bd0d96dc98de7bf9a7860de407508043c39
RUNTIME_SHA=f6bda1a13e30b1d58cb4303455cce92c1601a683
"$ENGINE_PYTHON" \
  "/root/jev-runtime/releases/$HARNESS_SHA/tests/integration/run_native_validation.py" \
  --engine vllm --model-path /root/jev-runtime/models/Qwen3-8B \
  --run-dir /root/jev-runtime/runs/choose-a-new-attempt-name \
  --gpus 3,4,5,6 --port 18795 --memory-fraction 0.065 \
  --reserve-mib 3072 --switches 1000 \
  --source-commit "$HARNESS_SHA" --runtime-source-commit "$RUNTIME_SHA"
```

The GPU list, port and memory fraction are examples tied to one observed DSW
allocation. Recheck current capacity before using them. vLLM's fraction uses total
device memory; this SGLang launch profile uses available memory. The list order is
the CUDA-visible order, and its length sets tensor parallelism. TP1, TP2 and TP4
are expressible; this does not certify every model on each topology.

The runner launches once, checks the recorded PID/start-tick/boot identity while
waiting for `/ready`, and fails early if that process exits. It never restarts a
service after a timeout. Topology evidence includes the live parent's CUDA mapping,
SGLang server parallel configuration or vLLM live rank-labelled workers in the owned
process group. NVML can use a different PID namespace; no worker-to-NVML mapping
is inferred. vLLM's log evidence is version-specific and fails closed when a
future log format no longer matches.

After topology verification it runs the four-output-type, native/attach parity,
bundle lifecycle and 1,000-switch contract, then legacy/wrong-readout rejection
checks. The postcheck requires zero leases and admission counters, healthy active
bundles and runtime/plugin import paths matching the runtime source checkout.
Finally an independent CPU Transformers forward checks the saved answer position.
The default absolute logprob tolerance of 0.15 is a development diagnostic, not a
release accuracy budget. A numerical mismatch remains separate from a functional
pass in `validation.json`; the overall command exits unsuccessfully if either
the reference check or cleanup fails.

Every path exits through owned-process cleanup after a process record exists,
including test failures and ordinary Python interruptions. `SIGKILL`, host loss
and failure before process identity capture cannot be cleaned automatically.
The stop operation verifies ownership before signaling; the runner then waits
for every non-zombie member of that process group to exit and records all selected
GPU UUIDs/free-memory snapshots. It does not kill an unidentified orphan or assert
that changing free memory belongs to this job. A missing record, surviving group,
changed identity or GPU UUID makes cleanup uncertified. Inspect and resolve it
before starting another allocation.

Keep failed attempts. Export only an explicit evidence allowlist: validation,
contract, precision-binding, profile, postcheck, topology, environment, CPU
reference and cleanup JSON. Never publish `keys.json`, unreviewed engine logs,
credential-bearing configuration or model weights. Reports establish the exact
tested model/revision/engine/profile and do not replace quality, controlled
performance, long-duration soak or broader LoRA certification.
