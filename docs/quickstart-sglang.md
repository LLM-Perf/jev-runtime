# SGLang quick start

This is the SGLang alternative to the [README quick start](../README.md#quick-start).
Run from the repository root on a Linux NVIDIA GPU host with Python 3.12. Keep the
environment separate from vLLM. The tested lane is **SGLang 0.5.19, CUDA 12.9,
Transformers 5.12.1**; newer versions are separate validation targets.

## 1. Install

If you already have that compatible SGLang environment, activate it and run:

```bash
python -m pip install 'transformers==5.12.1' -e '.[tokenizers]' -e packages/sglang
python -m pip check
```

For a fresh environment matching the tested CUDA 12.9 text-only lane, use the
[installation script](../deployment/install_sglang_cu129.sh). It checks out SGLang
v0.5.19, applies its documented CUDA 12 dependency substitutions and excludes
optional Rust extensions. It also removes upstream client, dataset, gRPC, audio,
and unused multimodal dependencies from this pinned text-only profile. Pillow and
torchvision remain because SGLang 0.5.19 imports them during ordinary server
startup. The script fails closed if the upstream dependency list changes.
It does not change the driver or system SGLang.
It downloads/builds large dependencies and needs Git, network access and a working
build toolchain.

```bash
export JEV_ROOT="$PWD/.jev/sglang-install"
export JEV_RELEASE="$PWD"
export JEV_PYTHON="$(command -v python3.12)"
bash deployment/install_sglang_cu129.sh
source "$JEV_ROOT/envs/sglang/bin/activate"
python -m pip install 'transformers==5.12.1'
python -m pip check
```

Use a fresh `JEV_ROOT`. This is a text-serving environment; multimodal and gRPC are
outside its validation scope. The installer verifies those excluded distributions
did not enter transitively. Check driver compatibility before starting the engine.

## 2. Prepare the model, tokenizer, config and keys

```bash
python examples/prepare_quickstart.py --engine sglang
source .jev/quickstart-sglang/env.sh
```

This uses the README's pinned SmolLM2 checkpoint and creates a separate tokenizer
profile and registry in this environment. For an already downloaded exact checkpoint,
add `--model-path /absolute/path/to/SmolLM2-1.7B-Instruct`. The local override does
not independently verify weight provenance.

## 3. Start the native plugin

```bash
python -m sglang.launch_server \
  --model-path "$JEV_MODEL_PATH" --served-model-name "$JEV_MODEL_ID" \
  --tokenizer-path "$JEV_TOKENIZER_PATH" \
  --host 127.0.0.1 --port "$JEV_PORT" --dtype bfloat16 \
  --context-length 2048 --max-running-requests 4 \
  --chunked-prefill-size 512 --max-total-tokens 2048 \
  --mem-fraction-static 0.8 --disable-cuda-graph --attention-backend triton
```

The generated environment sets `SGLANG_PLUGINS=jev_runtime` and `JEV_CONFIG` before
the engine imports its plugins. In the pinned 0.5.19 lane, memory budgeting uses
the available-memory baseline; this differs from vLLM's total-memory fraction.
The example assumes a dedicated GPU. Inspect available resources before sharing
a device and leave room for engine workspaces.

## 4. Send a request and update a bundle

In a second terminal, activate the **same SGLang environment**. For the fresh install
above, run from the repository root:

```bash
source .jev/sglang-install/envs/sglang/bin/activate
source .jev/quickstart-sglang/env.sh
curl --fail-with-body "$JEV_URL/ready" \
  -H "Authorization: Bearer $JEV_API_KEY"
jevctl decide examples/request.json --url "$JEV_URL"
```

Wait for `/ready` to return `"ready": true`. Use the README's
[live bundle update](../README.md#5-try-a-live-bundle-update) commands unchanged in
this terminal. `JEV_URL` selects port 30000 and includes the native plugin prefix.

Selected-token scoring here uses **zero generated completion tokens**; the vLLM
path accounts for one per scoring sequence. This difference is not an error or
evidence of a latency advantage.

Follow [quiescence and ordered shutdown](quiescence.md) before stopping the native
parent. The [engine guide](engine-integration.md) covers normalization hooks,
multi-worker integration, gateway limitations and tested scope.
