# Engine integration and validation boundaries

Target releases: SGLang 0.5.20 and vLLM 0.30.0. These are implementation targets,
not yet a certified GPU compatibility claim. Each engine uses its own environment.

## Gateway

Install core with its tokenizer extra, copy `examples/gateway.yaml`, and replace
the model revision with the immutable revision of the actual loaded weights.

```sh
pip install -e '.[tokenizers]'
jevctl inspect gateway.yaml
jevctl serve gateway.yaml
jevctl decide examples/request.json
```

SGLang attach uses `/generate` with zero output tokens and explicit token logprobs.
It verifies that the response contains exactly one score position and every label.
The vLLM attach adapter currently uses the plugin's complete-label scoring contract.
It never treats the built-in first-label scalar response as a whole distribution.

## vLLM native plugin

```sh
pip install -e '.[tokenizers]' -e packages/vllm
export VLLM_PLUGINS=jev_runtime_api
# Set JEV_API_KEY using your secret manager, not a checked-in file.
export JEV_CONFIG=/absolute/path/to/vllm-native.yaml
vllm serve /absolute/path/to/model
```

The config selects `backend: vllm`; its model and tokenizer identity must match the
engine. The endpoint prefix is `/plugins/jev-runtime`. Complete raw-label scores
are exposed under `/v1/scores`, and typed decisions under `/v1/decisions` beneath
that prefix. Without `JEV_CONFIG`, only the raw scoring contract is initialized.
`JEV_API_KEY` is mandatory for loading the plugin routes.

## SGLang native plugin

```sh
pip install -e '.[tokenizers]' -e packages/sglang
export SGLANG_PLUGINS=jev_runtime
export JEV_CONFIG=/absolute/path/to/sglang-native.yaml
# Set JEV_API_KEY using your secret manager.
python -m sglang.launch_server --model-path /absolute/path/to/model
```

The registered hook attaches API routes at global-state initialization and wraps
the existing API lifespan. The separate `jev-sglang` launcher offers a startup
integration path that installs the routes before invoking the upstream launcher.
Neither path replaces a scheduler or modifies installed engine source files.

Both paths still require real GPU startup and shutdown verification for each
supported engine version. Registration discovery alone does not prove this works.

## Bundle updates

Build a new immutable version, upload, prepare (runs real scoring), then activate
using the expected current generation. Rollback uses `activate` on the retained
prior version with the new expected generation. Disable stops new requests;
retire rejects versions with active routes or leases.

The registry supports multiple API processes using one **local** SQLite file.
It does not yet claim multi-node rollout support. Failed or unconfirmed remote
cancellations retain a lease. After a process crash, orphan recovery must confirm
engine cancellation before releasing the lease; automatic timeout is not proof.
