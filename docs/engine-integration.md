# Engine integration and validation boundaries

Development checks currently use SGLang 0.5.19 and vLLM 0.30.0 on CUDA 12.9.
SGLang 0.5.20 retired the CUDA 12 lane and remains a separate certification target.
Each engine uses its own environment. See `evidence/dsw/` for tested source commits.

Additional backends now use the `jev_runtime.backends` installed-plugin entry point.
The optional TokenSpeed package has a pinned native Engine launcher, startup
preflight, async dispatch and persistent completion receipts. It implements the complete
typed-decision API and bundle lifecycle with single-label readout. Model GPU
certification is tracked separately. See the
[TokenSpeed quick start](quickstart-tokenspeed.md) and [multi-engine limits](multi-engine.md).

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
vllm serve /absolute/path/to/model --shutdown-timeout 30
```

The config selects `backend: vllm`; its model and tokenizer identity must match the
engine. The endpoint prefix is `/plugins/jev-runtime`. Complete raw-label scores
are exposed under `/v1/scores`, and typed decisions under `/v1/decisions` beneath
that prefix. Without `JEV_CONFIG`, only the raw scoring contract is initialized.
`JEV_API_KEY` is mandatory for loading the plugin routes.
For the pinned vLLM version, zero is the native default shutdown budget and means
immediate abort. Use a positive `--shutdown-timeout` (30 seconds above) so native
API and engine processes have time to release resources after the
[Jev quiescence handshake](quiescence.md). The DSW launcher now defaults to 30 seconds
and accepts `--vllm-shutdown-timeout 1..300`; it records the exact budget. This flag
does not replace Jev drain, and does not configure a gateway or SGLang shutdown.
The native typed endpoints passed a two-API-worker DSW check with SmolLM2 at
`a283bd5`, including the publication barrier, 1,000 switches, K=32/64 and native
chat. The separate HTTP gateway currently requires one engine API worker so that
upstream external-request-ID cancellation reaches its owning OutputProcessor.

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

SGLang 0.5.19 has a selected-logprob normalization incompatibility: its producer
can return Python list rows while its consumers call `.tolist()` on every row.
The plugin uses official BEFORE hooks on the two affected consumers to preserve
host rows without changing tensor objects or numeric precision. Registration is
restricted to 0.5.19, including build suffixes. An installed-upstream CPU
reproduction and concurrent GPU serving passed at `533549a`; see the
[failure, fix and evidence](lora-crash-recovery.md). A standalone HTTP gateway
cannot repair an unmodified remote engine: mixed native/typed traffic on that
version needs the engine-side plugin or a separately validated upstream fix.

For SGLang 0.5.19 with multiple tokenizer workers, the parent plugin rewrites only
the matching Uvicorn ASGI import target to `jev_sglang.worker:app` using a registered
BEFORE hook. This installs the wrapper in every spawned worker before the host
lifespan initializes its tokenizer manager. Unrelated Uvicorn apps are unchanged.
The initial parent-only hook failed with missing plugin routes at `a283bd5`;
the new worker target passed at `c5f8867`, including two tokenizer workers,
1,000 switches, K=32/64 and native chat. A separate two-worker gateway attached
to that engine also passed its publication and switching checks. Multi-tokenizer HTTP/2
through Granian is explicitly rejected until its worker integration is validated.

Real startup, scoring and native chat coexistence have checkpoint-specific evidence
on both engines. Phi-3 mini and Phi-4 mini also passed native BF16 TP2/API1
checks at `737d814`; see [the TP2 report](phi-tp2-validation.md) for model revisions,
failed attempts and numerical limits. These checks do not certify other parallel
configurations, TP2 managed LoRA, controlled performance or full model accuracy.

## TokenSpeed native launcher and HTTP bridge

After installing the pinned TokenSpeed source and dependencies in their own
environment, run from the Jev checkout:

```sh
pip install -e '.[tokenizers]' -e packages/tokenspeed
python examples/prepare_tokenspeed.py
source .jev/quickstart-tokenspeed/env.sh
jev-tokenspeed --config "$JEV_CONFIG" --engine-config "$JEV_ENGINE_CONFIG" --check
jev-tokenspeed --config "$JEV_CONFIG" --engine-config "$JEV_ENGINE_CONFIG"
```

The current profile binds upstream `f4ac1affe11ad404720bcd150970487f75fbf59a`;
the legacy `7fa8acb1e885389825c077a6aec0326fbbbd7116` profile remains accepted.
GPU selection, source installation and exact configuration requirements are in
the [quick start](quickstart-tokenspeed.md). The native launcher owns its Engine
and exposes typed/admin/raw routes under `/plugins/jev-runtime`, with one HTTP
worker. It does not wrap an already-running TokenSpeed OpenAI service or expose chat.

Local `model_id` must match the Engine's absolute weight path, and `tokenizer`
must match its tokenizer path, which may be separate. The helper generates both
configs with these identities aligned. Requests submit asynchronously to the
Engine's owner loop; each label receives its own native request and usage entry.
`label_scoring: single` is retained across the HTTP bridge, so gateways account
for every expanded branch. For independent-candidate mode, each candidate scores
both true and false, requiring two branches.

For a separate gateway, install `jev-tokenspeed` there too. Use `backend: tokenspeed`,
set `engine_url` to the native launcher root (for example `http://127.0.0.1:8796`,
**without** the plugin prefix), and supply its API key through `JEV_ENGINE_API_KEY`.
Keep the gateway on its own listening port and local registry. The bridge appends
the scoring prefix itself; ordinary OpenAI endpoints cannot replace this contract.

Native cancellation waits for terminal one-token completion. The launcher persists
completed IDs beside the registry so the same identity can confirm completion
after response loss or restart; unknown/pending IDs stay unconfirmed. See
[receipt storage and recovery limits](operations.md#tokenspeed-operations).
CPU tests verify these paths; no GPU throughput or complete crash-recovery claim
follows from them. [Actual validation](tokenspeed-validation.md).

## Bundle updates

Build a new immutable version, upload, prepare (runs real scoring), then activate
using the expected current generation. Rollback uses `activate` on the retained
prior version with the new expected generation. Disable stops new requests;
retire rejects versions with active routes or leases.

The registry supports multiple API processes using one **local** SQLite file.
It does not yet claim multi-node rollout support. Failed or unconfirmed remote
cancellations retain a lease. After a process crash, orphan recovery must confirm
engine cancellation before releasing the lease; automatic timeout is not proof.
