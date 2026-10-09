# TokenSpeed quick start

Run Jev's Choice, Boolean, Score and Rank API on a pinned TokenSpeed native Engine.
The plugin adds bundle prepare/activate/rollback and a raw-scoring HTTP bridge.
It does not modify TokenSpeed's source or add a chat proxy.

**Status: experimental.** The current source profile is
[`f4ac1affe11ad404720bcd150970487f75fbf59a`](https://github.com/lightseekorg/tokenspeed/tree/f4ac1affe11ad404720bcd150970487f75fbf59a).
The previous `7fa8acb1e885389825c077a6aec0326fbbbd7116` profile remains accepted.
Source checks and CPU contracts do not certify GPU inference. TokenSpeed still has
**0/20 GPU-validated model cases** in this repository.

## 1. Install the engine in its own environment

The Jev launcher accepts NVIDIA `sm90`, `sm100`, `sm103`, or `sm107`, matching the
current pinned upstream hardware scope. **L20/L20Z, L40 and A100 are outside this
profile.** Other hardware, AMD, quantized weights, TP/DP/PP > 1, speculative
decoding, PD and managed LoRA are separate integration targets.

For **H100/H200, Linux, Python 3.11 and an installed CUDA Toolkit 12.9**, the
pinned upstream provides an
[experimental source installer](https://github.com/lightseekorg/tokenspeed/blob/f4ac1affe11ad404720bcd150970487f75fbf59a/docs/guides/hopper-cu129.md).
It builds native components and needs a compatible driver, C++ compiler, OpenSSL
development files and network access. The recipe uses Torch 2.14.0+cu126 with
CUDA Toolkit 12.9 for native builds; it is not a generic CUDA 12.9 wheel install.
On Blackwell, use the same commit's normal upstream installation instructions
and matching CUDA/kernel stack instead of this Hopper recipe.

From the Jev repository root, with `uv` installed:

```bash
export JEV_REPO="$PWD"
export JEV_TS_SOURCE="$JEV_REPO/.jev/tokenspeed-source"
mkdir -p "$JEV_REPO/.jev"
git clone https://github.com/lightseekorg/tokenspeed.git "$JEV_TS_SOURCE"
git -C "$JEV_TS_SOURCE" checkout --detach f4ac1affe11ad404720bcd150970487f75fbf59a
cd "$JEV_TS_SOURCE"
uv venv .venv-cu129 --python 3.11 --seed
source .venv-cu129/bin/activate
export CUDA_HOME=/usr/local/cuda-12.9
CUDA_VARIANT=cu129 bash test/ci_system/install_deps.sh
cd "$JEV_REPO"
python -m pip install -e '.[tokenizers]' -e packages/tokenspeed
python -m pip check
jevctl backend list
jev-tokenspeed --check-source "$JEV_TS_SOURCE/python/tokenspeed"
```

Use a fresh checkout and environment; set `CUDA_HOME` to the actual toolkit path.
Installing `jev-tokenspeed` alone installs the adapter, **not TokenSpeed or CUDA**.
The source-only check can also run on a CPU machine without importing TokenSpeed.

## 2. Prepare a model, separate tokenizer, config and credentials

```bash
python examples/prepare_tokenspeed.py
source .jev/quickstart-tokenspeed/env.sh
```

The helper downloads **Qwen/Qwen3-0.6B** at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, preserves its tokenizer and creates:

- `config.json`: Jev configuration and bootstrap alias `decision-model`.
- `engine.json`: the native TokenSpeed configuration.
- `env.sh`: separate API/admin credentials, readable by the current user only.
- A private run directory for the registry and native completion receipts.

It refuses to overwrite an existing output directory. For an already downloaded
exact checkpoint, add `--model-path /absolute/path/to/Qwen3-0.6B`. The offline
override checks required files, not weight provenance. To use a different model,
prepare its own immutable revision and tokenizer and edit both configurations;
the helper's default checkpoint is a validation target, not a GPU-certified model.

The Jev `model_id` must equal Engine `model`; both use the **absolute weight path**
for local weights. Jev `tokenizer` must equal Engine `tokenizer`; the generated
configuration correctly uses the separate preserved tokenizer path. The public
request still uses `"model": "decision-model"`.

## 3. Preflight and start

```bash
export CUDA_VISIBLE_DEVICES=0
jev-tokenspeed --config "$JEV_CONFIG" --engine-config "$JEV_ENGINE_CONFIG" --check
jev-tokenspeed --config "$JEV_CONFIG" --engine-config "$JEV_ENGINE_CONFIG"
```

Select an available supported GPU before running. The example uses one GPU,
BF16, eager execution, `triton_full`, output logprobs, context 4096, at most 16
native sequences and a 0.5 memory fraction. Adjust capacity to the target host.
`--check` checks source, visible GPU and resolved configuration without starting
the Engine or loading weights; it does not certify kernels, free VRAM or serving.
Actual startup validates the tokenizer, model precision and bootstrap canary.

There is one HTTP worker. Native requests run on the Engine's owner event loop;
waiting for GPU results does not reserve a thread in the HTTP executor pool.

## 4. Call the API and update bundles

In another terminal, activate the same environment and run from the Jev root:

```bash
source .jev/tokenspeed-source/.venv-cu129/bin/activate
source .jev/quickstart-tokenspeed/env.sh
curl --fail-with-body "$JEV_URL/ready" \
  -H "Authorization: Bearer $JEV_API_KEY"
jevctl decide examples/request.json --url "$JEV_URL"
```

Wait for `/ready` to return `"ready": true`. Use the README's
[live bundle update](../README.md#5-try-a-live-bundle-update) commands unchanged.
Python and TypeScript SDKs use the same `JEV_URL`. Bundle activation and rollback
are live; replacing plugin code, engine binaries or model weights needs a restart.

For a separate gateway, install this adapter there too, set `backend: tokenspeed`,
point `engine_url` to `http://127.0.0.1:8796`, and set `JEV_ENGINE_API_KEY` to the
native plugin API key. Use a different gateway listening port, registry and API/admin
credentials. An ordinary TokenSpeed OpenAI endpoint does not implement this scoring
contract and cannot serve as the bridge target.

## 5. Run strict functional validation

On a dedicated test instance started above, the shared live harness supports the
TokenSpeed launcher and reads the generated API/admin environment variables:

```bash
export JEV_TEST_COMMIT="$(git rev-parse HEAD)"
python -m tests.integration.live_contract \
  --run-dir "$JEV_RUN" --output "$JEV_RUN/contract.json" \
  --source-commit "$JEV_TEST_COMMIT" --runtime-source-commit "$JEV_TEST_COMMIT" \
  --minimum-requests 10000 --switches 1000 \
  --traffic-concurrency 4 --traffic-timeout 1800
python -m tests.integration.verify_traffic_evidence \
  --responses "$JEV_RUN/traffic-responses.jsonl.gz" --contract "$JEV_RUN/contract.json"
```

The harness creates temporary test bundles, exercises all four types and both
readout modes, verifies bridge cancellation after completion, and retains each
traffic request/response. It requires both request and switch counts, checks
identity, generation, probabilities and exact native sequence/completion usage,
and never retries failures into successes. A failed run retains its report and
may leave its test bundles active for diagnosis; use a fresh output directory for
each subsequent attempt. It never starts or stops the engine. Record the deployed
source commit separately if the harness and runtime differ.

These are functional checks. Bridge parity compares two clients of the same
backend, not an independent numerical reference. For an independent development
check, run the command below. It compares one saved answer position/label with
Transformers on CPU; the manifest records the configured checkpoint identity,
not a weight checksum. Wider numerical certification, task quality, performance
and soak testing remain separate. No real TokenSpeed GPU run of this
10,000-request test is claimed in this change.

```bash
python tests/integration/reference_logits.py \
  --model-path "$JEV_MODEL_PATH" --model-manifest "$JEV_RUN/model-source.json" \
  --contract-report "$JEV_RUN/contract.json" \
  --output "$JEV_RUN/reference-cpu.json" --device cpu --readout-dtype model
```

## Readout, drain and recovery boundaries

TokenSpeed's pinned input path rejects requested selected-token ID lists. The
adapter forces one candidate with a finite BF16-exact bias, then checks that the
actual sampled token matches and reads its **pre-bias raw logprob**. Missing,
nonfinite, positive, mismatched or nonterminal outputs fail explicitly.

For K labels, this means **K native requests and K generated completion tokens**.
Independent-candidate mode uses two native requests per candidate (true/false).
All branches and prompt usage count toward admission and billing. Prefix caching
may reuse work; no single-prefill gather, zero-decode or performance advantage is
claimed. Efficient joint selected-ID readout still needs upstream execution and
output support plus GPU validation.

Cancellation waits up to 4.5 seconds for native one-token completion. Upstream
frontend abort does not prove scheduler drain. The launcher now stores reservations
and completion receipts in `registry.tokenspeed-receipts.db` beside `registry.db`:

- A persisted terminal receipt confirms an already completed request after HTTP
  response loss, cache eviction, or a restart with the same model/engine identity.
- Unknown or pending requests still fail closed and retain their journal/lease;
  this is not full crash recovery or a native scheduler abort acknowledgement.
- Request IDs cannot be reused in the same receipt namespace. The namespace binds
  model/tokenizer/template identity, source profile, engine options and engine URL.
- Retain this database together with the registry, on local durable storage.
  Rows have no automatic expiry, so budget and monitor disk usage. Do not delete
  receipts while old gateway journals may still ask for cancellation. Full-sync
  SQLite writes have a cost; throughput has not been benchmarked.

Follow [quiescence and ordered shutdown](quiescence.md). A failed drain remains an
error; the launcher still shuts down its own Engine in `finally` and keeps pending
evidence. Source checks cover 24 critical files on the current profile (13 on the
legacy profile), not all kernels, dependencies or weights.

| Startup/operation failure | Action |
|---|---|
| `tokenspeed_source_mismatch` | Restore one of the pinned source profiles; do not bypass the hash guard |
| `tokenspeed_hardware` | Select a supported NVIDIA GPU; the existing L20Z DSW cannot validate this lane |
| Model or tokenizer mismatch | Align both configs, including the preserved tokenizer path and revision |
| `tokenspeed_profile` | Use the documented eager, single-GPU raw-scoring configuration |
| `duplicate_request` | Generate a fresh request ID; completed IDs are not reusable |
| `cancellation_unconfirmed` | Preserve journal/receipts and inspect native completion; do not treat it as successful drain |

See [validation evidence](tokenspeed-validation.md) for what actually ran.
