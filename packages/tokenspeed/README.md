# Jev Runtime for TokenSpeed

Turn a pinned TokenSpeed native Engine into a **Choice, Boolean, Score and Rank**
API with probabilities and live decision-bundle updates. This optional adapter
uses [Jev Runtime](https://github.com/LLM-Perf/jev-runtime)'s shared typed API,
admission, health and bundle lifecycle.

**Experimental:** CPU contracts and packaging are tested; real TokenSpeed GPU
model serving and performance have not been validated. GPU coverage remains 0/20.

## Install and start

First follow the [TokenSpeed quick start](https://github.com/LLM-Perf/jev-runtime/blob/main/docs/quickstart-tokenspeed.md)
to install upstream commit `f4ac1affe11ad404720bcd150970487f75fbf59a` and its native
dependencies in a separate environment. The launcher accepts NVIDIA sm90, sm100,
sm103 or sm107. L20Z is outside this profile. The legacy upstream source profile
`7fa8acb1e885389825c077a6aec0326fbbbd7116` remains accepted.

From the Jev repository root, in that activated environment:

```bash
python -m pip install -e '.[tokenizers]' -e packages/tokenspeed
python examples/prepare_tokenspeed.py
source .jev/quickstart-tokenspeed/env.sh
export CUDA_VISIBLE_DEVICES=0
jev-tokenspeed --config "$JEV_CONFIG" --engine-config "$JEV_ENGINE_CONFIG" --check
jev-tokenspeed --config "$JEV_CONFIG" --engine-config "$JEV_ENGINE_CONFIG"
```

The helper prepares a pinned Qwen3-0.6B checkpoint, a preserved tokenizer, matching
configs, a private run directory and separate API/admin keys. This package does
not install TokenSpeed or CUDA; the commands do not assume a published PyPI release.

In a second terminal, activate the same environment and source the same `env.sh`:

```bash
curl --fail-with-body "$JEV_URL/ready" \
  -H "Authorization: Bearer $JEV_API_KEY"
jevctl decide examples/request.json --url "$JEV_URL"
```

Wait for `"ready": true`. The default base URL is
`http://127.0.0.1:8796/plugins/jev-runtime`; the request alias is `decision-model`.
Use the shared [bundle update commands](https://github.com/LLM-Perf/jev-runtime#5-try-a-live-bundle-update)
to prepare, activate or roll back a version while requests retain their original bundle.

## Included and remaining scope

- Source/hardware/config preflight and native Engine owner-loop submission.
- Typed, admin and raw-scoring APIs; an optional separate Jev HTTP gateway.
- Durable completion receipts for response loss and confirmation of already completed
  requests after restart with the same model/engine identity.
- Strict traffic tooling for four types and live bundle switching, including exact
  per-label branch and token accounting.

K labels still require K native requests. Efficient joint selected-ID readout,
scheduler-confirmed abort, unresolved-request crash recovery and GPU certification
remain open. The launcher exposes one HTTP worker, eager execution and TP/DP/PP=1;
it does not add native chat, managed LoRA, quantization, PD or speculative decoding.

Keep the registry and its separate completion-receipt database together. Unknown
or pending requests fail closed; elapsed time does not prove drain. Bundle updates
are live, while plugin code, engine binaries and base model changes require restart.
See [operations](https://github.com/LLM-Perf/jev-runtime/blob/main/docs/operations.md#tokenspeed-operations-experimental)
and the [validation report](https://github.com/LLM-Perf/jev-runtime/blob/main/docs/tokenspeed-validation.md).
