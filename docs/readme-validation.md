# README quick-start validation

On 2026-09-30, the vLLM and SGLang native startup paths passed a DSW smoke check using
the new `examples/prepare_quickstart.py`. This validates the user workflow separately
from the larger [strict traffic campaign](strict-traffic-validation.md).

The README now also includes TokenSpeed setup. That path is outside
this historical GPU smoke check: its current evidence is the
[2026-10-09 CPU/source validation](tokenspeed-validation.md), with real TokenSpeed
GPU serving still pending. The saved vLLM/SGLang launch commands remain unchanged.

## What ran

- Preparation in each engine's Python 3.12 environment, using the offline model
  override for SmolLM2-1.7B-Instruct revision
  `31b70e2e869a7173562077fd711b654946d38674`.
- A newly generated tokenizer profile, config, registry location and independent
  API/admin keys for each engine. Secret files had mode `0600`; repeating preparation
  against an existing output directory was rejected.
- Startup commands extracted directly from the README/SGLang guide. The only
  command substitution was memory fraction: vLLM `0.8 → 0.07`, SGLang `0.8 → 0.65`
  for shared L20Z GPUs. Environment settings selected GPU 7/6 and ports 18795/18794.
- Readiness, bundle listing, the example decision, remote bundle build, upload,
  prepare, activation, another decision, rollback and a third decision. A fourth
  decision used the HTTP endpoint directly. Each request had two questions.
- Backend quiescence, signaling only the owned native parent, checking process-group
  exit, GPU memory restoration and the pre-existing service's PID/start identity
  and command hash.

| Check | vLLM | SGLang |
|---|---|---|
| Engine | 0.30.0+cu129 | 0.5.19 |
| Transformers | 5.17.0 | 5.12.1 |
| Torch | 2.13.0+cu129 | 2.13.0+cu129 |
| Typed decision responses validated | 4/4 | 4/4 |
| Bundle sequence | `default@1 → demo@2 → default@1` | `default@1 → demo@2 → default@1` |
| Generation sequence | `1 → 2 → 3` | `1 → 2 → 3` |
| Quiescence drained / pending work | true / 0 | true / 0 |
| Native parent exit code | 0 | 0 |
| Remaining owned process-group members | 0 | 0 |
| Free GPU memory before / after | 11,744 / 11,744 MiB | 10,544 / 10,544 MiB |

Both first responses selected `billing` and `true`. This is a single example, not
an accuracy evaluation. All eight responses pass the runtime schema, engine/type
and candidate checks. The local bundle CLI, API and tokenizer-profile regression
selection passes **17 tests**. Both existing engine environments pass `pip check`.

SGLang retains two shutdown tracebacks (`SystemExit: 0` and `CancelledError`) and
an NCCL `destroy_process_group` warning. The owned group exited and memory returned;
the result does not claim warning-free shutdown or universal signal safety.

## Reproduction and limits

The runtime source is `9da181342326793f6edddbd3f3ada4b0f7dbb320`, selected with
explicit Python import paths in the existing DSW environments. This README change
does not modify runtime code. Model files and engine dependencies were reused;
cloning, a fresh pip installation, Hub downloads, the dedicated-GPU `0.8` memory
setting and installation on another machine were **not rerun**. The vLLM wheel URL
was checked against the official v0.30.0 release assets.

The frozen Markdown snapshots preserve the exact command blocks used; subsequent
README edits add comparison/verification prose without changing those launch blocks.
Raw responses, launch commands, process records, configs and checked logs are in
[the evidence directory](../evidence/dsw/readme-quickstart-20260930).
Its [manifest](../evidence/dsw/readme-quickstart-20260930/manifest.json) covers 12 files.
Export scanned all included files against the four generated credentials; `env.sh`
and other key files are excluded. The recorded harness uses the explicit test-host
paths; the public-facing commands are in the README.

Verify artifact hashes, current command blocks and all eight response contracts:

```bash
.venv/bin/python evidence/harnesses/verify_readme_quickstart_20260930.py
```
