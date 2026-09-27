# Public-model pair: retained orchestration

The `.py.txt` files preserve the exact executed Python bytes, with SHA-256 values
in the retained reports. They use explicit DSW paths and immutable commits. They
are evidence scripts, not general commands to run against an unrelated service.

- `download_verify.py.txt`: `hf download` for two pinned public checkpoints,
  followed by size/LFS SHA-256/Git blob checks for all 22 selected files.
- `run_campaign.py.txt`: four BF16 TP4 native attempts with separate directories.
  Numerical failure is retained without being confused with functional success;
  the next attempt starts only after confirmed cleanup.
- `run_precision_followup.py.txt`: two additional explicit FP32-head Qwen profiles.
  Both reference failures remain separate from default BF16 evidence.
- `tokenizer_fidelity.py.txt`: reconstructs the saved R1 prompt with the checkpoint
  tokenizer, audits original serving input IDs, and creates a separate generic HF
  profile without changing `tokenizer.json` or model weights.
- `qwen_tokenizer_fidelity.py.txt`: verifies saved Qwen input IDs against the same
  checkpoint-byte reference approach.
- `run_tokenizer_followup.py.txt`: reruns both R1 engines using the explicit
  ByteLevel profile through launcher `b28bb46` and unchanged runtime `3377c95`.
- `tokenizer_components.py.txt`: records the direct AutoTokenizer pre-tokenizer,
  decoder and sample IDs in each environment. These CPU probes are explicitly
  distinguished from each engine's native host-tokenizer behavior.
- `export_evidence.py.txt`: allowlists JSON evidence, rechecks both source trees,
  verifies no recorded owned process remains live, and collects final GPU state.
  Credentials, raw engine logs, databases and model weights are excluded.

Every GPU attempt calls the committed `run_native_validation.py`: readiness,
actual topology, typed/native checks, 1,000 switches, precision-binding rejection,
health/drain, independent CPU reference and `finally` cleanup. The controller
records a nonzero exit for numerical failure; no retry replaces that evidence.

Recompute the saved evidence checks from the repository root:

```sh
.venv/bin/python evidence/harnesses/verify_public_pair_3377c95.py
```

The verifier preserves the original vLLM tokenizer failure despite its passing
raw-token API checks. Corrected tokenizer profiles and FP32-head attempts never
increase the model/engine denominator. A successful verification means evidence
consistency, not that every reference or release gate passed.
