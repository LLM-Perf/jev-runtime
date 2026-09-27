# Executed numerical-diagnosis scripts

The `.py.txt` files retain the exact bytes executed on DSW:

- `exploratory-probe.py.txt`: initial eight-way Qwen2.5 CPU/CUDA,
  eager/SDPA and BF16/FP32-head diagnosis. Its baseline is the original BF16
  serving report. Consequently, the raw FP32-head `abs_engine_errors` field is
  a cross-precision observation and is **excluded** from matched-profile
  comparisons. The subsequent committed harness enforces readout identity.
- `campaign.py.txt`: the second environment's helper check followed by all 18
  matched reference profiles. It records process identity, nonzero numerical
  exits, per-attempt GPU state and source hashes. A numerical failure does not
  stop the campaign, but failed cleanup or unexpected return codes do.
- `audit-export.py.txt`: after the campaign completes, rehashes all 22 pinned
  Qwen/R1 artifacts, checks historical owned identities and groups, and exports
  only the allowlisted structured evidence. It retains a separate export manifest.

The first helper check was run directly through the tracked SSH command and
completed with exit 0 before starting the campaign. Both helper runs use the
committed `tests/integration/check_reference_offload.py` at `5730449`.
Core/plugins stay at `3377c95`; no serving engine was started by this campaign.
Engine logprobs and exact input IDs come from the preserved earlier GPU runs.

Run `evidence/harnesses/verify_reference_diagnostics_5730449.py` to verify local
evidence/source/input hashes, recalculate numerical outcomes and probability
differences, and regenerate the summary. This consistency check intentionally
preserves failed profiles and does not pass a release gate.
